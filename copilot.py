"""
Technician copilot: "Belt loader BLR-BL-007 is leaking hydraulic oil.
What do I do, and is the part in stock?"

Retrieval-augmented generation (RAG):
  1. search the maintenance procedures (TF-IDF retrieval over manual sections)
  2. check the spare part's stock at the unit's station (and other stations)
  3. look up the unit's recent repair history
  4. answer with the procedure, the part status and a draft work order

Two answerers produce the same answer format:
  - ClaudeCopilot: Claude with read-only tools (needs an API key; local use)
  - rule_based_answer: retrieval only, no AI, no key (runs in the browser demo)

The draft work order always goes to a supervisor for approval.
"""
import json
import os
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from gse import FAILURE_MODES, PARTS

BASE = Path(__file__).parent
MODEL = os.environ.get("COPILOT_MODEL", "claude-opus-5-5")


class Fleet:
    """Read-only access to the fleet data (equipment, repairs, stock)."""

    def __init__(self, data_dir=BASE / "data"):
        self.equipment = pd.read_csv(data_dir / "equipment.csv")
        self.work_orders = pd.read_csv(data_dir / "work_orders.csv", keep_default_na=False)
        self.work_orders["stockout"] = self.work_orders["stockout"].astype(str).eq("True")
        self.parts = pd.read_csv(data_dir / "parts.csv")

    def unit(self, unit_id):
        rows = self.equipment[self.equipment["unit_id"] == unit_id]
        if rows.empty:
            raise ValueError(f"Unknown unit {unit_id}")
        return rows.iloc[0]

    def check_stock(self, part_no, station):
        if part_no not in PARTS:
            raise ValueError(f"Unknown part {part_no}")
        p = self.parts[self.parts["part_no"] == part_no]
        here = p[p["station"] == station].iloc[0]
        others = p[(p["station"] != station) & (p["on_hand"] > 0)]
        return {
            "part_no": part_no, "description": PARTS[part_no][0], "station": station,
            "on_hand": int(here["on_hand"]), "reorder_point": int(here["current_rop"]),
            "lead_time_days": int(here["lead_time_days"]),
            "other_stations_with_stock": {r.station: int(r.on_hand) for r in others.itertuples()},
        }

    def unit_history(self, unit_id, n=5):
        u = self.unit(unit_id)
        wos = self.work_orders[self.work_orders["unit_id"] == unit_id].tail(n)
        return {
            "unit_id": unit_id, "type": u["type_name"], "station": u["station"],
            "age_years": int(u["age_years"]), "days_since_pm": int(u["days_since_pm"]),
            "recent_repairs": [
                {"date": w["Breakdown reported"][:10], "failure": w["failure_desc"], "part": w["part_no"],
                 "downtime_h": w["downtime_h"]} for _, w in wos.iterrows()
            ],
        }


class Manuals:
    """Splits each procedure into sections and indexes them with TF-IDF."""

    def __init__(self, folder=BASE / "manuals"):
        self.chunks = []
        for path in sorted(folder.glob("*.md")):
            text = path.read_text()
            title = text.splitlines()[0].lstrip("# ").strip()
            codes = re.search(r"Applies to: (.+)", text).group(1).replace(" ", "").split(",")
            for section in text.split("\n## ")[1:]:
                head, _, body = section.partition("\n")
                self.chunks.append({"doc": path.stem, "title": title, "codes": codes,
                                    "section": head.strip(), "text": body.strip()})
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform([f"{c['title']} {c['section']} {c['text']}" for c in self.chunks])

    def search(self, query, k=3, codes=None):
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        ranked = sorted(range(len(self.chunks)), key=lambda i: -scores[i])
        hits = [dict(self.chunks[i], score=round(float(scores[i]), 3)) for i in ranked
                if codes is None or set(self.chunks[i]["codes"]) & set(codes)]
        return hits[:k]

    def procedure(self, doc):
        return {c["section"]: c["text"] for c in self.chunks if c["doc"] == doc}


def rule_based_answer(question, unit_id, fleet: Fleet, manuals: Manuals):
    """Retrieval-only answer: no AI model, no API key."""
    unit = fleet.unit(unit_id)
    type_modes = [code for code, m in FAILURE_MODES.items() if m[0] == unit["type_code"]]
    hits = manuals.search(question, k=3, codes=type_modes)
    if not hits:
        raise ValueError("No procedure found for this unit type")
    doc = hits[0]["doc"]
    # A procedure can cover several faults: pick the one whose description best matches the question
    candidates = [c for c in hits[0]["codes"] if c in type_modes]
    labels = [f"{FAILURE_MODES[c][1]} {PARTS[FAILURE_MODES[c][2]][0]}" for c in candidates]
    sims = cosine_similarity(manuals.vectorizer.transform([question]), manuals.vectorizer.transform(labels))[0]
    code = candidates[int(sims.argmax())]
    _, desc, part, _, repair_h = FAILURE_MODES[code]
    stock = fleet.check_stock(part, unit["station"])
    steps = manuals.procedure(doc)

    if stock["on_hand"] > 0:
        part_line = f"{part} ({stock['description']}): {stock['on_hand']} in stock at {unit['station']}."
    elif stock["other_stations_with_stock"]:
        src, qty = next(iter(stock["other_stations_with_stock"].items()))
        part_line = (f"{part} is OUT OF STOCK at {unit['station']}. {src} has {qty}: request a stock "
                     f"transfer before an emergency buy (normal lead time {stock['lead_time_days']} days).")
    else:
        part_line = (f"{part} is OUT OF STOCK at all stations: raise an emergency purchase now "
                     f"(normal lead time {stock['lead_time_days']} days).")

    return {
        "answerer": "Retrieval only (no AI)",
        "failure_code": code, "failure_desc": desc,
        "procedure": steps, "part_status": part_line, "stock": stock,
        "draft_work_order": (f"Unit {unit_id} ({unit['type_name']}, {unit['station']}). Reported: {question.strip()} "
                             f"Likely failure {code} - {desc}. Part {part}. Estimated repair {repair_h:.1f} h. "
                             f"Follow procedure '{hits[0]['title']}'."),
        "sources": [f"{h['title']} › {h['section']} (score {h['score']})" for h in hits],
        "trace": [],
    }


SYSTEM_PROMPT = """You are a maintenance copilot for ground-support-equipment technicians at an \
airport ground-handling station. A technician describes a fault on a unit.

Use the tools: look up the unit, search the maintenance procedures, and check stock for the \
part the procedure names. Then call submit_answer once.

Rules:
- Base the procedure ONLY on the procedure sections you retrieved, and cite them in sources. \
Never invent steps, torque values or limits; if the manuals don't cover it, say so.
- Always include the safety steps first.
- If the part is out of stock locally, say which other station has it, or that an emergency \
purchase is needed.
- The draft work order is checked by a supervisor; you cannot release a unit or order parts."""

TOOLS = [
    {"name": "unit_history", "description": "Type, station, age, days since PM and recent repairs of a unit.",
     "input_schema": {"type": "object", "properties": {"unit_id": {"type": "string"}},
                      "required": ["unit_id"], "additionalProperties": False}, "strict": True},
    {"name": "search_manuals", "description": "Search the maintenance procedures. Returns the best-matching sections with their text.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}},
                      "required": ["query"], "additionalProperties": False}, "strict": True},
    {"name": "check_stock", "description": "Stock of a spare part at a station, plus other stations that have it.",
     "input_schema": {"type": "object", "properties": {"part_no": {"type": "string", "enum": sorted(PARTS)},
                                                       "station": {"type": "string"}},
                      "required": ["part_no", "station"], "additionalProperties": False}, "strict": True},
    {"name": "submit_answer", "description": "Submit the final answer for the technician. Call once, at the end.",
     "input_schema": {"type": "object", "properties": {
         "failure_code": {"type": "string"}, "failure_desc": {"type": "string"},
         "procedure_summary": {"type": "string", "description": "Safety, diagnosis, repair and test steps from the retrieved sections."},
         "part_status": {"type": "string"},
         "draft_work_order": {"type": "string"},
         "sources": {"type": "array", "items": {"type": "string"}}},
         "required": ["failure_code", "failure_desc", "procedure_summary", "part_status", "draft_work_order", "sources"],
         "additionalProperties": False}, "strict": True},
]


class ClaudeCopilot:
    def __init__(self, fleet: Fleet, manuals: Manuals, client=None, model=MODEL, on_step=None):
        import anthropic  # imported here so the browser demo never needs it

        self.fleet, self.manuals = fleet, manuals
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.on_step = on_step or (lambda text: None)
        self.trace = []

    def _run(self, name, args):
        if name == "unit_history":
            return self.fleet.unit_history(args["unit_id"])
        if name == "search_manuals":
            return [{k: h[k] for k in ("title", "section", "text")} for h in self.manuals.search(args["query"], k=3)]
        if name == "check_stock":
            return self.fleet.check_stock(args["part_no"], args["station"])
        raise ValueError(f"Unknown tool {name}")

    def answer(self, question, unit_id):
        messages = [{"role": "user", "content": f"Unit {unit_id}: {question}"}]
        for _ in range(10):
            response = self.client.beta.messages.create(
                model=self.model, max_tokens=16000, system=SYSTEM_PROMPT, tools=TOOLS,
                messages=messages, output_config={"effort": "medium"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            )
            if response.stop_reason in ("refusal", "max_tokens"):
                raise RuntimeError(f"Stopped: {response.stop_reason}")
            messages.append({"role": "assistant", "content": response.content})
            calls = [b for b in response.content if b.type == "tool_use"]
            if not calls:
                messages.append({"role": "user", "content": "Please call submit_answer with your answer."})
                continue
            results = []
            for call in calls:
                if call.name == "submit_answer":
                    self.trace.append({"tool": call.name})
                    out = dict(call.input)
                    out.update(answerer=f"Claude ({self.model})", procedure=None, trace=self.trace)
                    return out
                self.on_step(f"{call.name}({json.dumps(call.input)})")
                try:
                    content, err = json.dumps(self._run(call.name, call.input)), False
                except Exception as exc:
                    content, err = f"Error: {exc}", True
                self.trace.append({"tool": call.name, "input": call.input})
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": content, "is_error": err})
            messages.append({"role": "user", "content": results})
        raise RuntimeError("No answer after 10 steps")


def has_claude_credentials():
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
