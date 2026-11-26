"""
Process mining of the repair (work-order) lifecycle, written with pandas so it
also runs in the browser.

The question it answers: when a unit breaks, where does the repair time go?

Each work order is a case; each lifecycle timestamp is an event. From that
event log we build:
  - step_times():    hours spent between each pair of steps, and its share of all downtime
  - variants():      the different paths repairs take (stock vs emergency purchase)
  - dfg_dot():       a directly-follows graph (DFG) to draw with Graphviz
"""
import pandas as pd

from gse import ACTIVITIES


def event_log(wos: pd.DataFrame) -> pd.DataFrame:
    """Long format: one row per (case, activity, timestamp). Emergency buys get an extra activity."""
    log = wos.melt(id_vars=["order_id", "stockout"], value_vars=ACTIVITIES,
                   var_name="activity", value_name="timestamp")
    log["timestamp"] = pd.to_datetime(log["timestamp"])
    emergency = log[(log["activity"] == "Part requested") & log["stockout"]].copy()
    emergency["activity"] = "Emergency purchase"
    emergency["timestamp"] += pd.Timedelta(minutes=30)
    log = pd.concat([log, emergency]).sort_values(["order_id", "timestamp"])
    return log[["order_id", "activity", "timestamp"]].reset_index(drop=True)


def _transitions(log: pd.DataFrame) -> pd.DataFrame:
    log = log.copy()
    log["next_activity"] = log.groupby("order_id")["activity"].shift(-1)
    log["next_time"] = log.groupby("order_id")["timestamp"].shift(-1)
    t = log.dropna(subset=["next_activity"]).copy()
    t["hours"] = (t["next_time"] - t["timestamp"]).dt.total_seconds() / 3600
    return t


def step_times(wos: pd.DataFrame) -> pd.DataFrame:
    """Hours between consecutive lifecycle steps (main path), with share of total downtime."""
    rows = []
    for a, b in zip(ACTIVITIES, ACTIVITIES[1:]):
        h = (pd.to_datetime(wos[b]) - pd.to_datetime(wos[a])).dt.total_seconds() / 3600
        rows.append({"step": f"{a} → {b}", "mean_hours": h.mean(), "median_hours": h.median(),
                     "total_hours": h.sum()})
    df = pd.DataFrame(rows)
    df["share_of_downtime"] = df["total_hours"] / df["total_hours"].sum()
    return df


def variants(wos: pd.DataFrame) -> pd.DataFrame:
    v = wos.assign(path=wos["stockout"].map({False: "Part from stock", True: "Emergency purchase"}),
                   code=wos["failure_code"].fillna("").eq("").map({True: "failure code blank", False: "failure code filled"}))
    out = v.groupby(["path", "code"]).agg(repairs=("order_id", "count"),
                                          mean_downtime_h=("downtime_h", "mean")).reset_index()
    out["share"] = out["repairs"] / out["repairs"].sum()
    return out.sort_values("repairs", ascending=False)


def dfg_dot(wos: pd.DataFrame) -> str:
    """Graphviz DOT for the directly-follows graph: edge label = cases and mean hours."""
    t = _transitions(event_log(wos))
    edges = t.groupby(["activity", "next_activity"]).agg(cases=("order_id", "count"),
                                                          mean_h=("hours", "mean")).reset_index()
    slowest = edges["mean_h"].max()
    lines = ["digraph { rankdir=TB; node [shape=box, style=rounded, fontsize=11, fontname=Helvetica];",
             'edge [fontsize=10, fontname=Helvetica];']
    for act in ACTIVITIES[:5] + ["Emergency purchase"] + ACTIVITIES[5:]:
        style = ', style="rounded,filled", fillcolor="#fde2e1"' if act == "Emergency purchase" else ""
        lines.append(f'"{act}" [label="{act}"{style}];')
    for _, e in edges.iterrows():
        hot = e["mean_h"] == slowest
        attr = ', color="#d64545", penwidth=2.5, fontcolor="#d64545"' if hot else ""
        lines.append(f'"{e.activity}" -> "{e.next_activity}" '
                     f'[label="{e.cases:,} cases\\n{e.mean_h:.1f} h avg"{attr}];')
    lines.append("}")
    return "\n".join(lines)
