"""
Tests: data, analyses, copilot (incl. the Claude loop with a fake client) and the dashboard.
No API key needed.   Run:  pytest -q
"""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from copilot import ClaudeCopilot, Fleet, Manuals, rule_based_answer
from data_generator import DAYS
from gse import ACTIVITIES, FAILURE_MODES
from predict import current_risk, train
from process_mining import dfg_dot, event_log, step_times
from spares import reorder_points

FLEET, MANUALS = Fleet(), Manuals()
WOS = FLEET.work_orders


def test_work_order_timestamps_are_in_order():
    times = WOS[ACTIVITIES].apply(pd.to_datetime)
    assert (times.diff(axis=1).iloc[:, 1:] >= pd.Timedelta(0)).all().all()


def test_stockouts_cause_longer_downtime():
    assert WOS[WOS["stockout"]]["downtime_h"].mean() > 2 * WOS[~WOS["stockout"]]["downtime_h"].mean()


def test_process_mining_finds_waiting_for_parts_as_bottleneck():
    steps = step_times(WOS)
    assert steps.sort_values("share_of_downtime").iloc[-1]["step"] == "Part requested → Part issued"
    assert abs(steps["share_of_downtime"].sum() - 1) < 1e-9
    assert "Emergency purchase" in set(event_log(WOS)["activity"])
    assert dfg_dot(WOS).startswith("digraph")


def test_reorder_point_formula():
    rop = reorder_points(WOS, FLEET.parts, DAYS)
    assert len(rop) == len(FLEET.parts)
    assert (rop["recommended_rop"] >= 0).all()
    # parts with stock-outs should not get a lower reorder point than today
    assert (rop[rop["stockouts"] > 5]["change"] >= 0).all()


def test_failure_model_beats_random():
    uw = pd.read_csv(ROOT / "data" / "unit_weeks.csv")
    model, m = train(uw)
    assert m["auc"] > 0.55
    assert m["top10_precision"] > 1.5 * m["base_rate"]
    risk = current_risk(model, uw, FLEET.equipment)
    assert len(risk) == 300 and risk["risk_7d"].between(0, 1).all()


def test_copilot_picks_the_right_fault():
    ids = FLEET.equipment.groupby("type_code")["unit_id"].first()
    cases = [("BL", "hydraulic oil leaking from the conveyor lift", "BL-HYD"),
             ("GPU", "output voltage fluctuating, aircraft rejected power", "GPU-AVR"),
             ("BT", "tractor will not start, battery weak", "BT-BATT"),
             ("PBT", "temperature gauge going red during pushback", "PBT-COOL"),
             ("ACU", "weak airflow and noise from duct hose", "ACU-HOSE")]
    for t, q, expected in cases:
        ans = rule_based_answer(q, ids[t], FLEET, MANUALS)
        assert ans["failure_code"] == expected, (q, ans["failure_code"])
        assert ans["part_status"] and ans["procedure"]["Safety"]


class FakeClient:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **kw):
        self.requests.append({**kw, "messages": list(kw["messages"])})
        return self.responses.pop(0)


def test_claude_copilot_loop_with_fake_client():
    use = lambda i, n, x: NS(type="tool_use", id=i, name=n, input=x)
    final = {"failure_code": "BL-HYD", "failure_desc": FAILURE_MODES["BL-HYD"][1],
             "procedure_summary": "Safety first...", "part_status": "In stock", "draft_work_order": "WO",
             "sources": ["Hydraulic leak › Repair"]}
    client = FakeClient([
        NS(stop_reason="tool_use", content=[use("a", "unit_history", {"unit_id": "BLR-BL-001"}),
                                            use("b", "search_manuals", {"query": "hydraulic leak"}),
                                            use("c", "check_stock", {"part_no": "P-SEAL-BL", "station": "BLR"})]),
        NS(stop_reason="tool_use", content=[use("d", "submit_answer", final)]),
    ])
    ans = ClaudeCopilot(FLEET, MANUALS, client=client).answer("leak", "BLR-BL-001")
    assert ans["failure_code"] == "BL-HYD"
    assert len(client.requests[1]["messages"][-1]["content"]) == 3   # all tool results in one message


def test_dashboard_runs_and_copilot_button_works():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    at.button[0].click().run()          # "Ask copilot"
    assert not at.exception
    assert any("Likely fault" in m.value for m in at.markdown)
