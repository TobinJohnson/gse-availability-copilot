"""
Spare-parts analysis: are the SAP MM reorder points right?

For every part at every station:
  d      = average parts used per day (from goods issues to work orders)
  sigma  = day-to-day variation in usage
  L      = supplier lead time in days
  safety stock   = z * sigma * sqrt(L)       (z = 1.65 for ~95% service level)
  reorder point  = d * L + safety stock

Then compare with the reorder point set in SAP today, and count what the
wrong ones cost: stock-outs, downtime hours spent waiting, emergency spend.
"""
import math

import pandas as pd

from gse import EMERGENCY_PREMIUM

Z_95 = 1.65


def reorder_points(wos: pd.DataFrame, parts: pd.DataFrame, days: int) -> pd.DataFrame:
    wos = wos.copy()
    wos["issue_date"] = pd.to_datetime(wos["Part issued"]).dt.normalize()
    wos["wait_h"] = (pd.to_datetime(wos["Part issued"]) - pd.to_datetime(wos["Part requested"])).dt.total_seconds() / 3600

    rows = []
    for _, p in parts.iterrows():
        used = wos[(wos["station"] == p["station"]) & (wos["part_no"] == p["part_no"])]
        daily = used.groupby("issue_date").size()
        daily = daily.reindex(pd.date_range(wos["issue_date"].min(), periods=days, freq="D"), fill_value=0)
        d, sigma, L = daily.mean(), daily.std(), p["lead_time_days"]
        safety = Z_95 * sigma * math.sqrt(L)
        new_rop = math.ceil(d * L + safety)
        outs = used[used["stockout"]]
        rows.append({
            "station": p["station"], "part_no": p["part_no"], "description": p["description"],
            "used_per_month": round(d * 30, 1), "lead_time_days": L,
            "current_rop": p["current_rop"], "recommended_rop": new_rop,
            "change": new_rop - p["current_rop"],
            "stockouts": len(outs),
            "hours_waiting_for_part": round(outs["wait_h"].sum()),
            "emergency_premium_usd": round(len(outs) * p["unit_cost_usd"] * EMERGENCY_PREMIUM),
            "extra_stock_value_usd": max(0, new_rop - p["current_rop"]) * p["unit_cost_usd"],
        })
    return pd.DataFrame(rows).sort_values("hours_waiting_for_part", ascending=False)


def downtime_pareto(wos: pd.DataFrame, by: str) -> pd.DataFrame:
    """Downtime hours ranked by a column (unit_id, part_no, type_code ...) with cumulative share."""
    p = wos.groupby(by)["downtime_h"].sum().sort_values(ascending=False).reset_index()
    p["cumulative_share"] = p["downtime_h"].cumsum() / p["downtime_h"].sum()
    return p
