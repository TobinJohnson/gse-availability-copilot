"""
Generate 18 months of synthetic, SAP-style maintenance data for a 300-unit
GSE fleet at three stations.

What it simulates, day by day:
  - each unit runs some hours; its chance of failing rises with age, with
    days since its last preventive maintenance (PM), and in summer/monsoon
  - PM is due every 90 days but is often skipped in peak months
  - a failure creates a work order (notification -> order -> diagnosis ->
    part request -> part issue -> repair -> back in service)
  - spare parts come from the station store; if the store is empty the part
    is bought urgently (emergency purchase, 2-5 days, 30% premium)
  - stores reorder when stock falls to the SAP reorder point

Output CSVs in data/ (all fictional):
  equipment.csv     equipment master (like SAP PM EQUI)
  work_orders.csv   one row per repair with every lifecycle timestamp
  parts.csv         part master + stock per station at the snapshot (like SAP MM)
  unit_weeks.csv    weekly features per unit + "failed in next 7 days" label

Run:  python data_generator.py
"""
import math
import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from gse import (EMERGENCY_LEAD_DAYS, EQUIPMENT_TYPES, FAILURE_MODES, PARTS,
                 PM_INTERVAL_DAYS, STATIONS)

DATA_DIR = Path(__file__).parent / "data"
START = datetime(2025, 5, 1)
DAYS = 549                                  # to 31 Oct 2026
SNAPSHOT = START + timedelta(days=DAYS)     # 1 Nov 2026
PEAK_MONTHS = {4, 5, 10, 11, 12}            # summer holidays and winter schedule


def season_factor(month):
    if month in (4, 5, 6):
        return 1.30      # heat: engines, hydraulics, tyres
    if month in (7, 8, 9):
        return 1.20      # monsoon: electrics, brakes
    return 1.0


def generate(seed=11):
    rng = random.Random(seed)

    # ---- equipment master -------------------------------------------------
    units = []
    for station in STATIONS:
        n = 0
        for code, (name, count, run_mean, _) in EQUIPMENT_TYPES.items():
            for _ in range(count):
                n += 1
                units.append({
                    "unit_id": f"{station}-{code}-{n:03d}",
                    "station": station,
                    "type_code": code,
                    "type_name": name,
                    "age_years": rng.randint(1, 14),
                    # some units are simply worse than others ("problem units")
                    "frailty": rng.lognormvariate(0, 0.55),
                    "run_mean": run_mean,
                    "last_pm": -rng.randint(0, PM_INTERVAL_DAYS - 1),
                    "down_until": -1,
                })

    modes_by_type = {}
    for fcode, (tcode, _, part, share, _) in FAILURE_MODES.items():
        modes_by_type.setdefault(tcode, []).append((fcode, share))

    stock = {(s, p): spec[3] + spec[4] for s in STATIONS for p, spec in PARTS.items()}
    open_orders = {}              # (station, part) -> arrival day
    run_hours = {u["unit_id"]: [0.0] * DAYS for u in units}
    pm_days = {u["unit_id"]: [] for u in units}
    fail_days = {u["unit_id"]: [] for u in units}
    work_orders = []
    wo_no = 4000000

    for day in range(DAYS):
        date = START + timedelta(days=day)
        sf = season_factor(date.month)

        # deliveries of normal replenishment orders
        for key, arrive in list(open_orders.items()):
            if arrive <= day:
                stock[key] += PARTS[key[1]][4]
                del open_orders[key]

        for u in units:
            uid = u["unit_id"]
            if u["down_until"] > day:
                continue

            # preventive maintenance when due (often skipped at peak)
            if day - u["last_pm"] >= PM_INTERVAL_DAYS:
                compliance = 0.45 if date.month in PEAK_MONTHS else 0.85
                if rng.random() < compliance:
                    u["last_pm"] = day
                    pm_days[uid].append(day)

            hours = max(0.0, rng.gauss(u["run_mean"], 1.5))
            run_hours[uid][day] = hours
            since_pm = day - u["last_pm"]
            overdue = max(0, since_pm - PM_INTERVAL_DAYS)
            rate = (EQUIPMENT_TYPES[u["type_code"]][3] / 1000) * 0.55 * u["frailty"] \
                * (1 + u["age_years"] / 10) * (0.6 + since_pm / 90) * (1 + overdue / 30) * sf
            if rng.random() >= 1 - math.exp(-rate * hours):
                continue

            # ---- a breakdown: build the work order ------------------------
            fails = modes_by_type[u["type_code"]]
            fcode = rng.choices([f for f, _ in fails], [w for _, w in fails])[0]
            _, desc, part, _, repair_h = FAILURE_MODES[fcode]
            key = (u["station"], part)

            t = date + timedelta(hours=rng.uniform(5, 23))
            times = {"Breakdown reported": t}
            t += timedelta(hours=rng.uniform(0.5, 6) + (rng.uniform(4, 10) if t.hour >= 20 else 0))
            times["Notification created"] = t
            t += timedelta(hours=rng.uniform(1, 4));   times["Work order released"] = t
            t += timedelta(hours=rng.uniform(1, 3.5)); times["Diagnosis done"] = t   # manual search
            t += timedelta(hours=rng.uniform(0.2, 1)); times["Part requested"] = t
            if stock[key] > 0:
                stock[key] -= 1
                stockout = False
                t += timedelta(hours=rng.uniform(0.5, 4))
            else:
                stockout = True                                   # emergency buy
                t += timedelta(days=rng.uniform(*EMERGENCY_LEAD_DAYS))
            times["Part issued"] = t
            t += timedelta(hours=repair_h * rng.uniform(0.8, 1.5)); times["Repair completed"] = t
            t += timedelta(hours=rng.uniform(0.5, 24));             times["Back in service"] = t

            if stock[key] <= PARTS[part][3] and key not in open_orders:
                lead = PARTS[part][2]
                open_orders[key] = day + max(1, round(rng.uniform(0.8, 1.3) * lead))

            u["down_until"] = (t - START).days + 1
            fail_days[uid].append(day)
            wo_no += 1
            row = {
                "order_id": wo_no, "unit_id": uid, "station": u["station"],
                "type_code": u["type_code"],
                # 30% of orders are closed with the failure code left blank
                "failure_code": "" if rng.random() < 0.30 else fcode,
                "true_failure": fcode, "failure_desc": desc, "part_no": part,
                "stockout": stockout,
                "downtime_h": round((t - times["Breakdown reported"]).total_seconds() / 3600, 1),
            }
            for act, ts in times.items():
                row[act] = ts.strftime("%Y-%m-%d %H:%M")
            work_orders.append(row)

    # ---- outputs ----------------------------------------------------------
    equipment = pd.DataFrame([{k: u[k] for k in ("unit_id", "station", "type_code", "type_name", "age_years")}
                              | {"days_since_pm": DAYS - u["last_pm"]} for u in units])
    wos = pd.DataFrame(work_orders)

    parts = []
    for s in STATIONS:
        for p, (desc, cost, lead, rop, qty) in PARTS.items():
            parts.append({"station": s, "part_no": p, "description": desc, "unit_cost_usd": cost,
                          "lead_time_days": lead, "current_rop": rop, "order_qty": qty,
                          "on_hand": stock[(s, p)]})
    parts = pd.DataFrame(parts)

    # weekly features per unit; label = any failure in the next 7 days
    rows = []
    for u in units:
        uid = u["unit_id"]
        pms, fds = pm_days[uid], fail_days[uid]
        for wk_start in range(84, DAYS + 1, 7):
            last_pm = max([d for d in pms if d < wk_start], default=-30)
            label = None
            if wk_start + 7 <= DAYS:
                label = int(any(wk_start <= d < wk_start + 7 for d in fds))
            rows.append({
                "unit_id": uid, "type_code": u["type_code"], "station": u["station"],
                "week_start": (START + timedelta(days=wk_start)).strftime("%Y-%m-%d"),
                "age_years": u["age_years"],
                "run_hours_28d": round(sum(run_hours[uid][max(0, wk_start - 28):wk_start]), 1),
                "days_since_pm": wk_start - last_pm,
                "failures_90d": sum(1 for d in fds if wk_start - 90 <= d < wk_start),
                "month": (START + timedelta(days=wk_start)).month,
                "failed_next_7d": label,
            })
    unit_weeks = pd.DataFrame(rows)
    return equipment, wos, parts, unit_weeks


def write_all(seed=11):
    DATA_DIR.mkdir(exist_ok=True)
    equipment, wos, parts, unit_weeks = generate(seed)
    equipment.to_csv(DATA_DIR / "equipment.csv", index=False)
    wos.to_csv(DATA_DIR / "work_orders.csv", index=False)
    parts.to_csv(DATA_DIR / "parts.csv", index=False)
    unit_weeks.to_csv(DATA_DIR / "unit_weeks.csv", index=False)
    return equipment, wos, parts, unit_weeks


if __name__ == "__main__":
    e, w, p, uw = write_all()
    print(f"{len(e)} units, {len(w)} work orders ({w['stockout'].mean():.0%} waited on a stock-out), "
          f"{len(uw)} unit-weeks, snapshot {SNAPSHOT:%d %b %Y}")
