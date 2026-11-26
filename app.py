"""
GSE Availability Copilot - maintenance & spares dashboard.

Run:  streamlit run app.py
"""
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from copilot import ClaudeCopilot, Fleet, Manuals, has_claude_credentials, rule_based_answer
from data_generator import DAYS, SNAPSHOT
from predict import current_risk, drivers, train
from process_mining import dfg_dot, step_times, variants
from spares import downtime_pareto, reorder_points

BASE = Path(__file__).parent
st.set_page_config(page_title="GSE Availability Copilot", page_icon="🛠️", layout="wide")


@st.cache_resource
def load():
    fleet = Fleet()
    unit_weeks = pd.read_csv(BASE / "data" / "unit_weeks.csv")
    model, metrics = train(unit_weeks)
    return fleet, Manuals(), unit_weeks, model, metrics


fleet, manuals, unit_weeks, model, metrics = load()
wos, eq, parts = fleet.work_orders, fleet.equipment, fleet.parts
years = DAYS / 365

with st.sidebar:
    st.header("Maintenance control")
    st.caption(f"300 GSE units at BLR, HYD and MAA · 18 months of repairs to {SNAPSHOT:%d %b %Y} · "
               "fictional, SAP-style data")
    station = st.selectbox("Station", ["All stations", "BLR", "HYD", "MAA"])
    creds = has_claude_credentials()
    use_ai = st.toggle("Use Claude for the copilot", value=creds, disabled=not creds,
                       help="Needs ANTHROPIC_API_KEY. Without it the copilot uses retrieval only.")
    if not creds:
        st.info("No API key: the copilot answers by retrieval from the procedures, without AI.")

if station != "All stations":
    wos, eq, parts = wos[wos["station"] == station], eq[eq["station"] == station], parts[parts["station"] == station]

st.title("🛠️ GSE Availability Copilot")
st.caption("Finds where repair time goes, fixes spare-part stocking, predicts failures and helps technicians fix faster.")

t_fleet, t_pm, t_spares, t_risk, t_copilot, t_case = st.tabs(
    ["Fleet overview", "Where repair time goes", "Spare parts", "Failure risk", "Technician copilot", "Business case"])

# ------------------------------------------------------------ fleet overview
with t_fleet:
    hours_total = len(eq) * DAYS * 24
    c = st.columns(5)
    c[0].metric("Fleet availability", f"{1 - wos['downtime_h'].sum() / hours_total:.1%}")
    c[1].metric("Repairs (18 months)", f"{len(wos):,}")
    c[2].metric("Avg downtime per repair", f"{wos['downtime_h'].mean():.0f} h")
    c[3].metric("Repairs waiting on a stock-out", f"{wos['stockout'].mean():.0%}")
    c[4].metric("Failure code left blank", f"{wos['failure_code'].eq('').mean():.0%}")

    left, right = st.columns(2)
    by_unit = downtime_pareto(wos, "unit_id")
    n50 = int((by_unit["cumulative_share"] <= 0.5).sum()) + 1
    with left:
        st.subheader("Downtime by unit")
        st.write(f"**{n50} of {len(eq)} units ({n50 / len(eq):.0%})** cause half of all downtime.")
        st.altair_chart(alt.Chart(by_unit.head(20)).mark_bar(color="#4c78a8").encode(
            x=alt.X("downtime_h:Q", title="Downtime hours"), y=alt.Y("unit_id:N", sort="-x", title=None),
            tooltip=["unit_id", alt.Tooltip("downtime_h:Q", format=",.0f")]), width="stretch")
    with right:
        st.subheader("Downtime by part needed")
        by_part = downtime_pareto(wos, "part_no")
        st.write("Failure codes are often blank, so the part issued is used to find the root cause.")
        st.altair_chart(alt.Chart(by_part).mark_bar(color="#e45756").encode(
            x=alt.X("downtime_h:Q", title="Downtime hours"), y=alt.Y("part_no:N", sort="-x", title=None),
            tooltip=["part_no", alt.Tooltip("downtime_h:Q", format=",.0f")]), width="stretch")

# ------------------------------------------------------- process mining
with t_pm:
    st.subheader("Where does repair time go?")
    steps = step_times(wos)
    worst = steps.sort_values("share_of_downtime", ascending=False).iloc[0]
    st.write(f"Process mining of {len(wos):,} work orders. The biggest block of downtime is "
             f"**{worst['step']}: {worst['share_of_downtime']:.0%}** of all repair hours, almost all of it "
             f"in repairs that waited for an emergency purchase.")
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(alt.Chart(steps).mark_bar(color="#4c78a8").encode(
            x=alt.X("share_of_downtime:Q", title="Share of all downtime", axis=alt.Axis(format="%")),
            y=alt.Y("step:N", sort=None, title=None),
            tooltip=["step", alt.Tooltip("mean_hours:Q", format=".1f", title="avg hours"),
                     alt.Tooltip("share_of_downtime:Q", format=".0%")]), width="stretch")
        st.write("**Process variants**")
        v = variants(wos)
        st.dataframe(v.assign(share=v["share"].map("{:.0%}".format),
                              mean_downtime_h=v["mean_downtime_h"].round(0)),
                     hide_index=True, width="stretch")
    with right:
        st.write("**Directly-follows graph** (cases and average hours per step; red = slowest)")
        st.graphviz_chart(dfg_dot(wos))

# ------------------------------------------------------------ spare parts
with t_spares:
    st.subheader("Are the SAP reorder points right?")
    st.latex(r"\text{Reorder point} = d \times L + z\,\sigma_d\sqrt{L}\quad (z = 1.65 \approx 95\%\ \text{service})")
    rop = reorder_points(wos, parts, DAYS)
    c = st.columns(4)
    c[0].metric("Reorder points to raise", f"{(rop['change'] > 0).sum()} of {len(rop)}")
    c[1].metric("Repair hours lost waiting for parts", f"{rop['hours_waiting_for_part'].sum():,.0f}")
    c[2].metric("Emergency-buy premium paid", f"${rop['emergency_premium_usd'].sum():,.0f}")
    c[3].metric("One-time extra stock to fix it", f"${rop['extra_stock_value_usd'].sum():,.0f}")
    st.dataframe(rop, hide_index=True, width="stretch", column_config={
        "emergency_premium_usd": st.column_config.NumberColumn("emergency premium ($)", format="$%d"),
        "extra_stock_value_usd": st.column_config.NumberColumn("extra stock ($)", format="$%d")})
    st.caption("Parts at the top caused the most waiting. Raising those reorder points is the quickest win; "
               "a few low-use parts can be lowered to free cash.")

# ----------------------------------------------------------- failure risk
with t_risk:
    st.subheader("Which units are likely to fail in the next 7 days?")
    c = st.columns(3)
    c[0].metric("Riskiest 10% of units: failure rate", f"{metrics['top10_precision']:.0%}",
                f"vs {metrics['base_rate']:.0%} average", delta_color="off")
    c[1].metric("Share of failures they contain", f"{metrics['top10_capture']:.0%}")
    c[2].metric("Model AUC (test weeks)", f"{metrics['auc']:.2f}")
    st.caption(f"Logistic regression trained on older weeks, tested on {metrics['test_period']}. "
               "Weekly failures are noisy: the value is in ranking which units to service first, not certainty.")
    risk = current_risk(model, unit_weeks, fleet.equipment)
    if station != "All stations":
        risk = risk[risk["station"] == station]
    left, right = st.columns([3, 2])
    with left:
        st.write(f"**Service these first** (week of {SNAPSHOT:%d %b %Y})")
        st.dataframe(risk.head(15)[["unit_id", "type_name", "age_years", "days_since_pm", "failures_90d", "risk_7d"]]
                     .assign(risk_7d=lambda d: d["risk_7d"].map("{:.0%}".format)),
                     hide_index=True, width="stretch")
    with right:
        st.write("**What drives risk** (model weights)")
        st.altair_chart(alt.Chart(drivers(model)).mark_bar(color="#4c78a8").encode(
            x=alt.X("weight:Q", title="Weight (standardised)"), y=alt.Y("feature:N", sort="-x", title=None)),
            width="stretch")

# ------------------------------------------------------------- copilot
with t_copilot:
    st.subheader("Ask the maintenance copilot")
    examples = {
        "Belt loader leaking hydraulic oil": ("BL", "Hydraulic oil leaking from the conveyor lift, platform drifts down"),
        "GPU rejected by aircraft": ("GPU", "Output voltage fluctuating, aircraft rejected ground power"),
        "Tractor won't start": ("BT", "Baggage tractor will not start in the morning, battery seems weak"),
        "Pushback tug overheating": ("PBT", "Temperature gauge goes red during pushback"),
    }
    pick = st.selectbox("Example", list(examples))
    t_code, q_default = examples[pick]
    units = fleet.equipment if station == "All stations" else fleet.equipment[fleet.equipment["station"] == station]
    unit_opts = list(units[units["type_code"] == t_code]["unit_id"]) + \
        list(units[units["type_code"] != t_code]["unit_id"])
    unit_id = st.selectbox("Unit", unit_opts)
    question = st.text_area("What's wrong?", q_default)

    if st.button("Ask copilot", type="primary"):
        try:
            if use_ai:
                with st.status("Copilot looking up procedures and stock…", expanded=True) as box:
                    ans = ClaudeCopilot(fleet, manuals, on_step=box.write).answer(question, unit_id)
                    box.update(label="Answer ready", state="complete")
            else:
                ans = rule_based_answer(question, unit_id, fleet, manuals)
            st.session_state["answer"] = ans
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")

    ans = st.session_state.get("answer")
    if ans:
        st.caption(ans["answerer"])
        st.write(f"**Likely fault:** {ans['failure_code']} - {ans['failure_desc']}")
        (st.success if "OUT OF STOCK" not in ans["part_status"] else st.warning)(ans["part_status"])
        if ans.get("procedure"):
            for section, text in ans["procedure"].items():
                st.markdown(f"**{section}**  \n{text}")
        else:
            st.markdown(ans["procedure_summary"])
        with st.expander("Sources"):
            for s in ans["sources"]:
                st.write("- " + s)
        draft = st.text_area("Draft work order (edit before approval)", ans["draft_work_order"])
        if st.button("✅ Supervisor approves work order"):
            log = st.session_state.setdefault("approved", [])
            log.append({"time": datetime.now().strftime("%H:%M"), "unit": unit_id,
                        "fault": ans["failure_code"], "work_order": draft})
        if st.session_state.get("approved"):
            st.write("**Approved this session**")
            st.dataframe(pd.DataFrame(st.session_state["approved"]), hide_index=True, width="stretch")

# -------------------------------------------------------- business case
with t_case:
    st.subheader("Business case (edit the assumptions)")
    rop_all = reorder_points(fleet.work_orders, fleet.parts, DAYS)
    measured_wait = rop_all["hours_waiting_for_part"].sum() / years
    measured_premium = rop_all["emergency_premium_usd"].sum() / years
    units_down = fleet.work_orders["downtime_h"].sum() / years / 8760
    st.write(f"From the data: **{measured_wait:,.0f} repair hours a year** waiting for parts, "
             f"**${measured_premium:,.0f} a year** in emergency-buy premiums, and downtime equal to "
             f"**{units_down:.0f} units out of service all year** across the 300-unit fleet.")
    c1, c2, c3 = st.columns(3)
    delay_min = c1.number_input("Equipment-caused delay minutes per day", value=60, step=10)
    delay_cut = c1.slider("Delay reduction", 5, 60, 30, help="From fewer breakdowns and faster repairs") / 100
    cost_min = c2.number_input("Cost per delay minute (\\$)", value=100, step=10)
    downtime_cut = c2.slider("Downtime reduction", 10, 70, 40,
                             help="Stock-outs fixed plus failures caught before peak") / 100
    units_freed = round(units_down * downtime_cut)
    lease = c3.number_input("Lease cost per unit per year (\\$)", value=20000, step=1000)
    premium_cut = c3.slider("Emergency buys avoided", 0, 100, 40) / 100

    rows = pd.DataFrame([
        ("Fewer equipment-caused delays", delay_min * 365 * delay_cut * cost_min),
        (f"Fewer standby units leased ({units_freed} units)", units_freed * lease),
        ("Less emergency purchasing (measured premium)", measured_premium * premium_cut),
    ], columns=["Value driver", "Annual value ($)"])
    st.dataframe(rows, hide_index=True, width="stretch",
                 column_config={"Annual value ($)": st.column_config.NumberColumn(format="$%d")})
    st.metric("Estimated annual value", f"${rows['Annual value ($)'].sum():,.0f}")
    st.caption(f"One-time cost: about ${rop_all['extra_stock_value_usd'].sum():,.0f} of extra stock to raise reorder "
               "points, plus integration with SAP PM/MM and technician training. All figures are simulated.")
