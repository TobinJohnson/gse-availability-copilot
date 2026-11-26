"""
Shared domain knowledge for a ground-support-equipment (GSE) fleet.

Fictional but realistic: equipment types, how they fail, which spare part
each failure needs, and what those parts cost and how long they take to buy.
"""

STATIONS = ["BLR", "HYD", "MAA"]  # three stations, 100 units each = 300-unit fleet

# type code -> (name, units per station, avg run hours per day, base failures per 1,000 run hours)
EQUIPMENT_TYPES = {
    "BL":  ("Belt loader",            20, 7.0, 2.2),
    "PBT": ("Pushback tug",            8, 5.0, 2.6),
    "GPU": ("Ground power unit",      14, 9.0, 1.8),
    "PS":  ("Passenger steps",        16, 6.0, 1.6),
    "BT":  ("Baggage tractor",        24, 8.0, 2.0),
    "CHL": ("Catering high-loader",    8, 5.0, 2.4),
    "ACU": ("Air-conditioning unit",  10, 6.0, 2.0),
}

# failure code -> (equipment type, description, part number, share of that type's failures, repair hours)
FAILURE_MODES = {
    "BL-HYD":  ("BL",  "Hydraulic leak on conveyor lift",   "P-SEAL-BL",  0.40, 3.0),
    "BL-BELT": ("BL",  "Conveyor belt torn or slipping",    "P-BELT-BL",  0.30, 4.0),
    "BL-STRT": ("BL",  "Engine does not start",             "P-STRT-12V", 0.30, 2.0),
    "PBT-BRK": ("PBT", "Brake wear, low braking force",     "P-BRK-PBT",  0.45, 3.5),
    "PBT-HOSE":("PBT", "Hydraulic hose burst",              "P-HOSE-PBT", 0.30, 2.5),
    "PBT-COOL":("PBT", "Engine overheating",                "P-PUMP-PBT", 0.25, 4.0),
    "GPU-AVR": ("GPU", "Output voltage out of limits",      "P-AVR-GPU",  0.45, 2.5),
    "GPU-CBL": ("GPU", "400 Hz cable damaged",              "P-CBL-GPU",  0.30, 1.5),
    "GPU-STRT":("GPU", "Engine does not start",             "P-STRT-24V", 0.25, 2.0),
    "PS-LIFT": ("PS",  "Lift cylinder leaking / drifting",  "P-SEAL-PS",  0.55, 3.0),
    "PS-TYRE": ("PS",  "Tyre damaged",                      "P-TYRE-S",   0.45, 1.0),
    "BT-TYRE": ("BT",  "Tyre damaged",                      "P-TYRE-S",   0.40, 1.0),
    "BT-BATT": ("BT",  "Battery flat / no start",           "P-BATT-12V", 0.35, 0.8),
    "BT-BRK":  ("BT",  "Brake wear",                        "P-BRK-BT",   0.25, 2.5),
    "CHL-HYD": ("CHL", "Scissor-lift hydraulic leak",       "P-SEAL-CHL", 0.60, 3.5),
    "CHL-SENS":("CHL", "Platform proximity sensor fault",   "P-SENS-CHL", 0.40, 1.5),
    "ACU-COMP":("ACU", "Compressor relay failure",          "P-RLY-ACU",  0.55, 2.0),
    "ACU-HOSE":("ACU", "Supply duct hose torn",             "P-DUCT-ACU", 0.45, 1.5),
}

# part number -> (description, unit cost USD, normal lead time days, current SAP reorder point, order quantity)
# Current reorder points were set once and never revisited: some are too low on purpose.
PARTS = {
    "P-SEAL-BL":  ("Belt loader lift seal kit",        120, 21, 1, 4),
    "P-BELT-BL":  ("Belt loader conveyor belt",        640, 30, 1, 2),
    "P-STRT-12V": ("Starter motor, 12 V",              380, 14, 3, 3),
    "P-BRK-PBT":  ("Pushback tug brake pad set",       260, 21, 1, 2),
    "P-HOSE-PBT": ("Pushback tug hydraulic hose",       90, 10, 3, 3),
    "P-PUMP-PBT": ("Pushback tug coolant pump",        520, 28, 1, 1),
    "P-AVR-GPU":  ("GPU voltage regulator module",     900, 35, 1, 2),
    "P-CBL-GPU":  ("400 Hz output cable",              450, 14, 3, 3),
    "P-STRT-24V": ("Starter motor, 24 V",              420, 14, 1, 2),
    "P-SEAL-PS":  ("Passenger steps lift seal kit",    110, 21, 1, 4),
    "P-TYRE-S":   ("Solid tyre, small GSE",            150, 10, 6, 10),
    "P-BATT-12V": ("Battery, 12 V",                    140,  7, 5, 6),
    "P-BRK-BT":   ("Baggage tractor brake set",        180, 14, 1, 3),
    "P-SEAL-CHL": ("High-loader scissor seal kit",     160, 28, 1, 2),
    "P-SENS-CHL": ("Platform proximity sensor",        210, 21, 1, 2),
    "P-RLY-ACU":  ("ACU compressor relay",             170, 21, 1, 2),
    "P-DUCT-ACU": ("ACU supply duct hose",             230, 14, 1, 2),
}

EMERGENCY_LEAD_DAYS = (2, 5)     # air-freighted part, min-max days
EMERGENCY_PREMIUM = 0.30         # 30% price premium on emergency buys
PM_INTERVAL_DAYS = 90            # preventive maintenance every 90 days

# Work-order lifecycle (SAP PM-style), in order
ACTIVITIES = [
    "Breakdown reported",
    "Notification created",
    "Work order released",
    "Diagnosis done",
    "Part requested",
    "Part issued",
    "Repair completed",
    "Back in service",
]
