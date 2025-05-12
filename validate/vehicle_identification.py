#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: vehicle_identification.py
# Author: Wadih Khairallah
# Description: 
# Created: 2025-04-21 22:04:52

import re
from datetime import datetime
from pprint import pprint

from datasets.vin_wmi_data import WMI_MAP

# ISO 3779 region codes by first VIN character (fallback)
REGION_MAP = {
    **dict.fromkeys(list("12345"), "North America"),
    **dict.fromkeys(list("67"),    "Oceania"),
    **dict.fromkeys(list("89"),    "South America"),
    **dict.fromkeys(list("ABCDEFGH"), "Africa"),
    **dict.fromkeys(list("JKLMNPR"),  "Asia"),
    **dict.fromkeys(list("STUVWXYZ"), "Europe"),
}

# Transliteration for check‐digit
_TRANSLITERATION = {
    **{str(i): i for i in range(10)},
    **dict(zip(
        list("ABCDEFGHJKLMNPRSTUVWXYZ"),
        [1,2,3,4,5,6,7,8,1,2,3,4,5,7,8,9,2,3,4,5,6,7,8,9]
    ))
}

# Position weights for check digit
_WEIGHTS = [8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2]

# Model‐year codes cycle every 30 years starting 1980
_BASE_YEAR_CODES = {
    **dict(zip(list("ABCDEFGHJKLMNPRSTVWXY"), range(1980, 2000))),
    **{str(i): 2000 + i for i in range(1, 10)}
}

def _compute_check_digit(vin):
    total = 0
    for i, ch in enumerate(vin):
        val = _TRANSLITERATION.get(ch, 0)
        total += val * _WEIGHTS[i]
    rem = total % 11
    return "X" if rem == 10 else str(rem)

def _resolve_model_year(code, ref_year=None):
    """
    VIN code repeats every 30 years. Returns the latest year <= ref_year+1.
    """
    if code not in _BASE_YEAR_CODES:
        return None
    base = _BASE_YEAR_CODES[code]
    if ref_year is None:
        ref_year = datetime.now().year
    # cycle forward in 30‑year steps until exceeding ref_year+1
    year = base
    while year + 30 <= ref_year + 1:
        year += 30
    return year

def validate_vin(vin_raw):
    """
    Validate VIN and extract metadata:
      - vin, valid, error
      - region, wmi, manufacturer, country
      - model_year, plant, vds, serial
      - check_digit: observed, expected, ok
    """
    vin = vin_raw.strip().upper()
    res = {
        "vin": vin,
        "valid": False,
        "error":    None,
        "region":   None,
        "wmi":      None,
        "manufacturer": None,
        "country":      None,
        "model_year":   None,
        "vds":       None,
        "plant":     None,
        "serial":    None,
        "check_digit": {"observed": None, "expected": None, "ok": None},
    }

    # 1) Format & forbidden letters
    if len(vin) != 17 or re.search(r"[IOQ]", vin):
        res["error"] = "invalid_format"
        return res

    # 2) Region & WMI
    res["region"] = WMI_MAP.get(vin[:3], {}).get("region", REGION_MAP.get(vin[0], "Unknown"))
    res["wmi"]    = vin[:3]

    # 3) Manufacturer & country
    info = WMI_MAP.get(vin[:3])
    if info:
        res["manufacturer"] = info["manufacturer"]
        res["country"]      = info["region"]
    else:
        res["manufacturer"] = None
        res["country"]      = res["region"]

    # 4) Check‑digit
    obs = vin[8]
    exp = _compute_check_digit(vin)
    ok  = (obs == exp)
    res["check_digit"].update({"observed": obs, "expected": exp, "ok": ok})
    if not ok:
        res["error"] = "bad_check_digit"
        return res

    # 5) VDS (4–8), Plant (11), Serial (12–17)
    res["vds"]   = vin[3:8]
    res["plant"] = vin[10]
    res["serial"]= vin[11:]

    # 6) Model year (10th char) with cycles
    res["model_year"] = _resolve_model_year(vin[9])

    res["valid"] = True
    return res

if __name__ == "__main__":
    print("=== VIN Validation & Metadata Tests ===")
    samples = [
        "1HGCM82633A004352", "WDBRF40J93F334589",
        "JH4KA4650MC000000", "5YJSA1CN5DFP01234",
        "1HGCM82633A00435X", "12345678901234567",
    ]
    for vin in samples:
        print(f"\nVIN → {vin}")
        pprint(validate_vin(vin))
