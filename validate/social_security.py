#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: social_security.py
# Author: Wadih Khairallah
# Description: 
# Created: 2025-04-21 21:59:10

import re
from datetime import date, timedelta

# Mapping of SSN area‑number ranges to (state, first_issue_date)
AREA_STATE = {
    (1,   3):   ("New Hampshire", date(1936, 11, 1)),
    (4,   7):   ("Maine",         date(1936, 11, 1)),
    (8,   9):   ("Vermont",       date(1936, 11, 1)),
    (10,  34):  ("Massachusetts", date(1936, 11, 1)),
    (35,  39):  ("Rhode Island",  date(1936, 11, 1)),
    (40,  49):  ("Connecticut",   date(1936, 11, 1)),
    (50,  134): ("New York",      date(1936, 11, 1)),
    (135, 158): ("New Jersey",    date(1936, 11, 1)),
    (159, 211): ("Pennsylvania",  date(1936, 11, 1)),
    (212, 220): ("Maryland",      date(1936, 11, 1)),
    (221, 222): ("Delaware",      date(1936, 11, 1)),
    (223, 231): ("Virginia",      date(1936, 11, 1)),
    (232, 236): ("West Virginia", date(1936, 11, 1)),
    (237, 246): ("North Carolina",date(1936, 11, 1)),
    (247, 251): ("South Carolina",date(1936, 11, 1)),
    (252, 260): ("Georgia",       date(1936, 11, 1)),
    (261, 267): ("Florida",       date(1936, 11, 1)),
    (268, 302): ("Ohio",          date(1936, 11, 1)),
    (303, 317): ("Indiana",       date(1936, 11, 1)),
    (318, 361): ("Illinois",      date(1936, 11, 1)),
    (362, 386): ("Michigan",      date(1936, 11, 1)),
    (387, 399): ("Wisconsin",     date(1936, 11, 1)),
    (400, 407): ("Kentucky",      date(1936, 11, 1)),
    (408, 415): ("Tennessee",     date(1936, 11, 1)),
    (416, 424): ("Alabama",       date(1936, 11, 1)),
    (425, 428): ("Mississippi",   date(1936, 11, 1)),
    (429, 432): ("Arkansas",      date(1936, 11, 1)),
    (433, 439): ("Louisiana",     date(1936, 11, 1)),
    (440, 448): ("Oklahoma",      date(1936, 11, 1)),
    (449, 467): ("Texas",         date(1936, 11, 1)),
    (468, 477): ("Minnesota",     date(1936, 11, 1)),
    (478, 485): ("Iowa",          date(1936, 11, 1)),
    (486, 500): ("Missouri",      date(1936, 11, 1)),
    (501, 502): ("North Dakota",  date(1936, 11, 1)),
    (503, 504): ("South Dakota",  date(1936, 11, 1)),
    (505, 508): ("Nebraska",      date(1936, 11, 1)),
    (509, 515): ("Kansas",        date(1936, 11, 1)),
    (516, 517): ("Montana",       date(1936, 11, 1)),
    (518, 519): ("Idaho",         date(1936, 11, 1)),
    (520, 520): ("Wyoming",       date(1936, 11, 1)),
    (521, 524): ("Colorado",      date(1936, 11, 1)),
    (525, 525): ("New Mexico",    date(1936, 11, 1)),
    (526, 527): ("Arizona",       date(1936, 11, 1)),
    (528, 529): ("Utah",          date(1936, 11, 1)),
    (530, 530): ("Nevada",        date(1936, 11, 1)),
    (531, 539): ("Washington",    date(1936, 11, 1)),
    (540, 544): ("Oregon",        date(1936, 11, 1)),
    (545, 573): ("California",    date(1936, 11, 1)),
    (574, 574): ("Alaska",        date(1936, 11, 1)),
    (575, 576): ("Hawaii",        date(1936, 11, 1)),
    (577, 579): ("District of Columbia", date(1936, 11, 1)),
    (580, 580): ("U.S. Virgin Islands",   date(1936, 11, 1)),
    (580, 584): ("Puerto Rico",           date(1936, 11, 1)),
    (586, 586): ("Guam, American Samoa & Philippines", date(1936, 11, 1)),
    (587, 588): ("Mississippi",   date(1936, 11, 1)),
    # --- Florida overflow block began ~1975
    (589, 595): ("Florida",       date(1975, 1, 1)),
    (596, 599): ("Puerto Rico",   date(1936, 11, 1)),
    (600, 601): ("Arizona",       date(1936, 11, 1)),
    (602, 626): ("California",    date(1936, 11, 1)),
    (627, 647): ("Texas",         date(1936, 11, 1)),
    (648, 649): ("New Mexico",    date(1936, 11, 1)),
    (650, 653): ("Colorado",      date(1936, 11, 1)),
    (654, 658): ("South Carolina",date(1936, 11, 1)),
    (659, 665): ("Louisiana",     date(1936, 11, 1)),
    (667, 675): ("Georgia",       date(1936, 11, 1)),
    (676, 679): ("Arkansas",      date(1936, 11, 1)),
    (680, 680): ("Nevada",        date(1936, 11, 1)),
    (681, 690): ("North Carolina",date(1936, 11, 1)),
    (691, 699): ("Virginia",      date(1936, 11, 1)),
    # Special / other issuers (all assumed 1936 start)
    (700, 728): ("Railroad Board",       date(1936, 11, 1)),
    (729, 733): ("Enumeration at Entry", date(1936, 11, 1)),
    (750, 751): ("Hawaii",               date(1936, 11, 1)),
    (752, 755): ("Mississippi",          date(1936, 11, 1)),
    (756, 763): ("Tennessee",            date(1936, 11, 1)),
    (764, 765): ("Arizona",              date(1936, 11, 1)),
    (766, 772): ("Florida",              date(1999, 1, 1)),  # second overflow
}

# SSA’s pre‑2011 group‑number issuance sequence
GROUP_SEQUENCE = [
     1,  3,  5,  7,  9,
    10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
    30, 32, 34, 36, 38, 40, 42, 44, 46, 48,
    50, 52, 54, 56, 58, 60, 62, 64, 66, 68,
    70, 72, 74, 76, 78, 80, 82, 84, 86, 88,
    90, 92, 94, 96, 98,
     2,  4,  6,  8,
    11, 13, 15, 17, 19, 21, 23, 25, 27, 29,
    31, 33, 35, 37, 39, 41, 43, 45, 47, 49,
    51, 53, 55, 57, 59, 61, 63, 65, 67, 69,
    71, 73, 75, 77, 79, 81, 83, 85, 87, 89,
    91, 93, 95, 97, 99,
]

def validate_ss(raw_ss: object) -> dict:
    """
    Validate a U.S. SSN and extract all available metadata,
    including an estimated issue date based on its area block.
    Estimated dates are formatted as MM-DD-YYYY.
    """
    raw = str(raw_ss or "").strip()
    ssn = re.sub(r"\D", "", raw)

    result = {
        "ssn": ssn,
        "valid": False,
        "area": None,
        "group": None,
        "serial": None,
        "state": None,
        "issue_group_pos": None,
        "randomized": False,
        "estimated_issue_date": None,
        "error": None
    }

    # 1) Format
    if not re.fullmatch(r"\d{9}", ssn):
        result["error"] = "invalid_format"
        return result

    # 2) Components
    area   = int(ssn[:3])
    group  = int(ssn[3:5])
    serial = int(ssn[5:])
    result.update({"area": area, "group": group, "serial": serial})

    # 3) Basic validity
    if area == 0 or area == 666 or 900 <= area <= 999:
        result["error"] = "invalid_area";   return result
    if group == 0:
        result["error"] = "invalid_group";  return result
    if serial == 0:
        result["error"] = "invalid_serial"; return result

    result["valid"] = True

    # 4) Lookup area block
    block_key = None
    for (start, end), (st, first_date) in AREA_STATE.items():
        if start <= area <= end:
            block_key = (start, end)
            result["state"] = st
            start_date = first_date
            break

    # 5) Pre‑2011 vs. randomized
    if block_key and group in GROUP_SEQUENCE:
        result["randomized"] = False
        pos = GROUP_SEQUENCE.index(group) + 1
        result["issue_group_pos"] = pos

        # Estimate within this block’s window
        cutoff = date(2011, 6, 25)
        span_days = (cutoff - start_date).days
        frac      = (pos - 1) / (len(GROUP_SEQUENCE) - 1)
        est_date  = start_date + timedelta(days=int(frac * span_days))
        result["estimated_issue_date"] = est_date.strftime("%m-%d-%Y")

    else:
        result["randomized"] = True

    return result

if __name__ == "__main__":
    import json
    ssn_tests = [
        "594-34-7877",    # Florida, pre‑2011
        "123-45-6789",    # valid format, but area 123 maps to New Hampshire
        "000-12-3456",    # invalid area
        "12345678",       # invalid format
    ]

    print("\n=== SSN Validation ===")
    for ssn in ssn_tests:
        result = validate_ss(ssn)
        print(f"\nInput: {ssn}")
        print(json.dumps(result, indent=2))

