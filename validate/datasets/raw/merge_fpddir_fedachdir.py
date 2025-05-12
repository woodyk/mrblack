#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: merge_fpddir_fedachdir.py
# Author: Wadih Khairallah
# Description: 
# Created: 2025-04-22 13:20:51
# Modified: 2025-04-22 14:27:42

import json
import sys

def parse_fpddir_line(line):
    return {
        "routing_number": line[0:9].strip(),
        "short_name": line[9:27].strip() or None,
        "customer_name_fpddir": line[27:63].strip() or None,
        "state_fpddir": line[63:65].strip() or None,
        "city_fpddir": line[65:85].strip() or None,
        "fedwire_eligibility": line[85:86].strip() or None,
        "fedach_participation": line[86:87].strip() or None,
        "change_date": line[87:95].strip() or None
    }

def parse_fedachdir_line(line):
    return {
        "routing_number": line[0:9].strip(),
        "record_type": line[9:10].strip() or None,
        "sending_point_routing": line[10:19].strip() or None,
        "file_date_sequence": line[19:26].strip() or None,
        "customer_name_fedach": line[35:71].strip() or None,
        "address": line[71:107].strip() or None,
        "city_fedach": line[107:127].strip() or None,
        "state_fedach": line[127:129].strip() or None,
        "zip_code": line[129:134].strip() or None,
        "phone_number": line[138:148].strip() or None,
        "status_code": line[148:150].strip() or None
    }

def main(fpddir_path, fedachdir_path):
    # Parse fpddir.txt
    fpddir_data = {}
    with open(fpddir_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_fpddir_line(line)
            routing_number = parsed["routing_number"]
            fpddir_data[routing_number] = parsed

    # Parse FedACHdir.txt
    fedachdir_data = {}
    with open(fedachdir_path, "r", encoding="utf-8") as f:
        for line in f:
            parsed = parse_fedachdir_line(line)
            routing_number = parsed["routing_number"]
            fedachdir_data[routing_number] = parsed

    # Get all unique routing numbers
    all_routing_numbers = set(fpddir_data.keys()) | set(fedachdir_data.keys())

    # Merge data
    merged_data = {}
    for routing_number in all_routing_numbers:
        entry = {}
        fedach = fedachdir_data.get(routing_number, {})
        fpddir = fpddir_data.get(routing_number, {})

        # Consolidated fields with preference for FedACHdir where applicable
        entry["customer_name"] = fedach.get("customer_name_fedach") or fpddir.get("customer_name_fpddir")
        entry["short_name"] = fpddir.get("short_name")
        entry["address"] = fedach.get("address")
        entry["city"] = fedach.get("city_fedach") or fpddir.get("city_fpddir")
        entry["state"] = fedach.get("state_fedach") or fpddir.get("state_fpddir")
        entry["zip_code"] = fedach.get("zip_code")
        entry["phone_number"] = fedach.get("phone_number")
        entry["fedwire_eligibility"] = fpddir.get("fedwire_eligibility")
        entry["fedach_participation"] = fpddir.get("fedach_participation")
        entry["change_date"] = fpddir.get("change_date")
        entry["record_type"] = fedach.get("record_type")
        entry["sending_point_routing"] = fedach.get("sending_point_routing")
        entry["file_date_sequence"] = fedach.get("file_date_sequence")
        entry["status_code"] = fedach.get("status_code")

        merged_data[routing_number] = entry

    # Output as pretty-printed JSON
    print(json.dumps(merged_data, indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py fpddir.txt FedACHdir.txt")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
