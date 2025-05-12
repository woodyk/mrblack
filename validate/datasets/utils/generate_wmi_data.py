#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: generate_wmi_data.py
# Author: Wadih Khairallah
# Description: 
# Created: 2025-04-21 22:32:10
# Modified: 2025-04-21 22:33:30

import requests
from bs4 import BeautifulSoup
import re
from collections import OrderedDict

WIKI_URL = (
    "https://en.wikibooks.org/wiki/"
    "Vehicle_Identification_Numbers_(VIN_codes)/World_Manufacturer_Identifier_(WMI)"
)

def scrape_wmi():
    resp = requests.get(WIKI_URL)
    resp.raise_for_status()

    # Parse with lxml only—no html5lib
    soup = BeautifulSoup(resp.text, "lxml")
    wmi_map = OrderedDict()

    # Walk through every wikitable on the page
    for table in soup.find_all("table", class_="wikitable"):
        # Find the nearest preceding heading (h2/h3/h4) with a mw-headline span
        header = table.find_previous_sibling(
            lambda tag: tag.name in ("h2", "h3", "h4")
            and tag.find("span", class_="mw-headline")
        )
        region = (
            header.find("span", class_="mw-headline").get_text(strip=True)
            if header else "Unknown"
        )

        # Iterate rows (skip the header row)
        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) < 2:
                continue

            raw_codes = cells[0].get_text(" ", strip=True)
            manuf     = cells[1].get_text(" ", strip=True)

            # Split on "/" or "," to handle multi‑code cells
            for code in re.split(r"[\/,]", raw_codes):
                code = code.strip()
                if not code or code.lower() == "nan":
                    continue
                wmi_map[code] = {"manufacturer": manuf, "region": region}

    return wmi_map

if __name__ == "__main__":
    wmi_map = scrape_wmi()

    # Emit to stdout
    print("# Auto‑generated WMI → manufacturer & region map")
    print("WMI_MAP = {")
    for code, info in wmi_map.items():
        print(f"    {code!r}: {info!r},")
    print("}")
