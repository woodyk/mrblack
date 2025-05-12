#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: banking.py
# Author: Wadih Khairallah
# Description: 
# Created: 2025-04-22 12:26:14

import re
import schwifty
import json
import pycountry

# Load the merged banking data from the provided JSON file
with open('datasets/banking.json', 'r') as f:
    BANKING_DATA = json.load(f)

# Luhn Check Function
def luhn_check(number: str) -> bool:
    """
    Validate using Luhn algorithm (checksum).

    Args:
        number (str): The number (credit card or routing number) to validate.

    Returns:
        bool: True if valid, False if invalid.
    """
    digits = [int(d) for d in number]
    total = sum(digits[-1::-2])  # Sum digits in odd positions (from right)
    for d in digits[-2::-2]:     # Double digits in even positions
        dbl = d * 2
        total += (dbl // 10) + (dbl % 10)

    return total % 10 == 0

def aba_checksum(routing_number: str) -> bool:
    """
    Validate ABA routing number using the ABA checksum algorithm.

    Args:
        routing_number (str): The 9-digit ABA routing number to validate.

    Returns:
        bool: True if valid, False if invalid.
    """
    # ABA Routing number weights: [3, 7, 1, 3, 7, 1, 3, 7, 1]
    weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]

    # Ensure the routing number has 9 digits
    if len(routing_number) != 9 or not routing_number.isdigit():
        return False

    # Calculate the weighted sum
    weighted_sum = sum(int(digit) * weight for digit, weight in zip(routing_number, weights))

    # Check if the sum modulo 10 is zero
    return weighted_sum % 10 == 0


def validate_routing_number(routing_number: str) -> dict:
    result = {
        "routing_number": routing_number,
        "valid": False,
        "institution_type": None,
        "bank_name": None,
        "address": None,
        "city": None,
        "state": None,
        "zip_code": None,
        "phone_number": None,
        "sending_point_routing": None,
        "error": None
    }

    routing_number = routing_number.strip()

    # Routing number length check (must be 9 digits)
    if len(routing_number) != 9 or not routing_number.isdigit():
        result["error"] = "Invalid length or non-numeric"
        return result

    # Apply ABA checksum algorithm
    if not aba_checksum(routing_number):
        result["error"] = "Invalid ABA checksum"
        return result

    # Normalize the routing number format (ensure string format)
    routing_number_str = str(routing_number)

    # Look up metadata in the BANKING_DATA from JSON file
    metadata = BANKING_DATA.get(routing_number_str)

    if metadata:
        result["valid"] = True
        # Extract the relevant metadata fields
        result["institution_type"] = metadata.get("record_type", None)
        result["bank_name"] = metadata.get("customer_name", None)
        result["address"] = metadata.get("address", None)
        result["city"] = metadata.get("city", None)
        result["state"] = metadata.get("state", None)
        result["zip_code"] = metadata.get("zip_code", None)
        result["phone_number"] = metadata.get("phone_number", None)
        result["sending_point_routing"] = metadata.get("sending_point_routing", None)
    else:
        result["error"] = "Bank data not found"

    return result

def validate_swift(swift_code: str) -> dict:
    result = {
        "swift": swift_code,
        "valid": False,
        "bank_code": None,
        "bank_name": None,
        "country_code": None,
        "country": None,
        "location_code": None,
        "branch_code": None,
        "error": None
    }

    swift_code = swift_code.strip().upper()

    # Length must be exactly 8 or 11
    if len(swift_code) not in [8, 11]:
        result["error"] = "Invalid length"
        return result

    # Validate format with regex
    if not re.match(r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$", swift_code):
        result["error"] = "Invalid format"
        return result

    try:
        swift_obj = schwifty.BIC(swift_code)
        result["valid"] = True
        result["bank_code"] = swift_obj.bank_code
        result["bank_name"] = swift_obj.bank_name  # Ensure bank name
        result["country_code"] = swift_obj.country_code
        result["country"] = pycountry.countries.get(alpha_2=swift_obj.country_code).name
        result["location_code"] = swift_obj.location_code
        result["branch_code"] = swift_obj.branch_code
    except ValueError as e:
        result["error"] = str(e)

    return result

def validate_iban(iban: str) -> dict:
    result = {
        "iban": iban,
        "valid": False,
        "bank_code": None,
        "bank_name": None,
        "branch_code": None,
        "account_number": None,
        "error": None
    }

    iban = iban.strip().upper()

    try:
        iban_obj = schwifty.IBAN(iban)
        result["valid"] = True
        result["bank_code"] = iban_obj.bank_code
        result["bank_name"] = iban_obj.bank_name  # Ensure bank name
        result["branch_code"] = iban_obj.branch_code
        result["account_number"] = iban_obj.account_code
    except ValueError as e:
        result["error"] = str(e)

    return result


# Credit Card Validation using Luhn and BIN patterns
NETWORK_REGEX = {
    "American Express": r"^3[47]\d{13}$",
    "MasterCard": r"^(?:5[1-5]\d{14}|2(?:2(?:2[1-9]|[3-9]\d)|[3-6]\d{2}|7(?:[01]\d|20))\d{12})$",
    "Visa": r"^4\d{12}(?:\d{3})?(?:\d{3})?$",
    "Discover": r"^(?:6011\d{12}|65\d{14}|64[4-9]\d{13}|622(?:12[6-9]|1[3-9]\d|[2-8]\d{2}|9(?:[01]\d|2[0-5]))\d{10})$",
    "Diners Club": r"^3(?:0[0-5]|[68]\d)\d{11}$",
    "JCB": r"^35(?:2[89]|[3-8]\d)\d{12,15}$",
    "Maestro": r"^(?:50[129]|5[6-9]\d|6\d{2})\d{10,17}$",
    "Visa Electron": r"^(?:4026|417500|4508|4844|491[37])\d{8,15}$",
    "UnionPay": r"^62\d{14,17}$",
}

def validate_credit_card(raw_cc: str) -> dict:
    """
    Normalize, classify, and validate a credit card number.
    Accepts raw_cc as any type (str, int, etc.) and coerces to string.
    Returns a dict:
      - valid: True if brand detected & Luhn check passes
      - luhn_valid: True/False
      - brand: network name or None
      - length: number of digits
      - iin: first 6 digits (or full number if shorter)
      - mii: first digit as int or None
      - mii_description: industry description for the MII digit
      - check_digit: last digit of the card number
    """
    # Major Industry Identifier descriptions
    MII_MAPPING = {
        0: "ISO/TC 68 and other industry assignments",
        1: "Airlines",
        2: "Airlines and future industry assignments",
        3: "Travel and entertainment",
        4: "Banking and financial",
        5: "Banking and financial",
        6: "Merchandising and banking",
        7: "Petroleum",
        8: "Healthcare, telecommunications and other",
        9: "National assignment"
    }

    # 1) Coerce to string and normalize: keep only digits
    raw = str(raw_cc or "")
    cc = re.sub(r"\D", "", raw)
    length = len(cc)
    iin = cc[:6] if length >= 6 else cc
    mii = int(cc[0]) if length >= 1 else None
    check_digit = int(cc[-1]) if length >= 1 else None

    # 2) Quick reject: most cards are 12–19 digits
    if not (12 <= length <= 19):
        return {
            "valid": False,
            "luhn_valid": False,
            "brand": None,
            "length": length,
            "iin": iin,
            "mii": mii,
            "mii_description": MII_MAPPING.get(mii),
            "check_digit": check_digit
        }

    # 3) Luhn check using the luhn_check function
    luhn_ok = luhn_check(cc)

    # 4) Brand detection by regex
    brand = None
    for name, pattern in NETWORK_REGEX.items():
        if re.match(pattern, cc):
            brand = name
            break

    valid = bool(brand) and luhn_ok

    return {
        "valid": valid,
        "luhn_valid": luhn_ok,
        "brand": brand,
        "length": length,
        "iin": iin,
        "mii": mii,
        "mii_description": MII_MAPPING.get(mii),
        "check_digit": check_digit
    }


if __name__ == "__main__":
    import json
    from pprint import pprint

    # Test data for each validator
    test_data = {
        "swift": [
            "DEUTDEFF",        # Valid SWIFT primary
            "NEDSZAJJXXX",     # Valid SWIFT branch
            "INVALID1",        # Invalid SWIFT
            "BOFAUS3NXXX",     # Valid US SWIFT branch
            "WFBIUS6S"
        ],
        "routing_number": [
            "011000015",       # Valid routing number (Federal Reserve Bank)
            "123456789",       # Invalid routing number (does not exist)
            "111000025",       # Valid routing number (Commercial Bank)
            "063107513",
            "121000248"
        ],
        "iban": [
            "GB29NWBK60161331926819",  # Valid UK IBAN
            "DE89370400440532013000",  # Valid German IBAN
            "FR7630006000011234567890189",  # Valid French IBAN
            "INVALIDIBAN123",    # Invalid IBAN
        ],
        "credit_card": [
            "4111111111111111",   # Valid Visa card
            "5105105105105100",   # Valid MasterCard
            "378282246310005",    # Valid American Express
            "123456789012345",    # Invalid card (too short)
            "0000000000000000",   # Invalid card (fails Luhn check)
        ]
    }

    # Test SWIFT/BIC Code Validation
    print("\n=== SWIFT Code Validation ===")
    for swift in test_data["swift"]:
        print(f"\nInput: {swift}")
        pprint(validate_swift(swift))

    # Test Routing Number Validation
    print("\n=== Routing Number Validation ===")
    for routing_number in test_data["routing_number"]:
        print(f"\nInput: {routing_number}")
        pprint(validate_routing_number(routing_number))

    # Test IBAN Validation
    print("\n=== IBAN Validation ===")
    for iban in test_data["iban"]:
        print(f"\nInput: {iban}")
        pprint(validate_iban(iban))

    # Test Credit Card Validation
    print("\n=== Credit Card Validation ===")
    for cc in test_data["credit_card"]:
        print(f"\nInput: {cc}")
        pprint(validate_credit_card(cc))

