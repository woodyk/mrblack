#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: passport.py
# Author: Wadih Khairallah
# Description: 
# Created: 2025-04-22 15:45:17
# Modified: 2025-04-22 15:48:22

import re

def validate_passport(passport_number: str) -> dict:
    """
    Validate a U.S. Passport number.

    Args:
        passport_number (str): The passport number to validate.

    Returns:
        dict: Validation result with metadata.
    """
    result = {
        "passport_number": passport_number,
        "valid": False,
        "error": None
    }

    # Normalize passport number (remove extra spaces)
    passport_number = passport_number.strip()

    # Step 1: Check if the input matches the U.S. passport number format
    if re.match(r'^[A-Z]\d{8}$', passport_number):
        result["valid"] = True
        return result

    result["error"] = "Invalid passport number format"
    return result


def validate_mrz(mrz: str) -> dict:
    """
    Validate a Machine Readable Zone (MRZ) string and extract metadata.

    Args:
        mrz (str): The MRZ string to validate.

    Returns:
        dict: Validation result with metadata.
    """
    result = {
        "valid": False,
        "error": None
    }

    # Normalize MRZ by removing non-alphanumeric characters (except '<')
    mrz = re.sub(r'[^A-Z0-9<]', '', mrz.upper())

    # Check if MRZ length is valid for TD3 format (88 characters)
    if len(mrz) != 88:
        result["error"] = "Invalid MRZ length"
        return result

    # Extract fields based on TD3 format
    doc_type = mrz[0:1]
    country_code = mrz[1:3]
    surname_given_names = mrz[3:44]
    passport_number = mrz[44:53]
    passport_check_digit = mrz[53:54]
    nationality = mrz[54:57]
    dob = mrz[57:63]
    gender = mrz[63:64]
    expiry_date = mrz[64:70]
    expiry_check_digit = mrz[70:71]
    optional_data = mrz[71:88]

    # Validate document type
    if doc_type != 'P':
        result["error"] = "Invalid document type"
        return result

    # Validate surname and given names (should include the '<<' separator)
    if '<<' not in surname_given_names:
        result["error"] = "Invalid surname and given names format"
        return result

    # Validate passport number and check digit
    if not passport_number.isalnum() or not passport_check_digit.isdigit():
        result["error"] = "Invalid passport number or check digit"
        return result

    # Validate nationality (must be 3 letters)
    if not nationality.isalpha() or len(nationality) != 3:
        result["error"] = "Invalid nationality code"
        return result

    # Validate date of birth (YYMMDD format)
    if not dob.isdigit() or len(dob) != 6:
        result["error"] = "Invalid date of birth"
        return result

    # Validate gender (M, F, X)
    if gender not in ['M', 'F', 'X']:
        result["error"] = "Invalid gender code"
        return result

    # Validate expiration date (YYMMDD format)
    if not expiry_date.isdigit() or len(expiry_date) != 6:
        result["error"] = "Invalid expiration date"
        return result

    # Validate check digits (simple modulus 10 check)
    check_digits = [passport_number, dob, expiry_date]
    for field in check_digits:
        if sum(int(digit) for digit in field) % 10 != int(passport_check_digit):
            result["error"] = "Check digit validation failed"
            return result

    # If all validations pass, populate metadata
    result["valid"] = True
    result["metadata"] = {
        "document_type": doc_type,
        "country_code": country_code,
        "surname_given_names": surname_given_names.replace('<', ' ').strip(),
        "passport_number": passport_number,
        "nationality": nationality,
        "dob": dob,
        "gender": gender,
        "expiry_date": expiry_date,
        "optional_data": optional_data
    }

    return result


if __name__ == "__main__":
    # Test with a valid U.S. passport number
    passport_number = "A12345678"
    print(f"Testing valid passport number: {passport_number}")
    print(validate_passport(passport_number))
    print()

    # Test with a valid MRZ string (formatted properly)
    mrz_string = """P<GBRSMITH<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<1234567890GBR6705089M1601019<<<<<<<<<<<<<<00"""
    print(f"Testing valid MRZ string:\n{mrz_string}")
    print(validate_mrz(mrz_string))
    print()

    # Test with an invalid passport number
    invalid_passport_number = "123456789"
    print(f"Testing invalid passport number: {invalid_passport_number}")
    print(validate_passport(invalid_passport_number))
    print()

    # Test with an invalid MRZ string (incorrect format)
    invalid_mrz_string = "INVALIDMRZSTRING"
    print(f"Testing invalid MRZ string: {invalid_mrz_string}")
    print(validate_mrz(invalid_mrz_string))
    print()

    # Test with an invalid MRZ string (incorrect length)
    invalid_mrz_length_string = """P<GBRSMITH<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<1234567890GBR6705089M1601019"""
    print(f"Testing invalid MRZ string with incorrect length:\n{invalid_mrz_length_string}")
    print(validate_mrz(invalid_mrz_length_string))
    print()

