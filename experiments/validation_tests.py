#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: validate.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-12-02 00:20:55
# Modified: 2025-03-24 13:57:24

#from cctek import CardCheck
#from schwifty import IBAN
#from schwifty import IBAN, BIC
from collections import Counter
from dateutil.parser import parse
from urllib.parse import urlparse
import dns.resolver
import idna
import ipaddress
import json
import math
import platform
import re
import requests
import select
import socket
import struct
import time
import uuid

def is_valid_date(self, date_str):
    try:
        parse(date_str, fuzzy=True)
        return True
    except:
        return False

def calculate_entropy(text):
    """
    Calculates the Shannon entropy of the input text.

    Args:
        text (str): Input text.

    Returns:
        float: Shannon entropy of the text.
    """
    if not text:
        return 0
    counter = Counter(text)
    length = len(text)
    entropy = -sum((count / length) * math.log2(count / length) for count in counter.values())
    return entropy

def is_url(string):
    """
    Checks if a given string is a URL.

    Args:
        string (str): The string to check.

    Returns:
        bool: True if the string is a URL, False otherwise.
    """
    try:
        result = urlparse(string)
        return all([result.scheme in ("http", "https"), result.netloc])
    except:
        return False

def is_phone_number(text, validate_country_code=False):
    """
    Validates if the given string matches the general pattern of an international phone number.

    Args:
        text (str): The string to validate.
        validate_country_code (bool): Whether to validate the country code.

    Returns:
        bool: True if the string matches a valid phone number pattern, False otherwise.
    """
    # Remove common separators and extensions
    cleaned_text = re.sub(r'[^\d+]', '', text)
    cleaned_text = re.sub(r'(?:ext\.?|x)\d+$', '', cleaned_text, flags=re.IGNORECASE)

    # Check total digit count (7-15 digits as per E.164)
    digits_only = re.sub(r'\D', '', cleaned_text)
    if not (7 <= len(digits_only) <= 15):
        return False

    # Optional: Validate the country code
    if validate_country_code:
        match = re.match(r'^\+(\d{1,3})', cleaned_text)
        if not match:
            return False  # Must have a country code if validation is enabled
        country_code = match.group(1)
        if country_code not in VALID_COUNTRY_CODES:
            return False

    # Check if the text matches the phone number pattern
    if not PHONE_NUMBER_REGEX.match(text):
        return False

    # Optional: Additional heuristic checks
    entropy = calculate_entropy(text)
    if entropy < 3.5:  # Adjust threshold based on empirical testing
        return False

    return True

# Regex for validating email format according to RFC 5321/5322
EMAIL_REGEX = re.compile(r"""
    ^                               # Start of string
    [a-zA-Z0-9._%+-]+               # Local part: alphanumeric + special characters ._%+-
    @                               # @ symbol
    (?:[a-zA-Z0-9-]+\.)+            # Subdomain(s): alphanumeric and -
    [a-zA-Z]{2,63}                  # TLD: 2-63 alphanumeric characters
    $                               # End of string
""", re.VERBOSE)

def has_mx_record(domain):
    """
    Checks if a domain has valid MX records.

    Args:
        domain (str): The domain to check.

    Returns:
        bool: True if the domain has MX records, False otherwise.
    """
    try:
        mx_records = dns.resolver.resolve(domain, 'MX')
        return len(mx_records) > 0
    except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer, dns.resolver.Timeout):
        return False  # Domain does not exist or no MX records found
    except Exception:
        return False  # Catch all other DNS errors

def get_valid_tlds():
    """
    Fetches the list of valid TLDs from IANA and returns them as a set.

    Returns:
        set: A set of valid TLDs.
    """
    try:
        # Fetch the IANA TLD list
        url = "https://data.iana.org/TLD/tlds-alpha-by-domain.txt"
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for HTTP errors

        # Split the response by lines and skip comments (lines starting with #)
        tlds = {line.strip().lower() for line in response.text.splitlines() if not line.startswith("#")}
        return tlds
    except requests.RequestException as e:
        print(f"Error fetching TLD list: {e}")
        return set()

def is_email(text):
    """
    Validates if the given string is a valid email address format.

    Args:
        text (str): The email address to validate.

    Returns:
        bool: True if the string is in a valid email format, False otherwise.
    """
    try:
        # Split the email into local and domain parts
        local, domain = text.split('@')

        # Validate email structure using regex
        if not EMAIL_REGEX.match(text):
            return False

        # Encode domain to Punycode for IDN support
        idna.encode(domain).decode('ascii')  # Raises an exception if invalid

        return True
    except ValueError:
        # Splitting '@' failed (e.g., missing '@')
        return False
    except idna.IDNAError:
        # Invalid internationalized domain name
        return False

# Regex for validating email format according to RFC 5321/5322
EMAIL_REGEX = re.compile(r"""
    ^                               # Start of string
    [a-zA-Z0-9._%+-]+               # Local part: alphanumeric + special characters ._%+-
    @                               # @ symbol
    (?:[a-zA-Z0-9-]+\.)+            # Subdomain(s): alphanumeric and -
    [a-zA-Z]{2,63}                  # TLD: 2-63 alphanumeric characters
    $                               # End of string
""", re.VERBOSE)

def is_email(text):
    """
    Validates if the given string is a valid email address format.

    Args:
        text (str): The email address to validate.

    Returns:
        bool: True if the string is in a valid email format, False otherwise.
    """
    try:
        # Split the email into local and domain parts
        local, domain = text.split('@')

        # Validate email structure using regex
        if not EMAIL_REGEX.match(text):
            return False

        # Encode domain to Punycode for IDN support
        idna.encode(domain).decode('ascii')  # Raises an exception if invalid

        return True
    except ValueError:
        # Splitting '@' failed (e.g., missing '@')
        return False
    except idna.IDNAError:
        # Invalid internationalized domain name
        return False

# Regex for domain validation
DOMAIN_REGEX = re.compile(r"""
    ^                                   # Start of string
    (?=.{1,253}$)                       # Entire domain length: 1-253 characters
    (?:                                 # Begin a label group
        [a-zA-Z0-9]                    # Label starts with alphanumeric
        [a-zA-Z0-9-]{0,61}             # Followed by alphanumeric or hyphen (max 63 chars total per label)
        [a-zA-Z0-9]                    # Label ends with alphanumeric
    \.)+                                # Repeat for subdomains, separated by dots
    [a-zA-Z]{2,63}                     # TLD: 2-63 alphanumeric characters
    $                                   # End of string
""", re.VERBOSE)

def is_domain(domain):
    """
    Validates if the given string is a valid domain name format.

    Args:
        domain (str): The domain name to validate.

    Returns:
        bool: True if the string is a valid domain name, False otherwise.
    """
    try:
        # Convert domain to Punycode for IDN support
        punycode_domain = idna.encode(domain).decode('ascii')

        # Validate domain using regex
        return bool(DOMAIN_REGEX.match(punycode_domain))
    except idna.IDNAError:
        # Invalid internationalized domain name
        return False

# Regex to validate SSN format
SSN_REGEX = re.compile(r"^\d{3}-?\d{2}-?\d{4}$")

def is_social_security(ssn):
    """
    Validates if the given text is a valid U.S. Social Security Number (SSN).

    Args:
        ssn (str): The input string to validate.

    Returns:
        bool: True if the input is a valid SSN, False otherwise.
    """
    # Remove dashes for processing
    cleaned_ssn = ssn.replace("-", "")

    # Ensure it matches the basic format of 9 digits
    if not SSN_REGEX.match(ssn) or len(cleaned_ssn) != 9:
        return False

    # Split into components
    area, group, serial = cleaned_ssn[:3], cleaned_ssn[3:5], cleaned_ssn[5:]

    # Area Number Rules
    if area == "000" or area == "666" or int(area) >= 900:
        return False

    # Group Number Rules
    if group == "00" or not (1 <= int(group) <= 99):
        return False

    # Serial Number Rules
    if serial == "0000" or not (1 <= int(serial) <= 9999):
        return False

    return True

def is_email_with_tld_check(text, tld_list, check_mx=False):
    """
    Validates if a given string is a valid email address with TLD validation.

    Args:
        text (str): The email address to validate.
        tld_list (set): A set of valid TLDs.
        check_mx (bool): Whether to validate the domain's MX records.

    Returns:
        bool: True if the string is a valid email address, False otherwise.
    """
    try:
        # Split local and domain parts
        local, domain = text.split('@')
        domain_parts = domain.split('.')

        # Ensure the TLD is in the valid list
        tld = domain_parts[-1].lower() if len(domain_parts) > 1 else None
        if not tld or tld not in tld_list:
            return False

        # Validate email format using regex
        if not EMAIL_REGEX.match(text):
            return False

        # Encode domain to Punycode for internationalized domains
        domain = idna.encode(domain).decode('ascii')

        # Optionally check MX records for the domain
        if check_mx and not has_mx_record(domain):
            return False

        return True
    except ValueError:
        return False  # Invalid email format (missing @ or domain)
    except Exception:
        return False

# Credit card catch all regex
# ^(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3(?:0[0-5]|[68][0-9])[0-9]{11}|6(?:011|5[0-9]{2})[0-9]{12}|(?:2131|1800|35\d{3})\d{11})$
def is_credit_card(card_number: str) -> bool:
    """
    Validates a credit card number using the Luhn algorithm.

    Args:
        card_number (str): The credit card number to validate.

    Returns:
        bool: True if the card number is valid, False otherwise.
    """
    # Step 1: Normalize input by removing all non-numeric characters
    normalized = re.sub(r"\D", "", card_number)

    # Step 2: Match against the credit card regex pattern
    card_pattern = re.compile(
        r"^(?:4[0-9]{12}(?:[0-9]{3})?"  # Visa
        r"|5[1-5][0-9]{14}"            # MasterCard
        r"|3[47][0-9]{13}"             # American Express
        r"|3(?:0[0-5]|[68][0-9])[0-9]{11}"  # Diners Club
        r"|6(?:011|5[0-9]{2})[0-9]{12}"  # Discover
        r"|(?:2131|1800|35\d{3})\d{11})$"  # JCB
    )

    # Step 2: Ensure the card number has a valid length (typically 13 to 19 digits)
    if not (13 <= len(normalized) <= 19):
        return False

    # Step 3: Apply the Luhn algorithm
    # Convert the card number to a list of digits
    digits = [int(d) for d in normalized]

    # Remove the check digit (last digit)
    check_digit = digits.pop()

    # Reverse the order of the remaining digits
    digits.reverse()

    # Process digits using the Luhn algorithm
    processed_digits = []
    for index, digit in enumerate(digits):
        if index % 2 == 0:  # Even index (0-based)
            doubled_digit = digit * 2
            if doubled_digit > 9:  # Subtract 9 if greater than 9
                doubled_digit -= 9
            processed_digits.append(doubled_digit)
        else:
            processed_digits.append(digit)

    # Calculate the total sum
    total = check_digit + sum(processed_digits)

    # Validate the total (must be divisible by 10)
    return total % 10 == 0
    

def validate_credit_card(card_number):
    """
    Validates a credit card number using pycctek.
    
    Args:
        card_number (str): The credit card number to validate.
        
    Returns:
        dict: Validation results, including Luhn check, issuer, and type.
    """
    # Step 1: Validate basic credit card structure and Luhn check
    if not CardCheck.is_valid(card_number):
        return {"valid": False, "reason": "Invalid credit card number format or failed Luhn check."}

    # Step 2: Retrieve card details
    card_data = CardCheck.get_card_data(card_number)
    if not card_data:
        return {"valid": False, "reason": "BIN/IIN not found in local data."}

    # Step 3: Return validation results
    return {
        "valid": True,
        "issuer": card_data.get("issuer", "Unknown"),
        "type": card_data.get("type", "Unknown"),
        "country": card_data.get("country", "Unknown"),
    }

def is_ipv4(address):
    """
    Validates if the given string is a valid IPv4 address.

    Args:
        address (str): The string to validate.

    Returns:
        bool: True if the string is a valid IPv4 address, False otherwise.
    """
    try:
        # Use socket's built-in validation
        socket.inet_pton(socket.AF_INET, address)
        return True
    except (socket.error, ValueError):
        return False

def is_ipv6(address):
    """
    Validates if the given string is a valid IPv6 address.

    Args:
        address (str): The string to validate.

    Returns:
        bool: True if the string is a valid IPv6 address, False otherwise.
    """
    try:
        # Use socket's built-in validation
        socket.inet_pton(socket.AF_INET6, address)
        return True
    except (socket.error, ValueError):
        return False

def is_mac_address(address):
    """
    Validates if the given string is a valid MAC address.

    Args:
        address (str): The string to validate.

    Returns:
        bool: True if the string is a valid MAC address, False otherwise.
    """
    # Regular expression for MAC address validation
    mac_regex = re.compile(r"^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$")
    return bool(mac_regex.match(address))

def is_hostname(hostname):
    """
    Validates if the given string is a valid hostname.

    Args:
        hostname (str): The string to validate.

    Returns:
        bool: True if the string is a valid hostname, False otherwise.
    """
    if len(hostname) > 253:
        return False
    labels = hostname.split(".")
    for label in labels:
        if len(label) < 1 or len(label) > 63:
            return False
        if not label.isalnum() and "-" in label:
            return False
        if label.startswith("-") or label.endswith("-"):
            return False
    return True

def is_cidr(cidr):
    """
    Validates if the given string is a valid CIDR notation.

    Args:
        cidr (str): The string to validate.

    Returns:
        bool: True if the string is a valid CIDR notation, False otherwise.
    """
    try:
        address, prefix = cidr.split("/")
        prefix = int(prefix)
        if is_ipv4(address) and 0 <= prefix <= 32:
            return True
        if is_ipv6(address) and 0 <= prefix <= 128:
            return True
    except (ValueError, IndexError):
        return False
    return False

def is_private_ip(address):
    """
    Checks if the given IP address is private.

    Args:
        address (str): The IP address to validate.

    Returns:
        bool: True if the address is private, False otherwise.
    """
    try:
        ip = ipaddress.ip_address(address)
        return ip.is_private
    except ValueError:
        return False

def is_loopback_ip(address):
    """
    Checks if the given IP address is a loopback address.

    Args:
        address (str): The IP address to validate.

    Returns:
        bool: True if the address is a loopback address, False otherwise.
    """
    try:
        ip = ipaddress.ip_address(address)
        return ip.is_loopback
    except ValueError:
        return False

def is_multicast_ip(address):
    """
    Checks if the given IP address is a multicast address.

    Args:
        address (str): The IP address to validate.

    Returns:
        bool: True if the address is a multicast address, False otherwise.
    """
    try:
        ip = ipaddress.ip_address(address)
        return ip.is_multicast
    except ValueError:
        return False

def calculate_checksum(packet):
    """
    Calculates the checksum of an ICMP packet.

    Args:
        packet (bytes): The packet to calculate the checksum for.

    Returns:
        int: The checksum.
    """
    if len(packet) % 2:
        packet += b'\x00'
    checksum = sum((packet[i] << 8) + packet[i+1] for i in range(0, len(packet), 2))
    checksum = (checksum >> 16) + (checksum & 0xFFFF)
    checksum += checksum >> 16
    return ~checksum & 0xFFFF

def create_icmp_packet(identifier, sequence_number, ipv6=False):
    """
    Creates an ICMP or ICMPv6 echo request packet.

    Args:
        identifier (int): Identifier for the packet.
        sequence_number (int): Sequence number of the packet.
        ipv6 (bool): Whether to create an ICMPv6 packet.

    Returns:
        bytes: The ICMP or ICMPv6 echo request packet.
    """
    if ipv6:
        header = struct.pack("!BBHHH", 128, 0, 0, identifier, sequence_number)  # ICMPv6 Type 128
    else:
        header = struct.pack("!BBHHH", 8, 0, 0, identifier, sequence_number)  # ICMP Type 8
    payload = b'PythonPing'
    checksum = calculate_checksum(header + payload)
    header = struct.pack("!BBHHH", header[0], header[1], checksum, identifier, sequence_number)
    return header + payload

def is_ping(address, timeout=2):
    """
    Checks if an IP address is reachable by sending an ICMP/ICMPv6 echo request.

    Args:
        address (str): The IP address to ping.
        timeout (int): Timeout in seconds for the response.

    Returns:
        bool: True if the address is reachable, False otherwise.
    """
    try:
        # Determine if the address is IPv4 or IPv6
        family = socket.AF_INET6 if ":" in address else socket.AF_INET
        proto = socket.IPPROTO_ICMP if family == socket.AF_INET else socket.IPPROTO_ICMPV6

        # Create a raw socket
        with socket.socket(family, socket.SOCK_RAW, proto) as sock:
            sock.settimeout(timeout)
            identifier = int(time.time() * 1000) & 0xFFFF  # Unique identifier
            sequence_number = 1
            packet = create_icmp_packet(identifier, sequence_number, ipv6=(family == socket.AF_INET6))

            # Resolve the address and send the packet
            sock.sendto(packet, (address, 0))
            start_time = time.time()

            while True:
                # Wait for a response
                ready = select.select([sock], [], [], timeout)
                if not ready[0]:
                    return False  # Timeout

                # Receive the packet
                recv_packet, addr = sock.recvfrom(1024)
                elapsed_time = time.time() - start_time

                # Unpack the response and validate it
                icmp_type = recv_packet[20] if family == socket.AF_INET else recv_packet[0]
                recv_id = struct.unpack("!H", recv_packet[24:26])[0] if family == socket.AF_INET else struct.unpack("!H", recv_packet[4:6])[0]

                if recv_id == identifier and (icmp_type == 0 or icmp_type == 129):  # Echo Reply
                    return True

                if elapsed_time > timeout:
                    return False
    except PermissionError:
        print("Error: Raw sockets require administrative privileges.")
        return False
    except Exception as e:
        print(f"Ping failed: {e}")
        return False

def is_port_open(ip, port, timeout=2):
    try:
        with socket.create_connection((ip, port), timeout):
            return True
    except socket.error:
        return False

def is_uuid(value):
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False

def is_json(value):
    try:
        json.loads(value)
        return True
    except ValueError:
        return False

def is_os_windows():
    return platform.system().lower() == "windows"

def is_public_ip(address):
    """
    Checks if the given IP address is public.

    Args:
        address (str): The IP address to validate.

    Returns:
        bool: True if the address is public, False otherwise.
    """
    try:
        ip = ipaddress.ip_address(address)
        return ip.is_global and not ip.is_private
    except ValueError:
        return False

def is_reserved_ip(address):
    """
    Checks if the given IP address is reserved.

    Args:
        address (str): The IP address to validate.

    Returns:
        bool: True if the address is reserved, False otherwise.
    """
    try:
        ip = ipaddress.ip_address(address)
        return ip.is_reserved
    except ValueError:
        return False

def is_broadcast_ip(address, network="255.255.255.255"):
    """
    Checks if the given IPv4 address is a broadcast address.

    Args:
        address (str): The IPv4 address to validate.
        network (str): The subnet to check against (default: "255.255.255.255").

    Returns:
        bool: True if the address is a broadcast address, False otherwise.
    """
    try:
        ip = ipaddress.ip_address(address)
        if ip.version == 4:
            net = ipaddress.ip_network(network, strict=False)
            return ip == net.broadcast_address
        return False  # IPv6 doesn't have a broadcast address
    except ValueError:
        return False

def is_geo_coordinate(coordinate):
    """
    Validates if the given string is a geographical coordinate in decimal degrees (DD) 
    or degrees, minutes, seconds (DMS) format, supporting variable decimal precision.

    Args:
        coordinate (str): The input string to validate.

    Returns:
        bool: True if the input is a valid geographical coordinate, False otherwise.
    """
    # Regex for Decimal Degrees (DD) format
    dd_regex = re.compile(
        r"""
        ^                       # Start of string
        (-?\d{1,2}(?:\.\d+)?),\s*    # Latitude: -90 to 90, optional decimal part
        (-?\d{1,3}(?:\.\d+)?)$       # Longitude: -180 to 180, optional decimal part
        """,
        re.VERBOSE
    )

    # Regex for Degrees, Minutes, Seconds (DMS) format
    dms_regex = re.compile(
        r"""
        ^                                       # Start of string
        (-?\d{1,2})°\s*(\d{1,2})'\s*(\d{1,2}(\.\d+)?)"\s*([NSns])?,?\s*  # Latitude with variable seconds precision
        (-?\d{1,3})°\s*(\d{1,2})'\s*(\d{1,2}(\.\d+)?)"\s*([EWew])?$      # Longitude with variable seconds precision
        """,
        re.VERBOSE
    )

    # Check Decimal Degrees (DD)
    match_dd = dd_regex.match(coordinate)
    if match_dd:
        latitude, longitude = float(match_dd.group(1)), float(match_dd.group(2))
        if -90.0 <= latitude <= 90.0 and -180.0 <= longitude <= 180.0:
            return True

    # Check Degrees, Minutes, Seconds (DMS)
    match_dms = dms_regex.match(coordinate)
    if match_dms:
        lat_deg, lat_min, lat_sec = int(match_dms.group(1)), int(match_dms.group(2)), float(match_dms.group(3))
        lon_deg, lon_min, lon_sec = int(match_dms.group(6)), int(match_dms.group(7)), float(match_dms.group(8))
        if (0 <= lat_deg <= 90 and 0 <= lon_deg <= 180 and
            0 <= lat_min < 60 and 0 <= lon_min < 60 and
            0 <= lat_sec < 60 and 0 <= lon_sec < 60):
            return True

    return False

def is_bank_account(account_number: str) -> bool:
    """
    Normalizes and validates if the given string is a plausible bank account number.
    Uses a catch-all regex for basic format validation and Schwifty for IBAN validation.

    Args:
        account_number (str): The input bank account number.

    Returns:
        bool: True if valid, False otherwise.
    """
    if not account_number or not isinstance(account_number, str):
        return False

    # Normalize: Remove spaces, dashes, and dots
    normalized = re.sub(r'[\s.\-]', '', account_number)

    # Catch-all regex validation
    if not re.match(r'^[A-Za-z0-9][A-Za-z0-9\s.\-]{7,33}[A-Za-z0-9]$', account_number):
        return False

    # Attempt IBAN validation using Schwifty
    try:
        iban = IBAN(normalized)
        return iban.validate()
    except ValueError:
        pass  # Not a valid IBAN

    # Add additional non-IBAN validation logic here if needed
    return False

def validate_vin(vin: str) -> bool:
    """
    Validates a VIN (Vehicle Identification Number) using the ISO 3779 checksum algorithm.

    Args:
        vin (str): The VIN to validate.

    Returns:
        bool: True if the VIN is valid, False otherwise.
    """
    if len(vin) != 17 or not vin.isalnum():
        return False

    # Define letter-to-number mapping
    letter_to_number = {
        "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8,
        "J": 1, "K": 2, "L": 3, "M": 4, "N": 5, "P": 7, "R": 9, "S": 2,
        "T": 3, "U": 4, "V": 5, "W": 6, "X": 7, "Y": 8, "Z": 9
    }

    # VIN weights
    weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]

    # Replace letters with numeric values
    vin_numeric = []
    for char in vin:
        if char.isdigit():
            vin_numeric.append(int(char))
        elif char.isalpha() and char in letter_to_number:
            vin_numeric.append(letter_to_number[char])
        else:
            return False

    # Calculate weighted sum
    weighted_sum = sum(value * weight for value, weight in zip(vin_numeric, weights))

    # Compute checksum
    checksum = weighted_sum % 11

    # Verify check digit
    check_digit = vin[8]
    if checksum == 10:
        return check_digit == "X"
    return str(checksum) == check_digit


def validate_asn(asn: int, bit: int = 32) -> bool:
    """
    Validate an Autonomous System Number (ASN).

    Args:
        asn (int): ASN to validate.
        bit (int): ASN bit size (16 or 32). Defaults to 32.

    Returns:
        bool: True if valid, False otherwise.
    """
    if bit == 16:
        return 0 <= asn <= 65535
    elif bit == 32:
        return 0 <= asn <= 4294967295
    else:
        raise ValueError("Invalid bit size. Only 16 or 32-bit ASNs are supported.")

def validate_asn_format(asn: str, bit: int = 32) -> bool:
    """
    Validate ASN format using regex.

    Args:
        asn (str): ASN to validate.
        bit (int): ASN bit size (16 or 32). Defaults to 32.

    Returns:
        bool: True if valid, False otherwise.
    """
    patterns = {
        16: r"^(0|[1-9][0-9]{0,4}|[1-5][0-9]{5}|6[0-4][0-9]{4}|65[0-4][0-9]{3}|655[0-2][0-9]{2}|6553[0-5])$",
        32: r"^(0|[1-9][0-9]{0,9}|[1-3][0-9]{9}|4[0-1][0-9]{8}|42[0-8][0-9]{7}|429[0-3][0-9]{6}|4294[0-8][0-9]{5}|42949[0-5][0-9]{4}|429496[0-6][0-9]{3}|4294967[0-9]{2}|42949672[0-8][0-9]|429496729[0-5])$",
    }
    pattern = patterns.get(bit)
    if not pattern:
        raise ValueError("Invalid bit size. Only 16 or 32-bit ASNs are supported.")
    return bool(re.match(pattern, asn.strip()))

# Examples
print(validate_asn(64512, 16))  # True (Private ASN)
print(validate_asn_format("4294967295", 32))  # True (Largest 32-bit ASN)
print(validate_asn(70000, 16))  # False (Out of range)

# Regex patterns for card types
card_patterns = {
    "Visa": r"^4[0-9]{12}(?:[0-9]{3})?$",
    "MasterCard": r"^5[1-5][0-9]{14}$",
    "American Express": r"^3[47][0-9]{13}$",
    "Discover": r"^6(?:011|5[0-9]{2})[0-9]{12}$",
    "Diners Club": r"^3(?:0[0-5]|[68][0-9])[0-9]{11}$",
    "JCB": r"^(?:2131|1800|35\d{3})\d{11}$",
    "UnionPay": r"^62[0-9]{14,17}$",
}

def luhn_algorithm(card_number: str) -> bool:
    """
    Validate credit card number using the Luhn Algorithm.

    Args:
        card_number (str): The credit card number to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    card_number = card_number.replace(" ", "")  # Remove spaces
    check_digit = int(card_number[-1])
    reversed_digits = [int(d) for d in card_number[-2::-1]]

    # Double every second digit
    for i in range(0, len(reversed_digits), 2):
        reversed_digits[i] *= 2
        if reversed_digits[i] > 9:
            reversed_digits[i] -= 9

    total_sum = sum(reversed_digits) + check_digit
    return total_sum % 10 == 0

def detect_card_type(card_number: str) -> str:
    """
    Detect the type of credit card based on regex patterns.

    Args:
        card_number (str): The credit card number to identify.

    Returns:
        str: Card type if matched, "Unknown" otherwise.
    """
    for card_type, pattern in card_patterns.items():
        if re.match(pattern, card_number):
            return card_type
    return "Unknown"

def validate_credit_card(card_number: str) -> dict:
    """
    Validate and identify a credit card number.

    Args:
        card_number (str): The credit card number to validate.

    Returns:
        dict: A dictionary with the validation results.
    """
    card_type = detect_card_type(card_number)
    is_valid = luhn_algorithm(card_number)
    return {
        "Card Number": card_number,
        "Card Type": card_type,
        "Valid": is_valid,
    }

# Example Usage
examples = [
    "4111111111111111",  # Visa
    "5555555555554444",  # MasterCard
    "378282246310005",   # American Express
    "6011111111111117",  # Discover
    "30569309025904",    # Diners Club
    "3530111333300000",  # JCB
]

for card in examples:
    print(validate_credit_card(card))

# IBAN regex for general format validation
iban_pattern = r"^[A-Z]{2}[0-9]{2}[A-Z0-9]{1,30}$"

# Country-specific IBAN lengths
iban_lengths = {
    "DE": 22,  # Germany
    "GB": 22,  # United Kingdom
    "FR": 27,  # France
    "ES": 24,  # Spain
    "NL": 18,  # Netherlands
    "IT": 27,  # Italy
    "CH": 21,  # Switzerland
    "AE": 23,  # UAE
    "SA": 24,  # Saudi Arabia
    "IN": 34,  # India
}

def is_valid_iban(iban):
    """Validates an IBAN number."""

    # Remove spaces and convert to uppercase
    iban = iban.replace(" ", "").upper()

    # Check if the IBAN has the correct format
    if not re.match(r"[A-Z]{2}\d{2}[A-Z\d]{11,30}", iban):
        return False

    # Move the first 4 characters to the end
    iban = iban[4:] + iban[:4]

    # Replace letters with their numerical equivalents
    iban = "".join(str(ord(c) - 55) if c.isalpha() else c for c in iban)

    # Check if the IBAN is valid using the modulo 97 operation
    return int(iban) % 97 == 1

# Example IBANs for testing
examples = [
    "DE44500105175407324931",  # Valid Germany IBAN
    "GB29NWBK60161331926819",  # Valid UK IBAN
    "FR7630006000011234567890189",  # Valid France IBAN
    "ES9121000418450200051332",  # Valid Spain IBAN
    "NL91ABNA0417164300",  # Valid Netherlands IBAN
    "INVALIDIBAN12345",  # Invalid IBAN
]

def validate_routing_number(routing_number: str) -> bool:
    """
    Validate a routing number using the checksum algorithm.

    Args:
        routing_number (str): 9-digit routing number.

    Returns:
        bool: True if valid, False otherwise.
    """
    if len(routing_number) != 9 or not routing_number.isdigit():
        return False

    digits = [int(d) for d in routing_number]
    checksum = (
        3 * (digits[0] + digits[3] + digits[6])
        + 7 * (digits[1] + digits[4] + digits[7])
        + (digits[2] + digits[5] + digits[8])
    )
    return checksum % 10 == 0

def parse_micr_line(micr_line: str) -> dict:
    """
    Parse and validate a MICR line.

    Args:
        micr_line (str): The MICR line to parse.

    Returns:
        dict: Parsed and validated MICR components.
    """
    # MICR pattern: ⎕<routing_number>⎕<account_number>|<check_number>
    micr_pattern = r"⎕(\d{9})⎕(\d+)\|(\d+)"
    match = re.match(micr_pattern, micr_line)

    if not match:
        return {"Valid": False, "Error": "Invalid MICR format"}

    routing_number, account_number, check_number = match.groups()

    # Validate components
    is_routing_valid = validate_routing_number(routing_number)
    is_account_valid = account_number.isdigit()
    is_check_valid = check_number.isdigit()

    return {
        "Valid": is_routing_valid and is_account_valid and is_check_valid,
        "Routing Number": routing_number,
        "Routing Valid": is_routing_valid,
        "Account Number": account_number,
        "Account Valid": is_account_valid,
        "Check Number": check_number,
        "Check Valid": is_check_valid,
    }

# Example MICR lines
micr_lines = [
    "⎕123456789⎕654321|1234",  # Valid MICR
    "⎕987654321⎕12345|6789",   # Invalid routing number
    "⎕12345678⎕654321|1234",   # Invalid MICR format
]

for micr in micr_lines:
    print(parse_micr_line(micr))

# Regex patterns for TIN validation
tin_patterns = {
    "US": r"^\d{3}-\d{2}-\d{4}$",  # SSN
    "India": r"^[A-Z]{5}[0-9]{4}[A-Z]$",  # PAN
    "UK": r"^[A-CEGHJPR-TW-Z]{2}[0-9]{6}[A-D]$",  # NIN
    "Germany": r"^\d{11}$",  # Tax ID
    "France": r"^\d{13}$",  # INSEE
    "Canada": r"^\d{3} \d{3} \d{3}$",  # SIN
    "Australia": r"^\d{8,9}$",  # TFN
    "Italy": r"^[A-Z]{6}[0-9]{2}[A-Z][0-9]{2}[A-Z][0-9]{3}[A-Z]$",  # Codice Fiscale
    "Spain": r"^[0-9]{8}[A-Z]$",  # NIF
    "Netherlands": r"^\d{9}$",  # BSN
    "EU_VAT": r"^[A-Z]{2}[A-Z0-9]{8,12}$",  # VAT
}

def validate_tin(tin: str, country: str) -> dict:
    """
    Validate a Tax Identification Number (TIN) based on country-specific rules.

    Args:
        tin (str): The TIN to validate.
        country (str): The country code (e.g., "US", "India").

    Returns:
        dict: Validation result including country, validity, and details.
    """
    pattern = tin_patterns.get(country)
    if not pattern:
        return {"TIN": tin, "Valid": False, "Error": f"No pattern available for {country}"}

    if re.match(pattern, tin):
        return {"TIN": tin, "Valid": True, "Country": country}
    else:
        return {"TIN": tin, "Valid": False, "Country": country, "Error": "Invalid format"}

# Example TINs for testing
examples = [
    ("123-45-6789", "US"),  # Valid SSN
    ("ABCDE1234F", "India"),  # Valid PAN
    ("AB123456C", "UK"),  # Valid NIN
    ("12345678901", "Germany"),  # Valid Tax ID
    ("INVALIDTIN", "US"),  # Invalid SSN
]

for tin, country in examples:
    print(validate_tin(tin, country))


def validate_passport(passport_number: str, country: str) -> dict:
    """
    Validate a passport number based on country-specific rules.

    Args:
        passport_number (str): The passport number to validate.
        country (str): The country code (e.g., "United States").

    Returns:
        dict: Validation result with details.
    """
    if country not in passport_validation:
        return {"Passport Number": passport_number, "Valid": False, "Error": "Country not supported"}

    validation_rule = passport_validation[country]
    pattern = validation_rule["Format"]
    if re.match(pattern, passport_number):
        return {"Passport Number": passport_number, "Valid": True, "Details": validation_rule["Details"]}
    else:
        return {"Passport Number": passport_number, "Valid": False, "Error": "Invalid format"}

# Example Usage
examples = [
    ("AA1234567", "United States"),
    ("AB123456", "Canada"),
    ("P12345678", "India"),
    ("INVALID", "United States"),
]
'''
for passport_number, country in examples:
    print(validate_passport(passport_number, country))
'''

def validate_swift(swift_code: str) -> dict:
    """
    Validate a SWIFT code based on international standards.

    Args:
        swift_code (str): The SWIFT code to validate.

    Returns:
        dict: Validation result with details.
    """
    # Normalize input
    swift_code = swift_code.strip().upper()

    # Length validation
    if len(swift_code) not in [8, 11]:
        return {"SWIFT Code": swift_code, "Valid": False, "Error": "Invalid length"}

    # Regex pattern for SWIFT code
    swift_pattern = r"^[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?$"

    # Match against the pattern
    if not re.match(swift_pattern, swift_code):
        return {"SWIFT Code": swift_code, "Valid": False, "Error": "Invalid format"}

    # Extract components
    bank_code = swift_code[:4]
    country_code = swift_code[4:6]
    location_code = swift_code[6:8]
    branch_code = swift_code[8:] if len(swift_code) == 11 else "N/A"

    # Validate country code (ISO 3166-1 alpha-2)
    valid_country_codes = {"US", "GB", "FR", "DE", "IN", "JP", "CN", "AU", "CA", "CH"}  # Example set
    if country_code not in valid_country_codes:
        return {"SWIFT Code": swift_code, "Valid": False, "Error": "Invalid country code"}

    # Return valid result
    return {
        "SWIFT Code": swift_code,
        "Valid": True,
        "Components": {
            "Bank Code": bank_code,
            "Country Code": country_code,
            "Location Code": location_code,
            "Branch Code": branch_code,
        }
    }

# Example SWIFT codes
examples = [
    "DEUTDEFF",        # Valid (8 characters)
    "DEUTDEFF500",     # Valid (11 characters)
    "INVALIDSW",       # Invalid
    "ABCDUS12",        # Invalid country code
]

for swift in examples:
    print(validate_swift(swift))

def validate_ssn(ssn: str) -> dict:
    """
    Validate a U.S. Social Security Number (SSN).

    Args:
        ssn (str): The SSN to validate.

    Returns:
        dict: Validation result with details.
    """
    # Normalize input
    ssn = ssn.replace("-", "").strip()

    # Check format
    if not re.match(r"^\d{9}$", ssn):
        return {"SSN": ssn, "Valid": False, "Error": "Invalid format"}

    # Extract components
    area, group, serial = int(ssn[:3]), int(ssn[3:5]), int(ssn[5:])

    # Area number validation
    if area == 0 or area == 666 or 900 <= area <= 999:
        return {"SSN": ssn, "Valid": False, "Error": "Invalid area number"}

    # Group number validation
    if group == 0:
        return {"SSN": ssn, "Valid": False, "Error": "Invalid group number"}

    # Serial number validation
    if serial == 0:
        return {"SSN": ssn, "Valid": False, "Error": "Invalid serial number"}

    # If all checks pass
    return {"SSN": ssn, "Valid": True, "Error": None}

# Example SSNs for testing
examples = [
    "123-45-6789",  # Valid
    "666-45-6789",  # Invalid area number
    "123-00-6789",  # Invalid group number
    "123-45-0000",  # Invalid serial number
    "123456789",    # Valid compact form
    "000-45-6789",  # Invalid area number
]

for ssn in examples:
    print(validate_ssn(ssn))

# IPv4 and IPv6 regex patterns
ipv4_pattern = r"^((25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$"
ipv6_pattern = r"^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(([0-9a-fA-F]{1,4}:){1,7}:|(:[0-9a-fA-F]{1,4}){1,7}))$"

def validate_ip_address(ip: str) -> dict:
    """
    Validate an IP address (IPv4 or IPv6).

    Args:
        ip (str): The IP address to validate.

    Returns:
        dict: Validation result with type and validity.
    """
    if re.match(ipv4_pattern, ip):
        return {"IP Address": ip, "Valid": True, "Type": "IPv4"}
    elif re.match(ipv6_pattern, ip):
        return {"IP Address": ip, "Valid": True, "Type": "IPv6"}
    else:
        return {"IP Address": ip, "Valid": False, "Type": "Invalid"}

# Examples
ip_addresses = [
    "192.168.1.1",          # Valid IPv4
    "255.255.255.255",      # Valid IPv4
    "0.0.0.0",              # Valid IPv4
    "2001:0db8:85a3::8a2e:0370:7334",  # Valid IPv6
    "::1",                  # Valid IPv6 (loopback)
    "12345::",              # Invalid IPv6
    "192.168.256.1",        # Invalid IPv4
]

for ip in ip_addresses:
    print(validate_ip_address(ip))

def validate_phone_number(phone_number: str) -> dict:
    """
    Validate a phone number (US and International).

    Args:
        phone_number (str): The phone number to validate.

    Returns:
        dict: Validation result with type and validity.
    """
    phone_number = phone_number.strip()
    pattern = r"^(\+?[1-9]\d{0,2})?[-. ]?(\(?[2-9]\d{2}\)?[-. ]?[2-9]\d{2}[-. ]?\d{4}|\d{7,15})$"

    if re.match(pattern, phone_number):
        if phone_number.startswith("+1") or re.match(r"^\(?[2-9]\d{2}\)?[-. ]?[2-9]\d{2}[-. ]?\d{4}$", phone_number):
            return {"Phone Number": phone_number, "Valid": True, "Type": "US"}
        else:
            return {"Phone Number": phone_number, "Valid": True, "Type": "International"}
    else:
        return {"Phone Number": phone_number, "Valid": False, "Type": "Invalid"}

# Example Phone Numbers
examples = [
    "+11234567890",     # US with country code
    "(123) 456-7890",   # Invalid US (area code starts with 1)
    "123-456-7890",     # Valid US
    "+442071838750",    # Valid International (UK)
    "+919876543210",    # Valid International (India)
    "987654321",        # Valid International (short form)
    "+1-800-555-1234",  # Valid US toll-free
    "18005551234",      # Valid US toll-free without separators
]

for phone in examples:
    print(validate_phone_number(phone))

# Comprehensive regex for email validation
email_regex = (
    r"^(?!(?:(?:\x22)?\x2e|\x22\x2e)(?:.*\x40))"
    r".+@"
    r"(?:(?!\x2e)[A-Za-z0-9][A-Za-z0-9-]{0,62}[A-Za-z0-9]?\x2e)"
    r"{1,126}[A-Za-z]{2,63}$"
)

def validate_email(email: str) -> dict:
    """
    Validate an email address based on RFC 5322 standards.

    Args:
        email (str): The email address to validate.

    Returns:
        dict: Validation result with details.
    """
    if re.match(email_regex, email):
        return {"Email": email, "Valid": True, "Error": None}
    else:
        return {"Email": email, "Valid": False, "Error": "Invalid email format"}

# Example emails for testing
emails = [
    "simple@example.com",            # Valid
    "very.common@example.com",       # Valid
    "disposable.style.email.with+symbol@example.com",  # Valid
    "other.email-with-hyphen@example.com",  # Valid
    "fully-qualified-domain@example.com",   # Valid
    "user.name+tag+sorting@example.com",    # Valid
    "user@localserver",             # Invalid (domain must have a TLD)
    "@missinglocal.com",            # Invalid (missing local part)
    "missingdomain@",               # Invalid (missing domain)
    "user.@example.com",            # Invalid (dot at end of local-part)
    "user@.com",                    # Invalid (dot at start of domain)
]

for email in emails:
    print(validate_email(email))

def validate_crypto_address(address: str, currency: str) -> dict:
    """
    Validate a cryptocurrency wallet address based on its type.

    Args:
        address (str): The wallet address to validate.
        currency (str): The cryptocurrency type (e.g., BTC, ETH, LTC).

    Returns:
        dict: Validation result with details.
    """
    # Define regex patterns for each cryptocurrency
    crypto_patterns = {
        "BTC": r"^(1|3|bc1)[a-zA-HJ-NP-Z0-9]{25,39}$",  # Bitcoin
        "ETH": r"^0x[a-fA-F0-9]{40}$",  # Ethereum
        "LTC": r"^(L|M|ltc1)[a-zA-HJ-NP-Z0-9]{25,39}$",  # Litecoin
        "DOGE": r"^D[a-zA-HJ-NP-Z0-9]{33}$",  # Dogecoin
        "XRP": r"^r[1-9A-HJ-NP-Za-km-z]{24,34}$",  # Ripple
        "USDT": r"^(0x[a-fA-F0-9]{40}|T[a-zA-HJ-NP-Z0-9]{33})$",  # USDT (Ethereum/Tron)
    }

    if currency not in crypto_patterns:
        return {"Address": address, "Valid": False, "Error": "Unsupported cryptocurrency"}
    
    pattern = crypto_patterns[currency]
    if re.match(pattern, address):
        return {"Address": address, "Valid": True, "Currency": currency}
    else:
        return {"Address": address, "Valid": False, "Error": "Invalid format"}

# Example Wallet Addresses
examples = [
    ("1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa", "BTC"),  # Valid Bitcoin
    ("0xde0B295669a9FD93d5F28D9Ec85E40f4cb697BAe", "ETH"),  # Valid Ethereum
    ("LZ3CyqZmbeBikM6SLQ4TB6TxCWAztvwNB3", "LTC"),  # Valid Litecoin
    ("D7Y55mD7K33UmSbHj8uhxek76NQC39Sy5j", "DOGE"),  # Valid Dogecoin
    ("r3AddbzLtR7GSdzXGUog2LU6zCwrKDc3vY", "XRP"),  # Valid Ripple
    ("T9yD14Nj9j7xAB4dbGeiX9h8unkKHxuWwb", "USDT"),  # Valid USDT (Tron)
    ("INVALIDADDRESS12345", "BTC"),  # Invalid Bitcoin
]

for address, currency in examples:
    print(validate_crypto_address(address, currency))

def validate_hashtag(hashtag: str) -> dict:
    """
    Validate a social media hashtag.

    Args:
        hashtag (str): The hashtag to validate.

    Returns:
        dict: Validation result with details.
    """
    hashtag = hashtag.strip()
    pattern = r"^#[A-Za-z0-9_]{1,139}$"

    if re.match(pattern, hashtag):
        return {"Hashtag": hashtag, "Valid": True, "Error": None}
    else:
        return {"Hashtag": hashtag, "Valid": False, "Error": "Invalid format"}

# Example Hashtags for Testing
examples = [
    "#validHashtag",       # Valid
    "#1234",               # Valid (numeric is allowed but not always meaningful)
    "#_underscoreStart",   # Valid
    "noHashSymbol",        # Invalid (missing `#`)
    "#Invalid Space",      # Invalid (contains space)
    "#invalid!Symbol",     # Invalid (contains `!`)
    "#",                   # Invalid (too short)
    "#HashtagWithMoreThan140Characters_" + "x" * 130,  # Invalid (too long)
]

for hashtag in examples:
    print(validate_hashtag(hashtag))


def luhn_checksum(card_number):
    def digits_of(n):
        return [int(d) for d in str(n)]

    digits = digits_of(card_number)
    odd_digits = digits[-1::-2]
    even_digits = digits[-2::-2]
    total = sum(odd_digits)
    for digit in even_digits:
        total += sum(digits_of(2 * digit))
    return total % 10 == 0

def is_luhn_valid(card_number):
    return luhn_checksum(card_number)

# Example usage
card_number = "79927398713"

if is_luhn_valid(card_number):
    print("Valid card number")
else:
    print("Invalid card number")

def lcg(seed, a, c, m):
    """
    Linear Congruential Generator (LCG)

    Args:
        seed (int): The initial seed value.
        a (int): The multiplier.
        c (int): The increment.
        m (int): The modulus.

    Returns:
        int: The next pseudo-random number in the sequence.
    """
    return (a * seed + c) % m

# Example usage
seed = 12345
a = 1103515245
c = 12345
m = 2**31

for _ in range(10):
    seed = lcg(seed, a, c, m)
    print(seed)

# Verhoeff algorithm checksum
def calc__verhoeff_checksum(number):
    """Calculates the Verhoeff checksum digit for a given number."""

    d = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
        [2, 3, 4, 0, 1, 7, 8, 9, 5, 6],
        [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
        [4, 0, 1, 2, 3, 9, 5, 6, 7, 8],
        [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
        [6, 5, 9, 8, 7, 1, 0, 4, 3, 2],
        [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
        [8, 7, 6, 5, 9, 3, 2, 1, 0, 4],
        [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    ]

    p = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
        [5, 8, 0, 3, 7, 9, 6, 1, 4, 2],
        [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
        [9, 4, 5, 3, 1, 2, 6, 8, 7, 0],
        [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
        [2, 7, 9, 3, 8, 0, 6, 4, 1, 5],
        [7, 0, 4, 6, 9, 1, 3, 2, 5, 8]
    ]

    number = str(number)
    c = 0
    for i, n in enumerate(reversed(number)):
        c = d[c][p[i % 8][int(n)]]
    return c

def validate(number):
    """Validates a number using the Verhoeff algorithm."""

    return luhn_checksum(number) == 0

if __name__ == "__main__":
    number = "123456789"
    checksum = calc__verhoeff_checksum(number)
    print("Checksum:", checksum)

    if validate(number + str(checksum)):
        print("Number is valid.")
    else:
        print("Number is invalid.")

def damm_checksum(number):
    """Calculates the Damm checksum for a given number."""

    # Damm operation table
    table = [
        [0, 3, 1, 7, 5, 9, 8, 6, 4, 2],
        [7, 0, 9, 2, 1, 5, 4, 8, 6, 3],
        [4, 2, 0, 6, 8, 7, 1, 3, 5, 9],
        [1, 7, 5, 0, 9, 8, 3, 4, 2, 6],
        [6, 1, 2, 3, 0, 4, 5, 9, 7, 8],
        [3, 6, 7, 4, 2, 0, 9, 5, 8, 1],
        [5, 8, 6, 9, 7, 2, 0, 1, 3, 4],
        [8, 9, 4, 5, 3, 6, 2, 0, 1, 7],
        [9, 4, 3, 8, 6, 1, 7, 2, 0, 5],
        [2, 5, 8, 1, 4, 3, 6, 7, 9, 0]
    ]

    interim = 0
    for digit in str(number):
        digit = int(digit)
        interim = table[interim][digit]

    return interim

# Example usage
number = 123456789
checksum = damm_checksum(number)
print("Damm checksum:", checksum)

# Example usage
number = 79927398713

def mod11_check(number_str):
    """
    Performs a standard Mod 11 check on the given number string.
    Returns True if the number is valid, False otherwise.
    """

    # Reverse the number string
    number_str = number_str[::-1]

    # Initialize the sum and weight
    total = 0
    weight = 2

    # Calculate the weighted sum
    for digit in number_str:
        total += int(digit) * weight
        weight += 1
        if weight > 10:
            weight = 2

    # Check if the result is divisible by 11
    return total % 11 == 0

# Example usage:
number = "123456789"
if mod11_check(number):
    print(f"{number} is a valid number according to Mod 11 check.")
else:
    print(f"{number} is not a valid number according to Mod 11 check.")

def db1a1_modulus_check(number_str):
    """
    Performs a DB1A1 modulus 10 check on the given number string.
    Returns True if the number is valid, False otherwise.
    """

    # Ensure the number string contains only digits
    if not number_str.isdigit():
        return False

    # Reverse the number string
    reversed_str = number_str[::-1]

    # Calculate the weighted sum
    total = 0
    for i, digit in enumerate(reversed_str):
        digit = int(digit)

        # Double every second digit
        if i % 2 == 1:
            digit *= 2

        # If doubling results in a two-digit number, add the digits
        if digit > 9:
            digit -= 9

        total += digit

    # Check if the total is a multiple of 10
    return total % 10 == 0

# Example usage
number = "1234567890"
if db1a1_modulus_check(number):
    print("Valid number")
else:
    print("Invalid number")

def validate_nuban(account_number):
    """
    Validates a Nigerian Uniform Bank Account Number (NUBAN).
    """

    # Check if the account number is a 10-digit string
    if not isinstance(account_number, str) or len(account_number) != 10:
        return False

    # Get the bank code from the first 3 digits
    bank_code = account_number[:3]

    # Get the serial number from the remaining 7 digits
    serial_number = account_number[3:]

    # Define the weights for the checksum calculation
    weights = [3, 7, 3, 3, 7, 3, 3, 7, 3, 3, 7, 3]

    # Calculate the checksum
    checksum = 0
    for i in range(10):
        digit = int(account_number[i])
        checksum += digit * weights[i]

    # Check if the checksum is divisible by 10
    return checksum % 10 == 0

account_number = "0441234567"  # Example NUBAN
if validate_nuban(account_number):
    print("Valid NUBAN")
else:
    print("Invalid NUBAN")

def validate_routing_number(routing_number):
    """
    Validates a US bank routing number using the checksum algorithm.

    Args:
        routing_number (str): The 9-digit routing number to validate.

    Returns:
        bool: True if the routing number is valid, False otherwise.
    """

    if not isinstance(routing_number, str) or len(routing_number) != 9 or not routing_number.isdigit():
        return False

    weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]
    checksum = sum(int(digit) * weight for digit, weight in zip(routing_number, weights)) % 10

    return checksum == 0

routing_number = "122105155"
if validate_routing_number(routing_number):
    print("Valid routing number")
else:
    print("Invalid routing number")
