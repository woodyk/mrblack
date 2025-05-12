#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: lucene_regex.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-12-02 18:22:17
# Modified: 2024-12-02 19:49:36

import re
import ipaddress

# Pattern groups
DATETIME_PATTERNS = [
    r"(?P<datetime>\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b)",  # ISO-8601
    r"(?P<datetime>\d{4}(/|-)(?:0[1-9]|1[0-2])(/|-)(?:0[1-9]|[12][0-9]|3[01])\b)",
    r"(?P<datetime>(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)\b)",
    r"(?P<datetime>\b(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}\b))",  # Standard datetime
]

MISC_PATTERNS = [
    r"(?P<unix_path>(?:[ \t\n]|^)/(?:[a-zA-Z0-9_.-]+/)*[a-zA-Z0-9_.-]+)",
    r"(?P<windows_path>([a-zA-Z]:\\|\\\\)[\w\\.-]+)",  # Windows file paths
    r"(?P<email>[\w.-]+@([\w-]+\.)+[\w-]+)",  # Email addresses
    r"(?P<url>([a-zA-Z]+):\/\/[a-zA-Z0-9\-._~:/?#[\]@!$&'()*+,;=%]+)",
]

IPV4_PATTERNS = [
    r"\b(?P<ipv4>(?:(\d{1,3}\.){3}\d{1,3}(\/\d{1,2}\b|\/|)))",  # IPv4 pattern
]

PATTERN_GROUPS = {
    "datetime": DATETIME_PATTERNS,
    "misc": MISC_PATTERNS,
    "ipv4": IPV4_PATTERNS,
}

def validate_ipv4(address):
    """
    Validate that a string is a complete IPv4 address (with optional subnet).
    """
    try:
        ipaddress.IPv4Network(address, strict=False)
        return True
    except ValueError:
        return False

def extract_non_empty_matches(matches):
    """
    Extract non-empty matches from grouped regex results.
    """
    extracted = []
    for match in matches:
        extracted.extend([item for item in match if item])
    return list(set(extracted))

def extract_matches_for_field(field_name, text, groups=PATTERN_GROUPS):
    """
    Extract matches for a specific field group.
    """
    if field_name not in groups:
        raise ValueError(f"Unknown field: {field_name}")
    patterns = groups[field_name]
    regex = re.compile("|".join(patterns))
    matches = regex.findall(text)
    return {m.strip() for m in extract_non_empty_matches(matches) if m.strip()}

def process_boolean_logic(query, text, groups=PATTERN_GROUPS):
    """
    Process Boolean logic queries (e.g., ipv4 OR url).
    """
    terms = re.split(r'\s+(AND|OR|NOT)\s+', query.strip())
    results = None
    current_operator = None

    for term in terms:
        term = term.strip()
        if term in {"AND", "OR", "NOT"}:
            current_operator = term
        else:
            if ':' not in term:
                raise ValueError(f"Invalid term format: {term}")
            field_name, field_term = term.split(':', 1)
            term_matches = extract_matches_for_field(field_name, text, groups)
            if results is None:
                results = term_matches
            elif current_operator == "AND":
                results = {match for match in results if match in term_matches}
            elif current_operator == "OR":
                results |= term_matches  # Union of results
            elif current_operator == "NOT":
                results -= term_matches  # Exclusion of matches

    # Apply validation for IPv4 if relevant
    if "ipv4" in query:
        results = {match for match in results if validate_ipv4(match)}

    return sorted(results) if results else []

def test_module(text, queries, groups=PATTERN_GROUPS):
    """
    Test the module with provided text and queries.
    """
    results = {}
    for query in queries:
        try:
            matches = process_boolean_logic(query, text, groups)
            results[query] = matches
        except Exception as e:
            results[query] = {"error": str(e)}
    return results

# Example text content
text_content = """
10.0.0.1
8.8.8.8
user@example.com
https://example.com
/home/user/documents
"""

# Example queries
queries = [
    "ipv4:ipv4 OR misc:url",
    "ipv4:ipv4 AND misc:email",
    "ipv4:ipv4 NOT misc:unix_path",
]

# Test the module
results = test_module(text_content, queries)
from pprint import pprint
pprint(results)

