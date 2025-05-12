#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: lucene_test.py
# Author: Wadih Khairallah
# Description: 
# Created: 2024-12-03 03:41:29
# Modified: 2024-12-03 03:45:41
import re

# Pattern groups
PATTERN_GROUPS = {
    "ipv4": [
        r"\b(?P<ipv4>(?:(\d{1,3}\.){3}\d{1,3}(\/\d{1,2}\b|\/|)))"  # IPv4 pattern
    ],
    "email": [
        r"(?P<email>[\w.-]+@([\w-]+\.)+[\w-]+)"  # Email addresses
    ],
    "datetime": [
        r"(?P<datetime>\b\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?\b)",  # ISO-8601
        r"(?P<datetime>\d{4}(/|-)(?:0[1-9]|1[0-2])(/|-)(?:0[1-9]|[12][0-9]|3[01])\b)",
        r"(?P<datetime>(?:[01][0-9]|2[0-3]):[0-5][0-9]:(?:[0-5][0-9]|60)\b)"
    ],
    "url": [
        r"(?P<url>([a-zA-Z]+):\/\/[a-zA-Z0-9\-._~:/?#[\]@!$&'()*+,;=%]+)"  # URL pattern
    ],
    "unix_path": [
        r"(?P<unix_path>(?:[ \t\n]|^)/(?:[a-zA-Z0-9_.-]+/)*[a-zA-Z0-9_.-]+)"  # Unix paths
    ]
}

def escape_lucene_syntax(text):
    """Escapes special characters in Lucene syntax."""
    return re.sub(r'([.+^$[\]{}=!<>|\\-])', r'\\\1', text)

def lucene_to_regex_refined(query, pattern_groups):
    """
    Refined function for Lucene-style queries, supporting AND, OR, NOT logic.
    """
    try:
        # Match field and value (e.g., "field:term")
        match = re.match(r'(\w+):(.+)', query)
        if not match:
            raise ValueError(f"Invalid Lucene query: {query}")

        field, value = match.groups()
        if field not in pattern_groups:
            raise ValueError(f"Unknown field: {field}")

        # Escape value and handle wildcards
        value = escape_lucene_syntax(value).replace('*', '.*').replace('?', '.')

        # Split value into components based on Boolean operators
        terms = re.split(r'\s+(AND|OR|NOT)\s+', value)
        regex_parts = []
        current_operator = None

        for term in terms:
            term = term.strip()
            if term in {"AND", "OR", "NOT"}:
                current_operator = term
            else:
                if current_operator == "AND":
                    regex_parts.append(f"(?=.*{term})")
                elif current_operator == "OR":
                    if regex_parts:
                        regex_parts[-1] = f"({regex_parts[-1]}|{term})"
                    else:
                        regex_parts.append(term)
                elif current_operator == "NOT":
                    regex_parts.append(f"^(?!.*{term}).*")
                else:
                    regex_parts.append(term)

        # Combine regex components
        combined_regex = "".join(regex_parts)
        patterns = pattern_groups.get(field, [])
        combined_patterns = [
            re.sub(r"\?P<(\w+)>", f"?P<\\1_{idx}>", pattern)
            for idx, pattern in enumerate(patterns)
        ]
        full_pattern = "|".join(combined_patterns)
        final_regex = f"(?=.*{combined_regex})({full_pattern})"

        return re.compile(final_regex)

    except re.error as e:
        raise ValueError(f"Regex generation error: {e}")

def extract_non_empty_matches(matches):
    """
    Extract non-empty matches from grouped regex results.
    """
    extracted = []
    for match in matches:
        extracted.extend([item for item in match if item])
    return list(set(extracted))  # Deduplicate results

# Load the text content from the file
file_path = "../SAMPLE_DATA.py"
try:
    with open(file_path, "r") as file:
        text_content = file.read()
except Exception as e:
    print(f"Error loading text file: {e}")
    text_content = ""  # Default to an empty string if file loading fails

# Sample test cases for validation
test_cases = [
    {
        "query": "ipv4:192.168.*",
        "description": "Match all IPv4 addresses starting with 192.168."
    },
    {
        "query": "email:bob",
        "description": "Match all email addresses containing 'bob'."
    },
    {
        "query": "datetime:2024*",
        "description": "Match all datetime strings starting with 2024."
    },
    {
        "query": "url:https",
        "description": "Match all URLs starting with 'https'."
    },
    {
        "query": "unix_path:/etc",
        "description": "Match all Unix paths containing '/etc'."
    },
    {
        "query": "ipv4:10.0.* AND NOT ipv4:10.0.0.*",
        "description": "Match all IPv4 addresses starting with 10.0. but exclude 10.0.0.*."
    },
    {
        "query": "url:google AND email:gmail",
        "description": "Match all URLs containing 'google' and email addresses containing 'gmail'."
    },
    {
        "query": "datetime:(2023 OR 2024) AND NOT datetime:2023-12-31",
        "description": "Match all datetimes from 2023 or 2024 but exclude 2023-12-31."
    },
]

# Validate each query and print results
test_results = {}
for test in test_cases:
    query = test["query"]
    description = test["description"]
    try:
        regex = lucene_to_regex_refined(query, PATTERN_GROUPS)
        matches = regex.findall(text_content)
        test_results[query] = {
            "description": description,
            "regex_pattern": regex.pattern if regex else "No regex generated",
            "matches": extract_non_empty_matches(matches),
        }
    except Exception as e:
        test_results[query] = {
            "description": description,
            "error": str(e)
        }

# Display test results
from pprint import pprint
pprint(test_results)

