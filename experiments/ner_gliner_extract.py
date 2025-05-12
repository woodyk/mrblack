#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: gliner.py
# Author: Wadih Khairallah
# Description: Process text data with NER detection, validation, and rich output
# Created: 2025-04-03 21:52:13
# Modified: 2025-04-23 17:52:45

import os
import warnings
import re
import logging
import sys
from gliner import GLiNER
from rich.console import Console
from rich.table import Table
from rich import box

# Set environment variables before any imports to ensure suppression
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
# Redirect stdout temporarily to suppress any early output
sys.stdout = open(os.devnull, 'w')
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
sys.stdout = sys.__stdout__  # Restore stdout after imports

def process_text_in_chunks(text, model, labels, chunk_size=500, max_length=None):
    all_entities = []
    if max_length is not None and len(text) > max_length:
        text = text[:max_length]
    
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i + chunk_size]
        entities = model.predict_entities(chunk, labels)
        for entity in entities:
            entity["start"] += i
            entity["end"] += i
            all_entities.append(entity)
    
    return all_entities

def validate_entities(entities):
    """
    Validate and correct entity labels based on patterns.
    
    Args:
        entities (list): List of entity dictionaries from GLiNER
    
    Returns:
        list: Corrected list of entities
    """
    # Common patterns for validation
    email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    url_pattern = re.compile(r'^(https?|ftp)://[^\s/$.?#].[^\s]*$')
    ip_pattern = re.compile(
        r'^(?:'
        r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)'
        r'|'
        r'(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}'
        r'|'
        r'(?:[0-9a-fA-F]{1,4}:){1,7}:)'
        r'$'
    )
    phone_pattern = re.compile(r'^\+?\d{1,4}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}$')
    date_pattern = re.compile(r'^\d{1,4}[-/]\d{1,2}[-/]\d{2,4}$|^(?:\d{1,2}\s)?[A-Za-z]+\s\d{1,2}(?:st|nd|rd|th)?\s\d{2,4}$')
    time_pattern = re.compile(r'^\d{1,2}:\d{2}(:\d{2})?\s?(AM|PM|am|pm)?$')
    ssn_pattern = re.compile(r'^\d{3}-\d{2}-\d{4}$')
    credit_card_pattern = re.compile(r'^(?:\d{4}[-\s]?){3}\d{4}$')

    corrected_entities = []
    for entity in entities:
        text = entity["text"].strip()
        label = entity["label"]

        # Correct mislabeled entities
        if email_pattern.match(text) and label != "EMAIL":
            entity["label"] = "EMAIL"
        elif url_pattern.match(text) and label != "URL":
            entity["label"] = "URL"
        elif ip_pattern.match(text) and label != "IP_ADDRESS":
            entity["label"] = "IP_ADDRESS"
        elif phone_pattern.match(text) and label != "PHONE":
            entity["label"] = "PHONE"
        elif date_pattern.match(text) and label != "DATE":
            entity["label"] = "DATE"
        elif time_pattern.match(text) and label != "TIME":
            entity["label"] = "TIME"
        elif ssn_pattern.match(text) and label != "SSN":
            entity["label"] = "SSN"
        elif credit_card_pattern.match(text) and label != "CREDIT_CARD":
            entity["label"] = "CREDIT_CARD"
        elif label == "VISA" and not text.isdigit():
            entity["label"] = "ORGANIZATION"

        corrected_entities.append(entity)
    
    return corrected_entities

def display_entities(entities):
    console = Console()
    table = Table(title="Detected Entities", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Entity Text", style="cyan", justify="left", width=40)
    table.add_column("Label", style="green", justify="center", width=20)
    table.add_column("Start", style="yellow", justify="right", width=10)
    table.add_column("End", style="yellow", justify="right", width=10)
    
    for entity in entities:
        table.add_row(
            entity["text"],
            entity["label"],
            str(entity["start"]),
            str(entity["end"])
        )
    
    console.print(table)
    console.print(f"[bold blue]Total entities detected: {len(entities)}[/bold blue]")

# Initialize model
model = GLiNER.from_pretrained("urchade/gliner_multi_pii-v1")

# Comprehensive list of PII entity types
labels = [
    "PERSON", "ORGANIZATION", "LOCATION", "PHONE", "EMAIL", "ADDRESS", "DATE", "TIME",
    "SSN", "CREDIT_CARD", "BANK_ACCOUNT", "PASSPORT", "DRIVER_LICENSE", "USERNAME",
    "URL", "IP_ADDRESS", "HEALTH_INSURANCE", "DOB", "MEDICATION", "CPF", "TAX_ID",
    "MEDICAL_CONDITION", "ID_CARD", "NATIONAL_ID", "IBAN", "CREDIT_CARD_EXP",
    "REGISTRATION", "STUDENT_ID", "INSURANCE", "FLIGHT", "BLOOD_TYPE", "CVV",
    "RESERVATION", "SOCIAL_MEDIA", "LICENSE_PLATE", "CNPJ", "POSTAL_CODE",
    "SERIAL_NUMBER", "VEHICLE_REG", "FAX", "VISA", "TRANSACTION", "BIRTH_CERTIFICATE"
]

file_path = sys.argv[1]

# Read input data
with open(file_path, "r") as f:
    data = f.read()

# Process and validate entities
entities = process_text_in_chunks(data, model, labels, chunk_size=500)
corrected_entities = validate_entities(entities)

# Display results
display_entities(corrected_entities)
