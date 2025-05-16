#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: ner_spacy_extract.py
# Author: Wadih Khairallah
# Description: 
# Created: 2025-05-15 20:06:09

import sys
import json
import spacy
from pathlib import Path

def load_text_file(file_path):
    """Read the content of a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except UnicodeDecodeError:
        print(f"Error: File '{file_path}' is not a valid UTF-8 text file.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file '{file_path}': {str(e)}")
        sys.exit(1)

def extract_entities(text, nlp, target_labels):
    """Extract entities from text using spaCy, filtering by target labels."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in target_labels:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
    return entities

def save_to_json(entities, output_file="output.json"):
    """Save extracted entities to a JSON file."""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(entities, f, indent=4, ensure_ascii=False)
        print(f"Results saved to '{output_file}'")
    except Exception as e:
        print(f"Error saving JSON to '{output_file}': {str(e)}")
        sys.exit(1)

def main():
    # Check for command-line argument
    if len(sys.argv) != 2:
        print("Usage: python extract_ner_labels.py <input_text_file>")
        sys.exit(1)

    # Get file path from command-line argument
    file_path = sys.argv[1]

    # Define target labels (modify as needed for your PII labels)
    target_labels = ["PERSON", "GPE", "DATE", "CARDINAL", "MONEY"]
    # Example custom labels if fine-tuned: ["EMAIL", "SSN", "ADDRESS"]

    # Load spaCy model
    try:
        nlp = spacy.load("en_core_web_sm")
        # If using a fine-tuned model, replace with:
        # nlp = spacy.load("./path/to/your/fine_tuned_model")
    except Exception as e:
        print(f"Error loading spaCy model: {str(e)}")
        sys.exit(1)

    # Read text file
    text = load_text_file(file_path)

    # Extract entities
    entities = extract_entities(text, nlp, target_labels)

    # Save results to JSON
    save_to_json(entities)

if __name__ == "__main__":
    main()
