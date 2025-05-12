#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: flair_test2.py
# Author: Wadih Khairallah
# Description: 
# Created: 2025-03-23 19:33:54
# Modified: 2025-04-03 22:31:28
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from flair.models import SequenceTagger
from flair.data import Sentence
from collections import defaultdict

def extract_entities(text):
    sentence = Sentence(text)
    tagger.predict(sentence)
    
    entity_groups = defaultdict(set)

    for entity in sentence.get_spans('ner'):
        label = entity.get_label("ner").value
        entity_groups[label].add(entity.text.strip())

    return entity_groups

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: flair_test.py <filename>")
        sys.exit(1)

    file_path = sys.argv[1]

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print("Loading model...")
        tagger = SequenceTagger.load("flair/ner-english-large")  # More accurate model
        #tagger = SequenceTagger.load("ner")
        print("Extracting named entities...\n")

        grouped_entities = extract_entities(text)

        if not grouped_entities:
            print("No named entities found.")
        else:
            for label in sorted(grouped_entities):
                print(f"=== {label} ===")
                for entity in sorted(grouped_entities[label]):
                    print(f"- {entity}")
                print()

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

