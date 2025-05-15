<img src="https://raw.githubusercontent.com/woodyk/mrblack/refs/heads/main/assets/mrblack-banner.png" alt="Mr. Black" style="width:100%; display:block; margin:0 auto;">

# Mr. Black

A comprehensive text extraction and PII detection toolkit for Python.

## Overview

Mr. Black is a powerful, versatile library for extracting text from virtually any source and detecting PII (Personally Identifiable Information) in extracted content. It provides a unified interface for text extraction from:

- Files (PDFs, DOCx, Excel, images, audio, and more)
- URLs and web pages (with JavaScript rendering)
- Screenshots
- Raw text

The library also includes robust PII detection capabilities with customizable regex patterns for various types of sensitive information.

## Features

### Text Extraction
- **Universal Text Extraction**: Extract text from almost any document format
- **Web Content**: Scrape and extract text from websites (with JS rendering)
- **OCR Capabilities**: Extract text from images and screenshots
- **Audio Transcription**: Convert audio to text
- **Password-Protected Files**: Support for extracting from encrypted documents
- **Metadata Extraction**: Get comprehensive file metadata
- **Text Analysis**: Summarization, language detection, and basic analytics

### PII Detection
- **Comprehensive Pattern Library**: Detect a wide range of PII types
- **Customizable Patterns**: Extend with your own regex patterns
- **Multiple Input Sources**: Scan files, URLs, text, or screen content
- **Batch Processing**: Process entire directories efficiently
- **Rich Output Options**: Formatted display or JSON output

## Installation

```bash
pip install mrblack
```

## Command Line Interface

Mr. Black provides comprehensive command-line utilities for both text extraction and PII detection.

### textextract CLI

```bash
textextract --help

usage: textextract [-h] [--metadata] [--summarize] [--sentences SENTENCES] [--analyze]
                   [--translate [LANG]] [--output OUTPUT] [--password PASSWORD] [--scrape]
                   [--max-pages MAX_PAGES] [--verbose] [--screenshot] [--chunked] [--no-js]
                   [--list-languages]
                   [source ...]

Extract and analyze text from any file, URL, directory or wildcard pattern

positional arguments:
  source                Path(s) to file(s), URL, directory, or wildcard pattern

options:
  -h, --help            show this help message and exit
  --metadata            Extract metadata instead of text
  --summarize           Summarize the extracted text
  --sentences SENTENCES
                        Number of sentences in summary (default: 5)
  --analyze             Perform text analysis
  --translate [LANG]    Translate text to specified language code (e.g., 'es'), or list available
                        languages if no code provided
  --output OUTPUT       Output file path (default: stdout)
  --password PASSWORD   Password for protected documents
  --scrape              Scrape multiple pages from a website (for URLs only)
  --max-pages MAX_PAGES
                        Maximum pages to scrape when using --scrape (default: 5)
  --verbose, -v         Increase verbosity (can be used multiple times)
  --screenshot          Capture and extract text from screen
  --chunked             Process large files in chunks to reduce memory usage
  --no-js               Disable JavaScript rendering for web pages
  --list-languages      List available translation languages
```

#### textextract Examples
```bash
# Basic text extraction
textextract document.pdf

# Extract from a URL
textextract https://example.com

# Capture and extract from screen
textextract screenshot

# Extract and summarize
textextract document.pdf --summarize --sentences 3

# Extract and translate
textextract document.pdf --translate es

# Extract metadata only
textextract document.pdf --metadata

# List available translation languages
textextract --list-languages

# Scrape multiple pages from a website
textextract https://example.com --scrape --max-pages 10

# Process files in chunks (for large files)
textextract large_document.pdf --chunked

# Process all files in a directory
textextract /path/to/documents/

# Process files matching a pattern
textextract "*.pdf"

# Save output to a file
textextract document.pdf --output results.txt
```

### pii CLI

```bash
pii --help

usage: pii [-h] [--labels [LABEL ...]] [--json] [--serial] [--save SAVE] [path]

Extract PII or patterns from files, dirs, URLs, or screenshots.

positional arguments:
  path                  File, directory, URL, or 'screenshot'

options:
  -h, --help            show this help message and exit
  --labels [LABEL ...]  Labels to extract; no args lists all labels
  --json                Output results as JSON
  --serial              Per-file results for directories
  --save SAVE           Save JSON output to specified file
```

#### pii Examples

```bash
# Detect PII in a file
pii resume.pdf

# Detect PII from a URL
pii https://example.com

# Detect PII from screen capture
pii screenshot

# List all available PII labels
pii --labels

# Filter for specific PII types
pii document.pdf --labels email phone_number credit_card

# Output results as JSON
pii document.pdf --json

# Save results to a file
pii document.pdf --json --save results.json

# Process an entire directory
pii /path/to/documents/

# Get per-file results for a directory
pii /path/to/documents/ --serial
```

## Mr. Black Library Usage

### Basic Text Extraction

```python
from mrblack import extract_text, text_from_url, text_from_screenshot

# Extract text from a file
content = extract_text('document.pdf')
print(content)

# Extract text from a URL
web_text = text_from_url('https://example.com')
print(web_text)

# Capture and extract text from the screen
screen_text = text_from_screenshot()
print(screen_text)
```

### Advanced Text Processing

```python
from mrblack import (
    summarize_text,
    analyze_text,
    translate_text,
    detect_language,
    extract_metadata
)

# Extract and summarize text
text = extract_text('article.pdf')
summary = summarize_text(text, sentences=5)
print(summary)

# Analyze text content
analysis = analyze_text(text)
print(f"Word count: {analysis['word_count']}")
print(f"Language detected: {analysis['language']}")
print(f"Most common words: {analysis['most_common_words'][:5]}")

# Translate text
translated = translate_text(text, target_lang='es')
print(translated)

# Extract metadata
metadata = extract_metadata('document.docx')
print(metadata)
```

### PII Detection

```python
from mrblack import (
    extract_pii_text,
    extract_pii_file,
    extract_pii_url,
    extract_pii_screenshot
)

# Detect PII in raw text
text = "Contact John Doe at john.doe@example.com or (123) 456-7890"
pii = extract_pii_text(text)
print(pii)
# Output: {'email': ['john.doe@example.com'], 'phone_number': ['(123) 456-7890']}

# Detect PII in a file
file_pii = extract_pii_file('resume.pdf')
print(file_pii)

# Detect PII on a website
url_pii = extract_pii_url('https://example.com/contact')
print(url_pii)

# Capture screen and detect PII
screen_pii = extract_pii_screenshot()
print(screen_pii)
```

### Filter By PII Types

```python
from mrblack import extract_pii_text

text = "My SSN is 123-45-6789 and my credit card is 4111-1111-1111-1111"
# Extract only specific PII types
pii = extract_pii_text(text, labels=["social_security", "credit_card"])
print(pii)
```


## Supported PII Types

Mr. Black can detect numerous types of PII and sensitive information:

| Category | PII Types |
|----------|-----------|
| Personal Identifiers | Email, Phone, Social Security Numbers, Passport Numbers |
| Financial | Credit Card Numbers, Bank Account Numbers, Routing Numbers, SWIFT Codes |
| Geographic | Postal/ZIP Codes, Addresses |
| Technical | IP Addresses (v4/v6), MAC Addresses, UUIDs |
| Temporal | Dates, Times, Datetime formats |
| Files/Paths | Windows Paths, Unix Paths |
| Technology | Protocol names, Programming Languages, File Formats, OS names |
| Miscellaneous | VIN Numbers, Hex Numbers, Environment Variables |

## Supported File Formats

Mr. Black supports text extraction from a wide range of file formats:

| Category | Supported Formats |
|----------|-------------------|
| Documents | PDF, DOC, DOCX, ODT, RTF, TXT |
| Spreadsheets | XLS, XLSX, CSV |
| Presentations | PPT, PPTX |
| Web | HTML, XML, JSON, YAML |
| Images | PNG, JPG, JPEG, GIF, TIFF, BMP, WebP |
| Audio | MP3, WAV, FLAC, AAC, OGG |
| E-Books | EPUB |
| Archives | ZIP (with extraction) |
