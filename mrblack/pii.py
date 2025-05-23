#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: pii.py
# Author: Wadih Khairallah
# Created: 2024-12-01
# Modified: 2025-05-17 18:03:48

import os
import re
import sys
import json
import shutil
import argparse
from collections import defaultdict
from typing import (
    Optional,
    Dict,
    List,
    Union,
)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from .patterns import PATTERNS
from .textextract import (
    clean_path,
    extract_text,
    text_from_screenshot,
    text_from_url,
)

console = Console()
print = console.print


def _clean_value(
    label: str,
    value: str
) -> Optional[str]:
    """
    Normalize and clean raw extracted values based on the label type.

    Args:
        label (str): The name of the PII label (e.g., 'phone_number').
        value (str): The raw matching string to clean.

    Returns:
        Optional[str]: The cleaned value, or None if it should be skipped.
    """
    value = value.strip().replace("\n", "")
    if label == "windows_path":
        return value.rstrip(".").replace("\\\\", "\\")
    if label == "phone_number":
        if re.search(r"\d+\.\d+,\s*-?\d+\.\d+", value):
            return None
        if "," in value and not re.match(
            r"^\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}$", value
        ):
            return None
        value = re.sub(r"^(\d{3})\)", r"(\1)", value)
        value = re.sub(r"\)(\d)", r") \1", value)
        return value
    return value


def extract_pii_text(
    text: str,
    labels: Optional[Union[List[str], str]] = None
) -> Dict[str, List[str]]:
    """
    Extract PII matches from provided text.

    Args:
        text (str): The input text to scan for patterns.
        labels (Optional[Union[List[str], str]]): Specific labels to filter on.

    Returns:
        Dict[str, List[str]]: Mapping of each label to a sorted list of
        matched and cleaned strings.
    """
    if isinstance(labels, str):
        labels = [labels]
    patterns = PATTERNS
    if labels:
        patterns = [
            p for p in PATTERNS
            if any(re.search(rf"\(\?P<{lbl}>", p) for lbl in labels)
        ]
    results: Dict[str, set] = defaultdict(set)
    for pattern in patterns:
        try:
            rx = re.compile(pattern)
            for m in rx.finditer(text):
                for lbl, val in m.groupdict().items():
                    if not val:
                        continue
                    cleaned = _clean_value(lbl, val)
                    if lbl == "url":
                        cleaned = cleaned.rstrip("),.**")
                    if cleaned is not None:
                        results[lbl].add(cleaned)
        except re.error as e:
            print(
                f"Invalid regex skipped: {pattern} → {e}",
                file=sys.stderr
            )
    return {lbl: sorted(vals) for lbl, vals in results.items()}


def extract_pii_file(
    file_path: str,
    labels: Optional[Union[List[str], str]] = None
) -> Optional[Dict[str, List[str]]]:
    """
    Extract PII from a single file's text content.

    Args:
        file_path (str): Path to the file.
        labels (Optional[Union[List[str], str]]): Labels to filter.

    Returns:
        Optional[Dict[str, List[str]]]: Extraction results, or None.
    """
    text = extract_text(file_path)
    if not text:
        return None
    data = extract_pii_text(text, labels)
    return data or None


def extract_pii_url(
    path: str,
    labels: Optional[Union[List[str], str]] = None
) -> Optional[Dict[str, List[str]]]:
    """
    Extract PII from the text at a URL.

    Args:
        path (str): The URL to fetch.
        labels (Optional[Union[List[str], str]]): Labels to filter.

    Returns:
        Optional[Dict[str, List[str]]]: Extraction results, or None.
    """
    text = text_from_url(path)
    if not text:
        return None
    data = extract_pii_text(text, labels)
    return data or None


def extract_pii_image(
    image_path: str,
    labels: Optional[Union[List[str], str]] = None
) -> Optional[Dict[str, List[str]]]:
    """
    Extract PII from an image using OCR.

    Args:
        image_path (str): Path to the image file.
        labels (Optional[Union[List[str], str]]): Labels to filter.

    Returns:
        Optional[Dict[str, List[str]]]: Extraction results, or None.
    """
    path = clean_path(image_path)
    if not path or not os.path.isfile(path):
        print(f"[red]Invalid image path:[/] {image_path}")
        return None
    text = extract_text(path)
    if not text:
        return None
    data = extract_pii_text(text, labels)
    return data or None


def extract_pii_screenshot(
    labels: Optional[Union[List[str], str]] = None
) -> Optional[Dict[str, List[str]]]:
    """
    Capture a screenshot and extract PII from its OCR text.

    Args:
        labels (Optional[Union[List[str], str]]): Labels to filter.

    Returns:
        Optional[Dict[str, List[str]]]: Extraction results, or None.
    """
    text = text_from_screenshot()
    if not text:
        return None
    data = extract_pii_text(text, labels)
    return data or None


def directory(
    directory_path: str,
    labels: Optional[Union[List[str], str]] = None,
    serial: bool = False
) -> Union[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
    """
    Recursively scan a directory for PII.

    Args:
        directory_path (str): Root directory to scan.
        labels (Optional[Union[List[str], str]]): Labels to filter.
        serial (bool): If True, per-file results; else aggregated.

    Returns:
        Union[Dict[str, List[str]], Dict[str, Dict[str, List[str]]]]:
            Aggregated or per-file mapping.
    """
    if serial:
        results: Dict[str, Dict[str, List[str]]] = {}
    else:
        results: Dict[str, set] = defaultdict(set)

    for root, _, files in os.walk(directory_path):
        for fname in files:
            path = os.path.join(root, fname)
            res = extract_pii_file(path, labels)
            if not res:
                continue
            if serial:
                results[path] = res
            else:
                for lbl, vals in res.items():
                    results[lbl].update(vals)

    if serial:
        return results
    return {lbl: sorted(vals) for lbl, vals in results.items()}


def display(
    results: Dict[str, List[str]],
    title: str = "PII Extraction Results"
) -> None:
    """
    Pretty-print extraction results in a formatted table.

    Args:
        results (Dict[str, List[str]]): The PII extraction output.
        title (str): Title for the display panel.
    """
    table = Table(
        title=title,
        box=box.ROUNDED,
        expand=True,
        show_lines=True
    )
    table.add_column("Label", style="bold cyan", no_wrap=True)
    table.add_column("Matches", overflow="fold")
    for lbl, matches in sorted(results.items()):
        if not matches:
            continue
        table.add_row(
            f"[magenta]{lbl}",
            "\n".join(f"[green]{m}" for m in matches)
        )
    print(Panel(table, border_style="blue"))


def _print_labels_in_columns(
    label_list: List[str]
) -> None:
    """
    Nicely print all available PII labels in columns.

    Args:
        label_list (List[str]): List of label names to display.
    """
    labels = sorted(label_list)
    width = shutil.get_terminal_size((80, 20)).columns
    col_width = max(len(l) for l in labels) + 4
    cols = max(1, width // col_width)
    rows = (len(labels) + cols - 1) // cols

    table = Table(box=box.ROUNDED, show_header=False)
    for _ in range(cols):
        table.add_column(no_wrap=True)

    for r in range(rows):
        row = []
        for c in range(cols):
            idx = c * rows + r
            row.append(labels[idx] if idx < len(labels) else "")
        table.add_row(*row)

    print(table)


def get_labels() -> List[str]:
    """
    List all named PII labels defined in the regex patterns.

    Returns:
        List[str]: Sorted list of label names.
    """
    lbls: set = set()
    for p in PATTERNS:
        m = re.search(r"\(\?P<(\w+)>", p)
        if m:
            lbls.add(m.group(1))
    return sorted(lbls)


def main() -> None:
    """
    Command-line interface entry point.
    Parses args and dispatches extraction routines.
    Supports piped input via stdin.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Extract PII or patterns from files, dirs, URLs, "
            "screenshots, or stdin piped data."
        )
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="File, directory, URL, 'screenshot', or omit to read from stdin"
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        metavar="LABEL",
        help="Labels to extract; no args lists all labels"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Per-file results for directories"
    )
    parser.add_argument(
        "--save",
        help="Save JSON output to specified file"
    )
    parser.add_argument(
        "--stdin-format",
        help="Specify format of stdin data (pdf, docx, txt, etc.) when reading from stdin"
    )
    args = parser.parse_args()

    if not any(vars(args).values()) and sys.stdin.isatty():
        parser.print_help()
        return

    if args.labels is not None and len(args.labels) == 0:
        all_labels = get_labels()
        if args.json:
            text = json.dumps(all_labels, indent=4, ensure_ascii=False)
            if args.save:
                with open(args.save, "w", encoding="utf-8") as f:
                    f.write(text)
            print(text)
        else:
            _print_labels_in_columns(all_labels)
        return

    if args.labels is not None:
        label_list: Optional[List[str]] = []
        for token in args.labels:
            for part in token.split(","):
                lbl = part.strip()
                if lbl:
                    label_list.append(lbl)
        label_list = label_list or None
    else:
        label_list = None

    # Handle stdin if no path is provided and data is being piped in
    if not args.path and not sys.stdin.isatty():
        print("[cyan]Reading data from stdin...[/cyan]")
        
        # Create a temporary file to store the piped data
        import tempfile
        from uuid import uuid4
        
        temp_path = os.path.join(tempfile.gettempdir(), f"pii_stdin_{uuid4().hex}")
        
        try:
            # Read all data from stdin
            stdin_data = sys.stdin.buffer.read()
            
            if not stdin_data:
                print("[red]Error:[/] No data received from stdin.")
                return
                
            # Write to temp file
            with open(temp_path, 'wb') as f:
                f.write(stdin_data)
                
            # Extract text from the temp file
            if args.stdin_format:
                # If format is specified, we could use it to inform extraction
                # This would need implementation in extract_text() to be fully utilized
                print(f"[cyan]Using specified format: {args.stdin_format}[/cyan]")
            
            # Extract PII from the text
            text = extract_text(temp_path)
            if not text:
                print("[red]Error:[/] Could not extract text from stdin data.")
                return
                
            func_result = extract_pii_text(text, label_list)
            
            # Clean up early if we have results
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
            # Handle empty results
            if not func_result:
                print("[yellow]No matches found in stdin data.[/yellow]")
                return
                
            # Output results
            if args.json:
                out = {"stdin": func_result}
                text = json.dumps(out, indent=4, ensure_ascii=False)
                if args.save:
                    with open(args.save, "w", encoding="utf-8") as f:
                        f.write(text)
                print(text)
            else:
                display(func_result, title="PII Extraction Results from stdin")
            return
            
        except Exception as e:
            print(f"[red]Error processing stdin data:[/] {e}")
            return
        finally:
            # Ensure temp file is cleaned up
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    print(f"[red]Error deleting temp file:[/] {e}")

    raw = args.path or ""
    func_result = None

    if re.match(r'^(?:http|ftp)s?://', raw):
        func_result = extract_pii_url(raw, label_list)
    elif raw.lower() in {"screenshot", "screen", "capture"}:
        func_result = extract_pii_screenshot(label_list)
    else:
        path = clean_path(raw)
        if not path:
            print(f"[red]Error:[/] Invalid path '{raw}'.")
            return
        if os.path.isdir(path):
            func_result = directory(path, label_list, serial=args.serial)
        elif os.path.isfile(path):
            func_result = extract_pii_file(path, label_list)
        else:
            print(f"[red]Error:[/] Unsupported input '{raw}'.")
            return

    if func_result is None:
        print("[yellow]No matches found.[/yellow]")
        return

    if args.json:
        out = func_result
        if isinstance(out, dict) and raw and not os.path.isdir(raw):
            key = "screenshot" if raw.lower().startswith("screen") else raw
            out = {key: out}
        text = json.dumps(out, indent=4, ensure_ascii=False)
        if args.save:
            with open(args.save, "w", encoding="utf-8") as f:
                f.write(text)
        print(text)
        return

    if isinstance(func_result, dict):
        if args.serial and all(isinstance(v, dict) for v in func_result.values()):
            for fp, res in func_result.items():
                display(res, title=f"Results for {fp}")
        else:
            display(func_result, title="Aggregated Results")
    else:
        display(func_result)


if __name__ == "__main__":
    main()

