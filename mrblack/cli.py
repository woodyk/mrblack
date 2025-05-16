#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: cli.py
# Project: mrblack
# Author: Based on work by Wadih Khairallah
# Created: 2025-05-15
# Modified: 2025-05-16 16:46:28
#
# Command line interface for mrblack toolkit

import os
import re
import sys
import json
import click
import importlib
import tempfile
from uuid import uuid4
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Union, Callable

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from rich.markdown import Markdown
from rich.json import JSON

# Import internal modules
try:
    from mrblack.textextract import (
        extract_text, extract_text_chunked, text_from_url, text_from_screenshot, 
        summarize_text, analyze_text, extract_metadata, 
        translate_text, list_available_languages, detect_language,
        is_url, clean_path, scrape_website, normalize_text,
        text_from_image, text_from_pdf, text_from_excel, text_from_docx,
        extract_document_structure
    )
    from mrblack.pii import (
        extract_pii_text, extract_pii_file, extract_pii_url, 
        extract_pii_image, extract_pii_screenshot, get_labels
    )
except ImportError:
    # Local development/direct run support
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from textextract import (
        extract_text, extract_text_chunked, text_from_url, text_from_screenshot,
        summarize_text, analyze_text, extract_metadata,
        translate_text, list_available_languages, detect_language,
        is_url, clean_path, scrape_website, normalize_text,
        text_from_image, text_from_pdf, text_from_excel, text_from_docx,
        extract_document_structure
    )
    from pii import (
        extract_pii_text, extract_pii_file, extract_pii_url,
        extract_pii_image, extract_pii_screenshot, get_labels
    )

# Setup console
console = Console()

# Constants
DEFAULT_MAX_PAGES = 5

# Utility functions
def handle_output(data: Any, save_path: Optional[str] = None, json_output: bool = False, raw_output: bool = False):
    """Handle output in either JSON, raw text, or rich formatted mode"""
    if json_output:
        json_data = json.dumps(data, indent=4, ensure_ascii=False)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(json_data)

        data = JSON(json_data)
    

    if save_path:
        if isinstance(data, str):
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(data)
        else:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        console.print(f"[green]Output saved to:[/] {save_path}")
        return data

    console.print(data)

    return data

def handle_stdin_data(stdin_format: Optional[str] = None) -> Optional[Tuple[str, str]]:
    """
    Process data from stdin if available and return the temp file path and detected format
    
    Returns:
        Optional[Tuple[str, str]]: (temp_file_path, format) or None if no stdin data
    """
    if sys.stdin.isatty():
        return None
        
    # Create a temporary file for the stdin data
    temp_path = os.path.join(tempfile.gettempdir(), f"mrblack_stdin_{uuid4().hex}")
    
    try:
        stdin_data = sys.stdin.buffer.read()
        if not stdin_data:
            console.print("[red]Error:[/] No data received from stdin.")
            return None
            
        # Write to temp file
        with open(temp_path, 'wb') as f:
            f.write(stdin_data)
            
        # Determine format if not specified
        format_type = stdin_format
        if not format_type:
            import magic
            mime_type = magic.from_file(temp_path, mime=True)
            format_type = mime_type.split('/')[-1]
            
        return temp_path, format_type
    except Exception as e:
        #console.print(f"[red]Error processing stdin data:[/] {e}")
        if os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
        return None

def display_pii_results(results: Dict[str, List[str]], title: str = "PII Extraction Results"):
    """Pretty-print PII extraction results in a formatted table"""
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
    
    console.print(Panel(table, border_style="blue"))
    return results

def display_labels_in_columns(label_list: List[str]):
    """Nicely print all available PII labels in columns"""
    import shutil
    
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
        
    console.print(table)
    return labels

def process_source(source: str, func: Callable, **kwargs) -> Any:
    """Process a source (file, URL, screenshot) with the given function"""
    if source.lower() in ("screenshot", "screen", "capture"):
        console.print("[cyan]Capturing screenshot...[/cyan]")
        return func(**kwargs)
    elif is_url(source):
        console.print(f"[cyan]Processing URL:[/] {source}")
        return func(source, **kwargs)
    else:
        path = clean_path(source)
        if not path:
            console.print(f"[red]Error:[/] Invalid path '{source}'")
            return None
        console.print(f"[cyan]Processing file:[/] {path}")
        return func(path, **kwargs)


# Main CLI group
@click.group(invoke_without_command=True)
@click.pass_context
@click.version_option(version="0.1.0")
def cli(ctx):
    """
    mrblack: Universal data extraction and analysis toolkit
    
    Extract text and data from any source, analyze content, and identify patterns.
    """
    # If no subcommand is provided, run basic extraction
    if ctx.invoked_subcommand is None:
        # Check if there are any arguments at all
        if len(sys.argv) <= 1:
            if not sys.stdin.isatty():
                # Data is being piped in, run default extraction on stdin
                ctx.invoke(extract)
            else:
                # Display help
                click.echo(ctx.get_help())
                return
        else:
            # Treat the first argument as a source for extraction
            source = sys.argv[1]
            
            # Remove the source from argv
            sys.argv.pop(1)
            
            # Check if --raw flag is in the remaining arguments
            raw_flag = '--raw' in sys.argv
            
            # Check if --json flag is in the remaining arguments
            json_flag = '--json' in sys.argv
            
            # Check if --output option is specified
            output_value = None
            for i, arg in enumerate(sys.argv):
                if arg == '--output' and i + 1 < len(sys.argv):
                    output_value = sys.argv[i + 1]
                    break
                elif arg.startswith('--output='):
                    output_value = arg.split('=', 1)[1]
                    break
            
            # Invoke extract with the processed arguments
            ctx.invoke(extract, source=source, raw=raw_flag, json=json_flag, output=output_value)


@cli.command()
@click.argument('source', required=False)
@click.option('--password', help='Password for protected documents')
@click.option('--chunked', is_flag=True, help='Process large files in chunks to reduce memory usage')
@click.option('--no-js', is_flag=True, help='Disable JavaScript rendering for web pages')
@click.option('--output', help='Save output to a file')
@click.option('--raw', is_flag=True, help='Output plain text without formatting')
@click.option('--stdin-format', help='Specify format for stdin data (pdf, docx, txt, etc.)')
def extract(source: Optional[str], password: Optional[str], chunked: bool, no_js: bool, output: Optional[str], raw: bool, stdin_format: Optional[str]):
    """Extract text from a file, URL, or screenshot"""
    # Handle stdin if no source is provided
    if not source:
        stdin_data = handle_stdin_data(stdin_format)
        if stdin_data:
            source = stdin_data[0]
            #console.print(f"[cyan]Processing stdin data (format: {stdin_data[1]})...[/cyan]")
        else:
            console.print("[yellow]No input source provided.[/yellow]")
            return
    
    # Process source based on type
    if source.lower() in ("screenshot", "screen", "capture"):
        console.print("[cyan]Capturing screenshot...[/cyan]")
        text = text_from_screenshot()
    elif is_url(source):
        console.print(f"[cyan]Extracting text from URL:[/] {source}")
        text = text_from_url(source, render_js=not no_js)
    else:
        path = clean_path(source)
        if not path:
            console.print(f"[red]Error:[/] Invalid path '{source}'")
            return
        
        console.print(f"[cyan]Extracting text from:[/] {path}")
        if chunked:
            text = extract_text_chunked(path)
        elif password:
            # This requires implementation of extract_text_with_password in textextract
            # For now, fall back to regular extraction with a warning
            console.print("[yellow]Warning:[/] Password-protected extraction not fully implemented")
            text = extract_text(path)
        else:
            text = extract_text(path)
    
    if text is not None:
        if raw:
            handle_output(text, output, False, raw)
        else:
            # Format for display with Rich panel
            console.print(Panel(text, title="Extracted Text", border_style="green", expand=False))
            console.print(f"[cyan]Total length:[/] {len(text)} characters")
    else:
        console.print("[red]No text could be extracted from the source.[/red]")


# Summarize command
@cli.command()
@click.argument('source', required=False)
@click.option('--sentences', type=int, default=5, help='Number of sentences in summary')
@click.option('--output', help='Save output to a file')
@click.option('--json', is_flag=True, help='Output results as JSON')
@click.option('--raw', is_flag=True, help='Output plain text without formatting')
@click.option('--stdin-format', help='Specify format for stdin data')
def summarize(source: Optional[str], sentences: int, output: Optional[str], 
               json: bool, raw: bool, stdin_format: Optional[str]):
    """Summarize text from a file, URL, or screenshot"""
    # Handle stdin if no source is provided
    if not source:
        stdin_data = handle_stdin_data(stdin_format)
        if stdin_data:
            source = stdin_data[0]
            #console.print(f"[cyan]Processing stdin data (format: {stdin_data[1]})...[/cyan]")
        else:
            console.print("[yellow]No input source provided.[/yellow]")
            return
    
    # First extract the text
    if source.lower() in ("screenshot", "screen", "capture"):
        text = text_from_screenshot()
    elif is_url(source):
        text = text_from_url(source)
    else:
        path = clean_path(source)
        if not path:
            console.print(f"[red]Error:[/] Invalid path '{source}'")
            return
        text = extract_text(path)
    
    if not text:
        console.print("[red]No text could be extracted from the source.[/red]")
        return
    
    # Summarize the text
    summary = summarize_text(text, sentences)
    
    if not json and not raw:
        console.print(Panel(
            summary, 
            title=f"Summary ({sentences} sentences)",
            border_style="green",
            expand=False
        ))
    
    # Handle output options
    handle_output(summary, output, json, raw)

# Analyze command
@cli.command()
@click.argument('source', required=False)
@click.option('--output', help='Save output to a file')
@click.option('--json', is_flag=True, help='Output results as JSON')
@click.option('--raw', is_flag=True, help='Output plain text without formatting')
@click.option('--stdin-format', help='Specify format for stdin data')
def analyze(source: Optional[str], output: Optional[str], json: bool, raw: bool, stdin_format: Optional[str]):
    """Analyze text to extract metrics and insights"""
    # Handle stdin if no source is provided
    if not source:
        stdin_data = handle_stdin_data(stdin_format)
        if stdin_data:
            source = stdin_data[0]
            #console.print(f"[cyan]Processing stdin data (format: {stdin_data[1]})...[/cyan]")
        else:
            console.print("[yellow]No input source provided.[/yellow]")
            return
    
    # First extract the text
    if source.lower() in ("screenshot", "screen", "capture"):
        text = text_from_screenshot()
    elif is_url(source):
        text = text_from_url(source)
    else:
        path = clean_path(source)
        if not path:
            console.print(f"[red]Error:[/] Invalid path '{source}'")
            return
        text = extract_text(path)
    
    if not text:
        console.print("[red]No text could be extracted from the source.[/red]")
        return
    
    # Analyze the text
    analysis = analyze_text(text)
    
    if not json and not raw:
        # Create a rich display of analysis results
        table = Table(title="Text Analysis Results", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        # Add general metrics
        for key, value in analysis.items():
            if key == "most_common_words":
                continue  # Handle separately
            table.add_row(key.replace("_", " ").title(), str(value))
        
        console.print(table)
        
        # Display most common words in a separate table
        if "most_common_words" in analysis:
            words_table = Table(title="Most Common Words", box=box.ROUNDED)
            words_table.add_column("Word", style="cyan")
            words_table.add_column("Count", style="green", justify="right")
            
            for word, count in analysis["most_common_words"]:
                words_table.add_row(word, str(count))
            
            console.print(words_table)
    
    # Handle output options
    handle_output(analysis, output, json, raw)

# Metadata command
@cli.command()
@click.argument('source', required=False)
@click.option('--output', help='Save output to a file')
@click.option('--json', is_flag=True, help='Output results as JSON')
@click.option('--raw', is_flag=True, help='Output plain text without formatting')
@click.option('--stdin-format', help='Specify format for stdin data')
def metadata(source: Optional[str], output: Optional[str], json: bool, raw: bool, stdin_format: Optional[str]):
    """Extract metadata from a file or URL"""
    # Handle stdin if no source is provided
    if not source:
        stdin_data = handle_stdin_data(stdin_format)
        if stdin_data:
            source = stdin_data[0]
            #console.print(f"[cyan]Processing stdin data (format: {stdin_data[1]})...[/cyan]")
        else:
            console.print("[yellow]No input source provided.[/yellow]")
            return
    
    # Process based on source type
    if is_url(source):
        # For URLs, use requests to get basic metadata
        import requests
        try:
            resp = requests.head(source)
            metadata = {
                "url": source,
                "status_code": resp.status_code,
                "content_type": resp.headers.get('Content-Type'),
                "content_length": resp.headers.get('Content-Length'),
                "server": resp.headers.get('Server'),
                "last_modified": resp.headers.get('Last-Modified'),
            }
        except Exception as e:
            console.print(f"[red]Error fetching URL metadata:[/] {e}")
            return
    else:
        path = clean_path(source)
        if not path:
            console.print(f"[red]Error:[/] Invalid path '{source}'")
            return
        
        metadata = extract_metadata(path)
    
    if not json and not raw:
        # Create a rich display of metadata
        table = Table(title="Metadata", box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        def add_metadata_to_table(data, prefix=""):
            for key, value in data.items():
                if isinstance(value, dict):
                    add_metadata_to_table(value, prefix=f"{key}.")
                else:
                    table.add_row(f"{prefix}{key}".replace("_", " ").title(), str(value))
        
        add_metadata_to_table(metadata)
        console.print(table)
    
    # Handle output options
    handle_output(metadata, output, json, raw)

# Translate command
@cli.command()
@click.argument('lang', required=False)
@click.argument('source', required=False)
@click.option('--output', help='Save output to a file')
@click.option('--json', is_flag=True, help='Output results as JSON')
@click.option('--raw', is_flag=True, help='Output plain text without formatting')
@click.option('--stdin-format', help='Specify format for stdin data')
def translate(lang: Optional[str], source: Optional[str], output: Optional[str], 
               json: bool, raw: bool, stdin_format: Optional[str]):
    """
    Translate text to another language or list available languages
    
    LANG: Target language code (e.g., 'en', 'es', 'fr')
    SOURCE: File, URL, or 'screenshot'
    """
    # If no language specified, list available languages
    if not lang:
        languages = list_available_languages()
        if json or raw:
            handle_output(languages, output, json, raw)
        else:
            # Display available languages
            table = Table(title="Available Translation Languages", box=box.ROUNDED)
            table.add_column("Code", style="cyan")
            table.add_column("Language", style="green")
            
            for code, name in sorted(languages.items(), key=lambda x: x[1]):
                table.add_row(code, name)
            
            console.print(table)
        return
    
    # If language but no source, check for stdin
    if not source:
        stdin_data = handle_stdin_data(stdin_format)
        if stdin_data:
            source = stdin_data[0]
            #console.print(f"[cyan]Processing stdin data (format: {stdin_data[1]})...[/cyan]")
        else:
            console.print("[yellow]No input source provided.[/yellow]")
            return
    
    # First extract the text
    if source.lower() in ("screenshot", "screen", "capture"):
        text = text_from_screenshot()
    elif is_url(source):
        text = text_from_url(source)
    else:
        path = clean_path(source)
        if not path:
            console.print(f"[red]Error:[/] Invalid path '{source}'")
            return
        text = extract_text(path)
    
    if not text:
        console.print("[red]No text could be extracted from the source.[/red]")
        return
    
    # First detect the original language
    source_lang = detect_language(text)
    
    # Translate the text
    console.print(f"[cyan]Translating from detected language '{source_lang}' to '{lang}'...[/cyan]")
    translated = translate_text(text, lang)
    
    if not translated:
        console.print(f"[red]Translation failed. Please check if the language code '{lang}' is valid.[/red]")
        return
    
    if not json and not raw:
        console.print(Panel(
            translated, 
            title=f"Translated Text ({source_lang} → {lang})",
            border_style="green",
            expand=False
        ))
    
    # Handle output options
    handle_output(translated, output, json, raw)

# Scrape command
@cli.command()
@click.argument('url')
@click.option('--max-pages', type=int, default=DEFAULT_MAX_PAGES, 
              help=f'Maximum pages to scrape (default: {DEFAULT_MAX_PAGES})')
@click.option('--stay-on-domain', is_flag=True, default=True, 
              help='Stay on the same domain while scraping')
@click.option('--output', help='Save output to a file')
@click.option('--json', is_flag=True, help='Output results as JSON')
@click.option('--raw', is_flag=True, help='Output plain text without formatting')
def scrape(url: str, max_pages: int, stay_on_domain: bool, 
            output: Optional[str], json: bool, raw: bool):
    """Scrape multiple pages from a website"""
    if not is_url(url):
        console.print(f"[red]Error:[/] '{url}' is not a valid URL.")
        return
    
    console.print(f"[cyan]Scraping website:[/] {url}")
    console.print(f"[cyan]Maximum pages:[/] {max_pages}")
    console.print(f"[cyan]Stay on domain:[/] {stay_on_domain}")
    
    results = scrape_website(url, max_pages=max_pages, stay_on_domain=stay_on_domain)
    
    if not results:
        console.print("[yellow]No pages were successfully scraped.[/yellow]")
        return
    
    if not json and not raw:
        # Display summary of scraped pages
        table = Table(title=f"Scraped {len(results)} Pages", box=box.ROUNDED)
        table.add_column("URL", style="cyan")
        table.add_column("Content Length", style="green", justify="right")
        
        for page_url, content in results.items():
            table.add_row(page_url, str(len(content)))
        
        console.print(table)
        
        # Show preview of first page
        first_url = next(iter(results))
        console.print(Panel(
            results[first_url], 
            title=f"Preview of first page: {first_url}",
            border_style="green",
            expand=False
        ))
    
    # Handle output options
    handle_output(results, output, json, raw)

# Screenshot command
@cli.command()
@click.option('--output', help='Save output to a file')
@click.option('--json', is_flag=True, help='Output results as JSON')
@click.option('--raw', is_flag=True, help='Output plain text without formatting')
def screenshot(output: Optional[str], json: bool, raw: bool):
    """Capture screenshot and extract text via OCR"""
    console.print("[cyan]Capturing screenshot...[/cyan]")
    
    text = text_from_screenshot()
    
    if not text:
        console.print("[red]No text could be extracted from the screenshot.[/red]")
        return
    
    if not json and not raw:
        console.print(Panel(
            text, 
            title="Text Extracted from Screenshot",
            border_style="green",
            expand=False
        ))
    
    # Handle output options
    handle_output(text, output, json, raw)

# PII command
@cli.command()
@click.argument('labels', required=False)
@click.argument('source', required=False)
@click.option('--serial', is_flag=True, help='Per-file results for directories')
@click.option('--output', help='Save output to a file')
@click.option('--json', is_flag=True, help='Output results as JSON')
@click.option('--raw', is_flag=True, help='Output plain text without formatting')
@click.option('--stdin-format', help='Specify format for stdin data')
def pii(labels: Optional[str], source: Optional[str], serial: bool, 
         output: Optional[str], json: bool, raw: bool, stdin_format: Optional[str]):
    """
    Extract PII (Personally Identifiable Information) from text
    
    LABELS: Optional comma-separated list of PII labels to extract
    SOURCE: File, URL, directory, or 'screenshot'
    """
    # Process labels argument
    label_list = None
    if labels:
        if labels.lower() == "list":
            # Just list available labels
            all_labels = get_labels()
            if json or raw:
                handle_output(all_labels, output, json, raw)
            else:
                display_labels_in_columns(all_labels)
            return
        else:
            # Parse comma-separated labels
            label_list = [label.strip() for label in labels.split(',') if label.strip()]
            
            # Check if the first "label" looks like a source path or URL
            if len(label_list) == 1 and (os.path.exists(label_list[0]) or is_url(label_list[0]) or 
                                         label_list[0].lower() in ("screenshot", "screen", "capture")):
                # This is actually a source, not a label
                source = label_list[0]
                label_list = None
    
    # If no source is provided, check stdin
    if not source:
        stdin_data = handle_stdin_data(stdin_format)
        if stdin_data:
            source = stdin_data[0]
            #console.print(f"[cyan]Processing stdin data (format: {stdin_data[1]})...[/cyan]")
        else:
            # If no labels were provided either, just list all available labels
            if not label_list:
                all_labels = get_labels()
                if json or raw:
                    handle_output(all_labels, output, json, raw)
                else:
                    display_labels_in_columns(all_labels)
                return
            else:
                console.print("[yellow]No input source provided.[/yellow]")
                return
    
    # Process the source
    if source.lower() in ("screenshot", "screen", "capture"):
        console.print("[cyan]Capturing screenshot for PII extraction...[/cyan]")
        result = extract_pii_screenshot(label_list)
    elif is_url(source):
        console.print(f"[cyan]Extracting PII from URL:[/] {source}")
        result = extract_pii_url(source, label_list)
    else:
        path = clean_path(source)
        if not path:
            console.print(f"[red]Error:[/] Invalid path '{source}'")
            return
        
        if os.path.isdir(path):
            console.print(f"[cyan]Recursively extracting PII from directory:[/] {path}")
            # This would require the directory function from pii.py
            # For now, we'll create a placeholder implementation
            from pii import directory as pii_directory
            result = pii_directory(path, label_list, serial=serial)
        else:
            console.print(f"[cyan]Extracting PII from file:[/] {path}")
            result = extract_pii_file(path, label_list)
    
    if not result:
        console.print("[yellow]No PII matches found.[/yellow]")
        return
    
    # Handle output
    if json or raw:
        handle_output(result, output, json, raw)
    else:
        if isinstance(result, dict):
            if serial and all(isinstance(v, dict) for v in result.values()):
                # Per-file results for directories
                for fp, res in result.items():
                    display_pii_results(res, title=f"PII Results for {fp}")
            else:
                # Single file or aggregated results
                display_pii_results(result)
        else:
            console.print(result)


# List command group for various listings
@cli.group()
def list():
    """List available options and capabilities"""
    pass

@list.command(name="languages")
@click.option('--output', help='Save output to a file')
@click.option('--json', is_flag=True, help='Output results as JSON')
@click.option('--raw', is_flag=True, help='Output plain text without formatting')
def list_languages(output: Optional[str], json: bool, raw: bool):
    """List available translation languages"""
    languages = list_available_languages()

    if json or raw:
        handle_output(languages, output, json, raw)
    else:
        # Display available languages
        table = Table(title="Available Translation Languages", box=box.ROUNDED)
        table.add_column("Code", style="cyan")
        table.add_column("Language", style="green")

        for code, name in sorted(languages.items(), key=lambda x: x[1]):
            table.add_row(code, name)

        console.print(table)

@list.command(name="pii-labels")
@click.option('--output', help='Save output to a file')
@click.option('--json', is_flag=True, help='Output results as JSON')
@click.option('--raw', is_flag=True, help='Output plain text without formatting')
def list_pii_labels(output: Optional[str], json: bool, raw: bool):
    """List available PII extraction labels"""
    all_labels = get_labels()

    if json or raw:
        handle_output(all_labels, output, json, raw)
    else:
        display_labels_in_columns(all_labels)


def main():
    try:
        cli()
    except Exception as e:
        console.print(f"[bold red]Error:[/] {str(e)}")
        if '--debug' in sys.argv:
            import traceback
            console.print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
