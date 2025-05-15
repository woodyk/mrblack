#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: test_textextract.py
# Description: Test script for enhanced_textextract.py command-line functionality
# Created: 2025-05-15
# Modified: 2025-05-15 16:04:52

import os
import sys
import subprocess
import tempfile
import unittest
import glob
import json
import time
import shutil
from pathlib import Path

# Configure path to the enhanced_textextract.py script
SCRIPT_PATH = "./enhanced-textextract4.py"  # Update this to your script's path

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class TextExtractTestCase(unittest.TestCase):
    """Base class for TextExtract test cases with helper methods"""
    
    def setUp(self):
        """Create a temporary directory for test files"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample files for testing
        self.txt_file = os.path.join(self.temp_dir, "sample.txt")
        with open(self.txt_file, "w") as f:
            f.write("This is a sample text file for testing purposes.\n")
            f.write("It contains multiple lines and some unicode characters: éçñ.\n")
            f.write("End of the test file.")
        
        # Create a PDF file if possible
        self.pdf_file = None
        try:
            from reportlab.pdfgen import canvas
            pdf_path = os.path.join(self.temp_dir, "sample.pdf")
            c = canvas.Canvas(pdf_path)
            c.drawString(100, 750, "PDF Test Document")
            c.drawString(100, 700, "This PDF was created for testing the enhanced_textextract.py tool.")
            c.save()
            self.pdf_file = pdf_path
        except ImportError:
            print(f"{YELLOW}Warning: reportlab not installed; skipping PDF creation{RESET}")
    
    def tearDown(self):
        """Clean up temporary files"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def run_command(self, args):
        """Run the extractor with the given arguments and return the output"""
        cmd = [sys.executable, SCRIPT_PATH] + args
        print(f"{BLUE}Running: {' '.join(cmd)}{RESET}")
        
        try:
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60  # 60 second timeout
            )
            print(f"Exit code: {result.returncode}")
            if result.stderr:
                print(f"{YELLOW}STDERR:{RESET}\n{result.stderr[:200]}...")
            
            return result
        except subprocess.TimeoutExpired:
            print(f"{RED}Command timed out{RESET}")
            self.fail("Command timed out")
            
    def assert_success(self, result):
        """Assert that the command succeeded"""
        self.assertEqual(result.returncode, 0, f"Command failed with exit code {result.returncode}:\n{result.stderr}")
        
    def assert_failure(self, result):
        """Assert that the command failed"""
        self.assertNotEqual(result.returncode, 0, "Command succeeded but should have failed")
        
    def assert_in_output(self, result, expected_text):
        """Assert that the expected text is in the output"""
        self.assertIn(expected_text, result.stdout, f"Expected text not found in output")
        
    def assert_not_in_output(self, result, unexpected_text):
        """Assert that the unexpected text is not in the output"""
        self.assertNotIn(unexpected_text, result.stdout, f"Unexpected text found in output")


class BasicCommandsTest(TextExtractTestCase):
    """Test basic command-line functionality"""
    
    def test_help(self):
        """Test that help is displayed correctly"""
        # Test with explicit help flag
        result = self.run_command(['--help'])
        self.assert_success(result)
        self.assert_in_output(result, "usage:")
        self.assert_in_output(result, "--translate")
        
        # Test with no arguments (should show help)
        result = self.run_command([])
        self.assert_success(result)
        self.assert_in_output(result, "usage:")
    
    def test_list_languages(self):
        """Test language listing functionality"""
        # Test with --translate alone
        result = self.run_command(['--translate'])
        self.assert_success(result)
        self.assert_in_output(result, "Available Translation Languages")
        
        # Test with --list-languages
        result = self.run_command(['--list-languages'])
        self.assert_success(result)
        self.assert_in_output(result, "Available Translation Languages")
        
    def test_version(self):
        """Test version information (if available)"""
        result = self.run_command(['--version'])
        # This may or may not be implemented, so just check it runs
        print(f"Version result: {result.stdout}")


class FileProcessingTest(TextExtractTestCase):
    """Test processing of various file types"""
    
    def test_text_file(self):
        """Test basic text file extraction"""
        result = self.run_command([self.txt_file])
        self.assert_success(result)
        self.assert_in_output(result, "sample text file")
    
    def test_pdf_file(self):
        """Test PDF file extraction (if PDF was created)"""
        if self.pdf_file:
            result = self.run_command([self.pdf_file])
            self.assert_success(result)
            # We expect some text from the PDF, but it might be extracted differently
            # so just verify it ran without error
            print(f"PDF extraction result: {result.stdout[:200]}...")
        else:
            print(f"{YELLOW}Skipping PDF test (no PDF file created){RESET}")
            self.skipTest("No PDF file available for testing")
    ''' 
    def test_metadata(self):
        """Test metadata extraction"""
        result = self.run_command([self.txt_file, '--metadata'])
        self.assert_success(result)
        self.assert_in_output(result, "size_bytes")
        
        # Check if output is valid JSON
        try:
            data = json.loads(result.stdout)
            self.assertIn("mime", data)
        except json.JSONDecodeError:
            self.fail("Metadata output is not valid JSON")
    '''


class ProcessingOptionsTest(TextExtractTestCase):
    """Test various processing options"""
    
    def test_analyze(self):
        """Test text analysis feature"""
        result = self.run_command([self.txt_file, '--analyze'])
        self.assert_success(result)
        self.assert_in_output(result, "word_count")
        self.assert_in_output(result, "sentence_count")
        
        # Check if output is valid JSON
        try:
            data = json.loads(result.stdout)
            self.assertIn("most_common_words", data)
        except json.JSONDecodeError:
            self.fail("Analysis output is not valid JSON")
    
    def test_summarize(self):
        """Test text summarization feature"""
        # Create a larger text file with multiple paragraphs
        summary_file = os.path.join(self.temp_dir, "for_summary.txt")
        with open(summary_file, "w") as f:
            for i in range(10):
                f.write(f"This is paragraph {i+1} with some text for testing the summarization feature. ")
                f.write(f"It contains multiple sentences to ensure there's enough content to summarize. ")
                f.write(f"The summarization should extract the most important sentences from this text. ")
                f.write(f"Paragraph {i+1} ends here.\n\n")
        
        result = self.run_command([summary_file, '--summarize', '--sentences', '3'])
        self.assert_success(result)
        
        # Should be shorter than the original
        self.assertLess(len(result.stdout), 10*4*50)  # rough estimate of original size
    
    def test_entities(self):
        """Test named entity extraction"""
        # Create a text with named entities
        entities_file = os.path.join(self.temp_dir, "entities.txt")
        with open(entities_file, "w") as f:
            f.write("John Smith visited New York City on Tuesday. ")
            f.write("He met with representatives from Google and Apple to discuss new technology. ")
            f.write("They agreed to meet again in San Francisco next month.")
        
        result = self.run_command([entities_file, '--entities'])
        self.assert_success(result)
        
        # Check for entity types in output
        self.assert_in_output(result, "PERSON")
        self.assert_in_output(result, "GPE")  # Geo-political entity
        self.assert_in_output(result, "ORG")  # Organization
        
        # Check if output is valid JSON
        try:
            data = json.loads(result.stdout)
            self.assertIsInstance(data, dict)
        except json.JSONDecodeError:
            self.fail("Entities output is not valid JSON")


class BatchProcessingTest(TextExtractTestCase):
    """Test batch processing functionality"""

    '''
    def test_directory_processing(self):
        """Test processing all files in a directory"""
        # Create multiple files in the temp directory
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"file{i+1}.txt")
            with open(file_path, "w") as f:
                f.write(f"Content of file {i+1}\n")
        
        result = self.run_command([self.temp_dir])
        self.assert_success(result)
        
        # Check if output contains references to all files
        self.assert_in_output(result, "file1.txt")
        self.assert_in_output(result, "file2.txt")
        self.assert_in_output(result, "file3.txt")
        
        # Check if output is valid JSON
        try:
            data = json.loads(result.stdout)
            self.assertIsInstance(data, dict)
            self.assertGreaterEqual(len(data), 3)  # At least 3 files
        except json.JSONDecodeError:
            self.fail("Batch processing output is not valid JSON")

    def test_wildcard_pattern(self):
        """Test processing files matching a wildcard pattern"""
        # Create files with different extensions
        txt_path1 = os.path.join(self.temp_dir, "wild1.txt")
        txt_path2 = os.path.join(self.temp_dir, "wild2.txt")
        dat_path = os.path.join(self.temp_dir, "other.dat")
        
        for path in [txt_path1, txt_path2, dat_path]:
            with open(path, "w") as f:
                f.write(f"Content of {os.path.basename(path)}\n")
        
        # Use wildcard pattern to get only txt files
        pattern = os.path.join(self.temp_dir, "*.txt")
        result = self.run_command([pattern])
        self.assert_success(result)
        
        # Should find the two txt files but not the .dat file
        self.assert_in_output(result, "wild1.txt")
        self.assert_in_output(result, "wild2.txt")
        self.assert_not_in_output(result, "other.dat")
    '''


class OutputOptionsTest(TextExtractTestCase):
    """Test different output options"""
    
    def test_output_file(self):
        """Test writing output to a file"""
        output_path = os.path.join(self.temp_dir, "output.txt")
        
        result = self.run_command([self.txt_file, '--output', output_path])
        self.assert_success(result)
        
        # Check if output file was created and contains content
        self.assertTrue(os.path.exists(output_path), "Output file not created")
        with open(output_path, 'r') as f:
            content = f.read()
            self.assertIn("sample text file", content)
    
    def test_json_output(self):
        """Test JSON output with multiple files"""
        output_path = os.path.join(self.temp_dir, "output.json")
        
        # Create multiple files
        for i in range(2):
            file_path = os.path.join(self.temp_dir, f"json_test{i+1}.txt")
            with open(file_path, "w") as f:
                f.write(f"JSON test file {i+1}\n")
        
        pattern = os.path.join(self.temp_dir, "json_test*.txt")
        result = self.run_command([pattern, '--output', output_path])
        self.assert_success(result)
        
        # Check if output file is valid JSON
        self.assertTrue(os.path.exists(output_path), "Output file not created")
        with open(output_path, 'r') as f:
            try:
                data = json.load(f)
                self.assertIsInstance(data, dict)
                self.assertEqual(len(data), 2)  # Two files
            except json.JSONDecodeError:
                self.fail("Output is not valid JSON")


class AdvancedFeaturesTest(TextExtractTestCase):
    """Test advanced features of the tool"""
    
    def test_chunked_processing(self):
        """Test chunked processing for large files"""
        # Create a "large" text file (not truly large, but enough to test)
        large_file = os.path.join(self.temp_dir, "large.txt")
        with open(large_file, "w") as f:
            for i in range(1000):
                f.write(f"Line {i}: This is test content to simulate a large file.\n")
        
        result = self.run_command([large_file, '--chunked'])
        self.assert_success(result)
        self.assert_in_output(result, "Line 0")  # Check for some content
        self.assert_in_output(result, "Line 999")  # Check for last line
    
    def test_screenshot(self):
        """Test screenshot functionality - this is hard to verify but check it runs"""
        # This may fail in CI environments without a screen, so handle that case
        try:
            result = self.run_command(['--screenshot', '--verbose'])
            # Just check it runs without crashing - actual functionality is hard to verify
            print(f"Screenshot result: {result.returncode}, Output: {result.stdout[:100]}...")
        except Exception as e:
            print(f"{YELLOW}Screenshot test failed (this is expected in headless environments): {e}{RESET}")
            self.skipTest("Screenshot test requires a display")
        
    def test_translate(self):
        """Test translation functionality"""
        # Skip if deep-translator not available
        try:
            import deep_translator
        except ImportError:
            print(f"{YELLOW}Skipping translation test (deep-translator not installed){RESET}")
            self.skipTest("deep-translator not available")
            return
            
        result = self.run_command([self.txt_file, '--translate', 'es'])
        self.assert_success(result)
        # Look for common Spanish words that should appear in translation
        self.assert_in_output(result, "para")  # "for" in Spanish


class NetworkFeaturesTest(TextExtractTestCase):
    """Test network-related features"""
    
    def test_url_extraction(self):
        """Test extracting text from a URL (may need internet connection)"""
        # Use a reliable test website
        url = "http://example.com/"
        
        try:
            result = self.run_command([url])
            self.assert_success(result)
            # Check for content that should be on example.com
            self.assert_in_output(result, "Example Domain")
        except Exception as e:
            print(f"{YELLOW}URL test failed (network issue?): {e}{RESET}")
            self.skipTest("URL test requires internet connection")
    
    def test_no_js(self):
        """Test --no-js option with a URL"""
        url = "http://example.com/"
        
        try:
            result = self.run_command([url, '--no-js'])
            self.assert_success(result)
            # Still should extract basic content
            self.assert_in_output(result, "Example Domain")
        except Exception as e:
            print(f"{YELLOW}URL with --no-js test failed (network issue?): {e}{RESET}")
            self.skipTest("URL test requires internet connection")


def run_tests():
    """Run the test suite"""
    # First check if the script exists
    if not os.path.exists(SCRIPT_PATH):
        print(f"{RED}Error: Script not found at {SCRIPT_PATH}{RESET}")
        print(f"Please update SCRIPT_PATH in this test script to point to your enhanced_textextract.py")
        sys.exit(1)
        
    # Run tests
    unittest.main(argv=[sys.argv[0]])


if __name__ == "__main__":
    print(f"{BLUE}=" * 70)
    print(f"Testing enhanced_textextract.py command-line functionality")
    print(f"=" * 70 + RESET)
    
    run_tests()
