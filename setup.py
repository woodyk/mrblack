#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: setup.py
# Author: Wadih Khairallah
# Description: 
# Created: 2025-04-28 14:40:57
# Modified: 2025-05-15 16:43:26

from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent

def read_requirements():
    return [
        line.strip()
        for line in (here / "requirements.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]

setup(
    name="mrblack",
    version="0.1.8",
    author="Wadih Khairallah",
    author_email="woodyk@gmail.com",
    description="Data extraction and text-extraction tools",
    long_description=(here / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/woodyk/mrblack",
    packages=find_packages(include=["mrblack", "mrblack.*"]),
    install_requires=read_requirements(),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "pii = mrblack.pii:main",
            "textextract = mrblack.textextract:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

