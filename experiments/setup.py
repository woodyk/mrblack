#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: setup.py
# Author: Wadih Khairallah
# Description: 
# Created: 2025-04-28 14:40:57
# Modified: 2025-05-17 21:03:12

from setuptools import setup, find_packages
from pathlib import Path

here = Path(__file__).parent

def read_requirements():
    return [
        line.strip()
        for line in (here / "requirements.txt").read_text().splitlines()
        if line and not line.startswith("#")
    ]

def get_version():
    version_file = here / "mrblack" / "__version__.py"
    version_ns = {}
    exec(version_file.read_text(), version_ns)
    return version_ns["__version__"]

setup(
    name="mrblack",
    version=get_version(),
    author="Wadih Khairallah",
    author_email="woodyk@gmail.com",
    description="Data extraction and analysis toolkit.",
    long_description=(here / "README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/woodyk/mrblack",
    packages=find_packages(include=["mrblack", "mrblack.*"]),
    install_requires=read_requirements(),
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "mrblack = mrblack.cli:main",
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

