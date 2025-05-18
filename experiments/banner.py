#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# File: banner.py
# Author: Wadih Khairallah
# Description: 
# Created: 2025-05-17 21:46:47
from pyfiglet import Figlet
from rich.console import Console
from rich.text import Text
from rich.color import Color
from rich.style import Style
from typing import List

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def adjust_brightness(rgb: tuple[int, int, int], factor: float) -> str:
    """Return a hex color with adjusted brightness"""
    r, g, b = rgb
    r = int(max(0, min(255, r * factor)))
    g = int(max(0, min(255, g * factor)))
    b = int(max(0, min(255, b * factor)))
    return f"#{r:02x}{g:02x}{b:02x}"

class GradientFigletBanner:
    def __init__(self, text: str, font: str = "slant", base_color: str = "#00ffcc"):
        self.text = text
        self.font = font
        self.base_color = base_color
        self.console = Console()
        self.figlet = Figlet(font=self.font)

    def render(self):
        banner_lines = self.figlet.renderText(self.text).splitlines()
        base_rgb = hex_to_rgb(self.base_color)
        total = len(banner_lines)

        for i, line in enumerate(banner_lines):
            factor = 0.5 + 0.5 * (i / max(1, total - 1))  # gradient from 50% to 100% brightness
            color = adjust_brightness(base_rgb, factor)
            styled_line = Text(line, style=Style(color=color))
            self.console.print(styled_line)

# Example usage:
if __name__ == "__main__":
    banner = GradientFigletBanner("Mr. Black", font="ansi_shadow", base_color="#00ffcc")
    banner.render()

