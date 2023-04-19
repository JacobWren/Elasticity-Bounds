"""Supress solver output."""

import io
import sys


def silence():
    # Create a text trap and redirect stdout
    text_trap = io.StringIO()
    sys.stdout = text_trap


def speak():
    sys.stdout = sys.__stdout__
