"""Pytest configuration: ensure the src/ tree takes precedence over any
installed miv_simulator package so tests always exercise the local sources."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
