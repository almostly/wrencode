"""What the test modules share: the ANSI codes wrencode emits and a stripper for them."""

from __future__ import annotations

import re

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLUE = "\033[34m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def strip_ansi(s: str) -> str:
    """Remove ANSI escape sequences from a string."""
    return re.sub(r"\x1b\[[0-9;]*m", "", s)
