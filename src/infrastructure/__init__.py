"""Adapters for every external concern: data, models, NLP, XAI, API.

This layer depends on :mod:`domain` and implements its protocols. Nothing in
``domain`` may import from here.
"""

from __future__ import annotations
