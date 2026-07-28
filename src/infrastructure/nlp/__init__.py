"""Natural-language feature extraction: sentiment, entities and topics.

Every component here degrades to a deterministic offline fallback when the
optional ``nlp`` extra is not installed.
"""

from __future__ import annotations
