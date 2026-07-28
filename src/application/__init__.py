"""Use cases orchestrating the domain over infrastructure adapters.

This layer may use ``pandas``; it is the seam where tabular data meets domain
types. It depends on ``domain`` protocols, never on concrete adapters.
"""

from __future__ import annotations
