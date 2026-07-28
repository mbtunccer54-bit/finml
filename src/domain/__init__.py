"""Pure business core.

This package must not import ``pandas``, ``numpy``, ``sklearn`` or any other
external runtime dependency — only the Python standard library. The rule is
enforced mechanically by ``scripts/check_domain_purity.py`` and by the
``test_domain_purity`` unit test.
"""

from __future__ import annotations

__all__ = ["entities", "exceptions", "repositories", "services", "value_objects"]
