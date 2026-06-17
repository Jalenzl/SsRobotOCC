"""Form-JSON parsing tests.

The previous ``CadAnalyzeOptions`` schema lived here as a stand-in; with the
feature recognition module removed, the schema is gone too. This file is kept
as a placeholder for future schema-specific tests; it is skipped until then.
"""

from __future__ import annotations

import pytest


pytestmark = pytest.mark.skip(
    reason="CadAnalyzeOptions has been removed along with feature recognition; "
    "re-enable tests once a new schema is added under app/models/cad.py."
)
