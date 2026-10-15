"""REMOVED: do not use.

The BI agent no longer retrieves verified question-and-SQL examples. It now retrieves view descriptions
(schema linking): see data/bi_schema_docs.json, scripts/index_metadata.py and src/utils/bi_context.py.

To check the descriptions against the real views run:
    uv run python scripts/validate_schema_docs.py

This stub can be deleted from the repository.
"""

import sys

if __name__ == "__main__":
    print(__doc__)
    sys.exit(0)
