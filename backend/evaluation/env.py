"""Load the evaluation environment, with a hard safety check against running on the live database.

The evaluation seeds synthetic data and runs adversarial prompts (including write attempts), so it must
use a SEPARATE Supabase project. Put that project's keys in backend/.env.eval; every other setting
(NIM key and embedding model, ...) still comes from backend/.env.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import dotenv_values

BACKEND = Path(__file__).resolve().parent.parent
REQUIRED = ("SUPABASE_URL", "SUPABASE_KEY", "SUPABASE_SERVICE_KEY", "SUPABASE_DB_URL")


def load_eval_env(allow_live: bool = False) -> None:
    eval_path = BACKEND / ".env.eval"
    if not eval_path.exists():
        raise SystemExit(
            "backend/.env.eval not found.\n"
            "Create a SEPARATE Supabase project for the evaluation, then copy backend/.env.eval.example "
            "to backend/.env.eval and fill it in (see evaluation/README.md)."
        )

    values = dotenv_values(eval_path)
    missing = [k for k in REQUIRED if not values.get(k)]
    if missing:
        raise SystemExit(f"backend/.env.eval is missing: {', '.join(missing)}")

    live = dotenv_values(BACKEND / ".env") if (BACKEND / ".env").exists() else {}
    for key in ("SUPABASE_URL", "SUPABASE_DB_URL"):
        same = (values.get(key) or "").strip().rstrip("/") == (live.get(key) or "").strip().rstrip("/")
        if same and live.get(key) and not allow_live:
            raise SystemExit(
                f"REFUSING TO RUN: {key} in .env.eval is identical to the one in .env (your LIVE project).\n"
                "The evaluation writes synthetic data and attempts writes. Use a separate Supabase project."
            )

    for key, value in values.items():
        if value is not None:
            os.environ[key] = value          # overrides .env and any stale shell variable
    os.environ["EVAL_MODE"] = "1"
    if str(BACKEND) not in sys.path:
        sys.path.insert(0, str(BACKEND))
