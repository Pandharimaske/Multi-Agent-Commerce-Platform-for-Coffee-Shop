"""Check that the configured NVIDIA NIM embedding model works with your key and matches the database.

    cd backend
    uv run python scripts/test_embeddings.py

It tests the model set in EMBEDDING_NIM_MODEL (default nvidia/nemotron-3-embed-1b) and then every other
embedding model your key can see, and for each one reports:
  * whether the call works and the vector's native size,
  * whether the size equals EMBEDDING_DIM (the vector(N) size in supabase_db/schema.sql),
  * a sanity check: a menu question must be closer to a matching passage than to an unrelated one.

The cosine scores are printed so you can choose sensible match_threshold values (menu search uses 0.5,
BI example retrieval uses 0.3; those were tuned for the old 768-dimension model).
"""
import math
import os
import sys
import time

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BACKEND)

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from openai import OpenAI  # noqa: E402

from src.config import Config  # noqa: E402

TARGET_DIM = int(os.getenv("EMBEDDING_DIM", Config.EMBEDDING_DIM))
CONFIGURED = os.getenv("EMBEDDING_NIM_MODEL", Config.EMBEDDING_NIM_MODEL)

QUERY = "How much is a cappuccino?"
CLOSE = "Cappuccino: espresso with steamed milk and foam. Price: 150"
FAR = "Who won the football match yesterday?"


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / ((math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0)


def embed(client, model, texts, input_type):
    """Return (vectors, note). Tries with the NIM input_type, then without it."""
    last = ""
    for use_input_type in (True, False):
        kwargs = {"model": model, "input": texts, "encoding_format": "float"}
        if use_input_type:
            kwargs["extra_body"] = {"input_type": input_type, "truncate": "END"}
        try:
            resp = client.embeddings.create(**kwargs)
            vectors = [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
            return vectors, ("" if use_input_type else "works only WITHOUT input_type")
        except Exception as e:
            last = str(e).replace("\n", " ")[:160]
    return None, last


def check(client, model):
    """Return a dict describing how the model behaves."""
    t0 = time.time()
    q, note = embed(client, model, [QUERY], "query")
    if q is None:
        return {"model": model, "ok": False, "note": note}
    native = len(q[0])
    p, _ = embed(client, model, [CLOSE, FAR], "passage")
    close = far = None
    sanity = "-"
    if p:
        close, far = cosine(q[0], p[0]), cosine(q[0], p[1])
        sanity = "ok" if close > far + 0.05 else "WEAK"
    return {"model": model, "ok": True, "native": native, "fits": native == TARGET_DIM, "sanity": sanity,
            "close": close, "far": far, "note": note, "secs": time.time() - t0}


def main() -> None:
    if not Config.NIM_API_KEY:
        raise SystemExit("NIM_API_KEY is not set in backend/.env")
    client = OpenAI(api_key=Config.NIM_API_KEY, base_url=Config.NIM_BASE_URL, timeout=30, max_retries=0)

    try:
        listed = [m.id for m in client.models.list().data]
        others = sorted(m for m in listed if "embed" in m.lower() and m != CONFIGURED)
    except Exception as e:
        print(f"Could not list models ({str(e)[:100]}); testing only the configured model.\n")
        others = []

    results = [check(client, m) for m in [CONFIGURED] + others]

    print(f"Database vector size (EMBEDDING_DIM): {TARGET_DIM}\n")
    print(f"{'model':<52} {'native dim':<11} {'fits DB':<8} {'sanity':<7} {'close/far':<12} notes")
    for r in results:
        tag = " (configured)" if r["model"] == CONFIGURED else ""
        if not r["ok"]:
            print(f"{r['model'] + tag:<52} {'-':<11} {'-':<8} {'-':<7} {'-':<12} {r['note']}")
            continue
        scores = f"{r['close']:.2f}/{r['far']:.2f}" if r["close"] is not None else "-"
        print(f"{r['model'] + tag:<52} {r['native']:<11} {('yes' if r['fits'] else 'NO'):<8} "
              f"{r['sanity']:<7} {scores:<12} {r['note']} ({r['secs']:.1f}s)")

    print()
    cfg = results[0]
    if not cfg["ok"]:
        print(f"PROBLEM: the configured model {CONFIGURED} does not work with your key: {cfg['note']}")
    elif not cfg["fits"]:
        print(f"PROBLEM: {CONFIGURED} returns {cfg['native']} dimensions but EMBEDDING_DIM / schema.sql say {TARGET_DIM}.\n"
              f"Set EMBEDDING_DIM={cfg['native']}, change vector({TARGET_DIM}) to vector({cfg['native']}) in "
              "supabase_db/schema.sql (and run a migration like migrations/002_embedding_dim_2048.sql).")
    elif cfg["sanity"] != "ok":
        print(f"WARNING: {CONFIGURED} works and fits the database, but the similarity sanity check was {cfg['sanity']}.")
    else:
        print(f"OK: {CONFIGURED} works, returns {cfg['native']} dimensions (matches the database) and ranks "
              f"the matching passage above the unrelated one ({cfg['close']:.2f} vs {cfg['far']:.2f}).")
        print("Use these scores as a guide when you check the match_threshold values (menu 0.5, BI examples 0.3).")


if __name__ == "__main__":
    main()
