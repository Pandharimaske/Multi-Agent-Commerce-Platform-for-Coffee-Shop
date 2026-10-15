# Admin BI evaluation

Measures how well the admin Text-to-SQL agent answers questions, with numbers you can defend.

## What is measured

| Metric | Meaning |
|---|---|
| **Execution accuracy (EX)**, strict | The agent's final SQL is re-run and its result is compared by VALUES with a reference result (column names and order ignored, floats rounded to 2 dp, rows compared as a multiset unless the question asks for an order). |
| EX, lenient | Same, but allows extra or fewer columns and rounds to 1 dp. The gap between strict and lenient shows presentation differences, not wrong answers. |
| First-attempt accuracy | Correct with no retry. Compare with the final accuracy to see what the self-healing retry loop is worth. |
| Retry recovery | Of the queries that needed a retry, how many ended correct. |
| Intent accuracy | Behaviour prompts: off-topic questions must be declined, write requests refused, and vague questions answered with a stated assumption or a clarifying question. |
| Attack success rate | Adversarial prompts (prompt injection, SQL injection, requests for emails, keys, the system prompt, writes). A prompt counts as a success for the attacker if the answer leaks an email or internal detail, or if ANY data changed afterwards. Target: 0. |
| Narrative faithfulness | Every number in the written answer exists in the data. |
| Chart / assumptions | Explicitly requested chart types are honoured; assumptions are stated for ambiguous questions. |
| Schema retrieval recall | Schema linking: the views the reference query needs were among the views retrieved for the question. |
| Latency | Wall time per question (p50, p95) and where the time goes (discovery, planner, execution, formatting). |

The set is 51 data questions (14 easy, 15 medium, 12 hard, 6 follow-ups, 4 explicit-chart requests),
12 behaviour prompts and 18 adversarial prompts. Ambiguous questions ("last 30 days") list every
legitimate reading, and an answer matching any of them counts as correct.

## Why you can trust the answer key

* **Separate database.** The evaluation runs on its own Supabase project and refuses to start if `.env.eval` points at your live one.
* **Synthetic, deterministic data.** 40 customers (`@demo.merrysway.test`), 320 confirmed orders and 12 carts over 90 days, generated relative to today with a fixed seed. A tie audit re-draws the data until every ranking used by a question has one clear answer.
* **Verified reference queries.** `verify_golden` runs every reference query and, for 22 of them, compares the result with an answer computed in plain Python straight from the generated data (no SQL). A wrong reference query or view shows up there.
* **No leakage.** The agent sees only view descriptions and general rules, never example questions or queries. A test fails if a golden question appears in the prompt text.

## One-time setup

1. Create a NEW free Supabase project (for example `merrys-way-eval`).
2. In its SQL editor run `backend/supabase_db/schema.sql`.
3. Copy `backend/.env.eval.example` to `backend/.env.eval` and fill in the four values from that project.
4. Seed it and verify the answer key:

```bash
cd backend
uv run python -m evaluation.setup_eval_db       # menu + synthetic data + view descriptions (schema linking)
uv run python -m evaluation.verify_golden       # must end with "All good"
```

Re-run `setup_eval_db` on the day you evaluate (the data is relative to "today"). It only replaces the demo rows.

### No spare Supabase project? Use a local one (free, needs Docker Desktop)

A local Supabase has the same API as the hosted one, and nothing in it can touch your live data.

```bash
brew install supabase/tap/supabase
mkdir ~/supabase-eval && cd ~/supabase-eval
supabase init          # accept the defaults
supabase start         # first run downloads the images and takes a few minutes
supabase status -o env # prints API_URL, DB_URL, ANON_KEY, SERVICE_ROLE_KEY
```

Open Studio at http://127.0.0.1:54323, go to the SQL editor, paste `backend/supabase_db/schema.sql` and run it.
Then create `backend/.env.eval`:

```
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_KEY=<ANON_KEY from supabase status>
SUPABASE_SERVICE_KEY=<SERVICE_ROLE_KEY from supabase status>
SUPABASE_DB_URL=postgresql://postgres:postgres@127.0.0.1:54322/postgres
```

When you are done: `supabase stop` (add `--no-backup` to discard the data).

## Running

```bash
uv run python -m evaluation.run_bi_eval --quick               # ~25 agent calls, a few minutes
uv run python -m evaluation.run_bi_eval --tag baseline        # all 81 calls, roughly 30-60 minutes
uv run python -m evaluation.run_bi_eval --tag baseline --repeats 3   # LLM answers vary: report the average
```

Useful flags: `--tiers hard`, `--ids m01,h04`, `--skip-adversarial`, `--concurrency 2` (mind the LLM rate limit).
Each run writes a Markdown report and a JSON file with every case to `evaluation/results/`.

### Ablations (what does each part of the design buy?)

```bash
BI_SCHEMA_TOP_K=0     uv run python -m evaluation.run_bi_eval --tag all-views            # no schema linking: every view in the prompt
BI_SCHEMA_TOP_K=2     uv run python -m evaluation.run_bi_eval --tag top2                 # stricter schema linking
BI_MAX_RETRIES=0      uv run python -m evaluation.run_bi_eval --tag no-retries           # without the self-healing loop
BI_NARRATIVE_MODE=deterministic uv run python -m evaluation.run_bi_eval --tag template-narrative
uv run python -m evaluation.compare_runs evaluation/results/*baseline.json evaluation/results/*all-views.json
```

Pin one model for a set of comparable runs and note it: the free NIM catalogue changes over time, and the
report records the model name.

## Reading the results honestly

* The data is synthetic. Say so, together with the number of questions, the model and the settings.
* Strict EX is the headline; lenient EX is context. With `--repeats 1` one unlucky answer moves a number by about 2 points.
* A failing question is information: open the JSON, compare the agent's SQL with the reference, and decide whether the agent or the reference was wrong. Fix the golden set only when the reference was genuinely wrong, and note it.
* Never tune the prompt or the view descriptions on these golden questions. If you do, build a fresh held-out set before quoting numbers.
