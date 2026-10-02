# Merry's Way: Improvement Plan (pre-evaluation)

Findings from a read-through of the backend (agents, graph, RAG, scripts, SQL, data files). The goal is to fix the inconsistencies **before** building the evaluation suite, so the eval measures the system you actually want to ship.

All paths are relative to `backend/` unless stated otherwise.

**Not read (verify yourself before acting on anything touching them):** `recommendation_management_agent/`, `order_management_agent/`, `memory_management_agent/`, `router_agent/`, `input_processor_agent/`, `utils/email_util.py`, `data/apriori_recommendations.json`, and the whole frontend. Your live Supabase data was not accessible, so anything about live DB contents is inferred from code.

---

## How to use this file

Work is split in two phases on purpose:

- **Phase 1, fix inconsistencies and bugs (do now, before eval).** These are correctness, safety and documentation problems. Fixing them does not need a baseline.
- **Phase 2, performance improvements (do *after* the baseline eval).** Changes like re-embedding products or adding few-shot SQL examples will move your metrics. Measure the baseline first so the eval can show "before to after" lift, which is a stronger resume line than a single number.

Priority: **P0** = security or safety, fix first. **P1** = correctness or consistency. **P2** = cleanup.

---

## Phase 1: Fix before evaluating

### P0-1. Anyone logged in can likely run arbitrary SQL via PostgREST

**Where:** `scripts/initialize_bi_agent.py`, the `execute_sql_query` function.

The function is `security definer` (bypasses RLS) and ends with `grant execute ... to authenticated, service_role`. Functions in the `public` schema are exposed by Supabase's REST API, so any logged-in customer with a valid JWT can probably call `POST /rest/v1/rpc/execute_sql_query` directly. That skips the `/admin/chat` admin check entirely and exposes every order and customer profile. Possibly also other schemas such as `auth`, since the function runs with its owner's privileges.

**Fix**
- [ ] `revoke execute on function execute_sql_query(text) from public, anon, authenticated;` then grant to `service_role` only. Do the same for `match_schema_metadata`.
- [ ] Create a dedicated read-only role for BI queries with `SELECT` only on the tables or views the agent should see. Run the dynamic SQL under that role (`SET LOCAL ROLE bi_readonly` inside the function).
- [ ] Add `SET LOCAL statement_timeout = '5s'` inside the function (currently `SELECT pg_sleep(1000)` is a DoS).
- [ ] Reject statements containing `;`, `--`, `/*`, or references to `auth.`, `pg_catalog`, `information_schema`.
- [ ] Allow `WITH ... SELECT` (see P1-6).

**Verify:** call the RPC with a normal customer JWT and confirm it is rejected. Try `SELECT email FROM auth.users`, `SELECT pg_sleep(30)` and `SELECT 1; DROP TABLE x`, and confirm each fails.

### P0-2. Allergen filter misses most nut products

**Where:** `src/recommender/hybrid_recommender.py`, `_is_safe`. It does `allergen.lower() in ingredient_text` (literal substring).

Your resume says "dietary safety independent of model behavior". With the current check and an allergy of `nuts`:

| Product | Ingredient / reason | Caught? |
|---|---|---|
| Hazelnut Biscotti | "Hazelnuts" | yes (contains "nuts") |
| Banana Nut Bread | "Walnuts" | yes |
| Chocolate Chip Biscotti | "Almonds" | **no** |
| Almond Croissant | "Almond Cream", "Almonds" | **no** |
| Pecan Brownie | "Pecans" | **no** |
| Hazelnut Syrup | "Hazelnut Extract" | **no** |
| Toffee Nut Syrup | no nut in ingredients, description says toasted nuts | **no** (data issue) |

Allergy `dairy` or `gluten` matches nothing, because no ingredient contains those words. The opposite problem also exists: allergy `milk` wrongly blocks "Oat Milk" products. The sample `nut_allergy` test in `scripts/ml/train.py` would leak the same way.

**Fix**
- [ ] Add an explicit curated `allergens` field per product in `data/products_data/products.jsonl` (for example `["tree_nuts"]`, `["dairy","gluten","eggs"]`). This is the real fix, because ingredient strings are not a reliable allergen source.
- [ ] Add an allergen taxonomy that maps user terms to canonical allergens, with synonyms:
  ```python
  ALLERGEN_SYNONYMS = {
      "tree_nuts": {"nut", "nuts", "tree nut", "almond", "hazelnut", "pecan", "walnut", "pistachio", "cashew"},
      "peanuts":   {"peanut", "peanuts", "groundnut"},
      "dairy":     {"dairy", "milk", "lactose", "cream", "butter", "cheese"},
      "gluten":    {"gluten", "wheat", "flour"},
      "eggs":      {"egg", "eggs"},
  }
  ```
  Normalize the user's stored allergies to canonical keys, then compare against the product's `allergens` tag set.
- [ ] Keep "Oat Milk / Almond Milk" from tripping the dairy rule (tag-based matching solves this).
- [ ] Re-seed products (add an `allergens jsonb` column to `coffee_shop_products` and update `seed_products.py`).
- [ ] Apply the same check at **order time**: if the cart contains an item matching the user's allergies, warn before confirming. The filter currently only guards recommendations. *Verify in `order_management_agent/`.*
- [ ] Make the details agent aware of the user's allergies when asked "does X contain nuts?". It should answer from the tags, not the LLM's guess.
- [ ] Normalize what the memory agent stores (for example "tree nuts", "peanut allergy", "lactose intolerant") into canonical keys at write time.

**Verify:** for every canonical allergen, assert zero unsafe products appear in recommendations across all 58 products. This becomes your first eval test.

### P0-3. PII redaction has a dead branch and leaks names

**Where:** `src/agents/admin/admin_agent/`, `redact_results`.

The `name` key is on the safe list, so the condition `k == "name" and not any(safe in k ...)` can never be true. The formatting prompt tells the LLM to alias columns as `name` and `value` for charts, so "top customers by spend" returns real customer names under the `name` key. Emails are masked by the `@` rule, but plain names are not.

**Fix**
- [ ] Best fix: create masked views (for example `v_orders_safe`, `v_profiles_safe`) that drop or hash PII, and expose **only those views** to the BI role. Redaction then does not depend on key-name guessing.
- [ ] Short-term: redact by content as well as by key, and stop whitelisting `name` blindly. Use column provenance where possible.
- [ ] Restrict which columns of `coffee_shop_profiles` are queryable at all (the BI metadata only needs aggregate-friendly fields such as `location`, `allergies`).

**Verify:** run "list my top 5 customers by spend" and confirm no real names or emails reach the LLM or the chart payload.

### P1-1. About-us has two contradictory sources of truth

**Where:** `src/tools/about_us.py` (hardcoded string) vs `data/products_data/Merrys_way_about_us.txt`, and the hours in `details_management_agent/prompt.py`.

| Fact | `about_us.py` | `Merrys_way_about_us.txt` |
|---|---|---|
| Founded | 2015 | 2019 |
| Closing time (Mon-Fri / Sat) | 8 PM | 9 PM |
| Sunday | until 6 PM | until 7 PM |
| Delivery areas | Camp, Shivaji Nagar | Baner, Kothrud |

The tool ignores the `.txt` entirely, though the README says it reads it. The prompt's hours match the hardcoded string.

**Fix**
- [ ] Decide which is canonical (**your decision**).
- [ ] Keep one source (a `data/shop_info.yaml` or the `.txt`), load it in `about_us.py`, and generate the hours line in the prompt from it.
- [ ] Update the README.

### P1-2. Details-agent prompt uses `$`; the tools return `₹`

**Where:** `details_management_agent/prompt.py` examples show `$4.50`-style prices; tool output uses `₹` and real prices are in the hundreds. The model can copy the wrong symbol or scale from the examples.

- [ ] Rewrite the examples with `₹` and realistic catalog prices (for example Cappuccino at its actual price).
- [ ] Optionally centralize currency formatting in one helper used by all tools and prompts.

### P1-3. The prompt, the code and the README disagree on tool-loop behavior

The prompt says each tool is called at most once. `MAX_ITERATIONS = 5` is in the code. The README says "can call `rag_tool` multiple times with refined queries".

- [ ] Pick one behavior and make the prompt, the code and the README match.
- [ ] Replace the synchronous `tool.invoke` inside the async agent with `ainvoke` (or `asyncio.to_thread`) so it does not block the event loop.

### P1-4. Product "availability" is not real

**Where:** the product tools read `is_available`, but that column does not exist in `coffee_shop_products`, so availability is always true.

- [ ] Either add the column (default `true`) and honor it in retrieval and ordering, or remove the field and stop answering availability questions as if it were data.

### P1-5. Semantic fallback in name lookup silently substitutes products

**Where:** `get_product_by_name` in `src/tools/product_info.py`: after the exact `ilike` fails, it falls back to semantic search at threshold 0.5. BGE similarities usually sit well above 0.5 even for unrelated text, so a query like "pizza" may return some real product.

- [ ] Raise the threshold or calibrate it. Plot the similarity score for known-good vs out-of-catalog queries.
- [ ] When a semantic fallback is used, return the matched name explicitly ("I couldn't find X, did you mean Y?") instead of presenting Y as X.
- [ ] Define the out-of-scope behavior: the agent should say the item is not on the menu.

### P1-6. Admin agent logic bugs

**Where:** `src/agents/admin/admin_agent/`.

- [ ] **Follow-up detection bug.** `discovery_node` checks `any(kw in query.lower() for kw in ["it","those","that",...])`. Substring matching means "it" matches "with", "item", "units", so history is nearly always prepended. Use word-boundary regex, or always pass the last user turn. Also, `history[-1]` is the assistant's narrative, not the user's last question. Prepend the last *user* message.
- [ ] **CTEs are rejected.** The RPC requires the query to start with `SELECT`, but LLMs often write `WITH ... SELECT`. Allow `WITH` in the RPC guard, or add "do not use CTEs" to the generation prompt. Right now it burns a retry.
- [ ] **Final failure path.** After the retries are exhausted, the formatter still runs with empty results. Add an explicit failure branch that tells the user the query could not be answered, so the model cannot invent a narrative.
- [ ] **Timezone and "today".** The prompt uses `datetime.now()` (server local time, likely UTC on Render), while Postgres `CURRENT_DATE` is UTC and the shop is in IST. "Today" and "this month" can be off by hours. Pass an explicit timezone (`Asia/Kolkata`) and make "now" injectable so tests can pin it.
- [ ] **Row cap.** The formatter only sees `results[:20]`. If more rows exist the narrative should say it is showing the top 20, or the prompt should require a `LIMIT`.
- [ ] **Dead code.** `discover_schema` has unused `top_k_tables` / `top_k_columns` parameters.
- [ ] **Unknown-error detection.** The error check looks for an `"error"` key in the first row. Prefer a structured RPC response (`{ok, rows, error}`).
- [ ] **Order timestamps.** `coffee_shop_orders` only has `updated_at` (no `created_at`). Date analytics depend on it, so a later `PATCH` would silently move an order's date. Consider adding `created_at` and `confirmed_at`.

### P1-7. `schema.sql` is not reproducible

`supabase_db/schema.sql` is missing things the code needs. A fresh Supabase project built from the repo will not work.

Missing or out of sync:
- [ ] `create extension if not exists vector;`
- [ ] `embedding vector(768)` on `coffee_shop_products`, plus an HNSW/IVFFlat index
- [ ] The `match_coffee_products` RPC (used by `src/rag/retriever.py`)
- [ ] The `append_chat_messages` RPC (README says it exists)
- [ ] `coffee_shop_schema_metadata`, `match_schema_metadata` and `execute_sql_query` (currently only in `scripts/initialize_bi_agent.py`)
- [ ] `coffee_shop_admin_sessions`

Make `schema.sql` the single source of truth, then verify by building a brand-new Supabase project from it only. You will need this for the eval database anyway.

### P1-8. `save_order` relies on a unique constraint that must NOT exist

`order_manager.save_order` upserts with `on_conflict="user_email,status"`, which needs a unique constraint on `(user_email, status)`. `schema.sql` has none, so it falls back to delete+insert. **Do not add that constraint**: `confirm_order` inserts a `confirmed` row each time, and the constraint would block a user's second confirmed order.

- [ ] Remove the dead upsert path, or implement "one pending order per user" with a partial unique index plus an RPC.
- [ ] Document the order status values (`pending | confirmed`), since the README says `active | cancelled`.

### P1-9. `/chat/stream` and `/chat` behave differently

The README promises a parallel `asyncio.gather` pre-load including Mem0 semantic memory. In the code, `/chat` does it but `/chat/stream` loads things sequentially and skips the Mem0 lookup. If the frontend uses `/chat/stream` (the README says it does), the semantic-memory feature does not run on the main path.

- [ ] Share one pre-load function between both endpoints, or correct the README.

### P1-10. Product data scripts are not idempotent or consistent

- [ ] `scripts/generate_more_products.py` has a hardcoded absolute path (`/Users/pandhari/Desktop/Coffee_Shop_ChatBot/...`) and **appends** to `products.jsonl`, so running it twice duplicates 40 products. Use a relative path and dedupe by name.
- [ ] `seed_products.py` writes Supabase Storage image URLs; `reseed_prices.py` writes `/images/...` paths. Whichever ran last wins. Pick one.
- [ ] `reseed_prices.py` includes a fallback `CREATE TABLE` with `BIGSERIAL` ids and `TEXT[]` columns that contradicts `schema.sql`. Remove it.
- [ ] Add a validation script that checks the catalog: unique names, required fields, allergen tags present, price > 0, ingredient list non-empty.

### P2-1. Leftover Pinecone code

Retrieval now uses Supabase pgvector (your resume is right; the README is stale), but Pinecone remnants are still around.

- [ ] Delete or archive `src/rag/vector_db_setup.py` and the Pinecone parts of `scripts/reseed_prices.py`.
- [ ] In `src/utils/util.py`, `initialize_all` / `shutdown_all` reference Pinecone helpers that no longer exist (a `NameError` if called). Remove them.
- [ ] Drop `pinecone` / `langchain-pinecone` from `pyproject.toml` if unused, and `PINECONE_*` from `.env.example` and the README.

### P2-2. README and resume drift

**Root and `backend/README.md`**
- [ ] Vector store: pgvector (not Pinecone) for product search.
- [ ] `ingredients` and `allergies` are `jsonb` (not `TEXT[]`) *if that is what the live schema uses. Verify, then align schema.sql and README.*
- [ ] Order statuses: `pending | confirmed`.
- [ ] Checkpointer: SQLite on Mac, Postgres on Linux (README says `MemorySaver`).
- [ ] `about_us_tool` source (after P1-1).
- [ ] Email provider: `config.py` has SMTP settings while the README says Resend. *Check `email_util.py`.*
- [ ] `coffee_shop_schema_metadata.id` type matches the README.

**Resume**
- [ ] "8-agent LangGraph StateGraph": the customer StateGraph has 7 nodes; 8 only if you count the separate admin graph. Reword (for example "7-node customer graph plus a separate 4-node admin BI graph").
- [ ] Do not claim allergen safety "independent of model behavior" until P0-2 is fixed and tested.

### P2-3. CI does almost nothing

`.github/workflows/ci.yml` only runs `py_compile` on two files.

- [ ] Add `pytest` with unit tests for deterministic logic: allergen filter, `scrub_sql`, `redact_results`, price lookup, catalog validation. These double as part of the eval harness.

---

## Phase 2: Improve after the baseline eval

Run the baseline first, then make these changes and report before/after.

### RAG
- **Embed richer product text.** `scripts/seed_products.py` currently embeds only product **names** (`embed_documents(names)`), unless you re-embedded manually. Queries like "something cold with chocolate" will retrieve poorly. `vector_db_setup.py` already has a `create_product_text` helper (name + category + description + ingredients) you can reuse. Re-embed and compare Hit@k and MRR.
- Tune `match_threshold` (currently 0.5) and `top_k` (5) on a validation split.
- Try the BGE query instruction prefix for short queries, and a keyword or metadata filter (category, price) alongside dense retrieval.
- Format ingredients as a comma-joined string in `rag_tool` output instead of a Python list repr.

### Text-to-SQL
- Add 5-10 few-shot examples to the generation prompt, especially for the JSONB `items` column (`jsonb_array_elements`, joining to products by name).
- Add value-domain hints to the schema metadata (valid `status` values, product categories, item JSON shape).
- Evaluate with and without chat history for follow-ups.
- `schema_metadata.json` covers only 3 tables, so "retrieved schema vs full schema" will likely show little difference. Report it as a minor ablation, not a headline.

### Recommender (later)
- `scripts/ml/train.py` reports a `coverage()` that always returns 1.0 (`len/len`) and no predictive metric, so there is nothing quantitative to cite yet.
- `data/popularity_recommendation.csv` covers only the original ~18 products. The 40 generated products have no popularity signal. *Check `apriori_recommendations.json` for the same gap.*
- The content score matches `likes` against product-name substrings only, so descriptive likes such as "sweet" contribute nothing, and `last_order` is passed as an empty string. The 0.2 / 0.5 / 0.3 weights look hand-picked.
- Plan: build an offline eval (time-split baskets, Precision@K / NDCG@K / hit rate) and grid-search the weights.

---

## Eval-readiness checklist (needed before the harness is built)

- [ ] `schema.sql` can build a working database from scratch (P1-7).
- [ ] Second Supabase project for evaluation, with its own `.env.eval`.
- [ ] Seed script generating a few hundred confirmed orders over a fixed 90-day window, using the real `items` shape (`name`, `quantity`, `per_unit_price`, `total_price`, `image_url`) and following the popularity CSV for the original products.
- [ ] "Now" is injectable and pinned for the admin agent (P1-6).
- [ ] Temperature 0, and the model that actually answered (Groq vs OpenRouter fallback) is logged on every call.
- [ ] One canonical about-us source (P1-1).
- [ ] Curated allergen tags in the catalog (P0-2).
- [ ] BI role locked down (P0-1), so the adversarial SQL tests mean something.

## Decisions only you can make

1. Which about-us version is correct (2015 vs 2019, closing hours, delivery areas)?
2. Do you want an `allergens` tag set in the catalog (recommended) or ingredient-synonym matching only?
3. Add a real availability column, or remove availability answers?
4. Evaluate against a fresh seeded Supabase project (recommended) or a copy of your live data?

## Suggested order of work

1. P0-1 (lock down the SQL RPC), because it is a live security issue.
2. P0-2 (allergen tags and taxonomy), then P0-3 (PII views).
3. P1-7 (reproducible `schema.sql`), then P1-1 to P1-6.
4. P1-8 to P1-10, then the P2 cleanup and docs.
5. Seed eval DB, then baseline eval, then Phase 2 changes, then final eval.
