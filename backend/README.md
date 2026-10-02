# ⚙️ Backend — Merry's Way

> FastAPI + LangGraph multi-agent backend powering the Coffee Shop AI Assistant.

---

## Contents

- [Project Layout](#-project-layout)
- [Agent Architecture](#-agent-architecture)
  - [Customer Pipeline](#customer-pipeline-7-nodes)
  - [Admin BI Pipeline](#admin-bi-pipeline-4-nodes)
- [API Endpoints](#-api-endpoints)
- [Data Layer](#-data-layer)
- [Database Schema](#-database-schema)
- [Environment Variables](#-environment-variables)
- [Local Development](#-local-development)
- [Key Design Decisions](#-key-design-decisions)

---

## 📁 Project Layout

```
backend/
├── api/
│   ├── routers/
│   │   ├── chat.py          # POST /chat, GET /chat/history, POST /chat/stream, POST /chat/resume
│   │   ├── admin.py         # POST /admin/chat, GET /admin/history
│   │   ├── orders.py        # GET /orders, PATCH /orders/:id
│   │   ├── products.py      # GET /products
│   │   ├── users.py         # GET /user/me, GET|PUT /user/preferences
│   │   └── auth.py          # POST /auth/login, /auth/signup, /auth/logout
│   ├── auth/
│   │   └── supabase_auth.py # JWT verification via Supabase, CurrentUser dependency
│   └── schemas.py           # Pydantic request/response models (ChatRequest, ChatResponse, etc.)
│
├── src/
│   ├── agents/
│   │   ├── input_processor_agent/   # Resolves pronouns + ambiguity
│   │   ├── memory_management_agent/ # Extracts + persists user preferences
│   │   ├── router_agent/            # 4-class intent classifier
│   │   ├── details_management_agent/# RAG agentic tool-calling loop
│   │   ├── order_management_agent/  # CRUD + HITL checkout
│   │   ├── recommendation_management_agent/ # Hybrid recommender
│   │   ├── general_agent/           # Small talk + streaming
│   │   └── admin/admin_agent/       # BI Text-to-SQL LangGraph
│   │
│   ├── graph/
│   │   ├── graph.py         # LangGraph StateGraph builder + OS-aware checkpointer
│   │   └── state.py         # CoffeeAgentState (shared state across all customer agents)
│   │
│   ├── memory/
│   │   ├── supabase_client.py  # Supabase anon + admin client singletons
│   │   ├── schemas.py          # UserMemory pydantic model
│   │   ├── memory_manager.py   # CRUD for coffee_shop_profiles table
│   │   └── mem0_manager.py     # Mem0 Cloud (MemoryClient v2) — semantic memory
│   │
│   ├── orders/
│   │   └── order_manager.py    # get_active_order, create/update order in Supabase
│   │
│   ├── sessions/
│   │   └── session_manager.py  # Atomic message append via RPC, load_messages
│   │
│   ├── rag/
│   │   ├── retriever.py        # pgvector query_products, retrieve_price_by_name
│   │   └── vector_db_setup.py  # (deprecated) one-time Pinecone indexing — pgvector used instead
│   │
│   ├── recommender/
│   │   └── hybrid_recommender.py # Fit, score (pop+apriori+content), recommend, persist
│   │
│   ├── tools/
│   │   ├── retriever_tool.py   # @tool: rag_tool (LangChain compatible)
│   │   ├── product_info.py     # @tool: product_info_tool
│   │   ├── about_us.py         # @tool: about_us_tool
│   │   └── schemas.py          # Tool input/output pydantic models
│   │
│   └── utils/
│       ├── util.py             # LLMPool, EmbeddingPool (thread-safe singletons, NVIDIA NIM)
│       ├── email_util.py       # Resend email receipts
│       └── logger.py           # Structured logger setup
│
├── scripts/
│   ├── index_metadata.py       # Seeds coffee_shop_schema_metadata with pgvector embeddings
│   ├── initialize_bi_agent.py  # Creates schema + RPC functions in Supabase
│   ├── seed_products.py        # Seeds products from products.jsonl into Supabase
│   └── migrate_images.py       # Uploads product images to Supabase Storage
│
├── data/
│   ├── products_data/
│   │   ├── products.jsonl          # 58-item product catalog (source of truth for seeding)
│   │   └── Merrys_way_about_us.txt # RAG context document for about-us queries
│   ├── apriori_recommendations.json # Pre-computed market basket associations
│   └── popularity_recommendation.csv # Product transaction frequency data
│
├── supabase_db/schema.sql      # Full DB schema with all tables and RPC functions
├── Dockerfile
├── pyproject.toml              # Dependencies managed via uv
└── render.yaml                 # Render deploy config
```

---

## 🤖 Agent Architecture

### Customer Pipeline (7 Nodes)

Every customer message flows through a **directed LangGraph `StateGraph`**. Agents share a single `CoffeeAgentState` object that carries the conversation forward.

```
CoffeeAgentState {
  user_input       : str           — the current message
  messages         : list[Message] — full conversation history
  user_memory      : UserMemory    — structured preferences (likes, allergies, etc.)
  semantic_memories: str           — Mem0 search results injected as context
  order            : list[Item]    — active cart
  final_price      : float
  response_message : str           — filled by the terminal specialist agent
}
```

#### Node 1 — Input Processor
- **Job**: Resolve ambiguity before anything else sees the message
- "I'll take two of those" → "I'll take two Cappuccinos"
- Uses the last 6 messages as a sliding context window

#### Node 2 — Memory Agent
- **Job**: Extract preferences from *every* message and persist them
- Writes to `coffee_shop_profiles` in Supabase (structured: likes, dislikes, allergies)
- Also writes to Mem0 Cloud for semantic long-term memory
- Runs even if there's no explicit preference — it extracts implicit signals

#### Node 3 — Router Agent
- **Job**: Classify intent → one of 4 specialist targets
- Uses `small_llm.with_structured_output(AgentDecision)` — returns typed JSON, not free text
- 4 routes: `details_management_agent` / `order_management_agent` / `recommendation_management_agent` / `general_agent`

#### Node 4 — Details Agent
- **Job**: Answer product/shop info questions using RAG
- Runs an **agentic tool-calling loop** — can call `rag_tool` multiple times with refined queries
- Tools: `rag_tool` (pgvector / Supabase), `product_info_tool` (Supabase), `about_us_tool` (txt file)
- Max 5 iterations to prevent infinite loops

#### Node 5 — Order Agent
- **Job**: Manage the full order lifecycle (add / update / remove / confirm / cancel)
- **Critical rule**: prices always fetched from Supabase — LLM-generated prices are never trusted
- Confirm step uses LangGraph `interrupt()` to pause the graph for human approval
- On resume: `POST /chat/resume` with the payment status

#### Node 6 — Recommendation Agent
- **Job**: Suggest products the user will like
- Hybrid scoring: `0.2 × popularity + 0.5 × apriori + 0.3 × content_similarity`
- **Allergen filter**: applied in Python *after* scoring, *before* LLM sees candidates
- Cart items excluded from recommendations automatically

#### Node 7 — General Agent
- **Job**: Handle greetings, order status, small talk
- Streams tokens via SSE (`astream_events` with `on_chat_model_stream`)

---

### Admin BI Pipeline (4 Nodes)

A completely separate LangGraph that handles SQL analytics. Lives at `src/agents/admin/admin_agent/`.

#### Node 1 — Discovery
```python
query → embed() → pgvector search on coffee_shop_schema_metadata
      → returns: global table overview + relevant column details
```

#### Node 2 — Generation
```python
(schema_context + query + history + error?) → NVIDIA NIM LLM → raw SQL
# error is injected on retries — LLM self-corrects
```

#### Node 3 — Execution
```python
scrub_sql(sql)                  # strip markdown fences + trailing semicolons
→ supabase.rpc("execute_sql_query", {"sql_query": sql})
→ redact_results(results)       # mask PII before LLM sees data
→ error? → back to Generation (max 2 retries)
```

#### Node 4 — Formatting
```python
(results + narrative request) → LLM → structured JSON:
{
  "narrative": "...",
  "chart_type": "bar | pie | line | table | none",
  "chart_data": [{"name": "...", "value": 123}],
  "sql": "..."
}
# React renders chart directly from this state
```

---

## 📡 API Endpoints

### Customer Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/chat` | ✅ JWT | Non-streaming chat (parallel pre-load, graph invoke) |
| `GET` | `/chat/history` | ✅ JWT | Restore session messages on page reload |
| `POST` | `/chat/stream` | ✅ JWT | SSE token streaming via `astream_events` |
| `POST` | `/chat/resume` | ✅ JWT | Resume graph after HITL interrupt (payment approval) |
| `POST` | `/chat/upload` | ✅ JWT | Upload image to Supabase Storage, returns public URL |

### Admin Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `POST` | `/admin/chat` | ✅ JWT + admin check | Run BI query → returns `AdminState` JSON |
| `GET` | `/admin/history` | ✅ JWT + admin check | Restore admin session conversation on refresh |

### User & Product Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/user/me` | ✅ JWT | Current user info + is_admin flag |
| `GET` | `/user/preferences` | ✅ JWT | Load UserMemory profile |
| `PUT` | `/user/preferences` | ✅ JWT | Update preferences (smart merge — won't wipe feedback) |
| `GET` | `/products` | ✅ JWT | Full product catalog from Supabase |
| `GET` | `/orders` | ✅ JWT | User's active order |

### Auth Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/auth/login` | Supabase email/password login |
| `POST` | `/auth/signup` | New user registration |
| `POST` | `/auth/logout` | Session invalidation |

---

## 🗄️ Data Layer

### Parallel Pre-load (Zero Sequential Latency)

Before the LangGraph is invoked on every `/chat` request, 5 async operations run **simultaneously**:

```python
_, user_memory, (order, final_price), messages, semantic_memories = await asyncio.gather(
    asyncio.to_thread(get_or_create_session, session_id, user_email),
    asyncio.to_thread(get_user_memory, user_email),
    asyncio.to_thread(get_active_order, user_email),
    asyncio.to_thread(load_messages, session_id),
    asyncio.to_thread(mem0_manager.search_memories, user_input, user_email)
)
```

The graph receives a fully-populated state with no sequential DB round trips.

### Atomic Message Persistence

Messages are never read-modify-written. A Supabase RPC (`append_chat_messages`) handles the append atomically on the database side — eliminates race conditions from concurrent requests.

```sql
-- Supabase function (JSONB concatenation, no Python read required)
UPDATE coffee_shop_sessions
SET messages = messages || p_new_messages::jsonb,
    last_active = NOW()
WHERE session_id = p_session_id;
```

---

## 🗃️ Database Schema

```sql
-- ── User profiles ─────────────────────────────────────────────────────────────
CREATE TABLE coffee_shop_profiles (
  user_email   TEXT PRIMARY KEY,
  name         TEXT,
  likes        TEXT[],
  dislikes     TEXT[],
  allergies    TEXT[],        -- hard constraint: Recommendation agent filters in Python
  last_order   TEXT,
  feedback     TEXT[],
  location     TEXT,
  is_admin     BOOLEAN DEFAULT FALSE,
  last_updated TIMESTAMPTZ DEFAULT NOW()
);

-- ── Chat sessions (single JSONB array — atomic RPC append) ────────────────────
CREATE TABLE coffee_shop_sessions (
  session_id  TEXT PRIMARY KEY,
  user_email  TEXT NOT NULL,
  messages    JSONB DEFAULT '[]',
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  last_active TIMESTAMPTZ DEFAULT NOW()
);

-- ── Admin BI sessions ─────────────────────────────────────────────────────────
CREATE TABLE coffee_shop_admin_sessions (
  session_id  TEXT PRIMARY KEY,
  user_email  TEXT NOT NULL,
  history     JSONB DEFAULT '[]',
  updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── Orders ────────────────────────────────────────────────────────────────────
CREATE TABLE coffee_shop_orders (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_email TEXT NOT NULL,
  items      JSONB NOT NULL DEFAULT '[]',
  total      FLOAT NOT NULL DEFAULT 0,
  status     TEXT NOT NULL DEFAULT 'pending',  -- pending | confirmed
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ── Products ──────────────────────────────────────────────────────────────────
CREATE TABLE coffee_shop_products (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name        TEXT NOT NULL,
  price       FLOAT NOT NULL,
  category    TEXT,
  description TEXT,
  ingredients TEXT[],
  rating      FLOAT,
  image_url   TEXT
);

-- ── BI Agent schema metadata (pgvector for semantic schema discovery) ─────────
CREATE TABLE coffee_shop_schema_metadata (
  id        BIGSERIAL PRIMARY KEY,
  content   TEXT,
  metadata  JSONB,
  embedding vector(768)
);

-- pgvector similarity search function used by Admin Discovery node
CREATE OR REPLACE FUNCTION match_schema_metadata(
  query_embedding vector, match_threshold float, match_count int
) RETURNS TABLE (id bigint, content text, metadata jsonb, similarity float) ...
```

Full schema with all RPC functions: [`supabase_db/schema.sql`](./supabase_db/schema.sql)

---

## 🔑 Environment Variables

Copy `.env.example` to `.env` and fill in:

```env
# ── LLM (NVIDIA NIM) ──────────────────────────────────────────────────────────
NIM_API_KEY=                    # nvapi-... key from build.nvidia.com
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=moonshotai/kimi-k3    # Primary (large) model
SMALL_LLM_MODEL=gpt-oss-20b     # Small model for router + structured output

# ── Embeddings ────────────────────────────────────────────────────────────────
HF_API_KEY=
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5

# ── Semantic Memory ───────────────────────────────────────────────────────────
MEM0_API_KEY=

# ── Supabase ──────────────────────────────────────────────────────────────────
SUPABASE_URL=
SUPABASE_KEY=                   # anon/public key — used only for auth verification
SUPABASE_SERVICE_KEY=           # service role key — used for all DB operations
SUPABASE_DB_URL=                # postgres:// URI — used by LangGraph PostgresSaver on Render

# ── Email Receipts (SMTP) ─────────────────────────────────────────────────────
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM_EMAIL=Coffee Shop <noreply@coffeeshop.com>

# ── CORS ──────────────────────────────────────────────────────────────────────
ALLOWED_ORIGINS=https://your-app.vercel.app  # comma-separated for multiple origins
```

---

## 🚀 Local Development

```bash
cd backend

# Install uv (Python package manager)
pip install uv

# Create virtualenv + install all dependencies
uv sync

# Activate (optional — uv run works without it)
source .venv/bin/activate

# Copy and fill environment
cp .env.example .env

# Run dev server (hot-reload on api/ and src/ changes)
uv run uvicorn api.main:app --reload --reload-dir api --reload-dir src

# API:  http://localhost:8000
# Docs: http://localhost:8000/docs  (Swagger UI)
```

### One-time Database Setup

```bash
# 1. Create tables + RPC functions in Supabase
#    (run the SQL in supabase_db/schema.sql via Supabase dashboard)

# 2. Seed product catalog
uv run python scripts/seed_products.py

# 3. Index schema metadata for Admin BI Agent
uv run python scripts/index_metadata.py

# 4. (Optional) Migrate product images to Supabase Storage
uv run python scripts/migrate_images.py
```

---

## 🔑 Key Design Decisions

### 1. Why LangGraph over plain LangChain?
LangGraph gives a stateful directed graph with explicit node transitions and a shared state object. This means:
- The Router node returns a *node name* — no if-else in application code
- The Memory agent writes `user_memory` to state; Recommendation reads it downstream — no second DB call
- `interrupt()` pauses graph execution at a node boundary, checkpoints state, and resumes later — impossible with a chain

### 2. OS-Aware Checkpointer
`psycopg`'s C extension segfaults on macOS ARM when forked by Uvicorn's multiprocessing. The checkpointer factory detects `platform.system()` and uses `MemorySaver` on Mac, `PostgresSaver` (via connection pool) on Linux/Render.

### 3. Allergen Safety is a Hard Constraint
Allergen filtering is done in Python *post-retrieval*, not by LLM reasoning. The LLM is probabilistic — it can be wrong. Python `in` checks are deterministic. The filter runs before candidates reach the LLM.

### 4. SQL Scrubbing for Admin Agent
LLMs often wrap SQL in markdown fences (` ```sql `) and add trailing semicolons. Both break Supabase's `execute_sql_query` RPC. A `scrub_sql()` function strips both before execution. If execution still fails, the error is injected into the next generation prompt — up to 2 retries.
