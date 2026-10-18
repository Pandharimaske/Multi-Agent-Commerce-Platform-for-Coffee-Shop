<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,13,14&height=180&section=header&text=Merry%27s%20Way&fontSize=72&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=Multi-Agent%20AI%20Platform%20for%20Coffee%20Shops&descAlignY=60&descSize=22" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/LangGraph-Multi--Agent-7c3aed?style=for-the-badge&logo=langchain&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
  <img src="https://img.shields.io/badge/Supabase-Postgres%20%2B%20pgvector%20%2B%20Auth-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white" />
  <img src="https://img.shields.io/badge/NVIDIA%20NIM-LLM-76B900?style=for-the-badge&logo=nvidia&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Render%20%2B%20Vercel-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
</p>

<p>
  <a href="https://coffee-shop-chat-bot.vercel.app"><strong>🌐 Live Demo</strong></a> &nbsp;·&nbsp;
  <a href="https://coffee-shop-chatbot.onrender.com/docs"><strong>📡 API Docs</strong></a> &nbsp;·&nbsp;
  <a href="https://www.youtube.com/watch?v=APp6CWFgrXw"><strong>🎥 Video Walkthrough</strong></a> &nbsp;·&nbsp;
  <a href="./backend/README.md"><strong>⚙️ Backend Docs</strong></a> &nbsp;·&nbsp;
  <a href="./frontend/README.md"><strong>🎨 Frontend Docs</strong></a>
</p>

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
  - [Customer Agent Pipeline](#customer-agent-pipeline)
  - [Checkout with Human-in-the-Loop](#checkout-with-human-in-the-loop)
  - [Admin BI Pipeline](#admin-bi-pipeline)
  - [Memory Architecture](#memory-architecture)
  - [Data Model](#data-model)
- [Evaluation](#evaluation)
- [Key Design Decisions](#key-design-decisions)
- [Tech Stack](#tech-stack)
- [API Overview](#api-overview)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Repository Structure](#repository-structure)
- [Known Limitations and Roadmap](#known-limitations-and-roadmap)
- [License](#license)

---

## Overview

**Merry's Way** is a two-sided AI platform for a coffee shop, built on two independent LangGraph agent pipelines that share one FastAPI backend.

| Interface | Users | What it does |
|-----------|-------|--------------|
| **💬 Customer assistant** | Shop customers | Natural-language ordering, personalised recommendations, product Q&A, persistent memory of preferences and allergies, live token streaming, and a human-in-the-loop checkout |
| **📊 Admin BI dashboard** | Shop owner | Ask business questions in plain English and get a narrative plus an auto-selected chart or table, generated from live SQL |

**Guiding principle: the LLM decides and talks, code computes and enforces.** Prices always come from the database, cart totals are calculated in code, cart summaries are deterministic text, and recommendation candidates are filtered in Python before the model sees them.

---

## Features

### Customer assistant

- **Guarded, context-aware input**: an input processor resolves references ("two of those" → "two Cappuccinos"), blocks off-topic or unsafe requests, and accepts an uploaded image.
- **Persistent memory**: likes, dislikes and allergies are extracted from conversation, **confirmed by the user before saving**, and remembered across sessions.
- **Intent routing**: a structured-output router sends each message to one of four specialists.
- **Grounded product answers**: a tool-calling RAG agent searches the menu (pgvector), looks products up by name, and answers shop-info questions.
- **Cart and checkout**: add, update, remove and confirm items through chat. Checkout pauses the graph for explicit payment approval, then confirms the order and emails an HTML receipt.
- **Hybrid recommendations**: popularity, Apriori market-basket rules and content similarity, with allergen, dislike and in-cart filtering applied in code.
- **Real-time UX**: Server-Sent Events stream progress and tokens; chat history survives page refreshes.

### Admin BI

- **Broad coverage**: sales and revenue (day, week, month, year, trends, comparisons, growth), products and categories, customers (top spenders, repeat and new customers, locations, allergy and preference patterns), busiest hours and weekdays, abandoned carts, and products bought together.
- **Knows its limits**: a structured planning step decides whether to answer with SQL, ask one clarifying question, or politely decline off-topic and write requests. Assumptions (such as "this year = 1 Jan to today") are stated in the answer.
- **Schema-linked Text-to-SQL**: every view is described in plain language and embedded (pgvector). For each question only the most relevant views are retrieved and put in the prompt, so the context stays small as the number of tables grows. The model also sees business definitions and the live menu. No example questions or queries are ever shown to it.
- **Measured**: a separate evaluation harness scores the agent on a synthetic database (see [Evaluation](#evaluation)).
- **Self-healing queries**: validation and database errors are fed back to the model for up to two retries.
- **Private by construction**: queries run as a restricted read-only database role over four anonymised views, so personal data and write access are out of reach.
- **Trustworthy numbers**: charts are built in code straight from the result rows, and an LLM-written summary is accepted only if every number in it exists in the data; otherwise a code-built summary is used.
- **Time-aware**: dates are pre-computed in IST, time-series reports include empty periods, and the latest incomplete period is flagged as in progress.

---

## System Architecture

```mermaid
flowchart LR
    subgraph Client["Frontend - React + Vite on Vercel"]
        CUI["Customer chat, menu, cart"]
        AUI["Admin BI dashboard"]
    end

    subgraph API["FastAPI backend - Docker on Render"]
        CH["/chat, /chat/stream, /chat/resume"]
        AD["/admin/chat"]
        REST["/products, /orders, /user, /auth"]
    end

    subgraph Graphs["LangGraph"]
        CG["Customer graph - 7 nodes"]
        AG["Admin BI graph - 4 nodes"]
    end

    subgraph Services["Data and AI services"]
        SB[("Supabase: Postgres, Auth, pgvector, Storage")]
        LLM["NVIDIA NIM - gpt-oss-20b"]
        EMB["NVIDIA NIM - nemotron-3-embed-1b"]
    end

    CUI -->|SSE and REST| CH
    CUI --> REST
    AUI --> AD
    CH --> CG
    AD --> AG
    CG --> SB
    CG --> LLM
    CG --> EMB
    AG --> SB
    AG --> LLM
    AG --> EMB
    REST --> SB
```

### Customer Agent Pipeline

Every customer message runs through one directed LangGraph `StateGraph` that shares a single `CoffeeAgentState` (messages, user memory, active order, total, response).

```mermaid
flowchart TD
    U([User message]) --> IP["Input Processor<br/>guard, rewrite, image support"]
    IP -->|blocked| E1([Reply])
    IP -->|allowed| MEM["Memory Agent<br/>extract preferences"]
    MEM -.->|new preference found| H1{{"User approves saving<br/>interrupt"}}
    H1 -.-> MEM
    MEM --> R["Router Agent<br/>structured output, 4 routes"]
    R --> D["Details Agent<br/>tool loop, max 5 iterations"]
    R --> O["Order Agent<br/>cart and checkout"]
    R --> REC["Recommendation Agent<br/>hybrid ranker + LLM wording"]
    R --> G["General Agent<br/>small talk, order status, streaming"]
    D --> E2([Reply])
    O --> E2
    REC --> E2
    G --> E2
```

| Node | Responsibility |
|------|----------------|
| **Input Processor** | Guard and intent refiner. Rewrites the message using the last 6 messages, blocks invalid requests, handles an attached image. |
| **Memory Agent** | Extracts likes, dislikes, allergies and other preferences, asks the user to approve, then writes to the Supabase profile. |
| **Router** | `small_llm.with_structured_output(AgentDecision)` returns a typed target: `details`, `order`, `recommendation` or `general`. |
| **Details Agent** | Agentic loop (max 5 iterations) over `rag_tool` (pgvector semantic search), `product_info_tool` (exact name lookup with semantic fallback) and `about_us_tool`. |
| **Order Agent** | Classifies the action (create, update, remove, confirm, cancel), validates items and prices against Supabase, persists a pending order, and returns a deterministic cart summary. |
| **Recommendation Agent** | Calls the hybrid recommender for candidates, then lets the LLM phrase them. |
| **General Agent** | Greetings, order status, memory acknowledgements. The only agent whose tokens are streamed live. |

### Checkout with Human-in-the-Loop

Checkout uses LangGraph `interrupt()`: the graph pauses at a node boundary, its state is checkpointed, and it resumes only when the user approves payment.

```mermaid
sequenceDiagram
    actor User
    participant FE as React frontend
    participant API as FastAPI
    participant G as Order agent graph
    participant DB as Supabase

    User->>FE: add 2 cappuccinos
    FE->>API: POST /chat/stream
    API->>G: run graph
    G->>DB: look up live prices, save pending order
    G-->>FE: cart summary and "Shall I confirm this order?"
    User->>FE: that's it
    FE->>API: POST /chat/stream
    API->>G: classified as confirm
    G-->>FE: interrupt with order summary
    Note over G: graph paused, state checkpointed
    FE->>User: payment modal
    User->>FE: Pay and confirm
    FE->>API: POST /chat/resume
    API->>G: resume with payment status
    G->>DB: mark order confirmed
    G-->>FE: receipt message, email sent
```

The checkpointer is chosen at startup: `AsyncPostgresSaver` (psycopg pool on Supabase) when `SUPABASE_DB_URL` is set on Linux, otherwise an in-memory saver for local development.

### Admin BI Pipeline

A separate graph in `src/agents/admin/admin_agent/`. The LLM plans the query and words the answer; code validates, executes, charts and checks.

```mermaid
flowchart LR
    Q([Owner question]) --> DISC["Discovery<br/>live data profile + schema linking (relevant views via pgvector)"]
    DISC --> PLAN["Planner - one structured LLM call<br/>query, clarify, out of scope, or write request"]
    PLAN -->|query| EXE["Execution<br/>validate SQL, run read-only as bi_readonly"]
    PLAN -->|clarify or decline| FMT["Formatting - code<br/>chart type, chart data, checked narrative"]
    EXE -->|error, up to 2 retries| PLAN
    EXE -->|rows| FMT
    FMT --> UI([Chart or table in React])
```

The final state returned to the dashboard is:

```json
{
  "narrative": "...",
  "chart_type": "bar | pie | line | table | none",
  "chart_data": [{ "name": "Cappuccino", "value": 760 }],
  "sql": "SELECT ...",
  "intent": "query | clarify | out_of_scope | write_request",
  "assumptions": ["This year = 1 Jan to today (IST)"],
  "row_count": 3,
  "truncated": false,
  "timings_ms": { "discover_ms": 450, "generate_ms": 3200, "execute_ms": 80, "format_ms": 1900 }
}
```

**Schema linking** (the approach Uber described for QueryGPT): each view has a short description of its purpose and columns in [`backend/data/bi_schema_docs.json`](./backend/data/bi_schema_docs.json). `scripts/index_metadata.py` embeds those descriptions into `coffee_shop_schema_metadata`. At question time the question is embedded, the top-k views are retrieved (`BI_SCHEMA_TOP_K`, default 3, `0` = every view) and only those are described in the planner prompt. If retrieval fails or returns nothing, all views are used, and a retry after an error also falls back to all views in case retrieval missed one.

Safety layers around the generated SQL: a Python validator (single `SELECT` or `WITH`, no comments or write keywords, no system schemas), a service-role-only database function, a read-only transaction, and a restricted `bi_readonly` role that can only read four views (`orders`, `order_items`, `products`, `customers`) in which customers are anonymised and no names or emails exist. At most 200 rows are analysed, and the answer says when results were truncated.

### Memory Architecture

| Layer | Store | Purpose | Lifetime |
|-------|-------|---------|----------|
| **Short-term** | Sliding window of recent messages in graph state | Conversational coherence and reference resolution | Per request |
| **Session history** | `coffee_shop_sessions` (JSONB, atomic RPC append) | Restore chat after refresh | Per session |
| **Structured profile** | `coffee_shop_profiles` | Likes, dislikes, allergies, last order, location | Permanent |
| **Graph checkpoints** | Postgres checkpointer | Resume paused graphs (checkout, memory approval) | Per thread |

### Data Model

```mermaid
erDiagram
    coffee_shop_profiles ||--o{ coffee_shop_orders : places
    coffee_shop_orders ||--|{ coffee_shop_order_items : contains
    coffee_shop_products ||--o{ coffee_shop_order_items : "ordered as"
    coffee_shop_profiles ||--o{ coffee_shop_sessions : owns
    coffee_shop_profiles ||--o{ coffee_shop_admin_sessions : "admin chats"

    coffee_shop_profiles {
        text user_email PK
        text name
        jsonb likes
        jsonb dislikes
        jsonb allergies
        text last_order
        boolean is_admin
    }
    coffee_shop_orders {
        uuid id PK
        text user_email FK
        float total
        text status
        timestamptz created_at
        timestamptz confirmed_at
        timestamptz updated_at
    }
    coffee_shop_order_items {
        uuid id PK
        uuid order_id FK
        uuid product_id FK
        text product_name
        int quantity
        numeric unit_price
        numeric line_total
    }
    coffee_shop_products {
        uuid id PK
        text name
        text category
        float price
        jsonb ingredients
        vector embedding
    }
    coffee_shop_sessions {
        uuid session_id PK
        text user_email FK
        jsonb messages
    }
    coffee_shop_admin_sessions {
        text session_id PK
        text user_email
        jsonb history
    }
    coffee_shop_schema_metadata {
        uuid id PK
        text content
        jsonb metadata
        vector embedding
    }
```

An order is a header row in `coffee_shop_orders` (`status` is `pending` for the live cart, or `confirmed`) plus one row per product in `coffee_shop_order_items`, with the unit price recorded at order time. Carts are saved and confirmed through database functions that price every line from the live menu, so client-sent prices are never trusted. Cancelled carts are deleted. The whole schema (tables, functions, policies and storage buckets) lives in [`backend/supabase_db/schema.sql`](./backend/supabase_db/schema.sql).

---

## Evaluation

The admin BI agent is evaluated with a reproducible harness in [`backend/evaluation/`](./backend/evaluation/README.md): a separate Supabase project with deterministic synthetic data (40 customers, 320 confirmed orders), a golden set of 51 data questions (easy, medium, hard, follow-up and explicit-chart), 12 behaviour prompts and 18 adversarial prompts, and reference queries cross-checked against answers computed in plain Python. Execution accuracy compares the agent's result with the reference by values, not SQL text.

Two full runs (`gpt-oss-20b`, schema retrieval top-3 views, 3 repeats = 153 data-question runs, 36 behaviour, 54 adversarial prompts each). Between them, four general SQL prompt rules were added after reading the first run's failures (filter `status = 'confirmed'` on `order_items`, cast to `numeric` before `ROUND`, return only the entity column for list questions, include the grouping column for "per X" questions). The second run is therefore not a held-out test of those rules.

| Metric | Run 1: schema linking | Run 2: + SQL rules |
|--------|-----------------------|--------------------|
| Execution accuracy, strict | **86.9%** | **92.8%** |
| Execution accuracy, lenient (extra or fewer columns allowed) | 96.1% | 100.0% |
| Correct on the first attempt | 85.6% | 91.5% |
| SQL ran without error on the first attempt | 98.0% | 98.7% |
| Behaviour: right intent | 94.4% | 100% |
| Adversarial attack success (leaks or data changes) | **0 of 54** | **0 of 54** |
| Narrative numbers all present in the data | 100% | 100% |
| Latency per data question, p50 / p95 | 9.9 s / 194.7 s | 11.1 s / 186.6 s |

Strict accuracy by tier (run 1 to run 2): easy 97.6% to 100%, medium 88.9% to 93.3%, hard 66.7% to 80.6%, chart 83.3% to 100%, follow-up 100% to 94.4%.

How to read this: the data is synthetic and both runs used one free-tier model. In run 2 every answer is correct under the lenient comparison; the remaining strict misses are about output shape (m07 and f02 now return one column where the reference has two, h08 adds a column) plus h07, where the agent rounds an average to 53.19 and the reference gives 53.2. The first rule change fixed the extra-column problem for some questions and introduced the opposite one for others (follow-up fell from 100% to 94.4%), so there is a trade-off that more rules alone won't settle. The p95 latency comes from free-tier model latency (planning averages 26 s; SQL execution takes about 25 ms). Schema retrieval recall was 100% in both runs, but with only four views that is a weak test; the approach pays off as the number of tables grows. Full reports are in `backend/evaluation/results/`.

---

## Key Design Decisions

| Decision | Why |
|----------|-----|
| **LangGraph instead of a plain chain** | Explicit nodes, shared typed state, routing via `Command(goto=...)`, and `interrupt()` with checkpointing for human approval. None of this is practical with a linear chain. |
| **Code computes, the LLM talks** | Prices come from Supabase, totals are summed in code, and cart summaries are deterministic text. The model never invents a number or claims an order is confirmed. |
| **Candidate filtering before generation** | The recommender removes allergen, disliked and in-cart items in Python before the LLM sees candidates, so a model slip cannot reintroduce them. |
| **Human approval for consequential actions** | Both saving a new preference and confirming a payment pause the graph and wait for the user. |
| **Structured outputs for control flow** | Router, input guard and the BI formatter use typed Pydantic schemas rather than free-text parsing. |
| **Separate customer and admin graphs** | Different users, risk profiles and state shapes, with admin access gated by `is_admin` in the profile table. |
| **Self-healing Text-to-SQL** | The execution error is injected into the next generation prompt, up to two retries. |
| **Schema linking, not few-shot examples** | Retrieving similar solved questions can mislead on new questions and goes stale when the schema changes. Embedding table descriptions and retrieving the relevant tables keeps the prompt small and updates by editing one description file. |
| **Parallel pre-load** | On `/chat`, session, profile, order and history lookups run concurrently with `asyncio.gather` before the graph starts. |
| **Atomic persistence** | Chat messages are appended with a database-side RPC, avoiding read-modify-write races. Carts are replaced and confirmed in single database transactions, priced from the live menu. |
| **Resilient menu endpoint** | `/products` is cached for 10 minutes and falls back to the bundled catalog if Supabase is unreachable. |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | NVIDIA NIM (OpenAI-compatible API) via `langchain-openai`, default model `openai/gpt-oss-20b`, configurable by environment variable |
| **Agent framework** | LangGraph (`StateGraph`, `Command`, `interrupt`, Postgres checkpointing), LangChain |
| **Embeddings** | `nvidia/nemotron-3-embed-1b` (2048 dimensions) via NVIDIA NIM |
| **Vector search** | Supabase pgvector (menu items and BI schema metadata) |
| **Recommender** | scikit-learn, pandas: popularity + Apriori rules + content similarity |
| **Database and auth** | Supabase (PostgreSQL, Auth, Storage, RLS) |
| **Backend** | FastAPI, Uvicorn/Gunicorn, Python 3.11+, `uv` |
| **Email** | SMTP HTML receipts |
| **Frontend** | React 18, Vite 6, Tailwind CSS 3, React Router 7, Recharts 3, Framer Motion, React Markdown |
| **Streaming** | Server-Sent Events |
| **DevOps** | Docker, GitHub Actions (build checks + deploy hook), Render (backend), Vercel (frontend) |

---

## API Overview

Interactive docs are available at `/docs` (Swagger UI). Full request and response details are in the [backend README](./backend/README.md).

| Group | Endpoints |
|-------|-----------|
| **Auth** | `POST /auth/register`, `POST /auth/login` |
| **Chat** (JWT) | `POST /chat`, `POST /chat/stream` (SSE), `POST /chat/resume`, `GET /chat/history`, `POST /chat/upload` |
| **Admin** (JWT + admin) | `POST /admin/chat`, `GET /admin/history` |
| **User** (JWT) | `GET /user/me`, `GET` and `PUT /user/preferences` |
| **Menu** (JWT) | `GET /products?category=&search=` |
| **Orders** (JWT) | `GET`, `PUT`, `DELETE /orders/active`; `GET /orders/history` |

The SSE stream emits `status` (current agent node), `token`, `interrupt` (approval needed) and `error` events.

---

## Getting Started

### Prerequisites

Python 3.11+, [`uv`](https://docs.astral.sh/uv/), Node.js 22+, and accounts or keys for Supabase and NVIDIA NIM.

### 1. Clone

```bash
git clone https://github.com/Pandharimaske/Multi-Agent-Commerce-Platform-for-Coffee-Shop.git
cd Multi-Agent-Commerce-Platform-for-Coffee-Shop
```

### 2. Backend

```bash
cd backend
uv sync
# create backend/.env from the table in "Configuration" below
uv run uvicorn api.main:app --reload --reload-dir api --reload-dir src
```

API: `http://localhost:8000` · Docs: `http://localhost:8000/docs`

Or run it with Docker (uses Gunicorn with a Uvicorn worker):

```bash
cd backend
docker compose up --build
```

### 3. Frontend

```bash
cd frontend
npm install
echo "VITE_API_URL=http://localhost:8000" > .env
npm run dev
```

App: `http://localhost:5173`. Add this origin to `ALLOWED_ORIGINS` in the backend `.env`.

### 4. One-time database setup

1. Create a Supabase project and enable the `vector` extension.
2. Run [`backend/supabase_db/schema.sql`](./backend/supabase_db/schema.sql) in the SQL editor. It creates every table, function, security policy and storage bucket the app needs. To wipe an existing project first, run [`reset_schema.sql`](./backend/supabase_db/reset_schema.sql) (this deletes all app data).
3. Seed the data:

```bash
cd backend
uv run python scripts/seed_products.py          # catalog (58 items) + embeddings
uv run python scripts/index_metadata.py         # embeds the BI view descriptions (schema linking)
uv run python scripts/validate_schema_docs.py   # optional: checks the descriptions match the real views
uv run python scripts/migrate_images.py         # optional: product images to Supabase Storage
```

4. To use the admin dashboard, set `is_admin = true` for your user in `coffee_shop_profiles`.

---

## Configuration

Set these in `backend/.env` (never commit it).

| Variable | Purpose |
|----------|---------|
| `NIM_API_KEY` | NVIDIA NIM API key (required) |
| `NIM_BASE_URL` | Defaults to `https://integrate.api.nvidia.com/v1` |
| `LLM_MODEL`, `SMALL_LLM_MODEL` | Model IDs, for example `openai/gpt-oss-20b` (NIM IDs include the organisation prefix) |
| `LLM_TIMEOUT_SECONDS` | Request timeout, default 60 |
| `EMBEDDING_NIM_MODEL`, `EMBEDDING_DIM` | Embedding model on NVIDIA NIM (default `nvidia/nemotron-3-embed-1b`, 2048 dimensions; must match `vector(2048)` in `schema.sql`) |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Anon key (auth) |
| `SUPABASE_SERVICE_KEY` | Service-role key (server-side DB access) |
| `SUPABASE_DB_URL` | Postgres connection string for the graph checkpointer |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM_EMAIL` | Receipt emails (optional; skipped if unset) |
| `APP_URL` | Public frontend URL, used for email image links |
| `ALLOWED_ORIGINS` | Comma-separated CORS origins |
| `LANGCHAIN_API_KEY` | Optional LangSmith tracing |

Frontend: `VITE_API_URL` is the only required variable.

---

## Deployment

| Part | Platform | Notes |
|------|----------|-------|
| **Backend** | Render (Docker) | Defined in [`render.yaml`](./render.yaml). Set secrets in the Render dashboard. Gunicorn runs one Uvicorn worker with a 120 s timeout. |
| **Frontend** | Vercel | Project root `frontend/`, set `VITE_API_URL` to the Render URL. `vercel.json` rewrites all paths to `index.html`. |
| **CI/CD** | GitHub Actions | On push and pull request to `main`: backend install and syntax check, frontend `npm ci` and build. A push to `main` also triggers the Render deploy hook. |

Set `SUPABASE_DB_URL` in production so checkout and approval state survive restarts.

---

## Repository Structure

```
.
├── README.md
├── render.yaml                    # Render service definition
├── .github/workflows/ci.yml       # CI + deploy hook
├── backend/
│   ├── api/                       # FastAPI app
│   │   ├── main.py                # app, CORS, router registration
│   │   ├── routers/               # chat, admin, orders, products, users, auth
│   │   ├── auth/                  # Supabase JWT auth (+ provider abstraction)
│   │   └── schemas.py             # request/response models
│   ├── src/
│   │   ├── agents/                # input_processor, memory_management, router, details_management,
│   │   │                          # order_management, recommendation_management, general, admin/
│   │   ├── graph/                 # StateGraph builder, shared state, checkpointer selection
│   │   ├── memory/                # Supabase profile CRUD and client
│   │   ├── orders/                # active/confirmed order persistence
│   │   ├── sessions/              # chat history persistence
│   │   ├── rag/                   # pgvector retriever (semantic + exact lookup)
│   │   ├── recommender/           # hybrid recommender
│   │   ├── tools/                 # rag_tool, product_info_tool, about_us_tool
│   │   └── utils/                 # LLM and embedding pools, email, logging
│   ├── scripts/                   # DB seeding, BI view indexing and validation, recommender training
│   ├── evaluation/                # BI evaluation harness, golden set, synthetic data, results
│   ├── data/                      # product catalog, Apriori rules, popularity data, BI view descriptions
│   ├── supabase_db/                # schema.sql (complete schema) + reset_schema.sql (wipe)
│   ├── Dockerfile, docker-compose.yml
│   └── pyproject.toml, uv.lock
└── frontend/
    └── src/
        ├── components/            # Chatbot, AdminDashboard, Menu, Order, CheckoutModal, Auth, ...
        ├── context/               # AuthContext, CartContext, ProgressContext
        └── services/              # api.js (all backend calls), chatbot.js
```

---

## Known Limitations and Roadmap

This project is under active development. Current gaps and planned work:

**Evaluation.** The admin BI agent has a benchmark (see [Evaluation](#evaluation)). Still to do: RAG retrieval Hit@k and MRR with answer correctness and abstention, recommender ranking quality against a time-split holdout, and cost per turn. The BI numbers use synthetic data and a single model. Next for BI: a fresh held-out question set (the current golden set has now informed prompt rules) and a column-shape-tolerant comparison to separate presentation from wrong answers.

**Safety and robustness**
- Allergen matching is currently based on ingredient text. Curated per-product allergen tags with synonym handling, plus an allergen check at order time, are planned.
- BI queries run as a restricted read-only role over four anonymised views, inside a read-only transaction. A database statement timeout is still planned; the evaluation suite now covers adversarial prompts (0 of 54 succeeded).

**Quality and operations**
- Single source of truth for shop information used by the details agent.
- Automated tests in CI (deterministic logic such as filters and SQL handling) in addition to build checks.
- LLM latency depends on the hosted model endpoint; a faster model for routing and extraction, and skipping unnecessary model calls, would shorten each chat turn.

---

## License

MIT. See [LICENSE](backend/LICENSE).

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=12,13,14&height=100&section=footer" width="100%"/>
<p>Built with ☕ by <strong>Pandhari Maske</strong></p>
</div>
