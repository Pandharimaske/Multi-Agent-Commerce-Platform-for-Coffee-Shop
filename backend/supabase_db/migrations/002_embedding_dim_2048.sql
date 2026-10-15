-- ============================================================
-- Move the vector columns from 768 to 2048 dimensions
-- (NVIDIA nvidia/nemotron-3-embed-1b, native size 2048)
-- ============================================================
-- Run ONCE in the Supabase SQL editor of an EXISTING project (your live one, and any
-- evaluation project that already has the old schema). A fresh install just runs schema.sql,
-- which already uses 2048.
--
-- Existing embeddings are DISCARDED: vectors from different models cannot be compared, and a
-- column cannot change its dimension while it holds data. Regenerate them straight afterwards:
--
--   cd backend
--   uv run python scripts/seed_products.py       # menu embeddings (upserts by name, keeps prices/orders)
--   uv run python scripts/index_metadata.py      # BI example embeddings
--
-- Until those two scripts have run, semantic menu search and BI example retrieval return nothing.
-- No orders, profiles, chat history or products are deleted by this file.
-- ============================================================

begin;

-- 1. Menu embeddings
alter table coffee_shop_products drop column if exists embedding;
alter table coffee_shop_products add column embedding vector(2048);

-- 2. BI example embeddings (index_metadata.py re-inserts every row, so clearing the table is safe)
delete from coffee_shop_schema_metadata;
alter table coffee_shop_schema_metadata drop column if exists embedding;
alter table coffee_shop_schema_metadata add column embedding vector(2048);

-- 3. Search functions (same bodies as schema.sql, new vector size)
create or replace function match_coffee_products(
  query_embedding  vector(2048),
  match_threshold  float,
  match_count      int
)
returns table (
  id           uuid,
  name         text,
  category     text,
  description  text,
  price        float,
  rating       float,
  ingredients  jsonb,
  image_url    text,
  similarity   float
)
language sql
stable
as $$
  select p.id, p.name, p.category, p.description, p.price, p.rating,
         p.ingredients, p.image_url,
         1 - (p.embedding <=> query_embedding) as similarity
  from coffee_shop_products p
  where p.embedding is not null
    and 1 - (p.embedding <=> query_embedding) > match_threshold
  order by p.embedding <=> query_embedding
  limit match_count;
$$;

create or replace function match_schema_metadata(
  query_embedding  vector(2048),
  match_threshold  float,
  match_count      int
)
returns table (
  id          uuid,
  content     text,
  metadata    jsonb,
  similarity  float
)
language sql
stable
as $$
  select m.id, m.content, m.metadata,
         1 - (m.embedding <=> query_embedding) as similarity
  from coffee_shop_schema_metadata m
  where 1 - (m.embedding <=> query_embedding) > match_threshold
  order by m.embedding <=> query_embedding
  limit match_count;
$$;

revoke all on function match_schema_metadata(vector, float, int) from public, anon, authenticated;
grant execute on function match_schema_metadata(vector, float, int) to service_role;

notify pgrst, 'reload schema';

commit;

-- Check afterwards:
--   select count(*) from coffee_shop_products where embedding is not null;   -- 58 after seed_products.py
--   select vector_dims(embedding) from coffee_shop_products limit 1;          -- 2048
