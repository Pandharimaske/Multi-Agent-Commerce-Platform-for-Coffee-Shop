-- ============================================================
-- RESET: deletes ALL Merry's Way app data and database objects
-- ============================================================
-- Run this ONLY when you want to start clean, then run schema.sql.
--
-- Deleted : profiles (incl. is_admin flags), products, orders, order items,
--           chat sessions, admin sessions, BI schema metadata, and every
--           function the app created.
-- Kept    : Supabase Auth users (they can still log in; schema.sql recreates
--           their profile rows), storage files, and LangGraph's own
--           checkpoint tables (created automatically by the backend).
--
-- After running this:  run schema.sql, then
--   uv run python scripts/seed_products.py
--   uv run python scripts/index_metadata.py
--   update coffee_shop_profiles set is_admin = true where user_email = 'you@example.com';
-- ============================================================

-- Drop every overload of the app's functions (also removes triggers that use set_updated_at).
do $$
declare
  r record;
begin
  for r in
    select p.oid::regprocedure::text as sig
    from pg_proc p
    join pg_namespace n on n.oid = p.pronamespace
    where n.nspname = 'public'
      and p.proname in (
        'execute_sql_query',
        'match_schema_metadata',
        'match_coffee_products',
        'append_chat_messages',
        'replace_pending_order',
        'confirm_pending_order',
        'set_updated_at'
      )
  loop
    execute 'drop function if exists ' || r.sig || ' cascade';
  end loop;
end;
$$;

drop table if exists
  coffee_shop_order_items,
  coffee_shop_orders,
  coffee_shop_orders_backup_001,
  coffee_shop_sessions,
  coffee_shop_admin_sessions,
  coffee_shop_schema_metadata,
  coffee_shop_products,
  coffee_shop_profiles
cascade;
