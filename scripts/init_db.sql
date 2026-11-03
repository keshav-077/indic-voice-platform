-- VAI-API Database Initialization
-- Run automatically on first docker compose up via init container
-- For production: use alembic migrations

-- API Keys table (for persistent key management)
CREATE TABLE IF NOT EXISTS api_keys (
    id          SERIAL PRIMARY KEY,
    key_hash    VARCHAR(64) UNIQUE NOT NULL,   -- SHA-256 of key + salt
    key_prefix  VARCHAR(8)  NOT NULL,          -- First 8 chars (for display)
    user_id     VARCHAR(128) NOT NULL,
    name        VARCHAR(256),                   -- Human label
    scopes      TEXT[] DEFAULT '{"api"}',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    last_used   TIMESTAMPTZ,
    expires_at  TIMESTAMPTZ,
    is_active   BOOLEAN DEFAULT TRUE,
    note        TEXT
);

CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);

-- Request logs (metadata only — no audio, no transcripts)
-- DPDP §8(3): data minimisation — only operational metadata
CREATE TABLE IF NOT EXISTS request_logs (
    id              BIGSERIAL PRIMARY KEY,
    request_id      VARCHAR(36),
    endpoint        VARCHAR(64) NOT NULL,        -- e.g. 'asr', 'mt'
    lang            VARCHAR(10),                 -- language code
    backend_used    VARCHAR(32),
    response_code   SMALLINT NOT NULL,
    latency_ms      INTEGER,
    caller_prefix   VARCHAR(8),                  -- first 8 chars of API key hash
    created_at      TIMESTAMPTZ DEFAULT NOW()
    -- NEVER store: audio, transcript, translation, user PII
);

CREATE INDEX IF NOT EXISTS idx_request_logs_created ON request_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_request_logs_endpoint ON request_logs(endpoint);

-- Automatic cleanup: delete logs older than 30 days
-- (DPDP §8: data retention limitation)
CREATE OR REPLACE FUNCTION cleanup_old_logs() RETURNS void AS $$
BEGIN
    DELETE FROM request_logs WHERE created_at < NOW() - INTERVAL '30 days';
END;
$$ LANGUAGE plpgsql;

-- Schedule cleanup (requires pg_cron extension in production)
-- SELECT cron.schedule('cleanup-logs', '0 2 * * *', 'SELECT cleanup_old_logs()');

-- Data erasure audit log (for DPDP §12 compliance)
CREATE TABLE IF NOT EXISTS erasure_requests (
    id          SERIAL PRIMARY KEY,
    user_id     VARCHAR(128) NOT NULL,
    requested_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status      VARCHAR(20) DEFAULT 'pending',  -- pending, completed, failed
    requested_by VARCHAR(128)                   -- admin who processed it
);

-- Grievances (DPDP §16)
CREATE TABLE IF NOT EXISTS grievances (
    id          SERIAL PRIMARY KEY,
    reference_id VARCHAR(36) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    contact_email VARCHAR(256),                 -- encrypted in production
    status      VARCHAR(20) DEFAULT 'open',
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    resolution  TEXT
);

-- Insert default admin API key (change immediately in production!)
-- Key: 'dev-admin-key-change-me' hashed with salt 'dev_salt_change_me'
-- DO $$
-- BEGIN
--     INSERT INTO api_keys (key_hash, key_prefix, user_id, name, scopes)
--     VALUES (
--         sha256('dev-admin-key-change-medev_salt_change_me'),
--         'dev-admi',
--         'admin',
--         'Development Admin Key — CHANGE IN PRODUCTION',
--         '{"api", "admin"}'
--     ) ON CONFLICT DO NOTHING;
-- END $$;
