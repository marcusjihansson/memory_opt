-- Enable pgvector extension for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- Note: The actual table creation is handled by the Python code
-- This file ensures the database and user exist before the application starts