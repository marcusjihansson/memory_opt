-- Create the memorydb database and user
CREATE DATABASE memorydb;
CREATE USER memory_user WITH PASSWORD 'memory_pass';
GRANT ALL PRIVILEGES ON DATABASE memorydb TO memory_user;

-- Connect to memorydb and enable required extensions
\c memorydb;

-- Enable pgvector extension for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- Note: The actual table creation is handled by the Python code
-- This file ensures the database and user exist before the application starts