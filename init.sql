-- Initialize the DuckLake catalog database
-- This script runs when PostgreSQL container starts for the first time

-- Create the main catalog database (already created via POSTGRES_DB)
-- But ensure we're connected to it
\c ducklake_catalog;

-- Create any additional schemas or configurations if needed
-- DuckLake will create its own tables as needed

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE ducklake_catalog TO ducklake;
GRANT ALL ON SCHEMA public TO ducklake;

-- Display confirmation
SELECT 'DuckLake catalog database initialized successfully' AS status;