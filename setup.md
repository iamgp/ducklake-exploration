# DuckLake Setup Guide

This guide provides step-by-step instructions for setting up DuckDB, DuckLake, Podman, PostgreSQL, and MinIO for lakehouse development.

## Prerequisites

- Python 3.10 or higher
- Podman container runtime
- UV package manager

## Installation

```bash
uv sync
```

## Service Setup

```bash
# Start all services
uv run task start

# Check service status
uv run task status
```

**Service Configuration:**
- **PostgreSQL** (Catalog): `ducklake_catalog` database on port 5432
- **MinIO** (Storage): S3-compatible storage on port 9000, console on 9001
- **Access**: MinIO console at http://localhost:9001 (minioadmin/minioadmin)

## DuckDB + DuckLake Configuration

### Extension Installation

DuckDB automatically installs and loads required extensions:

```python
# Install extensions
conn.execute("INSTALL ducklake")
conn.execute("INSTALL postgres")
conn.execute("INSTALL httpfs")

# Load extensions
conn.execute("LOAD ducklake")
conn.execute("LOAD postgres")
conn.execute("LOAD httpfs")
```

### MinIO S3 Configuration

Configure DuckDB to connect to MinIO:

```python
# S3 settings for MinIO
conn.execute("""
    SET s3_region='us-east-1';
    SET s3_access_key_id='minioadmin';
    SET s3_secret_access_key='minioadmin';
    SET s3_endpoint='localhost:9000';
    SET s3_use_ssl=false;
    SET s3_url_style='path';
""")
```

### DuckLake Initialization

Attach DuckLake with PostgreSQL catalog and S3 storage:

```python
# Initialize DuckLake with PostgreSQL catalog and S3 storage
db_name = "ducklake_demo"
s3_data_path = "s3://ducklake-bucket/data"

ducklake_init_query = f"""
ATTACH 'ducklake:postgres:dbname=ducklake_catalog
        user=ducklake
        password=ducklake123
        host=localhost
        port=5432' AS {db_name}
        (DATA_PATH '{s3_data_path}');
"""

conn.execute(ducklake_init_query)
conn.execute(f"USE {db_name}")
```

## Environment Variables

Customize service configuration with environment variables:

```bash
# PostgreSQL
export POSTGRES_DB=ducklake_catalog
export POSTGRES_USER=ducklake
export POSTGRES_PASSWORD=ducklake123
export POSTGRES_PORT=5432

# MinIO
export MINIO_USER=minioadmin
export MINIO_PASSWORD=minioadmin
export MINIO_PORT=9000
export MINIO_CONSOLE_PORT=9001
```

## Development Workflow

### Run Demo

```bash
# Start services and run demo
uv run task start
uv run ducklake-demo

# Or run Python script directly
python ducklake_demo.py
```

### Data Management

```bash
# Reset data only (keep services running)
uv run task reset-data

# Full reset (stop services, clear volumes, restart)
uv run task reset

# Complete cleanup
uv run task clean
```

### Monitoring

```bash
# Check service status
uv run task status

# View combined logs
uv run task logs

# View individual service logs
uv run task logs-postgres
uv run task logs-minio
```

## Troubleshooting

### Service Issues

1. **PostgreSQL Connection Failed**
   ```bash
   # Check if container is running
   podman ps --filter name=ducklake-postgres
   
   # Restart PostgreSQL
   uv run task stop-postgres
   uv run task start-postgres
   ```

2. **MinIO Access Issues**
   ```bash
   # Check container status
   podman ps --filter name=ducklake-minio
   
   # Restart MinIO
   uv run task stop-minio
   uv run task start-minio
   uv run task create-bucket
   ```

3. **Port Conflicts**
   ```bash
   # Use different ports
   export POSTGRES_PORT=5433
   export MINIO_PORT=9001
   export MINIO_CONSOLE_PORT=9002
   ```

### Clean Restart

```bash
# Complete cleanup and restart
uv run task clean
uv run task start
```

## Next Steps

1. Run the demo notebook: `uv run task dev`
2. Explore the tutorial sections for ACID transactions, time travel, and schema evolution
3. Check the performance analysis and Parquet file optimization examples
4. Experiment with your own lakehouse use cases

## Architecture

- **DuckDB**: Analytical query engine with columnar storage
- **DuckLake**: Iceberg-compatible lakehouse extension
- **PostgreSQL**: Catalog metadata storage
- **MinIO**: S3-compatible object storage
- **Podman**: Container runtime for services