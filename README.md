# DuckLake Tutorial

A comprehensive tutorial for exploring DuckDB's DuckLake capabilities using PostgreSQL catalog and MinIO object storage.

## Quick Start

### Automated Setup (Recommended)
```bash
./setup.sh
```

### Manual Setup

#### Prerequisites
- Python 3.10+
- Podman container runtime
- UV package manager

#### 1. Install Dependencies
```bash
uv sync
```

#### 2. Start Services
```bash
uv run task start
```

#### 3. Run Demo
```bash
uv run ducklake-demo
```

## What's Included

- **ducklake_demo.py** - CLI demo script with complete tutorial
- **ducklake_demo.ipynb** - Jupyter notebook version
- **helpers.py** - Utility functions for setup and demos
- **setup.sh** - Automated setup script
- **setup.md** - Detailed setup guide
- **pyproject.toml** - Project configuration and task definitions

## Tutorial Phases

### Phase 1: Foundation Setup
- PostgreSQL catalog configuration
- MinIO object storage setup
- DuckDB extensions installation
- DuckLake initialization

### Phase 2: Core Operations
- Table creation and data ingestion
- Basic queries and joins
- ACID transaction demonstrations

### Phase 3: Advanced Features
- Time travel and snapshots
- Schema evolution
- Performance analysis

### Phase 4: Real-World Patterns
- Data compression and storage optimization
- Parquet file analysis
- Maintenance operations

## Key Features Demonstrated

✅ **Local Development** - No cloud dependencies  
✅ **PostgreSQL Catalog** - Familiar SQL database for metadata  
✅ **MinIO Storage** - S3-compatible object storage  
✅ **ACID Transactions** - Full consistency guarantees  
✅ **Time Travel** - Historical data access via snapshots  
✅ **Schema Evolution** - Safe schema changes  
✅ **Columnar Storage** - Efficient Parquet-based storage  

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Python        │    │    DuckDB        │    │   PostgreSQL    │
│   Demo/Notebook │◄──►│    Engine        │◄──►│    Catalog      │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │      MinIO      │
                       │   S3 Storage    │
                       │ (ducklake-bucket)│
                       └─────────────────┘
```

## Available Tasks

All tasks are managed via taskipy in `pyproject.toml`:

```bash
# Service Management
uv run task start          # Start all services
uv run task stop           # Stop all services
uv run task status         # Check service status

# Individual Services
uv run task start-postgres # Start PostgreSQL only
uv run task start-minio    # Start MinIO only
uv run task create-bucket  # Create MinIO bucket

# Monitoring
uv run task logs           # View combined logs
uv run task logs-postgres  # PostgreSQL logs only
uv run task logs-minio     # MinIO logs only

# Data Management
uv run task reset-data     # Reset data only
uv run task reset          # Full reset
uv run task clean          # Complete cleanup
```

## Service Configuration

**PostgreSQL (Catalog)**
- Database: `ducklake_catalog`
- User: `ducklake`
- Password: `ducklake123`
- Port: `5432`

**MinIO (Storage)**
- User: `minioadmin`
- Password: `minioadmin`
- API Port: `9000`
- Console Port: `9001`
- Console URL: http://localhost:9001

## Environment Variables

Customize configuration in `.env` file:

```bash
# PostgreSQL
POSTGRES_DB=ducklake_catalog
POSTGRES_USER=ducklake
POSTGRES_PASSWORD=ducklake123
POSTGRES_PORT=5432

# MinIO
MINIO_USER=minioadmin
MINIO_PASSWORD=minioadmin
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001
```

## DuckLake Configuration

### Extension Installation
```python
conn.execute("INSTALL ducklake")
conn.execute("INSTALL postgres")
conn.execute("INSTALL httpfs")
conn.execute("LOAD ducklake")
conn.execute("LOAD postgres")
conn.execute("LOAD httpfs")
```

### S3 Configuration for MinIO
```python
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
```python
ducklake_init_query = f"""
ATTACH 'ducklake:postgres:dbname=ducklake_catalog
        user=ducklake
        password=ducklake123
        host=localhost
        port=5432' AS ducklake_demo
        (DATA_PATH 's3://ducklake-bucket/data');
"""
conn.execute(ducklake_init_query)
```

## Troubleshooting

### Service Issues
```bash
# Check container status
uv run task status

# Restart services
uv run task stop && uv run task start

# View logs
uv run task logs
```

### Port Conflicts
Set different ports in `.env` file:
```bash
POSTGRES_PORT=5433
MINIO_PORT=9001
MINIO_CONSOLE_PORT=9002
```

### Complete Reset
```bash
uv run task clean  # Stop services, remove volumes
uv run task start  # Restart everything
```

## Demo Usage

```bash
# Run complete demo
uv run ducklake-demo

# Run without data reset
uv run ducklake-demo --no-reset

# Manual reset only
uv run ducklake-demo reset

# Direct Python execution
python ducklake_demo.py
```

## Resources

- [DuckLake Documentation](https://duckdb.org/docs/stable/core_extensions/ducklake.html)
- [DuckDB SQL Reference](https://duckdb.org/docs/stable/sql/introduction.html)
- [MinIO Documentation](https://docs.min.io/)
- [UV Documentation](https://docs.astral.sh/uv/)
- [Taskipy Documentation](https://github.com/taskipy/taskipy)

## License

MIT License - see LICENSE file for details.