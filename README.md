# DuckLake Tutorial with marimo

A comprehensive tutorial for exploring DuckDB's DuckLake capabilities using a local PostgreSQL catalog and marimo notebooks.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- [uv](https://docs.astral.sh/uv/) - Python package manager
- Python 3.9+
- Git

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd ducklake
```

### 2. Install Dependencies
```bash
uv sync
```

### 3. Start Everything (One Command!)
```bash
uv run task start
```

This will:
- Start PostgreSQL container
- Wait for database initialization  
- Launch marimo notebook

**Alternative: Step by step**
```bash
# Start PostgreSQL only
uv run task docker-up

# Launch notebook (in separate terminal)
uv run task notebook
```

The notebook will open in your browser at `http://localhost:2718`

## What's Included

- **ducklake_tutorial.py** - Interactive marimo notebook with complete tutorial
- **docker-compose.yml** - PostgreSQL setup for DuckLake catalog
- **init.sql** - Database initialization script
- **PLAN.md** - Detailed implementation plan
- **pyproject.toml** - Project configuration and dependencies

## Tutorial Phases

### Phase 1: Foundation Setup
- Docker PostgreSQL configuration
- DuckDB extensions installation
- DuckLake initialization with PostgreSQL catalog

### Phase 2: Core Operations
- Table creation and data ingestion
- Basic queries and joins
- ACID transaction demonstrations

### Phase 3: Advanced Features
- Time travel and snapshots
- Schema evolution
- Multi-user access patterns

### Phase 4: Real-World Patterns
- Performance analysis
- Data maintenance operations
- Best practices and optimization

## Key Features Demonstrated

✅ **Local Development** - No cloud dependencies  
✅ **PostgreSQL Catalog** - Familiar SQL database for metadata  
✅ **ACID Transactions** - Full consistency guarantees  
✅ **Time Travel** - Historical data access via snapshots  
✅ **Schema Evolution** - Safe schema changes  
✅ **Interactive Learning** - Reactive marimo notebook environment  

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   marimo        │    │    DuckDB        │    │   PostgreSQL    │
│   Notebook      │◄──►│    Engine        │◄──►│    Catalog      │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Local Parquet  │
                       │     Files       │
                       │ (./ducklake_data)│
                       └─────────────────┘
```

## Troubleshooting

### PostgreSQL Connection Issues
```bash
# Check container status
uv run task docker-status

# View logs
uv run task docker-logs

# Restart container
uv run task docker-down && uv run task docker-up
```

### DuckDB Extension Issues
```python
# In Python/marimo
import duckdb
conn = duckdb.connect()
conn.execute("INSTALL ducklake")
conn.execute("INSTALL postgres") 
conn.execute("LOAD ducklake")
conn.execute("LOAD postgres")
```

### Port Conflicts
If port 5432 is in use, modify `docker-compose.yml`:
```yaml
ports:
  - "5433:5432"  # Use different host port
```

## Available Tasks

All tasks are managed via taskipy in `pyproject.toml`:

```bash
# Start PostgreSQL container
uv run task docker-up

# Stop PostgreSQL container  
uv run task docker-down

# Start marimo notebook
uv run task notebook

# Complete setup: start docker + marimo
uv run task start

# Check PostgreSQL status
uv run task docker-status

# View PostgreSQL logs
uv run task docker-logs

# Reset everything (stop containers, remove volumes)
uv run task reset

# Install with dev dependencies
uv run task dev-install

# Reset DuckLake data only (keep containers running)
uv run task reset-lake

# Complete clean reset (containers + data)
uv run task clean-reset
```

## Cleanup

Stop and remove containers:
```bash
uv run task docker-down

# Remove volumes (optional)
uv run task reset
```

## Resources

- [DuckLake Documentation](https://duckdb.org/docs/stable/core_extensions/ducklake.html)
- [marimo Documentation](https://docs.marimo.io/)
- [DuckDB SQL Reference](https://duckdb.org/docs/stable/sql/introduction.html)
- [uv Documentation](https://docs.astral.sh/uv/)
- [taskipy Documentation](https://github.com/taskipy/taskipy)

## License

MIT License - see LICENSE file for details.