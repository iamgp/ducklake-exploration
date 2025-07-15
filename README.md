# DuckLake Installer & Container Manager

A comprehensive installer and management tool for DuckLake with PostgreSQL catalog and MinIO object storage. Single-file installer that handles everything from initial setup to ongoing container management.

## Quick Start

### Installation
```bash
# Interactive installation
curl -sSL https://your-server/ducklake-installer.sh | bash

# Or download and run locally
wget https://your-server/ducklake-installer.sh
chmod +x ducklake-installer.sh
./ducklake-installer.sh
```

### Container Management
```bash
# Check status
./ducklake-installer.sh --status

# Start/stop services
./ducklake-installer.sh --start
./ducklake-installer.sh --stop

# View logs
./ducklake-installer.sh --logs postgres
```

## Installation Options

### Interactive Mode (Default)
```bash
./ducklake-installer.sh
```
Prompts for all configuration options with sensible defaults.

### Non-Interactive Mode
```bash
./ducklake-installer.sh --non-interactive
```
Uses default values, perfect for automation and CI/CD.

### Configuration File
```bash
# Create config file
cp ducklake.conf.example ducklake.conf
# Edit ducklake.conf with your settings
./ducklake-installer.sh --config ducklake.conf
```

### Dry Run Mode
```bash
./ducklake-installer.sh --dry-run
```
Preview all changes without executing them.

## Container Management Commands

### Status & Monitoring
```bash
# Show all DuckLake containers
./ducklake-installer.sh --status

# Show specific instance
./ducklake-installer.sh --status prod

# View real-time logs
./ducklake-installer.sh --logs postgres
./ducklake-installer.sh --logs minio
```

### Service Control
```bash
# Start all containers
./ducklake-installer.sh --start

# Start specific instance
./ducklake-installer.sh --start dev

# Stop all containers
./ducklake-installer.sh --stop

# Stop specific instance
./ducklake-installer.sh --stop prod

# Restart containers
./ducklake-installer.sh --restart
./ducklake-installer.sh --restart test
```

### Cleanup & Maintenance
```bash
# Remove containers and volumes
./ducklake-installer.sh --clean

# Clean specific instance
./ducklake-installer.sh --clean dev

# Complete uninstall
./ducklake-installer.sh --uninstall
```

## Configuration

### Default Configuration
```bash
INSTALL_DIR=./ducklake
INSTANCE_NAME=default
BUCKET_NAME=ducklake-bucket
DATA_PATH=data
DB_NAME=ducklake_catalog
DB_USER=ducklake
DB_PASS=ducklake123
POSTGRES_PORT=5432
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001
```

### Multiple Instances
Run multiple DuckLake instances with different configurations:

```bash
# Production instance
INSTANCE_NAME=prod POSTGRES_PORT=5432 ./ducklake-installer.sh --non-interactive

# Development instance  
INSTANCE_NAME=dev POSTGRES_PORT=5433 MINIO_PORT=9002 ./ducklake-installer.sh --non-interactive

# Manage instances separately
./ducklake-installer.sh --status prod
./ducklake-installer.sh --restart dev
```

## Prerequisites

The installer automatically handles prerequisites:

- **Python 3.10+** - Auto-detects and provides installation instructions
- **Podman** - Auto-installs on Ubuntu/Debian/RHEL/CentOS/Fedora
- **UV Package Manager** - Auto-downloads and installs
- **User Namespaces** - Auto-configures for rootless podman

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   DuckDB        │    │    PostgreSQL    │    │      MinIO      │
│   Client        │◄──►│    Catalog       │    │   S3 Storage    │
│                 │    │   (Metadata)     │    │   (Data Files)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                                └────────────────────────┘
                                    DuckLake Extension
```

## Service Details

### PostgreSQL Catalog
- **Purpose**: Stores table metadata, schemas, and transaction logs
- **Database**: `ducklake_catalog` 
- **Default Port**: `5432`
- **Credentials**: `ducklake` / `ducklake123`

### MinIO Object Storage
- **Purpose**: Stores actual data files (Parquet format)
- **Default Ports**: `9000` (API), `9001` (Console)
- **Credentials**: `minioadmin` / `minioadmin`
- **Console**: http://localhost:9001

## Client Connection

### Python/DuckDB Setup
```python
import duckdb

# Connect to DuckDB
conn = duckdb.connect()

# Install required extensions
conn.execute("INSTALL ducklake")
conn.execute("INSTALL postgres") 
conn.execute("INSTALL httpfs")
conn.execute("LOAD ducklake")
conn.execute("LOAD postgres")
conn.execute("LOAD httpfs")

# Configure S3 connection to MinIO
conn.execute("""
    SET s3_region='us-east-1';
    SET s3_access_key_id='minioadmin';
    SET s3_secret_access_key='minioadmin';
    SET s3_endpoint='localhost:9000';
    SET s3_use_ssl=false;
    SET s3_url_style='path';
""")

# Connect to DuckLake
conn.execute("""
    ATTACH 'ducklake:postgres:dbname=ducklake_catalog
            user=ducklake  
            password=ducklake123
            host=localhost
            port=5432' AS ducklake_demo
            (DATA_PATH 's3://ducklake-bucket/data');
""")

# Create and query tables
conn.execute("""
    CREATE TABLE ducklake_demo.sales (
        id INTEGER,
        product VARCHAR,
        amount DECIMAL(10,2),
        sale_date DATE
    );
""")
```

### Remote Connections
For remote server installations, replace `localhost` with your server IP:

```python
# Use server IP instead of localhost
conn.execute("""
    SET s3_endpoint='YOUR_SERVER_IP:9000';
""")

conn.execute("""
    ATTACH 'ducklake:postgres:dbname=ducklake_catalog
            user=ducklake  
            password=ducklake123
            host=YOUR_SERVER_IP
            port=5432' AS ducklake_demo
            (DATA_PATH 's3://ducklake-bucket/data');
""")
```

## Features

### Installation Features
- ✅ **Single-file installer** - No dependencies to download
- ✅ **Auto-detection** - Detects OS and installs requirements
- ✅ **Validation** - Validates all inputs and configurations
- ✅ **Rollback** - Automatic cleanup on installation failure
- ✅ **Logging** - Comprehensive logs for debugging

### Container Management
- ✅ **Multi-instance** - Manage multiple DuckLake deployments
- ✅ **Status monitoring** - Detailed container and instance status
- ✅ **Log streaming** - Real-time log viewing with Ctrl+C exit
- ✅ **Bulk operations** - Start/stop/restart all or filtered containers
- ✅ **Clean removal** - Complete cleanup of containers and volumes

### Robustness Features
- ✅ **Retry logic** - Automatic retries for transient failures
- ✅ **Timeout handling** - Network operations with configurable timeouts
- ✅ **Error recovery** - Detailed error messages with fix suggestions
- ✅ **Health checks** - Verifies services are actually ready
- ✅ **Port conflict detection** - Prevents installation conflicts

## Troubleshooting

### Installation Issues
```bash
# Check installation log
cat ./ducklake/ducklake-install.log

# Validate configuration
./ducklake-installer.sh --dry-run --config ducklake.conf

# Test podman setup
podman info
```

### Service Issues
```bash
# Check container status
./ducklake-installer.sh --status

# View service logs
./ducklake-installer.sh --logs postgres
./ducklake-installer.sh --logs minio

# Restart services
./ducklake-installer.sh --restart
```

### Port Conflicts
```bash
# Check what's using ports
ss -tuln | grep :5432
ss -tuln | grep :9000

# Use different ports
POSTGRES_PORT=5433 MINIO_PORT=9002 ./ducklake-installer.sh --non-interactive
```

### Complete Reset
```bash
# Clean everything and reinstall
./ducklake-installer.sh --clean
./ducklake-installer.sh --non-interactive
```

### Network Issues
```bash
# Test registry access
curl -I https://registry-1.docker.io

# Test with proxy
HTTP_PROXY=http://proxy:8080 ./ducklake-installer.sh

# Manual container pull
podman pull docker.io/library/postgres:15
```

## Advanced Usage

### CI/CD Integration
```bash
# Automated deployment
curl -sSL https://your-server/ducklake-installer.sh | bash -s -- --non-interactive

# With custom config
echo "INSTANCE_NAME=ci-test" > ci.conf
echo "POSTGRES_PORT=5433" >> ci.conf
./ducklake-installer.sh --config ci.conf --non-interactive
```

### Docker Compose Alternative
The installer creates equivalent functionality to this docker-compose.yml:

```yaml
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: ducklake_catalog
      POSTGRES_USER: ducklake
      POSTGRES_PASSWORD: ducklake123
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  minio:
    image: quay.io/minio/minio:latest
    command: server /data --console-address :9001
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data
```

### Production Considerations
- Change default passwords in configuration
- Use proper SSL certificates for remote access
- Configure firewall rules for ports
- Set up regular backups of PostgreSQL catalog
- Monitor container resource usage
- Use dedicated storage volumes for production data

## Help & Support

```bash
# Show all available options
./ducklake-installer.sh --help

# Get version information
./ducklake-installer.sh --status
```

