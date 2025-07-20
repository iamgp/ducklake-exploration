# @ducklakekit

A modern, interactive installer and management tool for DuckLake with PostgreSQL catalog and MinIO object storage. Features an enhanced terminal interface powered by [gum](https://github.com/charmbracelet/gum) with automatic fallback to a standard command-line interface.

## Quick Start

### Installation
```bash
# Interactive installation with enhanced interface
./ducklakekit

# Or use the direct gum interface
./ducklakekit-gum

# Non-interactive installation
./ducklakekit --non-interactive
```

### Container Management
```bash
# Check status
./ducklakekit --status

# Start/stop services
./ducklakekit --start
./ducklakekit --stop

# View logs
./ducklakekit --logs postgres
./ducklakekit --logs minio
```

## New Enhanced Interface

@ducklakekit now features a smart launcher that automatically ensures you have the best available interface:

- **Enhanced Interface**: Uses gum for beautiful, interactive menus and forms
- **Auto-Installation**: Automatically downloads and installs gum when possible
- **Gum Required**: Requires gum for the enhanced interactive experience
- **Consistent Styling**: Follows YADL style guidelines with DuckLake's signature yellow theme

### Interface Detection Flow
1. Checks if `gum` is available in PATH
2. If not available, attempts automatic installation
3. If installation succeeds, launches enhanced interface
4. If installation fails, provides helpful installation instructions

## Installation Options

### Interactive Mode (Default)
```bash
./ducklakekit
```
Beautiful interactive interface with guided setup and real-time validation.

### Non-Interactive Mode
```bash
./ducklakekit --non-interactive
```
Uses default values or configuration file, perfect for automation and CI/CD.

### Configuration File
```bash
# Copy example configuration
cp ducklakekit.conf.example ducklakekit.conf
# Edit ducklakekit.conf with your settings
./ducklakekit --config ducklakekit.conf
```

### Dry Run Mode
```bash
./ducklakekit --dry-run
```
Preview all changes without executing them.

## Container Management Commands

### Status & Monitoring
```bash
# Show all DuckLake containers
./ducklakekit --status

# Show specific instance
./ducklakekit --status prod

# View real-time logs
./ducklakekit --logs postgres
./ducklakekit --logs minio
```

### Service Control
```bash
# Start all containers
./ducklakekit --start

# Start specific instance
./ducklakekit --start dev

# Stop all containers
./ducklakekit --stop

# Stop specific instance  
./ducklakekit --stop prod

# Restart containers
./ducklakekit --restart
./ducklakekit --restart test
```

### Cleanup & Maintenance
```bash
# Remove containers and volumes
./ducklakekit --clean

# Clean specific instance
./ducklakekit --clean dev

# Complete uninstall
./ducklakekit --uninstall
```

## Configuration

### Default Configuration
```bash
INSTALL_DIR=./ducklakekit
INSTANCE_NAME=default
BUCKET_NAME=ducklakekit-bucket
DATA_PATH=data
DB_NAME=ducklakekit_catalog
DB_USER=ducklakekit
DB_PASS=ducklakekit123
POSTGRES_PORT=5432
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001
```

### Multiple Instances
Run multiple DuckLake instances with different configurations:

```bash
# Production instance
INSTANCE_NAME=prod POSTGRES_PORT=5432 ./ducklakekit --non-interactive

# Development instance  
INSTANCE_NAME=dev POSTGRES_PORT=5433 MINIO_PORT=9002 ./ducklakekit --non-interactive

# Manage instances separately
./ducklakekit --status prod
./ducklakekit --restart dev
```

## Prerequisites

The installer automatically handles prerequisites:

- **Python 3.10+** - Auto-detects and provides installation instructions
- **Podman** - Auto-installs on Ubuntu/Debian/RHEL/CentOS/Fedora  
- **UV Package Manager** - Auto-downloads and installs
- **User Namespaces** - Auto-configures for rootless podman
- **Gum (Optional)** - Auto-installs for enhanced interface

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
- **Database**: `ducklakekit_catalog` 
- **Default Port**: `5432`
- **Credentials**: `ducklakekit` / `ducklakekit123`

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
    ATTACH 'ducklake:postgres:dbname=ducklakekit_catalog
            user=ducklakekit  
            password=ducklakekit123
            host=localhost
            port=5432' AS ducklakekit_demo
            (DATA_PATH 's3://ducklakekit-bucket/data');
""")

# Create and query tables
conn.execute("""
    CREATE TABLE ducklakekit_demo.sales (
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
    ATTACH 'ducklake:postgres:dbname=ducklakekit_catalog
            user=ducklakekit  
            password=ducklakekit123
            host=YOUR_SERVER_IP
            port=5432' AS ducklakekit_demo
            (DATA_PATH 's3://ducklakekit-bucket/data');
""")
```

## Features

### Enhanced Interface Features
- ✅ **Interactive Menus** - Gum-powered selection and input forms
- ✅ **Auto-Installation** - Automatically installs gum when possible
- ✅ **Smart Fallback** - Seamless fallback to command-line interface
- ✅ **Consistent Styling** - YADL style guide with DuckLake yellow theme
- ✅ **Progress Indicators** - Real-time progress and status updates

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

### Interface Issues
```bash
# Install gum manually for enhanced interface
# Ubuntu/Debian
wget https://github.com/charmbracelet/gum/releases/download/v0.16.2/gum_0.16.2_amd64.deb
sudo dpkg -i gum_0.16.2_amd64.deb

# Use gum interface directly
./ducklakekit-gum

# Check gum installation
command -v gum && echo "Gum available" || echo "Gum not found"
```

### Installation Issues
```bash
# Check installation log
cat ./ducklakekit/ducklakekit-install.log

# Validate configuration
./ducklakekit --dry-run --config ducklakekit.conf

# Test podman setup
podman info
```

### Service Issues
```bash
# Check container status
./ducklakekit --status

# View service logs
./ducklakekit --logs postgres
./ducklakekit --logs minio

# Restart services
./ducklakekit --restart
```

### Port Conflicts
```bash
# Check what's using ports
ss -tuln | grep :5432
ss -tuln | grep :9000

# Use different ports
POSTGRES_PORT=5433 MINIO_PORT=9002 ./ducklakekit --non-interactive
```

### Complete Reset
```bash
# Clean everything and reinstall
./ducklakekit --clean
./ducklakekit --non-interactive
```

## Advanced Usage

### CI/CD Integration
```bash
# Automated deployment (installs gum if needed)
./ducklakekit --non-interactive

# Use gum interface directly if gum is pre-installed
./ducklakekit-gum --non-interactive

# With custom config
echo "INSTANCE_NAME=ci-test" > ci.conf
echo "POSTGRES_PORT=5433" >> ci.conf
./ducklakekit --config ci.conf --non-interactive
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
./ducklakekit --help

# Get version information
./ducklakekit --status

# Install gum for enhanced interface
# Visit: https://github.com/charmbracelet/gum
```

## File Structure

```
ducklakekit/
├── ducklakekit                 # Smart launcher (installs gum if needed)
├── ducklakekit-gum            # Enhanced gum interface
└── ducklakekit.conf.example   # Configuration template
```

The launcher automatically installs gum when needed and provides the enhanced interactive experience. All functionality is now powered by the gum interface for consistency and improved user experience.