#!/bin/bash

# DuckLake Complete Installer
# Single-file installer for DuckLake with PostgreSQL catalog and MinIO storage
# Run with: curl -sSL https://your-server/ducklake-installer.sh | bash

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Get user configuration
get_user_config() {
    print_status "DuckLake Configuration Setup"
    echo
    
    # Get installation directory
    read -p "Enter installation directory [./ducklake]: " INSTALL_DIR
    INSTALL_DIR=${INSTALL_DIR:-./ducklake}
    
    # Get instance name for this DuckLake setup
    read -p "Enter instance name [default]: " INSTANCE_NAME
    INSTANCE_NAME=${INSTANCE_NAME:-default}
    
    # Create directory if it doesn't exist
    if [ ! -d "$INSTALL_DIR" ]; then
        mkdir -p "$INSTALL_DIR"
        print_status "Created directory: $INSTALL_DIR"
    fi
    
    # Convert to absolute path and change to it
    INSTALL_DIR=$(realpath "$INSTALL_DIR")
    cd "$INSTALL_DIR"
    print_status "Using installation directory: $INSTALL_DIR"
    echo
    
    # Get bucket name
    read -p "Enter S3 bucket name [ducklake-bucket]: " BUCKET_NAME
    BUCKET_NAME=${BUCKET_NAME:-ducklake-bucket}
    
    # Get data path within bucket
    read -p "Enter data path within bucket [data]: " DATA_PATH
    DATA_PATH=${DATA_PATH:-data}
    
    # Get database name
    read -p "Enter catalog database name [ducklake_catalog]: " DB_NAME
    DB_NAME=${DB_NAME:-ducklake_catalog}
    
    # Get username
    read -p "Enter database username [ducklake]: " DB_USER
    DB_USER=${DB_USER:-ducklake}
    
    # Get password
    read -s -p "Enter database password [ducklake123]: " DB_PASS
    echo
    DB_PASS=${DB_PASS:-ducklake123}
    
    # Get port numbers for this instance
    read -p "Enter PostgreSQL port [5432]: " POSTGRES_PORT
    POSTGRES_PORT=${POSTGRES_PORT:-5432}
    
    read -p "Enter MinIO port [9000]: " MINIO_PORT
    MINIO_PORT=${MINIO_PORT:-9000}
    
    read -p "Enter MinIO console port [9001]: " MINIO_CONSOLE_PORT
    MINIO_CONSOLE_PORT=${MINIO_CONSOLE_PORT:-9001}
    
    # Export for use in other functions
    export BUCKET_NAME DATA_PATH DB_NAME DB_USER DB_PASS INSTALL_DIR INSTANCE_NAME POSTGRES_PORT MINIO_PORT MINIO_CONSOLE_PORT
    
    echo
    print_success "Configuration saved:"
    echo "  Installation: $INSTALL_DIR"
    echo "  Bucket: s3://$BUCKET_NAME/$DATA_PATH"
    echo "  Database: $DB_NAME"
    echo "  User: $DB_USER"
    echo
}

# Get server IP for remote connections
get_server_ip() {
    # Try multiple methods to get external IPv4 IP
    if command_exists curl; then
        curl -s -4 ifconfig.me 2>/dev/null || \
        curl -s -4 ipinfo.io/ip 2>/dev/null || \
        curl -s -4 icanhazip.com 2>/dev/null || \
        hostname -I | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -1
    else
        hostname -I | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -1
    fi
}

# Create embedded files function
create_files() {
    print_status "Creating configuration files..."
    
    # Create pyproject.toml
    cat > pyproject.toml << EOF
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "ducklake-server"
version = "0.1.0"
description = "DuckLake server with PostgreSQL catalog and MinIO storage"
requires-python = ">=3.10"

dependencies = [
    "duckdb>=1.3.0",
    "psycopg2-binary>=2.9.0",
    "taskipy>=1.12.0",
    "click>=8.0.0",
    "rich>=13.0.0",
]

[tool.hatch.build.targets.wheel]
packages = ["ducklake_server"]

[tool.taskipy.tasks]
# Core service management
start-postgres = "podman run -d --name ducklake-postgres-${INSTANCE_NAME} --replace -e POSTGRES_DB=${DB_NAME} -e POSTGRES_USER=${DB_USER} -e POSTGRES_PASSWORD=${DB_PASS} -e POSTGRES_HOST_AUTH_METHOD=trust -p ${POSTGRES_PORT}:5432 -v postgres_data_${INSTANCE_NAME}:/var/lib/postgresql/data -v $(pwd)/init.sql:/docker-entrypoint-initdb.d/init.sql docker.io/library/postgres:15"
start-minio = "podman run -d --name ducklake-minio-${INSTANCE_NAME} --replace -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin -p ${MINIO_PORT}:9000 -p ${MINIO_CONSOLE_PORT}:9001 -v minio_data_${INSTANCE_NAME}:/data quay.io/minio/minio:latest server /data --console-address :9001"
create-bucket = "sleep 5 && podman exec ducklake-minio-${INSTANCE_NAME} mc alias set local http://localhost:9000 minioadmin minioadmin && podman exec ducklake-minio-${INSTANCE_NAME} mc mb local/${BUCKET_NAME} 2>/dev/null || true"
start = "task start-postgres && task start-minio && task create-bucket"

stop-postgres = "podman stop ducklake-postgres-${INSTANCE_NAME} 2>/dev/null || true && podman rm ducklake-postgres-${INSTANCE_NAME} 2>/dev/null || true"
stop-minio = "podman stop ducklake-minio-${INSTANCE_NAME} 2>/dev/null || true && podman rm ducklake-minio-${INSTANCE_NAME} 2>/dev/null || true"
stop = "task stop-postgres && task stop-minio"

# Status and monitoring
status = "podman ps --filter name=ducklake-*-${INSTANCE_NAME}"
logs-postgres = "podman logs -f ducklake-postgres-${INSTANCE_NAME}"
logs-minio = "podman logs -f ducklake-minio-${INSTANCE_NAME}"

# Data management
clean = "task stop && podman volume rm minio_data_${INSTANCE_NAME} postgres_data_${INSTANCE_NAME} 2>/dev/null || true"
reset = "task clean && task start"
EOF

    # Create init.sql
    cat > init.sql << EOF
-- Initialize the DuckLake catalog database
-- Create user if it doesn't exist
DO \$\$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = '${DB_USER}') THEN
        CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';
    END IF;
END
\$\$;

-- Connect to the target database
\c ${DB_NAME};

-- Grant necessary permissions
GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};
GRANT ALL ON SCHEMA public TO ${DB_USER};
GRANT CREATE ON SCHEMA public TO ${DB_USER};

-- Create extension if available
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Display confirmation
SELECT 'DuckLake catalog database initialized successfully' AS status;
EOF

    # Create .env file
    cat > .env << EOF
# PostgreSQL Configuration
POSTGRES_DB=${DB_NAME}
POSTGRES_USER=${DB_USER}
POSTGRES_PASSWORD=${DB_PASS}
POSTGRES_PORT=5432

# MinIO Configuration
MINIO_USER=minioadmin
MINIO_PASSWORD=minioadmin
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001
EOF

    # Create dummy Python module to satisfy packaging
    mkdir -p ducklake_server
    echo "# DuckLake Server" > ducklake_server/__init__.py

    print_success "Configuration files created"
}

# Check and setup user namespaces for podman
setup_user_namespaces() {
    print_status "Checking user namespace configuration..."
    
    local user=$(whoami)
    local uid=$(id -u)
    local subuid_exists=false
    local subgid_exists=false
    
    # Check if user has subuid/subgid entries
    if [ -f /etc/subuid ] && grep -q "^${user}:" /etc/subuid; then
        subuid_exists=true
    fi
    
    if [ -f /etc/subgid ] && grep -q "^${user}:" /etc/subgid; then
        subgid_exists=true
    fi
    
    if [ "$subuid_exists" = false ] || [ "$subgid_exists" = false ]; then
        print_warning "User namespace configuration incomplete"
        echo "Adding user namespace entries..."
        
        # Try to add entries (may require sudo)
        if command_exists sudo; then
            if [ "$subuid_exists" = false ]; then
                echo "${user}:100000:65536" | sudo tee -a /etc/subuid >/dev/null
            fi
            if [ "$subgid_exists" = false ]; then
                echo "${user}:100000:65536" | sudo tee -a /etc/subgid >/dev/null
            fi
            print_success "User namespace entries added"
        else
            print_error "Cannot configure user namespaces without sudo access"
            echo "Please ask your administrator to add these lines:"
            echo "To /etc/subuid: ${user}:100000:65536"
            echo "To /etc/subgid: ${user}:100000:65536"
            exit 1
        fi
    else
        print_success "User namespaces properly configured"
    fi
}

# Install podman if missing
install_podman() {
    if ! command_exists podman; then
        print_status "Installing Podman..."
        
        # Detect OS and install podman
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            case $ID in
                ubuntu|debian)
                    if command_exists apt-get; then
                        sudo apt-get update
                        sudo apt-get install -y podman
                    else
                        print_error "apt-get not found on Debian/Ubuntu system"
                        exit 1
                    fi
                    ;;
                fedora|centos|rhel)
                    if command_exists dnf; then
                        sudo dnf install -y podman
                    elif command_exists yum; then
                        sudo yum install -y podman
                    else
                        print_error "Package manager not found on RHEL/CentOS/Fedora system"
                        exit 1
                    fi
                    ;;
                *)
                    print_error "Unsupported OS: $ID"
                    echo "Please install Podman manually: https://podman.io/docs/installation"
                    exit 1
                    ;;
            esac
        else
            print_error "Cannot detect OS. Please install Podman manually."
            exit 1
        fi
        
        if command_exists podman; then
            print_success "Podman installed successfully"
        else
            print_error "Failed to install Podman"
            exit 1
        fi
    else
        print_success "Podman found"
    fi
}

echo "=================================================="
echo "         DuckLake Complete Installer"
echo "=================================================="
echo ""

# Get user configuration
get_user_config

# Check prerequisites
print_status "Checking prerequisites..."

# Check if running as root (not recommended for podman)
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root. Podman rootless mode is recommended."
    echo "Consider running as a regular user instead."
fi

# Check Python version
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.10+ required. Found: $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Install podman if needed
install_podman

# Setup user namespaces
setup_user_namespaces

# Check UV package manager
if ! command_exists uv; then
    print_status "UV not found. Installing UV..."
    
    # Create temp directory for download
    TEMP_DIR=$(mktemp -d)
    trap "rm -rf $TEMP_DIR" EXIT
    
    # Download installer script
    if curl -LsSf https://astral.sh/uv/install.sh -o "$TEMP_DIR/install.sh"; then
        print_status "UV installer downloaded, running installation..."
        bash "$TEMP_DIR/install.sh"
        
        # Add to PATH and reload
        export PATH="$HOME/.local/bin:$PATH"
        
        # Source shell profile to persist PATH changes
        if [ -f "$HOME/.bashrc" ]; then
            if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc"; then
                echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
            fi
        fi
        
        if command_exists uv; then
            print_success "UV installed successfully"
        else
            print_error "Failed to install UV - check that $HOME/.local/bin is in your PATH"
            exit 1
        fi
    else
        print_error "Failed to download UV installer"
        exit 1
    fi
else
    print_success "UV found"
fi

# Test podman configuration
print_status "Testing podman configuration..."
if ! podman info >/dev/null 2>&1; then
    print_error "Podman configuration issues detected"
    echo ""
    echo "Common fixes for podman permission errors:"
    echo "1. Run: podman system reset (WARNING: removes all containers/images)"
    echo "2. Restart user session: sudo loginctl terminate-user $(whoami)"
    echo "3. For rootless mode: systemctl --user enable --now podman.socket"
    echo ""
    echo "After fixes, re-run this installer."
    exit 1
fi
print_success "Podman configuration verified"

# Test image pulling permissions
print_status "Testing container image access..."
if ! podman pull --quiet docker.io/library/hello-world:latest >/dev/null 2>&1; then
    print_warning "Container registry access may be limited"
    echo "This could be due to:"
    echo "1. Network restrictions"
    echo "2. Registry authentication required"
    echo "3. Corporate firewall/proxy"
    echo ""
    echo "Continuing anyway - images will be pulled during service start..."
else
    print_success "Container registry access verified"
    # Clean up test image
    podman rmi docker.io/library/hello-world:latest >/dev/null 2>&1 || true
fi

# Use the installation directory from config
DUCKLAKE_DIR="$INSTALL_DIR"

# Create all configuration files
create_files

# Install Python dependencies
print_status "Installing Python dependencies..."
# Force UV to use local directory, not parent venv
unset VIRTUAL_ENV
uv venv
uv sync

print_success "Python dependencies installed"

# Check for port conflicts
print_status "Checking for port conflicts..."
CONFLICTING_CONTAINERS=""

# Check each port for conflicts
for port in $POSTGRES_PORT $MINIO_PORT $MINIO_CONSOLE_PORT; do
    CONTAINER=$(podman ps --format "{{.Names}}" --filter "publish=$port" 2>/dev/null | head -1)
    if [ -n "$CONTAINER" ]; then
        CONFLICTING_CONTAINERS="$CONFLICTING_CONTAINERS $CONTAINER"
    fi
done

if [ -n "$CONFLICTING_CONTAINERS" ]; then
    print_warning "Found containers using required ports:"
    for container in $CONFLICTING_CONTAINERS; do
        PORTS=$(podman ps --format "{{.Ports}}" --filter "name=$container")
        echo "  $container: $PORTS"
    done
    echo ""
    read -p "Stop and remove conflicting containers? [y/N]: " STOP_CONFLICTING
    if [ "$STOP_CONFLICTING" = "y" ] || [ "$STOP_CONFLICTING" = "Y" ]; then
        print_status "Stopping conflicting containers..."
        for container in $CONFLICTING_CONTAINERS; do
            podman stop "$container" 2>/dev/null || true
            podman rm "$container" 2>/dev/null || true
            echo "  Removed: $container"
        done
    else
        print_error "Cannot continue with port conflicts"
        exit 1
    fi
fi

# Start services
print_status "Starting DuckLake services..."
uv run task start

# Wait for services with timeout
check_service() {
    local service_name="$1"
    local max_attempts=60
    local attempt=0
    
    print_status "Waiting for $service_name to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if podman ps --filter name="$service_name" --format "{{.Names}}" 2>/dev/null | grep -q "$service_name"; then
            return 0
        fi
        sleep 1
        attempt=$((attempt + 1))
        if [ $((attempt % 10)) -eq 0 ]; then
            echo -n "."
        fi
    done
    return 1
}

if check_service "ducklake-postgres-${INSTANCE_NAME}"; then
    print_success "PostgreSQL is running"
else
    print_error "PostgreSQL failed to start"
    echo "Check logs with: cd $DUCKLAKE_DIR && uv run task logs-postgres"
    exit 1
fi

if check_service "ducklake-minio-${INSTANCE_NAME}"; then
    print_success "MinIO is running"
else
    print_error "MinIO failed to start"
    echo "Check logs with: cd $DUCKLAKE_DIR && uv run task logs-minio"
    exit 1
fi

# Get server IP for remote connections
SERVER_IP=$(get_server_ip)
if [ -z "$SERVER_IP" ]; then
    SERVER_IP="YOUR_SERVER_IP"
    print_warning "Could not determine server IP automatically"
fi

# Print connection information
echo ""
echo "=================================================="
print_success "DuckLake Server Setup Complete!"
echo "=================================================="
echo ""
echo "Services Status:"
echo "  PostgreSQL Catalog: $SERVER_IP:$POSTGRES_PORT"
echo "  MinIO Object Storage: $SERVER_IP:$MINIO_PORT"
echo "  MinIO Web Console: http://$SERVER_IP:$MINIO_CONSOLE_PORT"
echo ""
echo "Credentials:"
echo "  PostgreSQL: ${DB_USER} / ${DB_PASS}"
echo "  MinIO: minioadmin / minioadmin"
echo ""
echo "Remote Connection Code:"
echo "Copy this into your local Python/DuckDB environment:"
echo ""
echo "=================================================="
cat << EOF
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

# Configure S3 connection to your MinIO server
conn.execute(f"""
    SET s3_region='us-east-1';
    SET s3_access_key_id='minioadmin';
    SET s3_secret_access_key='minioadmin';
    SET s3_endpoint='$SERVER_IP:$MINIO_PORT';
    SET s3_use_ssl=false;
    SET s3_url_style='path';
""")

# Connect to your DuckLake server
conn.execute(f"""
    ATTACH 'ducklake:postgres:dbname=${DB_NAME}
            user=${DB_USER}  
            password=${DB_PASS}
            host=$SERVER_IP
            port=$POSTGRES_PORT' AS remote_ducklake
            (DATA_PATH 's3://${BUCKET_NAME}/${DATA_PATH}');
""")

# Test the connection
print("Connected to remote DuckLake!")

# Example: Create a table and insert data
conn.execute("""
    CREATE TABLE IF NOT EXISTS remote_ducklake.test_table (
        id INTEGER,
        name VARCHAR,
        timestamp TIMESTAMP
    );
""")

conn.execute("""
    INSERT INTO remote_ducklake.test_table (id, name, timestamp) 
    VALUES (1, 'Remote Test', CURRENT_TIMESTAMP), (2, 'DuckLake Demo', CURRENT_TIMESTAMP);
""")

# Query the data
result = conn.execute("SELECT * FROM remote_ducklake.test_table").fetchall()
print("Data:", result)
EOF
echo "=================================================="
echo ""
echo "Server Management Commands:"
echo "  Status:    cd $DUCKLAKE_DIR && uv run task status"
echo "  Stop:      cd $DUCKLAKE_DIR && uv run task stop"
echo "  Restart:   cd $DUCKLAKE_DIR && uv run task start"
echo "  Logs:      cd $DUCKLAKE_DIR && uv run task logs-postgres"
echo "  Cleanup:   cd $DUCKLAKE_DIR && uv run task clean"
echo ""
echo "Web Interfaces:"
echo "  MinIO Console: http://$SERVER_IP:$MINIO_CONSOLE_PORT (minioadmin/minioadmin)"
echo ""
echo "Installation Directory: $DUCKLAKE_DIR"
echo ""
echo "=================================================="
print_success "Ready for remote DuckLake connections!"
echo "=================================================="