#!/bin/bash

# DuckLake Setup Script
# Installs dependencies and sets up the development environment

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

# Check prerequisites
print_status "Checking prerequisites..."

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

# Check UV
if ! command_exists uv; then
    print_status "UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
    
    if command_exists uv; then
        print_success "UV installed successfully"
    else
        print_error "Failed to install UV"
        exit 1
    fi
else
    print_success "UV found"
fi

# Check Podman
if ! command_exists podman; then
    print_error "Podman not found. Please install Podman first."
    echo "On Ubuntu/Debian: sudo apt-get install podman"
    echo "On macOS: brew install podman"
    echo "On other systems: https://podman.io/docs/installation"
    exit 1
else
    print_success "Podman found"
fi

# Set up environment variables
print_status "Setting up environment variables..."

# Create .env file with default values
cat > .env << EOF
# PostgreSQL Configuration
POSTGRES_DB=ducklake_catalog
POSTGRES_USER=ducklake
POSTGRES_PASSWORD=ducklake123
POSTGRES_PORT=5432

# MinIO Configuration
MINIO_USER=minioadmin
MINIO_PASSWORD=minioadmin
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001
EOF

print_success "Environment variables written to .env"

# Install Python dependencies
print_status "Installing Python dependencies..."
uv sync

print_success "Python dependencies installed"

# Start services
print_status "Starting services..."
uv run task start

# Wait for services to be ready
print_status "Waiting for services to be ready..."
sleep 5

# Check if services are running
if uv run task status | grep -q "ducklake-postgres"; then
    print_success "PostgreSQL is running"
else
    print_warning "PostgreSQL might not be running properly"
fi

if uv run task status | grep -q "ducklake-minio"; then
    print_success "MinIO is running"
else
    print_warning "MinIO might not be running properly"
fi

# Print final instructions
echo ""
echo "=================================================="
print_success "DuckLake setup complete!"
echo "=================================================="
echo ""
echo "Services running:"
echo "  - PostgreSQL: localhost:5432 (ducklake/ducklake123)"
echo "  - MinIO API: localhost:9000"
echo "  - MinIO Console: http://localhost:9001 (minioadmin/minioadmin)"
echo ""
echo "Next steps:"
echo "  1. Run the demo: uv run ducklake-demo"
echo "  2. Or run Python script: python ducklake_demo.py"
echo "  3. Check status: uv run task status"
echo "  4. View logs: uv run task logs"
echo ""
echo "Data management:"
echo "  - Reset data: uv run task reset-data"
echo "  - Full reset: uv run task reset"
echo "  - Stop services: uv run task stop"
echo "  - Clean up: uv run task clean"
echo ""
echo "Environment variables loaded from .env file"
echo "=================================================="