#!/bin/bash

# DuckLake Complete Installer
# Single-file installer for DuckLake with PostgreSQL catalog and MinIO storage
# Run with: curl -sSL https://your-server/ducklake-installer.sh | bash
# 
# Usage: ./ducklake-installer.sh [OPTIONS]
# Options:
#   --non-interactive    Run without prompts using defaults or config file
#   --dry-run           Preview changes without executing them
#   --config FILE       Use configuration file instead of prompts
#   --uninstall         Remove DuckLake installation
#   --help              Show this help message

set -e  # Exit on any error

# Global flags
NON_INTERACTIVE=false
DRY_RUN=false
CONFIG_FILE=""
UNINSTALL=false
COMMAND=""
INSTANCE_FILTER=""
LOG_SERVICE=""

# Error handling and logging
LOG_FILE=""
ROLLBACK_ACTIONS=()
TEMP_FILES=()

# Network timeouts (in seconds)
CURL_TIMEOUT=30
CURL_CONNECT_TIMEOUT=10

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local msg="$1"
    echo -e "${BLUE}[INFO]${NC} $msg"
    log_message "INFO" "$msg"
}

print_success() {
    local msg="$1"
    echo -e "${GREEN}[SUCCESS]${NC} $msg"
    log_message "SUCCESS" "$msg"
}

print_warning() {
    local msg="$1"
    echo -e "${YELLOW}[WARNING]${NC} $msg"
    log_message "WARNING" "$msg"
}

print_error() {
    local msg="$1"
    echo -e "${RED}[ERROR]${NC} $msg"
    log_message "ERROR" "$msg"
}

# Logging function
log_message() {
    local level="$1"
    local msg="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    if [ -n "$LOG_FILE" ]; then
        echo "[$timestamp] [$level] $msg" >> "$LOG_FILE"
    fi
}

# Setup logging
setup_logging() {
    if [ "$DRY_RUN" = false ]; then
        LOG_FILE="${INSTALL_DIR:-./ducklake}/ducklake-install.log"
        mkdir -p "$(dirname "$LOG_FILE")"
        echo "DuckLake Installation Log - $(date)" > "$LOG_FILE"
        log_message "INFO" "Installation started with PID: $$"
        log_message "INFO" "Command line: $0 $*"
    fi
}

# Cleanup function for exit
cleanup_on_exit() {
    local exit_code=$?
    
    if [ $exit_code -ne 0 ]; then
        print_error "Installation failed with exit code: $exit_code"
        
        if [ ${#ROLLBACK_ACTIONS[@]} -gt 0 ]; then
            print_status "Performing rollback..."
            for ((i=${#ROLLBACK_ACTIONS[@]}-1; i>=0; i--)); do
                eval "${ROLLBACK_ACTIONS[i]}" 2>/dev/null || true
            done
        fi
    fi
    
    # Clean up temporary files
    for temp_file in "${TEMP_FILES[@]}"; do
        rm -f "$temp_file" 2>/dev/null || true
    done
    
    if [ -n "$LOG_FILE" ] && [ $exit_code -ne 0 ]; then
        echo ""
        print_error "Installation failed. Check log file: $LOG_FILE"
        echo "Last 10 log entries:"
        tail -10 "$LOG_FILE" 2>/dev/null || true
    fi
}

# Set up exit trap
trap cleanup_on_exit EXIT

# Add rollback action
add_rollback() {
    local action="$1"
    ROLLBACK_ACTIONS+=("$action")
    log_message "DEBUG" "Added rollback action: $action"
}

# Retry function for transient failures
retry_command() {
    local max_attempts="$1"
    local delay="$2"
    local description="$3"
    shift 3
    local cmd="$*"
    
    local attempt=1
    while [ $attempt -le $max_attempts ]; do
        if [ $attempt -gt 1 ]; then
            print_status "Retry attempt $attempt/$max_attempts: $description"
            sleep $delay
        fi
        
        if eval "$cmd"; then
            return 0
        fi
        
        attempt=$((attempt + 1))
    done
    
    print_error "Failed after $max_attempts attempts: $description"
    return 1
}

# Container management functions
find_ducklake_containers() {
    local instance_filter="$1"
    local filter_pattern="ducklake-"
    
    if [ -n "$instance_filter" ]; then
        filter_pattern="ducklake-.*-${instance_filter}"
    fi
    
    podman ps -a --filter "name=${filter_pattern}" --format "{{.Names}}" 2>/dev/null | sort
}

find_ducklake_instances() {
    # Find all unique instance names from container names
    podman ps -a --filter "name=ducklake-" --format "{{.Names}}" 2>/dev/null | \
        sed -n 's/ducklake-.*-\(.*\)/\1/p' | sort -u
}

container_status() {
    local instance_filter="$1"
    
    print_status "DuckLake Container Status"
    echo
    
    local containers
    containers=$(find_ducklake_containers "$instance_filter")
    
    if [ -z "$containers" ]; then
        if [ -n "$instance_filter" ]; then
            print_warning "No DuckLake containers found for instance: $instance_filter"
        else
            print_warning "No DuckLake containers found"
        fi
        echo
        echo "Available instances:"
        local instances
        instances=$(find_ducklake_instances)
        if [ -n "$instances" ]; then
            echo "$instances" | sed 's/^/  /'
        else
            echo "  None"
        fi
        return 0
    fi
    
    # Show detailed status
    echo "Container Status:"
    printf "%-30s %-15s %-20s %s\n" "NAME" "STATUS" "PORTS" "IMAGE"
    echo "$(printf '%.80s' "$(printf '%*s' 80 '' | tr ' ' '-')")"
    
    echo "$containers" | while read -r container; do
        if [ -n "$container" ]; then
            local status ports image
            status=$(podman ps -a --filter "name=$container" --format "{{.Status}}" 2>/dev/null)
            ports=$(podman ps -a --filter "name=$container" --format "{{.Ports}}" 2>/dev/null)
            image=$(podman ps -a --filter "name=$container" --format "{{.Image}}" 2>/dev/null | sed 's/.*\///')
            
            printf "%-30s %-15s %-20s %s\n" "$container" "$status" "$ports" "$image"
        fi
    done
    
    echo
    
    # Show summary by instance
    local instances
    instances=$(find_ducklake_instances)
    if [ -n "$instances" ]; then
        echo "Instance Summary:"
        echo "$instances" | while read -r instance; do
            if [ -n "$instance" ]; then
                local running stopped
                running=$(podman ps --filter "name=ducklake-.*-${instance}" --format "{{.Names}}" 2>/dev/null | wc -l)
                stopped=$(podman ps -a --filter "name=ducklake-.*-${instance}" --format "{{.Names}}" 2>/dev/null | wc -l)
                stopped=$((stopped - running))
                
                printf "  %-15s Running: %d, Stopped: %d\n" "$instance" "$running" "$stopped"
            fi
        done
    fi
}

container_start() {
    local instance_filter="$1"
    
    if [ -n "$instance_filter" ]; then
        print_status "Starting DuckLake containers for instance: $instance_filter"
    else
        print_status "Starting all DuckLake containers"
    fi
    
    local containers
    containers=$(find_ducklake_containers "$instance_filter")
    
    if [ -z "$containers" ]; then
        if [ -n "$instance_filter" ]; then
            print_error "No DuckLake containers found for instance: $instance_filter"
        else
            print_error "No DuckLake containers found"
            echo "Run the installer first to create containers"
        fi
        return 1
    fi
    
    local started=0
    local failed=0
    
    echo "$containers" | while read -r container; do
        if [ -n "$container" ]; then
            echo -n "Starting $container... "
            if podman start "$container" >/dev/null 2>&1; then
                echo "✓"
                started=$((started + 1))
            else
                echo "✗"
                failed=$((failed + 1))
            fi
        fi
    done
    
    echo
    if [ $failed -eq 0 ]; then
        print_success "All containers started successfully"
    else
        print_warning "Some containers failed to start. Check logs with --logs"
    fi
    
    # Wait a moment and show status
    sleep 2
    container_status "$instance_filter"
}

container_stop() {
    local instance_filter="$1"
    
    if [ -n "$instance_filter" ]; then
        print_status "Stopping DuckLake containers for instance: $instance_filter"
    else
        print_status "Stopping all DuckLake containers"
    fi
    
    local containers
    containers=$(find_ducklake_containers "$instance_filter")
    
    if [ -z "$containers" ]; then
        if [ -n "$instance_filter" ]; then
            print_warning "No DuckLake containers found for instance: $instance_filter"
        else
            print_warning "No DuckLake containers found"
        fi
        return 0
    fi
    
    local stopped=0
    local failed=0
    
    echo "$containers" | while read -r container; do
        if [ -n "$container" ]; then
            echo -n "Stopping $container... "
            if podman stop "$container" >/dev/null 2>&1; then
                echo "✓"
                stopped=$((stopped + 1))
            else
                echo "✗"
                failed=$((failed + 1))
            fi
        fi
    done
    
    echo
    if [ $failed -eq 0 ]; then
        print_success "All containers stopped successfully"
    else
        print_warning "Some containers failed to stop"
    fi
}

container_restart() {
    local instance_filter="$1"
    
    print_status "Restarting DuckLake containers"
    container_stop "$instance_filter"
    sleep 2
    container_start "$instance_filter"
}

container_logs() {
    local service="$1"
    local instance_filter="$2"
    
    case "$service" in
        postgres|postgresql)
            local container_pattern="ducklake-postgres"
            ;;
        minio)
            local container_pattern="ducklake-minio"
            ;;
        *)
            print_error "Unknown service: $service"
            echo "Available services: postgres, minio"
            return 1
            ;;
    esac
    
    if [ -n "$instance_filter" ]; then
        container_pattern="${container_pattern}-${instance_filter}"
    fi
    
    local containers
    containers=$(podman ps -a --filter "name=${container_pattern}" --format "{{.Names}}" 2>/dev/null)
    
    if [ -z "$containers" ]; then
        print_error "No $service containers found"
        if [ -n "$instance_filter" ]; then
            echo "Instance filter: $instance_filter"
        fi
        return 1
    fi
    
    local container_count
    container_count=$(echo "$containers" | wc -l)
    
    if [ "$container_count" -eq 1 ]; then
        local container
        container=$(echo "$containers" | head -1)
        print_status "Showing logs for: $container"
        echo "Press Ctrl+C to exit"
        echo
        podman logs -f "$container"
    else
        echo "Multiple $service containers found:"
        echo "$containers" | nl
        echo
        read -p "Select container number [1]: " selection
        selection=${selection:-1}
        
        local container
        container=$(echo "$containers" | sed -n "${selection}p")
        
        if [ -n "$container" ]; then
            print_status "Showing logs for: $container"
            echo "Press Ctrl+C to exit"
            echo
            podman logs -f "$container"
        else
            print_error "Invalid selection"
            return 1
        fi
    fi
}

container_clean() {
    local instance_filter="$1"
    
    if [ -n "$instance_filter" ]; then
        print_status "Cleaning DuckLake containers and volumes for instance: $instance_filter"
    else
        print_status "Cleaning all DuckLake containers and volumes"
    fi
    
    # Stop containers first
    container_stop "$instance_filter"
    
    # Remove containers
    local containers
    containers=$(find_ducklake_containers "$instance_filter")
    
    if [ -n "$containers" ]; then
        echo
        print_status "Removing containers..."
        echo "$containers" | while read -r container; do
            if [ -n "$container" ]; then
                echo -n "Removing $container... "
                if podman rm "$container" >/dev/null 2>&1; then
                    echo "✓"
                else
                    echo "✗"
                fi
            fi
        done
    fi
    
    # Remove volumes
    echo
    print_status "Removing volumes..."
    local volume_pattern="*_data"
    if [ -n "$instance_filter" ]; then
        volume_pattern="*_data_${instance_filter}"
    fi
    
    local volumes
    volumes=$(podman volume ls --format "{{.Name}}" | grep -E "${volume_pattern}" 2>/dev/null || true)
    
    if [ -n "$volumes" ]; then
        echo "$volumes" | while read -r volume; do
            if [ -n "$volume" ]; then
                echo -n "Removing volume $volume... "
                if podman volume rm "$volume" >/dev/null 2>&1; then
                    echo "✓"
                else
                    echo "✗"
                fi
            fi
        done
    else
        echo "No volumes found to remove"
    fi
    
    echo
    print_success "Cleanup completed"
}

# Show help message
show_help() {
    cat << EOF
DuckLake Complete Installer & Container Manager

USAGE:
    $0 [OPTIONS] [COMMAND]

INSTALLATION OPTIONS:
    --non-interactive    Run without prompts using defaults or config file
    --dry-run           Preview changes without executing them
    --config FILE       Use configuration file instead of prompts
    --uninstall         Remove DuckLake installation
    --help              Show this help message

CONTAINER MANAGEMENT COMMANDS:
    --start [INSTANCE]   Start DuckLake containers
    --stop [INSTANCE]    Stop DuckLake containers
    --restart [INSTANCE] Restart DuckLake containers
    --status [INSTANCE]  Show container status
    --logs SERVICE       Show logs for service (postgres|minio)
    --clean [INSTANCE]   Stop containers and remove volumes

EXAMPLES:
    # Interactive installation
    $0

    # Non-interactive with defaults
    $0 --non-interactive

    # Use configuration file
    $0 --config ducklake.conf

    # Preview installation
    $0 --dry-run

    # Remove installation
    $0 --uninstall

    # Container management
    $0 --status                    # Show all DuckLake containers
    $0 --status default            # Show containers for 'default' instance
    $0 --start                     # Start all DuckLake containers
    $0 --start prod                # Start containers for 'prod' instance
    $0 --stop default              # Stop containers for 'default' instance
    $0 --restart                   # Restart all DuckLake containers
    $0 --logs postgres             # Show PostgreSQL logs
    $0 --logs minio                # Show MinIO logs
    $0 --clean                     # Clean up all containers and volumes

CONFIGURATION FILE FORMAT:
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

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --non-interactive)
                NON_INTERACTIVE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --uninstall)
                UNINSTALL=true
                shift
                ;;
            --start)
                COMMAND="start"
                INSTANCE_FILTER="$2"
                if [[ "$2" =~ ^-- ]] || [[ -z "$2" ]]; then
                    shift
                else
                    shift 2
                fi
                ;;
            --stop)
                COMMAND="stop"
                INSTANCE_FILTER="$2"
                if [[ "$2" =~ ^-- ]] || [[ -z "$2" ]]; then
                    shift
                else
                    shift 2
                fi
                ;;
            --restart)
                COMMAND="restart"
                INSTANCE_FILTER="$2"
                if [[ "$2" =~ ^-- ]] || [[ -z "$2" ]]; then
                    shift
                else
                    shift 2
                fi
                ;;
            --status)
                COMMAND="status"
                INSTANCE_FILTER="$2"
                if [[ "$2" =~ ^-- ]] || [[ -z "$2" ]]; then
                    shift
                else
                    shift 2
                fi
                ;;
            --logs)
                COMMAND="logs"
                LOG_SERVICE="$2"
                if [[ -z "$2" ]] || [[ "$2" =~ ^-- ]]; then
                    print_error "--logs requires a service name (postgres|minio)"
                    exit 1
                fi
                shift 2
                ;;
            --clean)
                COMMAND="clean"
                INSTANCE_FILTER="$2"
                if [[ "$2" =~ ^-- ]] || [[ -z "$2" ]]; then
                    shift
                else
                    shift 2
                fi
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Execute command with dry-run support
execute_cmd() {
    local cmd="$1"
    local description="$2"
    
    if [ "$DRY_RUN" = true ]; then
        print_status "[DRY RUN] Would execute: $description"
        echo "  Command: $cmd"
        return 0
    else
        if [ -n "$description" ]; then
            print_status "$description"
        fi
        eval "$cmd"
    fi
}

# Progress indicator for long operations
show_progress() {
    local pid=$1
    local message="$2"
    local delay=0.5
    local spinstr='|/-\'
    
    if [ "$DRY_RUN" = true ]; then
        return 0
    fi
    
    echo -n "$message "
    while [ "$(ps a | awk '{print $1}' | grep $pid)" ]; do
        local temp=${spinstr#?}
        printf " [%c]  " "$spinstr"
        local spinstr=$temp${spinstr%"$temp"}
        sleep $delay
        printf "\b\b\b\b\b\b"
    done
    printf "    \b\b\b\b"
    echo ""
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Validation functions
validate_port() {
    local port="$1"
    local name="$2"
    
    if ! [[ "$port" =~ ^[0-9]+$ ]] || [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
        print_error "Invalid $name port: $port (must be 1-65535)"
        return 1
    fi
    
    # Check if port is already in use
    if command_exists ss; then
        if ss -tuln | grep -q ":$port "; then
            print_warning "$name port $port is already in use"
            return 1
        fi
    elif command_exists netstat; then
        if netstat -tuln 2>/dev/null | grep -q ":$port "; then
            print_warning "$name port $port is already in use"
            return 1
        fi
    fi
    
    return 0
}

validate_path() {
    local path="$1"
    local name="$2"
    
    # Check for invalid characters
    if [[ "$path" =~ [[:cntrl:]] ]]; then
        print_error "Invalid $name path: contains control characters"
        return 1
    fi
    
    # Check if parent directory exists or can be created
    local parent_dir=$(dirname "$path")
    if [ ! -d "$parent_dir" ]; then
        if ! mkdir -p "$parent_dir" 2>/dev/null; then
            print_error "Cannot create parent directory for $name: $parent_dir"
            return 1
        fi
    fi
    
    return 0
}

validate_name() {
    local name="$1"
    local field="$2"
    
    # Check for valid identifier (alphanumeric, underscore, hyphen)
    if ! [[ "$name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        print_error "Invalid $field: $name (only alphanumeric, underscore, and hyphen allowed)"
        return 1
    fi
    
    return 0
}

validate_config() {
    local errors=0
    
    print_status "Validating configuration..."
    
    # Validate paths
    if ! validate_path "$INSTALL_DIR" "installation directory"; then
        errors=$((errors + 1))
    fi
    
    # Validate names
    if ! validate_name "$INSTANCE_NAME" "instance name"; then
        errors=$((errors + 1))
    fi
    
    if ! validate_name "$BUCKET_NAME" "bucket name"; then
        errors=$((errors + 1))
    fi
    
    if ! validate_name "$DB_NAME" "database name"; then
        errors=$((errors + 1))
    fi
    
    if ! validate_name "$DB_USER" "database user"; then
        errors=$((errors + 1))
    fi
    
    # Validate ports
    if ! validate_port "$POSTGRES_PORT" "PostgreSQL"; then
        errors=$((errors + 1))
    fi
    
    if ! validate_port "$MINIO_PORT" "MinIO"; then
        errors=$((errors + 1))
    fi
    
    if ! validate_port "$MINIO_CONSOLE_PORT" "MinIO Console"; then
        errors=$((errors + 1))
    fi
    
    # Check for port conflicts between services
    if [ "$POSTGRES_PORT" = "$MINIO_PORT" ] || [ "$POSTGRES_PORT" = "$MINIO_CONSOLE_PORT" ] || [ "$MINIO_PORT" = "$MINIO_CONSOLE_PORT" ]; then
        print_error "Port conflict: PostgreSQL ($POSTGRES_PORT), MinIO ($MINIO_PORT), and MinIO Console ($MINIO_CONSOLE_PORT) must use different ports"
        errors=$((errors + 1))
    fi
    
    # Validate password strength (basic check)
    if [ ${#DB_PASS} -lt 8 ]; then
        print_warning "Database password is less than 8 characters (current: ${#DB_PASS})"
        if [ "$NON_INTERACTIVE" = false ]; then
            read -p "Continue anyway? [y/N]: " continue_weak_pass
            if [ "$continue_weak_pass" != "y" ] && [ "$continue_weak_pass" != "Y" ]; then
                errors=$((errors + 1))
            fi
        fi
    fi
    
    if [ $errors -gt 0 ]; then
        print_error "Configuration validation failed with $errors error(s)"
        return 1
    fi
    
    print_success "Configuration validation passed"
    return 0
}

# Load configuration from file
load_config() {
    local config_file="$1"
    
    if [ ! -f "$config_file" ]; then
        print_error "Configuration file not found: $config_file"
        print_error "Create one using the example: cp ducklake.conf.example ducklake.conf"
        exit 1
    fi
    
    print_status "Loading configuration from: $config_file"
    
    # Source the config file safely
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        [[ $key =~ ^[[:space:]]*# ]] && continue
        [[ -z $key ]] && continue
        
        # Remove quotes from value
        value=$(echo "$value" | sed 's/^["'\'']//' | sed 's/["'\'']$//')
        
        case $key in
            INSTALL_DIR|INSTANCE_NAME|BUCKET_NAME|DATA_PATH|DB_NAME|DB_USER|DB_PASS|POSTGRES_PORT|MINIO_PORT|MINIO_CONSOLE_PORT)
                export "$key=$value"
                ;;
        esac
    done < "$config_file"
}

# Get user configuration
get_user_config() {
    # If config file specified, load it
    if [ -n "$CONFIG_FILE" ]; then
        load_config "$CONFIG_FILE"
        print_success "Configuration loaded from file"
        return 0
    fi
    
    # Set defaults for non-interactive mode
    if [ "$NON_INTERACTIVE" = true ]; then
        INSTALL_DIR=${INSTALL_DIR:-./ducklake}
        INSTANCE_NAME=${INSTANCE_NAME:-default}
        BUCKET_NAME=${BUCKET_NAME:-ducklake-bucket}
        DATA_PATH=${DATA_PATH:-data}
        DB_NAME=${DB_NAME:-ducklake_catalog}
        DB_USER=${DB_USER:-ducklake}
        DB_PASS=${DB_PASS:-ducklake123}
        POSTGRES_PORT=${POSTGRES_PORT:-5432}
        MINIO_PORT=${MINIO_PORT:-9000}
        MINIO_CONSOLE_PORT=${MINIO_CONSOLE_PORT:-9001}
        
        print_status "Using default configuration (non-interactive mode)"
    else
        print_status "DuckLake Configuration Setup"
        echo
        
        # Get installation directory
        read -p "Enter installation directory [./ducklake]: " INSTALL_DIR
        INSTALL_DIR=${INSTALL_DIR:-./ducklake}
        
        # Get instance name for this DuckLake setup
        read -p "Enter instance name [default]: " INSTANCE_NAME
        INSTANCE_NAME=${INSTANCE_NAME:-default}
        
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
    fi
    
    # Create directory if it doesn't exist
    if [ ! -d "$INSTALL_DIR" ]; then
        execute_cmd "mkdir -p \"$INSTALL_DIR\"" "Creating directory: $INSTALL_DIR"
    fi
    
    # Convert to absolute path and change to it
    INSTALL_DIR=$(realpath "$INSTALL_DIR")
    if [ "$DRY_RUN" = false ]; then
        cd "$INSTALL_DIR"
    fi
    print_status "Using installation directory: $INSTALL_DIR"
    echo
    
    # Export for use in other functions
    export BUCKET_NAME DATA_PATH DB_NAME DB_USER DB_PASS INSTALL_DIR INSTANCE_NAME POSTGRES_PORT MINIO_PORT MINIO_CONSOLE_PORT
    
    # Validate configuration
    if ! validate_config; then
        exit 1
    fi
    
    echo
    print_success "Configuration:"
    echo "  Installation: $INSTALL_DIR"
    echo "  Instance: $INSTANCE_NAME"
    echo "  Bucket: s3://$BUCKET_NAME/$DATA_PATH"
    echo "  Database: $DB_NAME"
    echo "  User: $DB_USER"
    echo "  PostgreSQL Port: $POSTGRES_PORT"
    echo "  MinIO Port: $MINIO_PORT"
    echo "  MinIO Console Port: $MINIO_CONSOLE_PORT"
    echo
}

# Get server IP for remote connections
get_server_ip() {
    local ip=""
    
    # Try multiple methods to get external IPv4 IP with timeouts
    if command_exists curl; then
        ip=$(curl -s -4 --connect-timeout $CURL_CONNECT_TIMEOUT --max-time $CURL_TIMEOUT ifconfig.me 2>/dev/null) || \
        ip=$(curl -s -4 --connect-timeout $CURL_CONNECT_TIMEOUT --max-time $CURL_TIMEOUT ipinfo.io/ip 2>/dev/null) || \
        ip=$(curl -s -4 --connect-timeout $CURL_CONNECT_TIMEOUT --max-time $CURL_TIMEOUT icanhazip.com 2>/dev/null)
    fi
    
    # Fallback to local IP
    if [ -z "$ip" ]; then
        ip=$(hostname -I | grep -oE '([0-9]{1,3}\.){3}[0-9]{1,3}' | head -1)
    fi
    
    echo "$ip"
}

# Uninstall DuckLake
uninstall_ducklake() {
    print_status "Starting DuckLake uninstallation..."
    
    # Try to find existing installations
    local found_installations=()
    
    # Look for common installation directories
    for dir in "./ducklake" "$HOME/ducklake" "/opt/ducklake"; do
        if [ -d "$dir" ] && [ -f "$dir/pyproject.toml" ]; then
            found_installations+=("$dir")
        fi
    done
    
    if [ ${#found_installations[@]} -eq 0 ]; then
        print_warning "No DuckLake installations found"
        return 0
    fi
    
    echo "Found DuckLake installations:"
    for i in "${!found_installations[@]}"; do
        echo "  $((i+1)). ${found_installations[$i]}"
    done
    echo
    
    if [ "$NON_INTERACTIVE" = false ]; then
        read -p "Select installation to remove [1]: " selection
        selection=${selection:-1}
    else
        selection=1
    fi
    
    if [ "$selection" -ge 1 ] && [ "$selection" -le ${#found_installations[@]} ]; then
        local install_dir="${found_installations[$((selection-1))]}"
        
        print_status "Removing DuckLake installation: $install_dir"
        
        # Change to installation directory
        if [ "$DRY_RUN" = false ]; then
            cd "$install_dir"
        fi
        
        # Stop services if they exist
        if [ -f "pyproject.toml" ]; then
            execute_cmd "uv run task stop 2>/dev/null || true" "Stopping DuckLake services"
            execute_cmd "uv run task clean 2>/dev/null || true" "Cleaning up containers and volumes"
        fi
        
        # Remove installation directory
        execute_cmd "rm -rf \"$install_dir\"" "Removing installation directory"
        
        print_success "DuckLake uninstalled successfully"
    else
        print_error "Invalid selection"
        exit 1
    fi
}

# Create embedded files function
create_files() {
    if [ "$DRY_RUN" = true ]; then
        print_status "[DRY RUN] Would create configuration files in: $INSTALL_DIR"
        return 0
    fi
    
    print_status "Creating configuration files..."
    
    # Detect if we need special userns flags for high UIDs
    local current_uid=$(id -u)
    local postgres_userns_flag=""
    local minio_userns_flag=""
    
    if [ "$current_uid" -gt 100000 ]; then
        postgres_userns_flag="--userns=host"
        minio_userns_flag="--userns=keep-id"
        print_status "Detected high UID ($current_uid), using --userns=host for PostgreSQL and --userns=keep-id for MinIO"
    fi

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
start-postgres = "podman run -d --name ducklake-postgres-${INSTANCE_NAME} --replace $postgres_userns_flag -e POSTGRES_DB=${DB_NAME} -e POSTGRES_USER=${DB_USER} -e POSTGRES_PASSWORD=${DB_PASS} -e POSTGRES_HOST_AUTH_METHOD=trust -p ${POSTGRES_PORT}:5432 -v postgres_data_${INSTANCE_NAME}:/var/lib/postgresql/data -v $(pwd)/init.sql:/docker-entrypoint-initdb.d/init.sql docker.io/library/postgres:15"
start-minio = "podman run -d --name ducklake-minio-${INSTANCE_NAME} --replace $minio_userns_flag -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin -p ${MINIO_PORT}:9000 -p ${MINIO_CONSOLE_PORT}:9001 -v minio_data_${INSTANCE_NAME}:/data quay.io/minio/minio:latest server /data --console-address :9001"
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
        if [ "$DRY_RUN" = true ]; then
            print_status "[DRY RUN] Would install Podman"
            return 0
        fi
        
        print_status "Installing Podman..."
        
        # Detect OS and install podman
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            case $ID in
                ubuntu|debian)
                    if command_exists apt-get; then
                        execute_cmd "sudo apt-get update && sudo apt-get install -y podman" "Installing Podman via apt"
                    else
                        print_error "apt-get not found on Debian/Ubuntu system"
                        exit 1
                    fi
                    ;;
                fedora|centos|rhel)
                    if command_exists dnf; then
                        execute_cmd "sudo dnf install -y podman" "Installing Podman via dnf"
                    elif command_exists yum; then
                        execute_cmd "sudo yum install -y podman" "Installing Podman via yum"
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

# Parse command line arguments first
parse_args "$@"

# Handle container management commands
if [ -n "$COMMAND" ]; then
    case "$COMMAND" in
        status)
            container_status "$INSTANCE_FILTER"
            exit 0
            ;;
        start)
            container_start "$INSTANCE_FILTER"
            exit 0
            ;;
        stop)
            container_stop "$INSTANCE_FILTER"
            exit 0
            ;;
        restart)
            container_restart "$INSTANCE_FILTER"
            exit 0
            ;;
        logs)
            container_logs "$LOG_SERVICE" "$INSTANCE_FILTER"
            exit 0
            ;;
        clean)
            container_clean "$INSTANCE_FILTER"
            exit 0
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            exit 1
            ;;
    esac
fi

# Handle uninstall mode
if [ "$UNINSTALL" = true ]; then
    uninstall_ducklake
    exit 0
fi

echo "=================================================="
echo "         DuckLake Complete Installer"
echo "=================================================="
if [ "$DRY_RUN" = true ]; then
    echo "                   [DRY RUN MODE]"
    echo "=================================================="
fi
echo ""

# Get user configuration
get_user_config

# Setup logging after we know the install directory
setup_logging

# Check prerequisites
print_status "Checking prerequisites..."

# Check if running as root (not recommended for podman)
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root. Podman rootless mode is recommended."
    echo "Consider running as a regular user instead."
fi

# Check Python version with enhanced error messages
check_python() {
    if ! command_exists python3; then
        print_warning "Python 3 not found in system PATH"
        echo "UV will manage Python installation automatically"
        return 0
    fi
    
    local python_version
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))' 2>/dev/null)
    
    if [ -z "$python_version" ]; then
        print_warning "Failed to determine Python version, UV will handle Python setup"
        return 0
    fi
    
    local python_major python_minor
    python_major=$(echo "$python_version" | cut -d. -f1)
    python_minor=$(echo "$python_version" | cut -d. -f2)
    
    if [ "$python_major" -ge 3 ] && [ "$python_minor" -ge 10 ]; then
        print_success "Python $python_version found"
        log_message "INFO" "Python version: $python_version"
        return 0
    else
        print_warning "Python 3.10+ preferred. Found: $python_version"
        echo "UV will manage Python version automatically"
        return 0
    fi
}

check_python

# Install podman if needed
install_podman

# Setup user namespaces
setup_user_namespaces

# Check UV package manager with enhanced error handling
install_uv() {
    if [ "$DRY_RUN" = true ]; then
        print_status "[DRY RUN] Would install UV package manager"
        return 0
    fi
    
    print_status "UV not found. Installing UV..."
    
    # Create temp directory for download
    TEMP_DIR=$(mktemp -d)
    TEMP_FILES+=("$TEMP_DIR")
    
    # Download installer script with timeout and retry
    local install_script="$TEMP_DIR/install.sh"
    
    if ! retry_command 3 2 "UV installer download" \
        "curl -LsSf --connect-timeout $CURL_CONNECT_TIMEOUT --max-time $CURL_TIMEOUT https://astral.sh/uv/install.sh -o \"$install_script\""; then
        print_error "Failed to download UV installer after multiple attempts"
        echo "Manual installation: curl -LsSf https://astral.sh/uv/install.sh | sh"
        return 1
    fi
    
    # Verify download
    if [ ! -f "$install_script" ] || [ ! -s "$install_script" ]; then
        print_error "UV installer download appears to be corrupted"
        return 1
    fi
    
    # Install UV
    if ! bash "$install_script"; then
        print_error "UV installation failed"
        echo "Try manual installation: curl -LsSf https://astral.sh/uv/install.sh | sh"
        return 1
    fi
    
    # Add to PATH and reload
    export PATH="$HOME/.local/bin:$PATH"
    add_rollback "export PATH=\"\${PATH//:$HOME\/.local\/bin/}\""
    
    # Source shell profile to persist PATH changes
    if [ -f "$HOME/.bashrc" ]; then
        if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$HOME/.bashrc"; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
            add_rollback "sed -i '/export PATH=\"\$HOME\/.local\/bin:\$PATH\"/d' \"$HOME/.bashrc\""
        fi
    fi
    
    # Verify installation
    if command_exists uv; then
        print_success "UV installed successfully"
        return 0
    else
        print_error "UV installation completed but command not found"
        echo "Try adding $HOME/.local/bin to your PATH manually"
        return 1
    fi
}

if ! command_exists uv; then
    if ! install_uv; then
        exit 1
    fi
else
    print_success "UV found"
    # Check UV version for compatibility
    UV_VERSION=$(uv --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    if [ -n "$UV_VERSION" ]; then
        log_message "INFO" "UV version: $UV_VERSION"
    fi
fi

# Test podman configuration with enhanced diagnostics
test_podman_config() {
    if [ "$DRY_RUN" = true ]; then
        print_status "[DRY RUN] Would test podman configuration"
        return 0
    fi
    
    print_status "Testing podman configuration..."
    
    # Test basic podman functionality
    if ! podman info >/dev/null 2>&1; then
        print_error "Podman configuration issues detected"
        echo ""
        echo "Diagnostic information:"
        
        # Check if running as root
        if [ "$EUID" -eq 0 ]; then
            echo "- Running as root (rootful mode)"
        else
            echo "- Running as user: $(whoami) (rootless mode)"
        fi
        
        # Check user namespaces
        if [ ! -f /etc/subuid ] || ! grep -q "^$(whoami):" /etc/subuid; then
            echo "- Missing subuid configuration"
        fi
        
        if [ ! -f /etc/subgid ] || ! grep -q "^$(whoami):" /etc/subgid; then
            echo "- Missing subgid configuration"
        fi
        
        echo ""
        echo "Common fixes for podman permission errors:"
        echo "1. Check podman configuration: podman info"
        echo "2. Restart user session: sudo loginctl terminate-user $(whoami)"
        echo "3. For rootless mode: systemctl --user enable --now podman.socket"
        echo "4. Check user namespaces: cat /etc/subuid /etc/subgid"
        echo ""
        echo "After fixes, re-run this installer."
        return 1
    fi
    
    print_success "Podman configuration verified"
    
    # Get podman version for logging
    local podman_version
    podman_version=$(podman --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
    if [ -n "$podman_version" ]; then
        log_message "INFO" "Podman version: $podman_version"
    fi
    
    return 0
}

# Test image pulling permissions with timeout
test_registry_access() {
    if [ "$DRY_RUN" = true ]; then
        print_status "[DRY RUN] Would test container registry access"
        return 0
    fi
    
    print_status "Testing container registry access..."
    
    # Test with timeout and specific error handling
    local test_image="docker.io/library/hello-world:latest"
    
    if timeout 60 podman pull --quiet "$test_image" >/dev/null 2>&1; then
        print_success "Container registry access verified"
        # Clean up test image
        podman rmi "$test_image" >/dev/null 2>&1 || true
        add_rollback "podman rmi $test_image >/dev/null 2>&1 || true"
        return 0
    else
        print_warning "Container registry access may be limited"
        echo ""
        echo "This could be due to:"
        echo "1. Network restrictions or firewall"
        echo "2. Registry authentication required"
        echo "3. Corporate proxy settings"
        echo "4. DNS resolution issues"
        echo ""
        echo "Troubleshooting steps:"
        echo "1. Test network: curl -I https://registry-1.docker.io"
        echo "2. Check proxy settings: echo \$HTTP_PROXY \$HTTPS_PROXY"
        echo "3. Try manual pull: podman pull $test_image"
        echo ""
        echo "Continuing anyway - images will be pulled during service start..."
        return 0
    fi
}

if ! test_podman_config; then
    exit 1
fi

test_registry_access

# Use the installation directory from config
DUCKLAKE_DIR="$INSTALL_DIR"

# Create all configuration files
create_files
add_rollback "rm -rf \"$INSTALL_DIR\" 2>/dev/null || true"

# Install Python dependencies with error handling
install_python_deps() {
    if [ "$DRY_RUN" = true ]; then
        print_status "[DRY RUN] Would install Python dependencies"
        return 0
    fi
    
    print_status "Installing Python dependencies..."
    
    # Force UV to use local directory, not parent venv
    unset VIRTUAL_ENV
    
    # Create virtual environment with Python 3.10+ and error handling
    if ! uv venv --python 3.10; then
        print_warning "Failed to create venv with Python 3.10, trying default Python"
        if ! uv venv; then
            print_error "Failed to create virtual environment"
            echo "Try manually: cd $INSTALL_DIR && uv venv"
            return 1
        fi
    fi
    add_rollback "rm -rf \"$INSTALL_DIR/.venv\" 2>/dev/null || true"
    
    # Install dependencies with retry
    if ! retry_command 3 5 "Python dependency installation" "uv sync"; then
        print_error "Failed to install Python dependencies"
        echo ""
        echo "Troubleshooting steps:"
        echo "1. Check internet connection"
        echo "2. Try manual installation: cd $INSTALL_DIR && uv sync"
        echo "3. Check UV configuration: uv --version"
        echo "4. Clear UV cache: uv cache clean"
        return 1
    fi
    
    print_success "Python dependencies installed"
    return 0
}

if ! install_python_deps; then
    exit 1
fi

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
if [ "$DRY_RUN" = true ]; then
    print_status "[DRY RUN] Would start DuckLake services"
else
    print_status "Starting DuckLake services..."
    (uv run task start) &
    show_progress $! "Starting PostgreSQL and MinIO containers..."
fi

# Enhanced service checking with better error handling
check_service() {
    local service_name="$1"
    local max_attempts=60
    local attempt=0
    local service_type="$2"
    
    if [ "$DRY_RUN" = true ]; then
        print_status "[DRY RUN] Would check $service_name status"
        return 0
    fi
    
    print_status "Waiting for $service_name to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        # Check if container exists and is running
        if podman ps --filter name="$service_name" --format "{{.Names}}" 2>/dev/null | grep -q "$service_name"; then
            # Additional health check for specific services
            case "$service_type" in
                "postgres")
                    if podman exec "$service_name" pg_isready -U "$DB_USER" >/dev/null 2>&1; then
                        return 0
                    fi
                    ;;
                "minio")
                    if podman exec "$service_name" mc --version >/dev/null 2>&1; then
                        return 0
                    fi
                    ;;
                *)
                    return 0
                    ;;
            esac
        fi
        
        sleep 1
        attempt=$((attempt + 1))
        if [ $((attempt % 10)) -eq 0 ]; then
            echo -n "."
        fi
    done
    return 1
}

# Check services with better error handling and recovery suggestions
check_postgres() {
    if check_service "ducklake-postgres-${INSTANCE_NAME}" "postgres"; then
        print_success "PostgreSQL is running and accepting connections"
        add_rollback "podman stop ducklake-postgres-${INSTANCE_NAME} 2>/dev/null || true"
    else
        print_error "PostgreSQL failed to start or is not accepting connections"
        echo ""
        echo "Troubleshooting steps:"
        echo "1. Check PostgreSQL logs: cd $DUCKLAKE_DIR && uv run task logs-postgres"
        echo "2. Verify port $POSTGRES_PORT is not in use: ss -tuln | grep :$POSTGRES_PORT"
        echo "3. Check container status: podman ps -a --filter name=ducklake-postgres-${INSTANCE_NAME}"
        echo "4. Try restarting: cd $DUCKLAKE_DIR && uv run task stop && uv run task start"
        
        # Try to get more specific error information
        if podman ps -a --filter name="ducklake-postgres-${INSTANCE_NAME}" --format "{{.Status}}" | grep -q "Exited"; then
            echo "5. Container exited - check logs for startup errors"
            podman logs "ducklake-postgres-${INSTANCE_NAME}" 2>/dev/null | tail -10 || true
        fi
        
        return 1
    fi
}

check_minio() {
    if check_service "ducklake-minio-${INSTANCE_NAME}" "minio"; then
        print_success "MinIO is running"
        add_rollback "podman stop ducklake-minio-${INSTANCE_NAME} 2>/dev/null || true"
    else
        print_error "MinIO failed to start"
        echo ""
        echo "Troubleshooting steps:"
        echo "1. Check MinIO logs: cd $DUCKLAKE_DIR && uv run task logs-minio"
        echo "2. Verify ports $MINIO_PORT and $MINIO_CONSOLE_PORT are not in use"
        echo "3. Check container status: podman ps -a --filter name=ducklake-minio-${INSTANCE_NAME}"
        echo "4. Try restarting: cd $DUCKLAKE_DIR && uv run task stop && uv run task start"
        
        # Try to get more specific error information
        if podman ps -a --filter name="ducklake-minio-${INSTANCE_NAME}" --format "{{.Status}}" | grep -q "Exited"; then
            echo "5. Container exited - check logs for startup errors"
            podman logs "ducklake-minio-${INSTANCE_NAME}" 2>/dev/null | tail -10 || true
        fi
        
        return 1
    fi
}

# Check services with retry logic
if ! retry_command 3 5 "PostgreSQL startup" "check_postgres"; then
    exit 1
fi

if ! retry_command 3 5 "MinIO startup" "check_minio"; then
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