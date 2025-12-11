#!/bin/bash
# =============================================================================
# SCL Internet Ingestion - Quick Start Script
# =============================================================================
#
# This script helps you build an SCL knowledge base from internet-scale data.
#
# Data Sources (in order of ease):
# 1. ArXiv - Scientific papers (easiest, ~1000 papers/hour)
# 2. Wikipedia - Encyclopedia (~6M articles, ~20GB dump)
# 3. Common Crawl - Web archive (petabytes available)
#
# Usage:
#   ./ingest_internet.sh [command]
#
# Commands:
#   start-server    Start the SCL API server
#   arxiv           Ingest ArXiv papers (CS by default)
#   wikipedia       Download and ingest Wikipedia
#   common-crawl    Ingest from Common Crawl
#   overnight       Run overnight batch processing
#   stats           Show database statistics
#   search          Search the knowledge base
# =============================================================================

set -e

# Configuration
DB_PATH="${SCL_DB:-scl_atoms.db}"
API_URL="${SCL_API_URL:-http://localhost:8000}"
VENV_PATH="./venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Activate virtual environment
activate_venv() {
    if [ -d "$VENV_PATH" ]; then
        source "$VENV_PATH/bin/activate"
    else
        log_error "Virtual environment not found. Run: python -m venv venv && pip install -r requirements.txt"
        exit 1
    fi
}

# Check if server is running
check_server() {
    if curl -s "$API_URL/health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Start the SCL server
cmd_start_server() {
    log_info "Starting SCL API server..."
    activate_venv
    
    if check_server; then
        log_warn "Server already running at $API_URL"
        return 0
    fi
    
    # Start in background
    nohup python scl_serve.py > scl_server.log 2>&1 &
    SERVER_PID=$!
    echo $SERVER_PID > .server.pid
    
    # Wait for server to start
    log_info "Waiting for server to start..."
    for i in {1..30}; do
        if check_server; then
            log_success "Server started (PID: $SERVER_PID)"
            return 0
        fi
        sleep 1
    done
    
    log_error "Server failed to start. Check scl_server.log"
    exit 1
}

# Stop the server
cmd_stop_server() {
    if [ -f .server.pid ]; then
        PID=$(cat .server.pid)
        if kill -0 $PID 2>/dev/null; then
            log_info "Stopping server (PID: $PID)..."
            kill $PID
            rm .server.pid
            log_success "Server stopped"
        else
            log_warn "Server not running"
            rm .server.pid
        fi
    else
        log_warn "No server PID file found"
    fi
}

# Ingest ArXiv papers
cmd_arxiv() {
    CATEGORY="${1:-cs}"
    LIMIT="${2:-1000}"
    
    log_info "Ingesting ArXiv papers (category: $CATEGORY, limit: $LIMIT)"
    activate_venv
    
    if ! check_server; then
        log_error "SCL server not running. Start it with: $0 start-server"
        exit 1
    fi
    
    python scl_crawler.py --arxiv "$CATEGORY" --limit "$LIMIT" --db "$DB_PATH" --api-url "$API_URL"
}

# Download and ingest Wikipedia
cmd_wikipedia() {
    DUMP_PATH="${1:-}"
    
    if [ -z "$DUMP_PATH" ]; then
        log_info "Wikipedia dump not provided. Downloading..."
        
        # Create data directory
        mkdir -p data
        
        # Download latest dump (this is large ~20GB!)
        DUMP_URL="https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2"
        DUMP_PATH="data/enwiki-latest-pages-articles.xml.bz2"
        
        if [ -f "$DUMP_PATH" ]; then
            log_warn "Dump already exists: $DUMP_PATH"
        else
            log_warn "This will download ~20GB. Press Ctrl+C to cancel."
            sleep 5
            
            log_info "Downloading Wikipedia dump..."
            wget -c "$DUMP_URL" -O "$DUMP_PATH"
        fi
    fi
    
    log_info "Ingesting Wikipedia from: $DUMP_PATH"
    activate_venv
    
    if ! check_server; then
        log_error "SCL server not running. Start it with: $0 start-server"
        exit 1
    fi
    
    python scl_crawler.py --wikipedia-dump "$DUMP_PATH" --db "$DB_PATH" --api-url "$API_URL"
}

# Ingest from Common Crawl
cmd_common_crawl() {
    SEGMENTS="${1:-5}"
    DOCS="${2:-500}"
    
    log_info "Ingesting from Common Crawl (segments: $SEGMENTS, docs/segment: $DOCS)"
    activate_venv
    
    if ! check_server; then
        log_error "SCL server not running. Start it with: $0 start-server"
        exit 1
    fi
    
    python scl_crawler.py --common-crawl --segments "$SEGMENTS" --limit "$DOCS" --db "$DB_PATH" --api-url "$API_URL"
}

# Run overnight batch processing
cmd_overnight() {
    log_info "Starting overnight batch processing..."
    activate_venv
    
    if ! check_server; then
        log_warn "Starting SCL server first..."
        cmd_start_server
    fi
    
    # Create default schedule if not exists
    if [ ! -f "schedule.yaml" ]; then
        python scl_scheduler.py --create-sample
        mv schedule_sample.yaml schedule.yaml
        log_info "Created default schedule.yaml"
    fi
    
    python scl_scheduler.py --overnight --schedule schedule.yaml --db "$DB_PATH" --api-url "$API_URL"
}

# Show statistics
cmd_stats() {
    activate_venv
    
    # Check all .db files
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    SCL DATABASE STATISTICS                        ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    
    for db in *.db; do
        if [ -f "$db" ]; then
            echo ""
            echo "Database: $db"
            python scl_crawler.py --stats --db "$db" 2>/dev/null | grep -v "^INFO" | tail -n +2
        fi
    done
}

# Search the knowledge base
cmd_search() {
    QUERY="$1"
    
    if [ -z "$QUERY" ]; then
        log_error "Usage: $0 search \"your query\""
        exit 1
    fi
    
    activate_venv
    python scl_crawler.py --search "$QUERY" --db "$DB_PATH"
}

# Show help
cmd_help() {
    echo ""
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║           SCL Internet Ingestion - Build Your Own GPT            ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  start-server              Start the SCL API server"
    echo "  stop-server               Stop the SCL API server"
    echo "  arxiv [category] [limit]  Ingest ArXiv papers (default: cs, 1000)"
    echo "  wikipedia [dump_path]     Ingest Wikipedia (downloads if no path)"
    echo "  common-crawl [segs] [docs] Ingest from Common Crawl"
    echo "  overnight                 Run overnight batch processing"
    echo "  stats                     Show database statistics"
    echo "  search \"query\"            Search the knowledge base"
    echo "  help                      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start-server           # Start the API server"
    echo "  $0 arxiv cs 500           # Ingest 500 CS papers from ArXiv"
    echo "  $0 arxiv physics 1000     # Ingest 1000 physics papers"
    echo "  $0 wikipedia              # Download and ingest Wikipedia"
    echo "  $0 common-crawl 10 1000   # Ingest 10 segments, 1000 docs each"
    echo "  $0 stats                  # Show how many atoms you have"
    echo "  $0 search \"relativity\"    # Search for atoms about relativity"
    echo ""
    echo "Environment Variables:"
    echo "  SCL_DB        Database path (default: scl_atoms.db)"
    echo "  SCL_API_URL   API server URL (default: http://localhost:8000)"
    echo ""
    echo "Recommended Order:"
    echo "  1. Start with ArXiv (fast, high quality)"
    echo "  2. Add Wikipedia (comprehensive, takes time)"
    echo "  3. Common Crawl for scale (web-scale data)"
    echo ""
}

# Main
case "${1:-help}" in
    start-server)
        cmd_start_server
        ;;
    stop-server)
        cmd_stop_server
        ;;
    arxiv)
        cmd_arxiv "$2" "$3"
        ;;
    wikipedia)
        cmd_wikipedia "$2"
        ;;
    common-crawl)
        cmd_common_crawl "$2" "$3"
        ;;
    overnight)
        cmd_overnight
        ;;
    stats)
        cmd_stats
        ;;
    search)
        cmd_search "$2"
        ;;
    help|--help|-h)
        cmd_help
        ;;
    *)
        log_error "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
