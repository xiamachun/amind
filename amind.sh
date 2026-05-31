#!/usr/bin/env bash
#
# amind — Agent's Mind Service Control
# Usage: ./amind.sh {start|stop|restart|status|build|rebuild|logs|test|health}
#
set -euo pipefail

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
# Prefer local binary (release package) over build directory
if [[ -x "${SCRIPT_DIR}/amind" ]]; then
    BINARY="${SCRIPT_DIR}/amind"
else
    BINARY="${BUILD_DIR}/amind"
fi
CONFIG="${SCRIPT_DIR}/amind.conf"
PID_FILE="${SCRIPT_DIR}/.amind.pid"
LOG_FILE="${SCRIPT_DIR}/amind.log"
DATA_DIR="${SCRIPT_DIR}/amind_data"

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# ── Helpers ─────────────────────────────────────────────────────────────────

# Get port from config file (default 8080)
get_port() {
    if [[ -f "$CONFIG" ]]; then
        grep -E '^port\s*=' "$CONFIG" | sed 's/.*=\s*//' | tr -d '[:space:]'
    fi
}

PORT=$(get_port)
PORT=${PORT:-8080}

get_webui_port() {
    if [[ -f "$CONFIG" ]]; then
        grep -E '^webui_port\s*=' "$CONFIG" | sed 's/.*=\s*//' | tr -d '[:space:]'
    fi
}
WEBUI_PORT=$(get_webui_port)
WEBUI_PORT=${WEBUI_PORT:-11011}

# Check if amind is running, echo PID if so
get_pid() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
        rm -f "$PID_FILE"
    fi
    return 1
}

# CPU count (cross-platform)
ncpu() {
    sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4
}

# ── Commands ────────────────────────────────────────────────────────────────

do_build() {
    echo -e "${YELLOW}[BUILD]${NC} Building amind (Release)..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DAMIND_ENABLE_ASAN=OFF \
        -DAMIND_BUILD_TESTS=ON \
        2>&1 | tail -3

    local start_time=$SECONDS
    cmake --build . --target amind -j"$(ncpu)" 2>&1
    local elapsed=$((SECONDS - start_time))

    if [[ -x "$BINARY" ]]; then
        local size
        size=$(du -h "$BINARY" | cut -f1)
        echo -e "${GREEN}[BUILD]${NC} Complete in ${elapsed}s — ${BINARY} (${size})"
    else
        echo -e "${RED}[BUILD]${NC} Build failed!"
        return 1
    fi
}

do_rebuild() {
    echo -e "${YELLOW}[REBUILD]${NC} Cleaning build directory..."
    if [[ -d "$BUILD_DIR" ]]; then
        rm -rf "${BUILD_DIR:?}"/*
    fi
    do_build
}

do_start() {
    if pid=$(get_pid); then
        echo -e "${YELLOW}[START]${NC} amind is already running (PID: ${pid})"
        return 0
    fi

    # Safety check: detect port occupancy before starting
    local port_pid
    port_pid=$(lsof -ti "tcp:${PORT}" 2>/dev/null || true)
    if [[ -n "$port_pid" ]]; then
        local port_cmd
        port_cmd=$(ps -o command= -p "$port_pid" 2>/dev/null | head -1)
        echo -e "${YELLOW}[START]${NC} Port ${PORT} occupied by PID ${port_pid} (${port_cmd})"
        echo -e "  Killing occupying process..."
        kill "$port_pid" 2>/dev/null || true
        sleep 1
        # Force kill if still alive
        if kill -0 "$port_pid" 2>/dev/null; then
            kill -9 "$port_pid" 2>/dev/null || true
            sleep 1
        fi
        # Also kill parent if it's a supervisor script
        local port_ppid
        port_ppid=$(ps -o ppid= -p "$port_pid" 2>/dev/null | tr -d ' ')
        if [[ -n "$port_ppid" && "$port_ppid" != "1" ]]; then
            local parent_cmd
            parent_cmd=$(ps -o command= -p "$port_ppid" 2>/dev/null || true)
            if [[ "$parent_cmd" == *"supervisor"* || "$parent_cmd" == *"amind"* ]]; then
                echo -e "  Also killing parent supervisor (PID ${port_ppid})"
                kill "$port_ppid" 2>/dev/null || true
            fi
        fi
        echo -e "${GREEN}[START]${NC} Port ${PORT} cleared."
    fi

    # Auto-build if binary missing
    if [[ ! -x "$BINARY" ]]; then
        echo -e "${YELLOW}[START]${NC} Binary not found, building first..."
        do_build
    fi

    echo -e "${GREEN}[START]${NC} Starting amind..."

    # Use config file if exists
    local args=()
    if [[ -f "$CONFIG" ]]; then
        args+=("$CONFIG")
        echo -e "  Config : ${CONFIG}"
    else
        echo -e "  Config : (using defaults)"
    fi

    echo -e "  Listen : 0.0.0.0:${PORT}"
    echo -e "  WebUI  : http://localhost:${WEBUI_PORT}"
    echo -e "  Log    : ${LOG_FILE}"
    echo -e "  Data   : ${DATA_DIR}"

    # Create data dir
    mkdir -p "$DATA_DIR"

    # Launch in background
    cd "$SCRIPT_DIR"
    nohup "$BINARY" "${args[@]}" >> "$LOG_FILE" 2>&1 &
    local new_pid=$!
    echo "$new_pid" > "$PID_FILE"

    # Wait and verify
    sleep 1
    if kill -0 "$new_pid" 2>/dev/null; then
        echo -e "${GREEN}[START]${NC} amind started (PID: ${new_pid})"

        # Quick health check
        sleep 1
        if command -v curl &>/dev/null; then
            local health
            health=$(curl -s --connect-timeout 2 "http://127.0.0.1:${PORT}/v1/health" 2>/dev/null || echo "")
            if [[ -n "$health" ]]; then
                echo -e "  Health : ${GREEN}${health}${NC}"
            fi
            # WebUI check
            local webui_check
            webui_check=$(curl -s --connect-timeout 2 -o /dev/null -w "%{http_code}" "http://127.0.0.1:${WEBUI_PORT}/" 2>/dev/null || echo "000")
            if [[ "$webui_check" == "200" ]]; then
                echo -e "  WebUI  : ${GREEN}http://localhost:${WEBUI_PORT}${NC}"
            else
                echo -e "  WebUI  : ${YELLOW}loading...${NC}"
            fi
        fi
    else
        echo -e "${RED}[START]${NC} amind failed to start!"
        echo -e "  Check log: ${LOG_FILE}"
        tail -5 "$LOG_FILE" 2>/dev/null | sed 's/^/  > /'
        rm -f "$PID_FILE"
        return 1
    fi
}

do_stop() {
    if ! pid=$(get_pid); then
        # PID file says not running, but check if port is occupied (orphan process)
        local port_pid
        port_pid=$(lsof -ti "tcp:${PORT}" 2>/dev/null || true)
        if [[ -n "$port_pid" ]]; then
            echo -e "${YELLOW}[STOP]${NC} No PID file, but port ${PORT} occupied by PID ${port_pid}"
            echo -e "  Killing orphan process..."
            kill "$port_pid" 2>/dev/null || true
            sleep 1
            if kill -0 "$port_pid" 2>/dev/null; then
                kill -9 "$port_pid" 2>/dev/null || true
            fi
            echo -e "${GREEN}[STOP]${NC} Orphan killed."
        else
            echo -e "${YELLOW}[STOP]${NC} amind is not running."
        fi
        return 0
    fi

    echo -e "${YELLOW}[STOP]${NC} Stopping amind (PID: ${pid})..."

    # Graceful shutdown via SIGTERM
    kill "$pid"

    local wait_count=0
    local max_wait=10
    while kill -0 "$pid" 2>/dev/null; do
        sleep 1
        wait_count=$((wait_count + 1))
        printf "\r  Waiting... %d/%ds" "$wait_count" "$max_wait"
        if [[ $wait_count -ge $max_wait ]]; then
            echo ""
            echo -e "${RED}[STOP]${NC} Graceful shutdown timed out, force killing..."
            kill -9 "$pid" 2>/dev/null || true
            sleep 1
            break
        fi
    done
    echo ""

    rm -f "$PID_FILE"
    echo -e "${GREEN}[STOP]${NC} amind stopped."
}

do_restart() {
    do_stop
    sleep 1
    do_start
}

do_status() {
    echo -e "${BOLD}amind Status${NC}"
    echo -e "────────────────────────────────"

    # Process status
    if pid=$(get_pid); then
        echo -e "  Process : ${GREEN}running${NC} (PID: ${pid})"

        # Memory/CPU usage
        if command -v ps &>/dev/null; then
            local rss cpu
            rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
            cpu=$(ps -o %cpu= -p "$pid" 2>/dev/null | tr -d ' ')
            if [[ -n "$rss" ]]; then
                local mem_mb=$((rss / 1024))
                echo -e "  Memory  : ${mem_mb} MB"
                echo -e "  CPU     : ${cpu}%"
            fi
        fi

        # Uptime
        if command -v ps &>/dev/null; then
            local etime
            etime=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
            if [[ -n "$etime" ]]; then
                echo -e "  Uptime  : ${etime}"
            fi
        fi

        # Health check
        if command -v curl &>/dev/null; then
            local health
            health=$(curl -s --connect-timeout 2 "http://127.0.0.1:${PORT}/v1/health" 2>/dev/null || echo "")
            if [[ -n "$health" ]]; then
                echo -e "  Health  : ${GREEN}${health}${NC}"
            else
                echo -e "  Health  : ${RED}unreachable${NC} (port ${PORT})"
            fi

            # Metrics
            local metrics
            metrics=$(curl -s --connect-timeout 2 "http://127.0.0.1:${PORT}/v1/metrics" 2>/dev/null || echo "")
            if [[ -n "$metrics" ]]; then
                echo -e "  Metrics : ${metrics}"
            fi
        fi
    else
        echo -e "  Process : ${RED}stopped${NC}"
    fi

    # Config
    echo -e "  Config  : ${CONFIG}"
    echo -e "  Port    : ${PORT}"
    echo -e "  WebUI   : http://localhost:${WEBUI_PORT}"
    echo -e "  Data    : ${DATA_DIR}"
    echo -e "  Log     : ${LOG_FILE}"

    # Data dir size
    if [[ -d "$DATA_DIR" ]]; then
        local data_size
        data_size=$(du -sh "$DATA_DIR" 2>/dev/null | cut -f1)
        echo -e "  DataSize: ${data_size}"
    fi

    # Log file size
    if [[ -f "$LOG_FILE" ]]; then
        local log_size
        log_size=$(du -sh "$LOG_FILE" 2>/dev/null | cut -f1)
        echo -e "  LogSize : ${log_size}"
    fi
}

do_health() {
    if ! command -v curl &>/dev/null; then
        echo -e "${RED}curl not found${NC}"
        return 1
    fi

    echo -e "${BOLD}Health Check${NC}"
    echo -e "────────────────────────────────"

    # /v1/health
    local health
    health=$(curl -s --connect-timeout 3 "http://127.0.0.1:${PORT}/v1/health" 2>/dev/null || echo "UNREACHABLE")
    echo -e "  /v1/health  : ${health}"

    # /v1/metrics
    local metrics
    metrics=$(curl -s --connect-timeout 3 "http://127.0.0.1:${PORT}/v1/metrics" 2>/dev/null || echo "UNREACHABLE")
    echo -e "  /v1/metrics : ${metrics}"

    # WebUI
    local webui_code
    webui_code=$(curl -s --connect-timeout 3 -o /dev/null -w "%{http_code}" "http://127.0.0.1:${WEBUI_PORT}/" 2>/dev/null || echo "000")
    echo -e "  WebUI (:${WEBUI_PORT}) : HTTP ${webui_code}"
}

do_logs() {
    local lines="${2:-50}"
    if [[ ! -f "$LOG_FILE" ]]; then
        echo -e "${YELLOW}No log file found at ${LOG_FILE}${NC}"
        return 0
    fi

    if [[ "${2:-}" == "-f" || "${2:-}" == "--follow" ]]; then
        echo -e "${CYAN}[LOGS]${NC} Following ${LOG_FILE} (Ctrl+C to stop)..."
        tail -f "$LOG_FILE"
    else
        echo -e "${CYAN}[LOGS]${NC} Last ${lines} lines of ${LOG_FILE}:"
        echo -e "────────────────────────────────"
        tail -n "$lines" "$LOG_FILE"
    fi
}

do_test() {
    echo -e "${YELLOW}[TEST]${NC} Running tests..."
    if [[ ! -d "$BUILD_DIR" ]]; then
        do_build
    fi
    cd "$BUILD_DIR"
    cmake --build . -j"$(ncpu)" 2>&1 | tail -3
    ctest --output-on-failure 2>&1
}

do_clean() {
    echo -e "${YELLOW}[CLEAN]${NC} Cleaning..."
    if [[ -d "$BUILD_DIR" ]]; then
        rm -rf "${BUILD_DIR:?}"
        echo -e "  Removed: ${BUILD_DIR}"
    fi
    if [[ -f "$LOG_FILE" ]]; then
        rm -f "$LOG_FILE"
        echo -e "  Removed: ${LOG_FILE}"
    fi
    if [[ -f "$PID_FILE" ]]; then
        rm -f "$PID_FILE"
        echo -e "  Removed: ${PID_FILE}"
    fi
    echo -e "${GREEN}[CLEAN]${NC} Done."
}

# ── Main Dispatch ───────────────────────────────────────────────────────────

case "${1:-help}" in
    start)    do_start ;;
    stop)     do_stop ;;
    restart)  do_restart ;;
    status)   do_status ;;
    build)    do_build ;;
    rebuild)  do_rebuild ;;
    health)   do_health ;;
    logs)     do_logs "$@" ;;
    test)     do_test ;;
    clean)    do_clean ;;
    *)
        echo -e "${BOLD}amind${NC} — Agent's Mind Service Control"
        echo ""
        echo -e "Usage: ${CYAN}$0${NC} <command>"
        echo ""
        echo -e "${BOLD}Service Commands:${NC}"
        echo "  start       Start amind server (auto-builds if needed)"
        echo "  stop        Stop amind gracefully (SIGTERM → 10s → SIGKILL)"
        echo "  restart     Stop then start"
        echo "  status      Show process, memory, health, metrics"
        echo "  health      Quick health + metrics check"
        echo ""
        echo -e "${BOLD}Build Commands:${NC}"
        echo "  build       Incremental Release build"
        echo "  rebuild     Clean + full Release build"
        echo "  test        Build and run all tests"
        echo "  clean       Remove build/, logs, pid file"
        echo ""
        echo -e "${BOLD}Debug Commands:${NC}"
        echo "  logs        Show last 50 lines of log"
        echo "  logs -f     Follow log in real-time"
        echo "  logs 100    Show last 100 lines"
        echo ""
        echo -e "${BOLD}Examples:${NC}"
        echo "  $0 build && $0 start    # Build and start"
        echo "  $0 status               # Check everything"
        echo "  $0 logs -f              # Watch logs live"
        echo "  $0 restart              # Restart after code change"
        ;;
esac
