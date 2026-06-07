<p align="center">
  <img src=".github/images/logo.png" alt="Amind Logo" width="200">
</p>

<h1 align="center">Amind — Agent's Mind</h1>

<p align="center">
  <a href="https://isocpp.org/"><img src="https://img.shields.io/badge/C%2B%2B-20-blue.svg" alt="C++20"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License"></a>
  <img src="https://img.shields.io/badge/version-v0.1-orange.svg" alt="Version">
</p>

<p align="center"><b>Production-grade memory engine for AI Agents.</b></p>

<p align="center"><a href="README_CN.md">中文文档</a></p>

Amind gives AI Agents a persistent, version-tracked memory system with metacognitive capabilities. Unlike simple vector databases, Amind understands memory ownership, tracks version history, detects conflicts, and can self-assess "what I know and what I don't know."

## Key Features

- **Owner-based Memory Classification** — User / Project / Agent / Session / Shared ownership types
- **Two-phase Async Storage** — Stage 1 fast write ~1ms; Stage 2 background LLM refinement / dedup / conflict detection
- **Version History Chain** — Each update creates a new version linked by parent_id, fully traceable
- **Conflict Detection & Resolution** — Semantic similarity + LLM judgment, six-level resolution strategy
- **Intent-aware Retrieval** — LLM analyzes query intent → multi-path parallel retrieval → RRF score fusion
- **Metacognition** — Coverage statistics, confidence map, proactive recall suggestions
- **Lineage Tracking** — Raw/Derived dual-layer memory + reverse index + cascade invalidation propagation
- **Write Gate** — Quality filtering + duplicate detection + shadow mode for gradual rollout
- **Forget Engine** — 7-signal weighted scoring + audit log + automatic GC
- **Consolidation Worker** — Periodic consolidation (Top-K promotion + cross-session dedup + drift check)
- **Feature Gate** — Runtime feature toggles + global shadow mode + emergency switch
- **Freshness Barrier** — Recall waits for pending reconcile, ensuring real-time consistency
- **Provider Abstraction** — LLM/Embedding/Rerank interfaces decoupled, supports Ollama/OpenAI/Anthropic and more
- **Bearer Token Auth** — API Key management + SHA-256 hash verification, multi-key rotation
- **Built-in WebUI Dashboard** — Visual browsing of memories, knowledge graph, audit logs and system status
- **Zero External HTTP Dependencies** — Pure POSIX socket REST server implementation

## Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│  API          REST Server  │  WebUI  │  Auth             │
├──────────────────────────────────────────────────────────┤
│  Intelligence   Capture Pipeline │ Retrieval │ MetaCog   │
├──────────────────────────────────────────────────────────┤
│  Async          TaskQueue  │  TaskExecutor (thread pool)  │
├──────────────────────────────────────────────────────────┤
│  Index          HNSW Vector │ LSM-Tree │ Graph Store     │
├──────────────────────────────────────────────────────────┤
│  Persist        WAL │ MemTable │ SSTable │ PageCache     │
└──────────────────────────────────────────────────────────┘
```

## Quick Start: Download Pre-built Binary

No compilation needed — download and run:

```bash
# Linux x86_64
curl -LO https://github.com/xiamachun/amind/releases/latest/download/amind-v0.1.0-linux-amd64.tar.gz
tar xzf amind-v0.1.0-linux-amd64.tar.gz && cd amind-v0.1.0-linux-amd64

# Initialize config
cp amind.conf.example amind.conf
vim amind.conf  # Fill in your LLM API Key and Embedding config

# Start
./amind.sh start
./amind.sh status    # Check running status
# WebUI: http://localhost:11011
```

<details>
<summary>macOS Users</summary>

```bash
# macOS Apple Silicon (M1/M2/M3/M4)
curl -LO https://github.com/xiamachun/amind/releases/latest/download/amind-v0.1.0-darwin-arm64.tar.gz
tar xzf amind-v0.1.0-darwin-arm64.tar.gz && cd amind-v0.1.0-darwin-arm64

# Remove macOS Gatekeeper quarantine flag (required on first download)
xattr -cr .

# Initialize config and start
cp amind.conf.example amind.conf
vim amind.conf  # Fill in your LLM API Key and Embedding config
./amind.sh start
```

> If you still see a "cannot verify developer" dialog, click "Done" then go to **System Settings → Privacy & Security** and click "Allow Anyway" at the bottom.

</details>

<details>
<summary>All platforms</summary>

| Platform | Download | Use Case |
|----------|----------|----------|
| **Linux x86_64** | [amind-v0.1.0-linux-amd64.tar.gz](https://github.com/xiamachun/amind/releases/latest/download/amind-v0.1.0-linux-amd64.tar.gz) | Cloud servers, VPS, Docker |
| **Linux ARM64** | [amind-v0.1.0-linux-arm64.tar.gz](https://github.com/xiamachun/amind/releases/latest/download/amind-v0.1.0-linux-arm64.tar.gz) | AWS Graviton, Raspberry Pi, Oracle ARM |
| **macOS Apple Silicon** | [amind-v0.1.0-darwin-arm64.tar.gz](https://github.com/xiamachun/amind/releases/latest/download/amind-v0.1.0-darwin-arm64.tar.gz) | M1/M2/M3/M4 Mac |

> **System requirements:** Linux (Ubuntu 20.04+ / Debian 11+ / CentOS 9+ / RHEL 9+), macOS 12+.

</details>

## Build from Source

### Prerequisites

- macOS 12+ or Linux (Ubuntu 22.04+ / Debian 12+ / CentOS Stream 9+)
- C++20 compiler (Clang 14+ / GCC 11+)
- CMake 3.20+

### Install Dependencies

<details>
<summary><b>macOS (Homebrew)</b></summary>

```bash
brew install spdlog nlohmann-json xxhash crc32c msgpack-cxx lz4 \
    googletest google-benchmark openssl
```

</details>

<details>
<summary><b>Ubuntu 22.04+ / Debian 12+</b></summary>

```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake g++ \
    libgtest-dev libxxhash-dev libcrc32c-dev \
    libmsgpack-dev nlohmann-json3-dev \
    libspdlog-dev libfmt-dev liblz4-dev libssl-dev
```

</details>

<details>
<summary><b>CentOS Stream 9 / RHEL 9</b></summary>

```bash
# Enable EPEL and CRB repositories
sudo dnf install -y gcc-c++ cmake make epel-release
sudo dnf config-manager --set-enabled crb

# Install available dependencies
sudo dnf install -y \
    spdlog-devel json-devel xxhash-devel msgpack-devel \
    lz4-devel gtest-devel openssl-devel

# crc32c is not available via dnf — build from source
cd /tmp && git clone https://github.com/google/crc32c.git
cd crc32c && git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCRC32C_BUILD_TESTS=OFF -DCRC32C_BUILD_BENCHMARKS=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_SHARED_LIBS=ON
make -j$(nproc) && sudo make install && sudo ldconfig
```

</details>

### Configure

```bash
cp amind.conf.example amind.conf
# Edit amind.conf with your LLM and Embedding provider settings
# Required: llm_host, llm_api_key, embedding_host, embedding_model
```

### Build

<details>
<summary><b>macOS (Apple Silicon / M-series)</b></summary>

> **Important:** If both Intel Homebrew (`/usr/local`) and ARM Homebrew (`/opt/homebrew`) are installed, you **must** explicitly specify the ARM paths to avoid architecture mismatch linker errors.

```bash
git clone <repo-url> amind && cd amind
mkdir -p build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMIND_ENABLE_TLS=ON \
    -DBREW_PREFIX=/opt/homebrew \
    -DOPENSSL_ROOT_DIR=/opt/homebrew/opt/openssl@3 \
    -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_PREFIX_PATH=/opt/homebrew

make -j$(sysctl -n hw.ncpu)
```

</details>

<details>
<summary><b>macOS (Intel)</b></summary>

```bash
git clone <repo-url> amind && cd amind
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DAMIND_ENABLE_TLS=ON
make -j$(sysctl -n hw.ncpu)
```

</details>

<details>
<summary><b>Ubuntu / Debian</b></summary>

```bash
git clone <repo-url> amind && cd amind
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMIND_BUILD_BENCHMARKS=OFF \
    -DAMIND_ENABLE_TLS=ON \
    -DBREW_PREFIX=/usr
make -j$(nproc)
```

</details>

<details>
<summary><b>CentOS Stream 9 / RHEL 9</b></summary>

```bash
git clone <repo-url> amind && cd amind
mkdir -p build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DAMIND_BUILD_BENCHMARKS=OFF \
    -DAMIND_ENABLE_TLS=ON \
    -DBREW_PREFIX=/usr
make -j$(nproc)
```

</details>

<details>
<summary><b>Convenience script (all platforms)</b></summary>

```bash
./amind.sh build
```

</details>

### Build Notes

| Platform | Notes |
|----------|-------|
| **Linux (GCC 11)** | Minimum GCC 11 required for C++20; Ubuntu 22.04 ships GCC 11 by default |
| **Linux** | `BREW_PREFIX` auto-detects to `/usr`; override manually with `-DBREW_PREFIX=/usr` if needed |
| **Linux** | Set `AMIND_BUILD_BENCHMARKS=OFF` unless Google Benchmark is manually installed |
| **macOS (Apple Silicon)** | Must specify `-DBREW_PREFIX=/opt/homebrew -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_PREFIX_PATH=/opt/homebrew` if dual Homebrew is present |
| **macOS (Intel)** | Homebrew path auto-detected (`/usr/local`), no extra flags needed |
| **TLS/HTTPS** | If your LLM provider uses HTTPS (e.g., OpenAI), you must enable `-DAMIND_ENABLE_TLS=ON` |

### Run

```bash
# Direct execution
./build/amind amind.conf

# Or use the service script
./amind.sh start     # Start in background
./amind.sh status    # Health check
./amind.sh stop      # Graceful shutdown
```

### API Examples

All API endpoints require Bearer Token authentication (except `/v1/health`):

```bash
# Set your token (matches api_token in amind.conf)
TOKEN="your-api-token"

# Health check (no auth required)
curl http://localhost:8080/v1/health

# Store a memory
curl -X POST http://localhost:8080/v1/memories \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"agent-1","user_id":"user-1","scope":"private","memory_type":"preference","content":"User prefers dark theme","importance":0.8}'

# Recall memories
curl -X POST http://localhost:8080/v1/memories/recall \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"agent-1","user_id":"user-1","scope":"private","memory_type":"preference","query":"What are the user UI preferences?","top_k":5}'

# View version history
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/v1/memories/123456789/history

# Metacognition — conflict detection
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/v1/metacognition/conflicts?agent_id=agent-1
```

#### API Parameter Migration Guide

| Old Parameter | New Parameters | Description |
|---------------|----------------|-------------|
| `namespace` | `agent_id` + `user_id` | 实现多用户多 Agent 物理隔离。`agent_id` 标识 Agent，`user_id` 标识用户 |
| `owner` (user/agent/system) | `scope` + `memory_type` | `scope`: `private`（私有）或 `agent_shared`（Agent 共享）；`memory_type`: `preference`/`fact`/`domain_knowledge`/`procedure`/`episodic`/`ephemeral` |

> 详见 [CHANGELOG](docs/CHANGELOG.md) 了解完整的变更历史。

<details>
<summary>Full API Reference</summary>

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/health` | Health check (no auth) |
| GET | `/v1/metrics` | System metrics |
| POST | `/v1/memories` | Store a memory |
| POST | `/v1/memories/recall` | Semantic recall |
| GET | `/v1/memories/{id}` | Get a single memory |
| GET | `/v1/memories/{id}/history` | Version history |
| POST | `/v1/memories/{id}/feedback` | Feedback / correction |
| DELETE | `/v1/memories/namespace/{ns}` | Bulk delete by namespace |
| POST | `/v1/intercept` | Conversation intercept (auto-extract + inject memories) |
| POST | `/v1/sessions/start` | Start a session |
| GET | `/v1/metacognition/conflicts` | Conflict detection |
| GET | `/v1/metacognition/coverage` | Coverage statistics |
| GET | `/v1/pipeline/stats` | Pipeline statistics |
| GET | `/v1/pipeline/reconcile-log` | Reconciliation log |
| GET | `/v1/gate/log` | WriteGate audit log |
| GET | `/v1/forget/log` | Forget engine audit log |
| POST | `/v1/admin/forget/run` | Manually trigger forget GC |
| POST | `/v1/admin/consolidation/run` | Manually trigger consolidation |
| GET | `/v1/backup/export` | Data export |
| POST | `/v1/backup/import` | Data import |
| POST/GET/DELETE | `/v1/auth/keys` | API Key management |
| GET | `/v1/variables` | Runtime variable inspection |
| PUT | `/v1/variables/{name}` | Runtime variable modification |
| POST | `/v1/config/reload` | Hot-reload configuration |

</details>


## Configuration

See [amind.conf.example](amind.conf.example) for the full configuration template. Key settings:

| Setting | Description |
|---------|-------------|
| `listen_port` | REST server port (default 8080) |
| `api_token` | Master auth token (**change the default!**) |
| `llm_host` / `llm_api_key` | LLM provider host and API key |
| `embedding_host` / `embedding_model` | Embedding model configuration |
| `write_gate_enabled` | Enable write gate |
| `reconcile_enabled` | Enable reconciler |
| `forget_score_enabled` | Enable forget engine |
| `consolidation_enabled` | Enable consolidation worker |

All settings can be overridden via `AMIND_*` environment variables.



## License

MIT — See [LICENSE](LICENSE)
