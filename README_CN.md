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

<p align="center"><a href="README.md">English Documentation</a></p>

Amind 让 AI Agent 拥有持久化、可版本追踪、具备元认知能力的记忆系统。不同于简单的向量数据库，Amind 理解记忆的归属关系、追踪版本历史、检测冲突矛盾，并能自我评估"我知道什么、不知道什么"。

## 核心特性

- **Owner-based 记忆分类** — User / Project / Agent / Session / Shared 五种归属
- **两阶段异步存储** — Stage 1 快速写入 ~1ms，Stage 2 后台 LLM 精炼/去重/冲突检测
- **版本历史链** — 每次更新创建新版本，parent_id 链接，完整演化可追溯
- **冲突检测与消解** — 语义相似度 + LLM 判断，六级消解策略自动处理矛盾
- **意图感知检索** — LLM 分析查询意图 → 多路并行检索 → RRF 分数融合
- **元认知** — 覆盖率统计、可信度地图、主动回忆建议
- **谱系追踪** — Raw/Derived 双层记忆 + 反向索引 + 级联失效传播
- **写入门控** — 质量过滤 + 重复检测 + shadow mode 灰度验证
- **遗忘引擎** — 7 信号加权评分 + 审计日志 + 自动 GC
- **巩固工人** — 定期巩固（Top-K 升级 + 跨 session 去重 + 漂移校验）
- **灰度门控** — 运行时特性开关 + 全局 shadow mode + 紧急开关
- **Freshness Barrier** — Recall 前等待 pending reconcile，确保检索结果实时一致
- **Provider 抽象** — LLM/Embedding/Rerank 接口解耦，支持 Ollama/OpenAI/Anthropic 等
- **Bearer Token 认证** — API Key 管理 + SHA-256 哈希校验，支持多 Key 轮换
- **内置 WebUI 仪表盘** — 可视化浏览记忆、知识图谱、审计日志与系统状态
- **零外部 HTTP 依赖** — 纯 POSIX socket 实现 REST server

## 架构概览

```
┌──────────────────────────────────────────────────────────┐
│  API          REST Server  │  WebUI  │  Auth             │
├──────────────────────────────────────────────────────────┤
│  Intelligence   Capture Pipeline │ Retrieval │ MetaCog   │
├──────────────────────────────────────────────────────────┤
│  Async          TaskQueue  │  TaskExecutor (thread pool) │
├──────────────────────────────────────────────────────────┤
│  Index          HNSW Vector │ LSM-Tree │ Graph Store     │
├──────────────────────────────────────────────────────────┤
│  Persist        WAL │ MemTable │ SSTable │ PageCache     │
└──────────────────────────────────────────────────────────┘
```

## 快速开始：下载预编译版本

无需编译，直接下载运行：

```bash
# Linux x86_64
VERSION=$(curl -s https://api.github.com/repos/xiamachun/amind/releases/latest | grep '"tag_name"' | cut -d'"' -f4)
curl -LO "https://github.com/xiamachun/amind/releases/download/${VERSION}/amind-${VERSION}-linux-amd64.tar.gz"
tar xzf amind-${VERSION}-linux-amd64.tar.gz && cd amind-${VERSION}-linux-amd64

# 初始化配置
cp amind.conf.example amind.conf
vim amind.conf  # 填入你的 LLM API Key 和 Embedding 配置

# 启动
./amind.sh start
./amind.sh status    # 确认运行状态
# WebUI: http://localhost:11011
```

<details>
<summary>macOS 用户</summary>

```bash
# macOS Apple Silicon (M1/M2/M3/M4)
VERSION=$(curl -s https://api.github.com/repos/xiamachun/amind/releases/latest | grep '"tag_name"' | cut -d'"' -f4)
curl -LO "https://github.com/xiamachun/amind/releases/download/${VERSION}/amind-${VERSION}-darwin-arm64.tar.gz"
tar xzf amind-${VERSION}-darwin-arm64.tar.gz && cd amind-${VERSION}-darwin-arm64

# 清除 macOS Gatekeeper 隔离标记（首次下载需要）
xattr -cr .

# 初始化配置并启动
cp amind.conf.example amind.conf
vim amind.conf  # 填入你的 LLM API Key 和 Embedding 配置
./amind.sh start
```

> 如果仍然弹出"无法验证开发者"对话框，点击"完成"后去 **系统设置 → 隐私与安全性** 页面底部点击"仍然允许"。

</details>

<details>
<summary>所有平台下载</summary>

| 平台 | 下载 | 适用场景 |
|------|------|----------|
| **Linux x86_64** | [amind-linux-amd64.tar.gz](https://github.com/xiamachun/amind/releases/latest) | 云服务器、VPS、Docker |
| **Linux ARM64** | [amind-linux-arm64.tar.gz](https://github.com/xiamachun/amind/releases/latest) | AWS Graviton、树莓派、Oracle ARM |
| **macOS Apple Silicon** | [amind-darwin-arm64.tar.gz](https://github.com/xiamachun/amind/releases/latest) | M1/M2/M3/M4 Mac |

> **系统要求：** Linux（Ubuntu 20.04+ / Debian 11+ / CentOS 9+ / RHEL 9+），macOS 12+。

</details>

## 从源码构建

### 环境要求

- macOS 12+ 或 Linux (Ubuntu 22.04+ / Debian 12+ / CentOS Stream 9+)
- C++20 编译器 (Clang 14+ / GCC 11+)
- CMake 3.20+

### 安装依赖

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
# 启用 EPEL 和 CRB 仓库
sudo dnf install -y gcc-c++ cmake make epel-release
sudo dnf config-manager --set-enabled crb

# 安装可用依赖
sudo dnf install -y \
    spdlog-devel json-devel xxhash-devel msgpack-devel \
    lz4-devel gtest-devel openssl-devel

# crc32c 需要从源码编译安装
cd /tmp && git clone https://github.com/google/crc32c.git
cd crc32c && git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCRC32C_BUILD_TESTS=OFF -DCRC32C_BUILD_BENCHMARKS=OFF \
    -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_SHARED_LIBS=ON
make -j$(nproc) && sudo make install && sudo ldconfig
```

</details>

### 配置

```bash
cp amind.conf.example amind.conf
# 编辑 amind.conf，填入你的 LLM 和 Embedding provider 配置
# 必须配置: llm_host, llm_api_key, embedding_host, embedding_model
```

### 构建

<details>
<summary><b>macOS (Apple Silicon / M 系列)</b></summary>

> **重要：** 如果同时安装了 Intel Homebrew (`/usr/local`) 和 ARM Homebrew (`/opt/homebrew`)，**必须**显式指定 ARM 路径，否则会出现架构不匹配的链接错误。

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
<summary><b>便捷脚本（全平台）</b></summary>

```bash
./amind.sh build
```

</details>

### 编译注意事项

| 平台 | 注意点 |
|------|--------|
| **Linux (GCC 11)** | 最低需要 GCC 11 以支持 C++20；Ubuntu 22.04 自带 GCC 11 |
| **Linux** | `BREW_PREFIX` 会自动设为 `/usr`，也可手动指定 `-DBREW_PREFIX=/usr` |
| **Linux** | 关闭 `AMIND_BUILD_BENCHMARKS=OFF`，除非手动安装了 Google Benchmark |
| **macOS (Apple Silicon)** | 如存在双 Homebrew 环境，须指定 `-DBREW_PREFIX=/opt/homebrew -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_PREFIX_PATH=/opt/homebrew` |
| **macOS (Intel)** | Homebrew 路径自动检测 (`/usr/local`)，无需额外参数 |
| **TLS/HTTPS** | 如果 LLM provider 是 HTTPS（如 OpenAI），必须开启 `-DAMIND_ENABLE_TLS=ON` |

### 运行

```bash
# 直接运行
./build/amind amind.conf

# 或使用服务脚本
./amind.sh start     # 后台启动
./amind.sh status    # 健康检查
./amind.sh stop      # 优雅停止
```

### API 示例

所有 API 端点均需要 Bearer Token 认证（`/v1/health` 除外）：

```bash
# 设置 Token（对应 amind.conf 中的 api_token）
TOKEN="your-api-token"

# 健康检查（无需认证）
curl http://localhost:8080/v1/health

# 存储记忆
curl -X POST http://localhost:8080/v1/memories \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"agent-1","user_id":"user-1","scope":"private","memory_type":"preference","content":"用户偏好暗色主题","importance":0.8}'

# 检索记忆
curl -X POST http://localhost:8080/v1/memories/recall \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"agent_id":"agent-1","user_id":"user-1","scope":"private","memory_type":"preference","query":"用户的界面偏好是什么？","top_k":5}'

# 查看版本历史
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/v1/memories/123456789/history

# 元认知 — 冲突检测
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8080/v1/metacognition/conflicts?agent_id=agent-1
```

<details>
<summary>完整 API 列表</summary>

| 方法 | 端点 | 说明 |
|------|------|------|
| GET | `/v1/health` | 健康检查（免认证） |
| GET | `/v1/metrics` | 系统指标 |
| POST | `/v1/memories` | 存储记忆 |
| POST | `/v1/memories/recall` | 语义检索 |
| GET | `/v1/memories/{id}` | 获取单条记忆 |
| GET | `/v1/memories/{id}/history` | 版本历史 |
| POST | `/v1/memories/{id}/feedback` | 反馈/修正 |
| DELETE | `/v1/memories/namespace/{ns}` | 按 namespace 批量删除 |
| POST | `/v1/intercept` | 对话拦截（自动提取+注入记忆） |
| POST | `/v1/sessions/start` | 创建会话 |
| GET | `/v1/metacognition/conflicts` | 冲突检测 |
| GET | `/v1/metacognition/coverage` | 覆盖率统计 |
| GET | `/v1/pipeline/stats` | 管线统计 |
| GET | `/v1/pipeline/reconcile-log` | 调和日志 |
| GET | `/v1/gate/log` | WriteGate 审计日志 |
| GET | `/v1/forget/log` | 遗忘引擎审计日志 |
| POST | `/v1/admin/forget/run` | 手动触发遗忘 GC |
| POST | `/v1/admin/consolidation/run` | 手动触发巩固 |
| GET | `/v1/backup/export` | 数据导出 |
| POST | `/v1/backup/import` | 数据导入 |
| POST/GET/DELETE | `/v1/auth/keys` | API Key 管理 |
| GET | `/v1/variables` | 运行时变量查看 |
| PUT | `/v1/variables/{name}` | 运行时变量修改 |
| POST | `/v1/config/reload` | 热重载配置 |

</details>

## 配置说明

参见 [amind.conf.example](amind.conf.example) 获取完整配置模板。关键配置项：

| 配置项 | 说明 |
|--------|------|
| `listen_port` | REST 服务端口（默认 8080） |
| `api_token` | 主认证 Token（**请务必修改默认值**） |
| `llm_host` / `llm_api_key` | LLM 提供商地址和密钥 |
| `embedding_host` / `embedding_model` | Embedding 模型配置 |
| `write_gate_enabled` | 是否启用写入门控 |
| `reconcile_enabled` | 是否启用调和器 |
| `forget_score_enabled` | 是否启用遗忘引擎 |
| `consolidation_enabled` | 是否启用巩固工人 |

支持 `AMIND_*` 环境变量覆盖配置文件中的值。

## License

MIT — 详见 [LICENSE](LICENSE)
