# amind Python SDK

Python client for [amind](../../README.md) — Agent's Mind memory engine.

## Installation

```bash
cd sdk/python
pip install -e .
```

## Quick Start

```python
from amind import AmindClient

client = AmindClient(host="127.0.0.1", port=8080)

# Store a memory
resp = client.store("User prefers dark theme", agent_id="agent-1", owner="user")
print(f"Stored: {resp.memory_ids}")

# Recall memories
results = client.recall("What theme does the user like?", agent_id="agent-1", top_k=5)
for mem in results:
    print(f"  [{mem.score:.2f}] {mem.content}")

# Session management
session = client.session_start(agent_id="agent-1")
client.session_turn(session.session_id, user_input="Hi!", agent_response="Hello!")
client.session_close(session.session_id)

client.close()
```

## Async Usage

```python
import asyncio
from amind import AsyncAmindClient

async def main():
    async with AsyncAmindClient() as client:
        results = await client.recall("user preferences")
        for mem in results:
            print(mem.content)

asyncio.run(main())
```

## API Reference

See `amind/client.py` for the full synchronous API and `amind/async_client.py` for the async variant.

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/
```
