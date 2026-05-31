# proto/amind.proto

Interface specification for the amind memory engine.

## Current Status

- **Server**: REST only (POSIX socket, HTTP/1.1)
- **Proto**: Interface contract for SDK code generation and documentation

The `.proto` file defines the canonical request/response schemas. It can be used to:
1. Generate client stubs in Go, Python, Java, etc. (`protoc --<lang>_out`)
2. Serve as machine-readable API documentation
3. Validate request payloads in integration tests

## Why no gRPC server?

amind's primary consumer (pyclaw) uses HTTP. Adding a gRPC listener would require:
- grpc++ dependency (~50MB)
- Dual-server maintenance burden
- No current consumer needs binary framing or HTTP/2 streaming

If high-throughput inter-service communication becomes necessary (e.g., Go/Java
agents calling amind directly), the proto definitions are ready for server-side
implementation.
