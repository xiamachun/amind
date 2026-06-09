// amind — Agent's Mind
// Production-grade memory engine for AI Agents

#include "server/config.h"
#include "server/engine.h"
#include "server/rest_server.h"
#include "web/web_server.h"

#include <csignal>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <spdlog/spdlog.h>

static std::atomic<bool> g_running{true};
static amind::RestServer* g_server = nullptr;

static void signalHandler(int sig) {
    spdlog::info("Received signal {}, shutting down...", sig);
    g_running = false;
    if (g_server) g_server->stop();
}

int main(int argc, char* argv[]) {
    // Default config path
    std::string config_path = "amind.conf";
    if (argc > 1) {
        config_path = argv[1];
    }

    // Setup logging
    spdlog::set_level(spdlog::level::info);
    spdlog::info("amind v{} — Agent's Mind", amind::AMIND_VERSION);
    spdlog::info("Loading config from: {}", config_path);

    // Load config
    auto config_result = amind::AppConfig::load(config_path);
    if (!config_result.ok()) {
        spdlog::error("Failed to load config: {}", config_result.error().toString());
        return 1;
    }

    // Set log level from config
    auto log_level = config_result->get("log_level", "info");
    if (log_level == "debug") spdlog::set_level(spdlog::level::debug);
    else if (log_level == "warn") spdlog::set_level(spdlog::level::warn);
    else if (log_level == "error") spdlog::set_level(spdlog::level::err);

    // Create engine
    amind::Engine engine(std::move(*config_result), config_path);
    auto init_result = engine.init();
    if (!init_result.ok()) {
        spdlog::error("Engine init failed: {}", init_result.error().toString());
        return 1;
    }

    // Setup signal handlers
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    // Ignore SIGPIPE — writes to a closed socket return EPIPE instead of
    // killing the process. Without this, a remote peer closing during a
    // long LLM stream silently terminates amind.
    signal(SIGPIPE, SIG_IGN);

    // Last-ditch handler for uncaught C++ exceptions — log + abort instead
    // of std::terminate's silent default. Helps post-mortem debugging.
    std::set_terminate([]() {
        try {
            auto e = std::current_exception();
            if (e) std::rethrow_exception(e);
            spdlog::critical("std::terminate called with no exception");
        } catch (const std::exception& ex) {
            spdlog::critical("Uncaught exception: {}", ex.what());
        } catch (...) {
            spdlog::critical("Uncaught non-std exception");
        }
        std::abort();
    });

    // Start REST server
    auto host = engine.config().get("host", "0.0.0.0");
    auto port = engine.config().getInt("port", 8080);


    // Start WebUI server (background thread)
    std::unique_ptr<amind::WebServer> web_server;
    if (engine.config().get("webui_enabled", "true") == "true") {
        auto webui_port = engine.config().getInt("webui_port", 11011);
        auto webui_root = engine.config().get("webui_root", "web/dist");
        auto api_token = engine.config().get("api_token", "");
        auto max_web_conn = engine.config().getInt("max_web_connections", 256);
        web_server = std::make_unique<amind::WebServer>(webui_root, webui_port, port, api_token, max_web_conn);
        web_server->startAsync();
    }

    auto max_conn = engine.config().getInt("max_connections", 128);
    auto req_timeout = engine.config().getInt("request_timeout_ms", 30000);
    amind::RestServer server(engine, host, port, max_conn, req_timeout);
    g_server = &server;
    spdlog::info("Starting REST server on {}:{}...", host, port);

    auto server_result = server.start();
    g_server = nullptr;
    if (!server_result.ok()) {
        spdlog::error("Server failed: {}", server_result.error().toString());
    }

    // Graceful shutdown: stop WebUI first, then engine
    if (web_server) web_server->stop();
    engine.shutdown();
    spdlog::info("amind shutdown complete");
    return 0;
}
