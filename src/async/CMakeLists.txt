# ── amind Async Library ───────────────────────────────────────────────────
add_library(amind_async
    task_queue.cpp
    task_executor.cpp
)

target_include_directories(amind_async
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(amind_async
    PUBLIC
        amind_core
        spdlog::spdlog
)
