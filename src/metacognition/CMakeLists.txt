# ── amind MetaCognition Library ───────────────────────────────────────────
add_library(amind_metacognition
    metacognition.cpp
)

target_include_directories(amind_metacognition
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/..
)

target_link_libraries(amind_metacognition
    PUBLIC
        amind_core
        amind_memory
        amind_graph
)
