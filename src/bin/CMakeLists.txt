# Configure server
add_executable(
    hyriseServer

    server.cpp
)
target_link_libraries(
    hyriseServer
    hyrise
    hyriseBenchmarkLib
)

# Configure playground
add_executable(
    hyrisePlayground

    playground.cpp
)

target_link_libraries(
    hyrisePlayground
    hyrise
    hyriseBenchmarkLib
)

add_executable(
    hyriseBalkesenComparison

    balkesen_files/cpu_mapping.cpp
    balkesen_comparison.cpp
)

target_link_libraries(
    hyriseBalkesenComparison
    hyrise
    hyriseBenchmarkLib
)

add_executable(
    hyriseSimdSortBench

    sort_bench.cpp
)

target_link_libraries(
    hyriseSimdSortBench
    hyrise
    hyriseBenchmarkLib
)

target_include_directories(hyriseBalkesenComparison
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/balkesen_files
)

# Configure client
add_executable(
    hyriseClient

    client.cpp
)

target_link_libraries(
    hyriseClient
    hyrise
)

# Configure Console
add_executable(
    hyriseConsole

    console/console.cpp
    console/console.hpp
    console/pagination.cpp
    console/pagination.hpp
)

target_link_libraries(
    hyriseConsole

    PRIVATE
    hyriseBenchmarkLib
    ncurses
)

# Link readline as a system library to ignore '-Wreserved-identifier' warnings (compile errors due to -Werror)
target_link_libraries_system(
    hyriseConsole
    readline
)
