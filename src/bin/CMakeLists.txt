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

add_executable(
    hyriseSIMDMergeTest
    simd_sort_test.cpp
)

add_executable(
    hyriseBenchmarkSIMDMerge
    merge_benchmark.cpp
)

add_executable(
    hyriseBenchmarkPartitioning
    partitioning_benchmark.cpp
)

target_link_libraries(
    hyrisePlayground
    hyrise
    hyriseBenchmarkLib
)

target_link_libraries(
    hyriseBenchmarkPartitioning
    hyrise
    hyriseBenchmarkLib
)

target_link_libraries(
    hyriseSIMDMergeTest
    hyrise
    hyriseBenchmarkLib
)

target_link_libraries(
    hyriseBenchmarkSIMDMerge
    hyrise
    hyriseBenchmarkLib
)

add_executable(
    hyriseBenchmarkSIMD
    cxxopts.hpp
    benchmark.cpp
)

target_link_libraries(
    hyriseBenchmarkSIMD
    hyrise
    hyriseBenchmarkLib
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
