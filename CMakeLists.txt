set(PN nonamengine)
cmake_minimum_required(VERSION 3.20)

project(${PN}
        VERSION 0.1.0
        LANGUAGES CXX)

find_package(nlohmann_json 3.11 REQUIRED)
find_package(CLI11 REQUIRED)
include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(nn
    src/model.cpp
    src/nn.cpp
    src/tensor.cpp
    src/utility.cpp
)

target_include_directories(nn PUBLIC
    src/include
)

target_link_libraries(nn PUBLIC
    nlohmann_json::nlohmann_json
)

add_executable(${PN}
    src/main.cpp
)

target_link_libraries(${PN} PRIVATE
    m
    nn
)

add_executable(nonamengine_benchmark benchmarks/main.cpp benchmarks/runner.cpp src/benchmarking.cpp)
target_include_directories(nonamengine_benchmark PRIVATE src)
target_link_libraries(nonamengine_benchmark PRIVATE nn CLI11::CLI11)
# Keep the observer in its own compilation unit so results remain observable.
set_target_properties(nonamengine_benchmark PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
target_compile_definitions(nonamengine_benchmark PRIVATE
    NONAMENGINE_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}"
    NONAMENGINE_BUILD_TYPE="$<CONFIG>")

if (MSVC)
    target_compile_options(${PN} PRIVATE /W4 /WX)
else()
    target_compile_options(${PN} PRIVATE -Wall -Wextra -Wpedantic -Werror -g)
endif()

if(BUILD_TESTING)
    foreach(case help quick matmul training missing_filter zero_samples zero_iterations negative_warmup)
        add_test(NAME benchmark_cli_${case} COMMAND ${CMAKE_COMMAND}
            -DPROGRAM=$<TARGET_FILE:nonamengine_benchmark> -DCASE=${case}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark_cli_test.cmake)
        set_tests_properties(benchmark_cli_${case} PROPERTIES TIMEOUT 35)
    endforeach()
    add_executable(benchmark_runner_test tests/benchmark_runner_test.cpp
        benchmarks/runner.cpp src/benchmarking.cpp)
    target_include_directories(benchmark_runner_test PRIVATE src src/include)
    foreach(case statistics validation repetitions exceptions throughput)
        add_test(NAME benchmark_runner_${case} COMMAND benchmark_runner_test ${case})
    endforeach()
    add_executable(benchmarking_test tests/benchmarking_test.cpp src/benchmarking.cpp)
    target_include_directories(benchmarking_test PRIVATE src/include)
    foreach(case result_contract clear empty single order batches names ownership
                 instances growth timing durations end_without_start double_end
                 nested_start active_retrieval)
        add_test(NAME benchmarking_${case} COMMAND benchmarking_test ${case})
        set_tests_properties(benchmarking_${case} PROPERTIES LABELS benchmarking TIMEOUT 10)
    endforeach()
    foreach(case help load missing arguments iterations_zero iterations_negative
                 batch-size_zero batch-size_negative)
        add_test(NAME cli_${case} COMMAND ${CMAKE_COMMAND}
            -DPROGRAM=$<TARGET_FILE:nonamengine>
            -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
            -DCASE=${case}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_test.cmake)
    endforeach()
    add_test(NAME cli_training COMMAND ${CMAKE_COMMAND}
        -DPROGRAM=$<TARGET_FILE:nonamengine>
        -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -DBINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_training_test.cmake)
    set_tests_properties(cli_training PROPERTIES TIMEOUT 100)
    add_executable(tensor_test tests/tensor_test.cpp)
    target_link_libraries(tensor_test PRIVATE nn)
    foreach(case construction arithmetic structural)
        add_test(NAME tensor_${case} COMMAND tensor_test ${case})
    endforeach()
    add_executable(nn_test tests/nn_test.cpp)
    target_link_libraries(nn_test PRIVATE nn)
    foreach(case dense activations loss gradients training nonlinear_training utilities)
        add_test(NAME nn_${case} COMMAND nn_test ${case})
    endforeach()
    add_executable(model_load_test tests/model_load_test.cpp)
    target_link_libraries(model_load_test PRIVATE nn)
    foreach(case loading malformed exporting)
        add_test(NAME model_${case} COMMAND model_load_test ${case})
    endforeach()
    add_executable(model_io_test tests/model_io_test.cpp)
    target_link_libraries(model_io_test PRIVATE nn)
    foreach(case reading writing offsets closing)
        add_test(NAME model_io_${case} COMMAND model_io_test ${case})
    endforeach()
    add_executable(model_export_test
        tests/model_export_test.cpp
    )
    target_link_libraries(model_export_test PRIVATE nn)
    add_test(NAME model_export COMMAND model_export_test)
endif()
