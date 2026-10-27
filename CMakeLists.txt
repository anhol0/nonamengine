set(PN nonamengine)
cmake_minimum_required(VERSION 3.20)

project(${PN}
        VERSION 0.1.0
        LANGUAGES CXX)

find_package(nlohmann_json 3.11 REQUIRED)
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

if (MSVC)
    target_compile_options(${PN} PRIVATE /W4 /WX)
else()
    target_compile_options(${PN} PRIVATE -Wall -Wextra -Wpedantic -Werror -g)
endif()

if(BUILD_TESTING)
    add_executable(tensor_test tests/tensor_test.cpp)
    target_link_libraries(tensor_test PRIVATE nn)
    foreach(case construction arithmetic structural)
        add_test(NAME tensor_${case} COMMAND tensor_test ${case})
    endforeach()
    add_executable(nn_test tests/nn_test.cpp)
    target_link_libraries(nn_test PRIVATE nn)
    foreach(case dense activations loss gradients training utilities)
        add_test(NAME nn_${case} COMMAND nn_test ${case})
    endforeach()
    add_executable(model_export_test
        tests/model_export_test.cpp
    )
    target_link_libraries(model_export_test PRIVATE nn)
    add_test(NAME model_export COMMAND model_export_test)
endif()
