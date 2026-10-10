set(PN nonamengine)
cmake_minimum_required(VERSION 3.20)

project(${PN}
        VERSION 0.1.0
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(nn
    src/matrix.cpp
    src/model.cpp
    src/nn.cpp
    src/utility.cpp
)

add_executable(${PN}
    src/main.cpp
)

target_include_directories(${PN} PRIVATE
    src/include
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
