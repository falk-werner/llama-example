cmake_minimum_required(VERSION 4.2)
project(llama-example)

set(CMAKE_CXX_STANDARD 17)

find_package(PkgConfig REQUIRED)
find_package(nlohmann_json REQUIRED)
pkg_check_modules(LLAMA REQUIRED IMPORTED_TARGET llama)

add_executable(gguf-info src/gguf_info.cpp)
target_link_libraries(gguf-info PRIVATE PkgConfig::LLAMA nlohmann_json::nlohmann_json)

add_executable(llm_relation src/llm_relation.cpp)
target_link_libraries(llm_relation PRIVATE PkgConfig::LLAMA)

add_executable(categorize src/categorize.cpp)
target_link_libraries(categorize PRIVATE PkgConfig::LLAMA nlohmann_json::nlohmann_json)
