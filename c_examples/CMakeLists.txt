cmake_minimum_required(VERSION 3.15)
include(FetchContent)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


project(
  KernelEvaluatorDemo
  VERSION 1.0
  LANGUAGES C)

# Get Corrosion, a CMake module to include Rust projects.

FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5 # Optionally specify a commit hash, version tag or branch here
)

FetchContent_MakeAvailable(Corrosion)

corrosion_import_crate(MANIFEST_PATH ../Cargo.toml CRATE_TYPES cdylib)

add_executable(evaluate evaluate.c)
add_executable(assemble assemble.c)

target_include_directories(evaluate PUBLIC ../include)
target_include_directories(assemble PUBLIC ../include)

target_link_libraries(evaluate PUBLIC green_kernels)
target_link_libraries(assemble PUBLIC green_kernels)



