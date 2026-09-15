cmake_minimum_required(VERSION 3.12)
project(g1_run_policy_test)

# 离线对拍程序，不依赖 DDS / 机器人。构建:
#   cd deploy/robots/g1_29dof/test && mkdir -p build && cd build && cmake .. && make
# 运行:
#   ./test_run_policy ../../config/policy/run/0908

set(CMAKE_CXX_STANDARD 17)

find_package(yaml-cpp REQUIRED)

set(DEPLOY_DIR ${PROJECT_SOURCE_DIR}/../../..)
set(ORT_DIR ${DEPLOY_DIR}/thirdparty/onnxruntime-linux-x64-1.22.0)

include_directories(
  ${PROJECT_SOURCE_DIR}/../include
  /usr/include/eigen3
  ${DEPLOY_DIR}/include
  ${ORT_DIR}/include
)

add_executable(test_run_policy test_run_policy.cpp)
target_link_libraries(test_run_policy yaml-cpp fmt pthread ${ORT_DIR}/lib/libonnxruntime.so.1.22.0)
set_target_properties(test_run_policy PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)

# 手柄路径行为测试：./test_gait_command ../../config/policy/run/0908/params/deploy.yaml
add_executable(test_gait_command test_gait_command.cpp)
target_link_libraries(test_gait_command yaml-cpp fmt pthread)

enable_testing()
add_executable(test_walk_handoff test_walk_handoff.cpp)
target_link_libraries(test_walk_handoff pthread)
add_test(NAME walk_handoff COMMAND test_walk_handoff)

add_executable(test_walk_policy test_walk_policy.cpp)
target_link_libraries(test_walk_policy yaml-cpp fmt pthread ${ORT_DIR}/lib/libonnxruntime.so.1.22.0)
set_target_properties(test_walk_policy PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
add_test(NAME walk_policy COMMAND test_walk_policy
  ${PROJECT_SOURCE_DIR}/../config/policy/velocity
  ${PROJECT_SOURCE_DIR}/../config/policy/mimic/jump3/params/jump3_waist15.csv)
add_test(NAME gait_command COMMAND test_gait_command ${PROJECT_SOURCE_DIR}/../config/policy/run/0908/params/deploy.yaml)

find_package(Boost REQUIRED COMPONENTS program_options)
add_executable(test_state_walk test_state_walk.cpp ../src/State_Walk.cpp)
target_include_directories(test_state_walk BEFORE PRIVATE ${PROJECT_SOURCE_DIR}/fakes)
target_link_libraries(test_state_walk yaml-cpp fmt pthread Boost::program_options ${ORT_DIR}/lib/libonnxruntime.so.1.22.0)
set_target_properties(test_state_walk PROPERTIES BUILD_RPATH ${ORT_DIR}/lib)
add_test(NAME state_walk COMMAND test_state_walk ${PROJECT_SOURCE_DIR}/..)
# The existing Run reference is checkpoint-specific. Enable its independent
# ONNX parity test only when the reference and deployed Run model are paired.
option(G1_TEST_RUN_REFERENCE "Test the separately supplied Run model/reference pair" OFF)
if(G1_TEST_RUN_REFERENCE)
  add_test(NAME run_policy COMMAND test_run_policy ${PROJECT_SOURCE_DIR}/../config/policy/run/0908)
endif()

# Background handoff trace must preserve complete rows and flush on shutdown.
add_executable(test_walk_trace test_walk_trace.cpp)
target_link_libraries(test_walk_trace fmt pthread)
add_test(NAME walk_trace COMMAND test_walk_trace)
