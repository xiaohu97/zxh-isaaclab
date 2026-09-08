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
