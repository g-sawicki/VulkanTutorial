cmake_minimum_required(VERSION 3.31)
project("VulkanTutorial")

file(GLOB SRC_FILES
     "src/*.h"
     "src/*.cpp"
)

add_executable (VulkanTutorial ${SRC_FILES})
set_property(TARGET VulkanTutorial PROPERTY CXX_STANDARD 23)

# vulkan
find_package(Vulkan REQUIRED)
target_link_libraries(VulkanTutorial PUBLIC Vulkan::Vulkan)

# glfw
set(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW build documentation" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "GLFW build documentation" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "GLFW build documentation" FORCE)
add_subdirectory(vendor/glfw)

# glm
target_include_directories(VulkanTutorial PUBLIC vendor/glm)

target_link_libraries(VulkanTutorial PUBLIC glfw)

add_custom_target(copy_assets
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_LIST_DIR}/copy_assets.cmake
)
add_dependencies(VulkanTutorial copy_assets)