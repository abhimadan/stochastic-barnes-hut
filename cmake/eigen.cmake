if (TARGET Eigen3::Eigen)
  return()
endif()

message(STATUS "Third-party: creating target 'Eigen3::Eigen'")

include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG tags/3.4.0
  GIT_SHALLOW TRUE
)

FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_MakeAvailable(eigen)
endif()

add_library(Eigen3_Eigen INTERFACE)
# add_library(Eigen3::Eigen ALIAS Eigen3_Eigen)

include(GNUInstallDirs)
target_include_directories(Eigen3_Eigen SYSTEM INTERFACE
    $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
