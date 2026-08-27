cmake_minimum_required(VERSION 3.10.2)

project(oculusteleop)

# --- Portability: Fetch Meta OpenXR SDK via FetchContent ---
include(FetchContent)
FetchContent_Declare(
  meta_openxr_sdk
  GIT_REPOSITORY https://github.com/meta-quest/Meta-OpenXR-SDK.git
  GIT_TAG        v85
)
FetchContent_MakeAvailable(meta_openxr_sdk)

# --- Define Paths based on FetchContent download ---
set(META_SDK_SOURCE_DIR ${meta_openxr_sdk_SOURCE_DIR})
set(META_SDK_SAMPLES_DIR ${META_SDK_SOURCE_DIR}/Samples)

# --- Add SDK Components ---
# We include 3rdParty and SampleXrFramework from the fetched SDK.
# These define 'samplexrframework' and its dependencies (ktx, stb, etc).
add_subdirectory(${META_SDK_SAMPLES_DIR}/3rdParty ${CMAKE_BINARY_DIR}/meta_3rdparty)
add_subdirectory(${META_SDK_SAMPLES_DIR}/SampleXrFramework ${CMAKE_BINARY_DIR}/meta_framework)

# --- Project Source Files ---
set(SRC_FILES
    src/main.cpp
    src/Remotes.h
)

# --- Build Target ---
if(ANDROID)
    add_library(${PROJECT_NAME} MODULE ${SRC_FILES})

    # Standard Android and SDK libraries
    target_link_libraries(${PROJECT_NAME} PRIVATE
        android
        EGL
        GLESv3
        log
        ktx
        samplexrframework
    )

    # Required for NativeActivity linkage
    set_target_properties(${PROJECT_NAME} PROPERTIES LINK_FLAGS "-u ANativeActivity_onCreate")

    # Ensure NDK glue is found
    target_include_directories(${PROJECT_NAME} PUBLIC ${ANDROID_NDK}/sources/android/native_app_glue)

elseif(WIN32)
    add_executable(${PROJECT_NAME} ${SRC_FILES})
    target_link_libraries(${PROJECT_NAME} PRIVATE samplexrframework)
    add_definitions(-D_USE_MATH_DEFINES)

    # Assets handling for Windows: copy from the fetched SDK sample folder
    add_custom_command(TARGET ${PROJECT_NAME} PRE_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy_directory
        "${META_SDK_SAMPLES_DIR}/XrSamples/XrControllers/assets"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/assets"
        VERBATIM)
endif()

# --- Common Configuration ---
target_include_directories(${PROJECT_NAME} PRIVATE src)
