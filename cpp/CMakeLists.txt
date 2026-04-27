cmake_minimum_required(VERSION 3.14)
project(PaddleOCR-json CXX C)

set(DEMO_NAME "PaddleOCR-json")

# 令CMake将最后的可执行文件放在 build/bin 里面
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 编译相关参数
option(WITH_MKL          "使用MKL或OpenBlas，默认使用MKL。"                                  ON)
option(WITH_GPU          "使用GPU或CPU，默认使用CPU。"                                      OFF)
option(WITH_STATIC_LIB   "编译成static library或shared library，默认编译成static library。"   ON)
option(WITH_TENSORRT     "使用TensorRT，默认关闭。"                                         OFF)
option(USE_FREETYPE      "启用FreeType支持"                                                OFF)

# 功能相关参数
option(ENABLE_CLIPBOARD         "启用剪贴板功能。默认关闭。"        OFF)
option(ENABLE_REMOTE_EXIT       "启用远程关停服务器命令。默认开启。"  ON)
option(ENABLE_JSON_IMAGE_PATH   "启用json命令image_path。默认开启。" ON)

# CMake功能相关参数
option(INSTALL_WITH_TOOLS       "CMake安装时附带工具文件。默认开启。"      ON)

# 依赖库路径相关参数
SET(PADDLE_LIB "" CACHE PATH "paddle_inference的路径")
SET(OPENCV_DIR "" CACHE PATH "库的路径。Linux下，如果已经安装到系统之中就不用指定了。")
SET(CUDA_LIB "" CACHE PATH "库的路径")
SET(CUDNN_LIB "" CACHE PATH "库的路径")
SET(TENSORRT_DIR "" CACHE PATH "使用TensorRT编译并设置其路径")

# compiler flags
if (UNIX AND NOT APPLE) # Linux
    execute_process (
        COMMAND "uname" "-a"
        OUTPUT_VARIABLE UNAME_RESULT
    )
    string(STRIP "${UNAME_RESULT}" UNAME_RESULT)
    if (UNAME_RESULT MATCHES ".*WSL.*")
        message(STATUS "Compiling under WSL")
        add_definitions(-DUNDER_WSL)
    endif()
endif()

if (ENABLE_CLIPBOARD AND WIN32)
    add_definitions(-DENABLE_CLIPBOARD)
elseif () # 在其他系统上禁用ENABLE_CLIPBOARD
    set(ENABLE_CLIPBOARD OFF)
endif()

if (ENABLE_REMOTE_EXIT)
    add_definitions(-DENABLE_REMOTE_EXIT)
endif()

if (ENABLE_JSON_IMAGE_PATH)
    add_definitions(-DENABLE_JSON_IMAGE_PATH)
endif()

macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
        if(${flag_var} MATCHES "/MD")
            string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
        endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

if (WITH_MKL)
    ADD_DEFINITIONS(-DUSE_MKL)
endif()

if(NOT DEFINED PADDLE_LIB)
    message(FATAL_ERROR "please set PADDLE_LIB with -DPADDLE_LIB=/path/paddle/lib")
endif()

if(NOT DEFINED OPENCV_DIR)
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
endif()

# 输出功能设置
message(STATUS "Features:")
message(STATUS "    ENABLE_CLIPBOARD: ${ENABLE_CLIPBOARD}")
message(STATUS "    ENABLE_REMOTE_EXIT: ${ENABLE_REMOTE_EXIT}")
message(STATUS "    ENABLE_JSON_IMAGE_PATH: ${ENABLE_JSON_IMAGE_PATH}")
message(STATUS "CMake Features:")
message(STATUS "    INSTALL_WITH_TOOLS: ${INSTALL_WITH_TOOLS}")

# 加载依赖
set(OPENCV_COMPONENTS_TO_LOAD
    core
    imgproc
    imgcodecs
)

if(DEFINED OPENCV_DIR AND NOT "${OPENCV_DIR}" STREQUAL "")
    message(STATUS "Setting OPENCV_DIR: ${OPENCV_DIR}")
    find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR} COMPONENTS ${OPENCV_COMPONENTS_TO_LOAD} NO_DEFAULT_PATH)
    set(CUSTOM_OPENCV_DIR ${OPENCV_DIR})
else()
    find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_COMPONENTS_TO_LOAD})
endif()
include_directories(${OpenCV_INCLUDE_DIRS})

if(USE_FREETYPE)
    if(NOT "opencv_freetype" IN_LIST OpenCV_LIBS)
        message(FATAL_ERROR "OpenCV was not compiled with the freetype module (opencv_freetype) !")
    endif()
    add_definitions(-DUSE_FREETYPE)
endif()

# Paddle Inference
include_directories("${PADDLE_LIB}/paddle/include")
link_directories("${PADDLE_LIB}/paddle/lib")

if (WIN32)
    add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
    if(WITH_MKL)
        set(FLAG_OPENMP "/openmp")
    endif()
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    if (WITH_STATIC_LIB)
        safe_set_static_flag()
        add_definitions(-DSTATIC_LIB)
        add_definitions(-DYAML_CPP_STATIC_DEFINE)
    endif()
    message(STATUS "cmake c debug flags:" ${CMAKE_C_FLAGS_DEBUG})
    message(STATUS "cmake c release flags:" ${CMAKE_C_FLAGS_RELEASE})
    message(STATUS "cmake cxx debug flags:" ${CMAKE_CXX_FLAGS_DEBUG})
    message(STATUS "cmake cxx release flags:" ${CMAKE_CXX_FLAGS_RELEASE})
else()
    if(WITH_MKL)
        set(FLAG_OPENMP "-fopenmp")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG_OPENMP} -std=c++11")
    set(CMAKE_STATIC_LIBRARY_PREFIX "")
    message(STATUS "cmake cxx flags:" ${CMAKE_CXX_FLAGS})
endif()

if (WITH_GPU)
    if (NOT DEFINED CUDA_LIB OR ${CUDA_LIB} STREQUAL "")
        message(FATAL_ERROR "please set CUDA_LIB with -DCUDA_LIB=/path/cuda-8.0/lib64")
    endif()
    if (NOT WIN32)
        if (NOT DEFINED CUDNN_LIB)
            message(FATAL_ERROR "please set CUDNN_LIB with -DCUDNN_LIB=/path/cudnn_v7.4/cuda/lib64")
        endif()
    endif(NOT WIN32)
endif()

# 加载gflags
set(GFLAGS_BUILD_STATIC_LIBS ON)
set(GFLAGS_HIDE_PATH_IN_HELP "Basename")
set(GFLAGS_USE_SHORTEST_DOUBLE_PRECISION ON)
set(GFLAGS_NAMESPACE "google")
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third_party/gflags")
include_directories(${gflags_BINARY_DIR}/include)

# 加载paddle_inference里面的其他第三方库
include_directories("${PADDLE_LIB}/third_party/install/protobuf/include")
include_directories("${PADDLE_LIB}/third_party/install/glog/include")
include_directories("${PADDLE_LIB}/third_party/install/gflags/include")
include_directories("${PADDLE_LIB}/third_party/install/xxhash/include")
include_directories("${PADDLE_LIB}/third_party/install/zlib/include")
include_directories("${PADDLE_LIB}/third_party/install/onnxruntime/include")
include_directories("${PADDLE_LIB}/third_party/install/paddle2onnx/include")
include_directories("${PADDLE_LIB}/third_party/install/yaml-cpp/include")
include_directories("${PADDLE_LIB}/third_party/install/openvino/include")
include_directories("${PADDLE_LIB}/third_party/install/tbb/include")
include_directories("${PADDLE_LIB}/third_party/boost")
include_directories("${PADDLE_LIB}/third_party/eigen3")
include_directories("${CMAKE_SOURCE_DIR}/paddleocr")
include_directories("${CMAKE_SOURCE_DIR}/")

link_directories("${PADDLE_LIB}/third_party/install/zlib/lib")
link_directories("${PADDLE_LIB}/third_party/install/protobuf/lib")
link_directories("${PADDLE_LIB}/third_party/install/glog/lib")
link_directories("${PADDLE_LIB}/third_party/install/gflags/lib")
link_directories("${PADDLE_LIB}/third_party/install/xxhash/lib")
link_directories("${PADDLE_LIB}/third_party/install/onnxruntime/lib")
link_directories("${PADDLE_LIB}/third_party/install/paddle2onnx/lib")
link_directories("${PADDLE_LIB}/third_party/install/yaml-cpp/lib")
link_directories("${PADDLE_LIB}/third_party/install/openvino/intel64")
link_directories("${PADDLE_LIB}/third_party/install/tbb/lib")
link_directories("${PADDLE_LIB}/paddle/lib")

if(WITH_MKL)
    include_directories("${PADDLE_LIB}/third_party/install/mklml/include")
    if (WIN32)
        set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/mklml.lib
                    ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5md.lib)
    else ()
        set(MATH_LIB ${PADDLE_LIB}/third_party/install/mklml/lib/libmklml_intel${CMAKE_SHARED_LIBRARY_SUFFIX}
                    ${PADDLE_LIB}/third_party/install/mklml/lib/libiomp5${CMAKE_SHARED_LIBRARY_SUFFIX})
        link_directories("${PADDLE_LIB}/third_party/install/onednn/lib")
    endif ()
    set(MKLDNN_PATH "${PADDLE_LIB}/third_party/install/onednn")
    if(EXISTS ${MKLDNN_PATH})
        include_directories("${MKLDNN_PATH}/include")
        if (WIN32)
            set(MKLDNN_LIB ${MKLDNN_PATH}/lib/mkldnn.lib)
        else ()
            set(MKLDNN_LIB ${MKLDNN_PATH}/lib/libdnnl.so.3)
        endif ()
    endif()
else()
    if (WIN32)
        set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/openblas${CMAKE_STATIC_LIBRARY_SUFFIX})
    else ()
        set(MATH_LIB ${PADDLE_LIB}/third_party/install/openblas/lib/libopenblas${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif ()
endif()

if(WITH_STATIC_LIB)
    if(WIN32)
        set(DEPS
            ${PADDLE_LIB}/paddle/lib/paddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
    else()
        set(DEPS
            ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
else()
    if(WIN32)
        set(DEPS
            ${PADDLE_LIB}/paddle/lib/paddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
    else()
        set(DEPS
            ${PADDLE_LIB}/paddle/lib/libpaddle_inference${CMAKE_SHARED_LIBRARY_SUFFIX})
    endif()
endif(WITH_STATIC_LIB)

if (NOT WIN32)
    set(DEPS ${DEPS}
        ${MATH_LIB} ${MKLDNN_LIB}
        glog gflags z xxhash yaml-cpp
        )
    if(EXISTS "${PADDLE_LIB}/third_party/install/snappystream/lib")
        set(DEPS ${DEPS} snappystream)
    endif()
    if (EXISTS "${PADDLE_LIB}/third_party/install/snappy/lib")
        set(DEPS ${DEPS} snappy)
    endif()
else() # WIN32
    set(DEPS ${DEPS}
        ${MATH_LIB} ${MKLDNN_LIB}
        glog gflags_static libprotobuf xxhash yaml-cpp)
    set(DEPS ${DEPS} libcmt shlwapi)
    if (EXISTS "${PADDLE_LIB}/third_party/install/snappy/lib")
        set(DEPS ${DEPS} snappy)
    endif()
    if(EXISTS "${PADDLE_LIB}/third_party/install/snappystream/lib")
        set(DEPS ${DEPS} snappystream)
    endif()
endif(NOT WIN32)

if(WITH_GPU)
    if(NOT WIN32)
        if (WITH_TENSORRT)
            set(DEPS ${DEPS} ${TENSORRT_DIR}/lib/libnvinfer${CMAKE_SHARED_LIBRARY_SUFFIX})
            set(DEPS ${DEPS} ${TENSORRT_DIR}/lib/libnvinfer_plugin${CMAKE_SHARED_LIBRARY_SUFFIX})
        endif()
        set(DEPS ${DEPS} ${CUDA_LIB}/libcudart${CMAKE_SHARED_LIBRARY_SUFFIX})
        set(DEPS ${DEPS} ${CUDNN_LIB}/libcudnn${CMAKE_SHARED_LIBRARY_SUFFIX})
    else()
        set(DEPS ${DEPS} ${CUDA_LIB}/cudart${CMAKE_STATIC_LIBRARY_SUFFIX} )
        set(DEPS ${DEPS} ${CUDA_LIB}/cublas${CMAKE_STATIC_LIBRARY_SUFFIX} )
        set(DEPS ${DEPS} ${CUDNN_LIB}/cudnn${CMAKE_STATIC_LIBRARY_SUFFIX})
    endif()
endif()

if (NOT WIN32)
    set(EXTERNAL_LIB "-ldl -lrt -lgomp -lz -lm -lpthread")
    set(DEPS ${DEPS} ${EXTERNAL_LIB})
endif()

# 第三方依赖: abseil-cpp, clipper, nlohmann json
set(THIRD_PARTY_PATH ${CMAKE_CURRENT_LIST_DIR}/third_party)

# abseil-cpp (static only to avoid runtime conflicts with system abseil)
set(BUILD_SHARED_LIBS_SAVED ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(${THIRD_PARTY_PATH}/absl)
set(BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS_SAVED})
include_directories(${THIRD_PARTY_PATH}/absl)

# clipper
add_subdirectory(${THIRD_PARTY_PATH}/clipper)
include_directories(${POLYCLIPPING_INCLUDE_DIR})

# nlohmann json (header only)
include_directories(${THIRD_PARTY_PATH})

# 添加库到 DEPS
set(DEPS ${DEPS} ${OpenCV_LIBS})
set(DEPS ${DEPS} absl::statusor)
set(DEPS ${DEPS} polyclipping)

if(UNIX)
    find_package(Iconv REQUIRED)
    set(DEPS ${DEPS} Iconv::Iconv)
endif()

# 收集 v3.5.0 源文件
file(GLOB_RECURSE PADDLEOCR_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/paddleocr/*.cc")
# 排除 args.cc，因为当前项目已有自己的参数定义
list(FILTER PADDLEOCR_SRC_LIST EXCLUDE REGEX ".*utils/args\\.cc$")

# 收集项目自定义源文件
AUX_SOURCE_DIRECTORY(./src SRCS)

add_executable(${DEMO_NAME} ${SRCS} ${PADDLEOCR_SRC_LIST})
target_link_libraries(${DEMO_NAME} ${DEPS})

# 设置需要安装的共享库
message(STATUS "Collecting PaddleOCR shared libraries")
file(GLOB_RECURSE PADDLE_LIB_FILES "${PADDLE_LIB}/**/*${CMAKE_SHARED_LIBRARY_SUFFIX}*")
foreach(ITEM ${PADDLE_LIB_FILES})
    list(APPEND LIBS_TO_INSTALL ${ITEM})
endforeach()

# OpenCV
include(cmake/opencv-install-utils.cmake)
if(DEFINED CUSTOM_OPENCV_DIR AND NOT "${CUSTOM_OPENCV_DIR}" STREQUAL "")
    message(STATUS "Collecting custom OpenCV shared libraries")
    set(OPENCV_MISS_LIBS FALSE)

    collect_opencv_lib(${CUSTOM_OPENCV_DIR} ${CMAKE_SHARED_LIBRARY_SUFFIX} "opencv_core" opencv_core_miss_lib opencv_core_regular opencv_core_symlink)
    collect_opencv_lib(${CUSTOM_OPENCV_DIR} ${CMAKE_SHARED_LIBRARY_SUFFIX} "opencv_imgcodecs" opencv_imgcodecs_miss_lib opencv_imgcodecs_regular opencv_imgcodecs_symlink)
    collect_opencv_lib(${CUSTOM_OPENCV_DIR} ${CMAKE_SHARED_LIBRARY_SUFFIX} "opencv_imgproc" opencv_imgproc_miss_lib opencv_imgproc_regular opencv_imgproc_symlink)

    if(opencv_core_miss_lib OR opencv_imgcodecs_miss_lib OR opencv_imgproc_miss_lib)
        set(OPENCV_MISS_LIBS TRUE)
    endif()

    if(DEFINED OPENCV_MISS_LIBS AND OPENCV_MISS_LIBS)
        message(STATUS "Missing OpenCV independent shared libraries, try opencv_world")
        collect_opencv_lib(${CUSTOM_OPENCV_DIR} ${CMAKE_SHARED_LIBRARY_SUFFIX} "opencv_world" opencv_world_miss_lib opencv_world_regular opencv_world_symlink)
        if(opencv_world_miss_lib)
            message(FATAL_ERROR "Cannot find OpenCV independent shared libraries or opencv_world. Please make sure your custom OpenCV directory contains the missing libraries.")
        else()
            message(STATUS "Found opencv_world")
            list(APPEND LIBS_TO_INSTALL ${opencv_world_regular})
            list(JOIN opencv_world_symlink "/" opencv_world_symlink_str)
            list(APPEND SYMS_TO_INSTALL ${opencv_world_symlink_str})
        endif()
    else()
        list(APPEND LIBS_TO_INSTALL ${opencv_core_regular})
        list(JOIN opencv_core_symlink "/" opencv_core_symlink_str)
        list(APPEND SYMS_TO_INSTALL ${opencv_core_symlink_str})

        list(APPEND LIBS_TO_INSTALL ${opencv_imgcodecs_regular})
        list(JOIN opencv_imgcodecs_symlink "/" opencv_imgcodecs_symlink_str)
        list(APPEND SYMS_TO_INSTALL ${opencv_imgcodecs_symlink_str})

        list(APPEND LIBS_TO_INSTALL ${opencv_imgproc_regular})
        list(JOIN opencv_imgproc_symlink "/" opencv_imgproc_symlink_str)
        list(APPEND SYMS_TO_INSTALL ${opencv_imgproc_symlink_str})
    endif()
endif()

foreach(ITEM ${LIBS_TO_INSTALL})
    add_custom_command(TARGET ${DEMO_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ITEM}
        "$<TARGET_FILE_DIR:${DEMO_NAME}>"
    )
endforeach()

# CMake TARGET安装路径设置
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    if(NOT CMAKE_TOOLCHAIN_FILE)
        if(WIN32)
            set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "安装路径" FORCE)
        else()
            set(CMAKE_INSTALL_PREFIX "/usr/" CACHE PATH "安装路径" FORCE)
        endif()
    else()
        set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "安装路径" FORCE)
    endif()
endif()

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

install(
    TARGETS ${DEMO_NAME}
    CONFIGURATIONS Debug Release
    LIBRARY DESTINATION "lib"
    ARCHIVE DESTINATION "lib"
    RUNTIME DESTINATION "bin"
)
if(WIN32)
    install(
        DIRECTORY "$<TARGET_FILE_DIR:${DEMO_NAME}>/"
        DESTINATION "bin"
        FILES_MATCHING
        PATTERN "*${CMAKE_SHARED_LIBRARY_SUFFIX}*"
        PATTERN "${DEMO_NAME}*" EXCLUDE
    )
else()
    install(
        DIRECTORY "$<TARGET_FILE_DIR:${DEMO_NAME}>/"
        DESTINATION "lib"
        FILES_MATCHING
        PATTERN "*${CMAKE_SHARED_LIBRARY_SUFFIX}*"
        PATTERN "${DEMO_NAME}*" EXCLUDE
    )

    foreach(ITEM_STR ${SYMS_TO_INSTALL})
        string(REPLACE "/" ";" ITEM_LIST ${ITEM_STR})
        list(GET ITEM_LIST 0 ITEM_REGULAR_NAME)
        list(REMOVE_AT ITEM_LIST 0)

        foreach(ITEM_SYMLINK_NAME ${ITEM_LIST})
            install(CODE "\
                execute_process(COMMAND ${CMAKE_COMMAND} -E chdir \$\{CMAKE_INSTALL_PREFIX\}/lib \
                ${CMAKE_COMMAND} -E create_symlink ${ITEM_REGULAR_NAME} ${ITEM_SYMLINK_NAME})"
            )
        endforeach()
    endforeach()

    if (INSTALL_WITH_TOOLS)
        install(
            FILES
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/linux_dist_tools/run.sh"
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/linux_dist_tools/install_env.sh"
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
            DESTINATION "."
        )
        install(
            FILES
                "${CMAKE_CURRENT_SOURCE_DIR}/tools/linux_dist_tools/README.txt"
            PERMISSIONS OWNER_READ OWNER_WRITE GROUP_READ GROUP_WRITE WORLD_READ
            DESTINATION "."
        )
    endif()
endif()
