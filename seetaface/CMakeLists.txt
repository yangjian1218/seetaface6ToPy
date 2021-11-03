cmake_minimum_required (VERSION 2.6)
project(myctypes)
#set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11" -O2)
add_compile_options(-Wall -std=c++11 -O2)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
#set(OpenCV_DIR D:/opencv/opencv-4.5.2)
#find_package(OpenCV REQUIRED)

find_package(OpenMP)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)           #设置库文件的输出目录
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)       #设置可执行文件的输出目录
# set(SRC_LIST hello.cpp)
# 添加头文件
# include_directories( ${OpenCV_INCLUDE_DIRS} )
# include_directories(D:/opencv/opencv-4.5.2/include)       #向工程添加多个特定的头文件搜索路径
# link_directories(D:/opencv/opencv-4.5.2/x64/mingw/lib )   #向工程添加多个特定的库文件搜索路径/
#添加头文件搜索路径
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
#添加动态库搜索文件
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)



add_library( SeetaFaceAPI SHARED SeetaFaceAPI.cpp)
target_link_libraries( SeetaFaceAPI  SeetaAuthorize tennis SeetaQualityAssessor300 SeetaFaceDetector600 SeetaAgePredictor600 SeetaEyeStateDetector200
	SeetaFaceAntiSpoofingX600 SeetaFaceLandmarker600 SeetaGenderPredictor600 SeetaMaskDetector200 SeetaFaceRecognizer610 SeetaFaceTracking600)


# add_executable(main SeetaFaceAPI.cpp)


# 链接OpenCV库
# target_link_libraries( hello opencv_calib3d452 opencv_core452 opencv_dnn452 opencv_features2d452 opencv_highgui452 opencv_imgcodecs452 )

# target_link_libraries( main  ${OpenCV_LIBS} SeetaAuthorize tennis SeetaQualityAssessor300 SeetaFaceDetector600 SeetaAgePredictor600 SeetaEyeStateDetector200
#	SeetaFaceAntiSpoofingX600 SeetaFaceLandmarker600 SeetaGenderPredictor600 SeetaMaskDetector200 SeetaFaceRecognizer610 SeetaFaceTracking600)  # 只要连接lib部分. windows下为静态库.
#
# target_link_libraries( main SeetaFaceAPI )