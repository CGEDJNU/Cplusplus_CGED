PROJECT(cnn:cged)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

INCLUDE_DIRECTORIES(${source_directory})
ADD_EXECUTABLE(text-bigram text-bigram.cc)
target_link_libraries(text-bigram cnn ${Boost_LIBRARIES})