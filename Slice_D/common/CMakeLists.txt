cmake_minimum_required(VERSION 3.12.1 FATAL_ERROR)


# 项目信息
project (common)

# 查找目录下的所有源文件
# 并将名称保存到 DIR_SRCS 变量
aux_source_directory(. DIR_SRCS)

# 指定生成目标
# add_executable(common ${DIR_SRCS})
# add_library(common ${DIR_SRCS})