project(rvs_laboratories_2)

find_package(CUDA REQUIRED)

cuda_add_executable(rvs_laboratories_2 rvs_laboratories_2/main.cu)
