#include <iostream>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <string.h>
// app 2, part of a 2-part IPC example
#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#define DSIZE 1

#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)

__global__ void set_kernel(volatile int *d, int val){
  *d = val;
}
void InitShm(int* shm_fd, void** shm_ptr, size_t size, const char* name) {
    *shm_fd = shm_open(name, O_CREAT | O_RDWR, 0666);
    if (*shm_fd == -1) {
        return;
    }

    // Set the size of the shared memory object
    if (ftruncate(*shm_fd, size) == -1) {
        close(*shm_fd);
        *shm_fd = -1;
        return;
    }

    // Map the shared memory object with read-write permissions
    *shm_ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, *shm_fd, 0);
    if (*shm_ptr == MAP_FAILED) {
        close(*shm_fd);
        *shm_fd = -1;
        return;
    }
}
int sharedMemoryCreate(int* shm_fd, void** shm_ptr, size_t size, const char* name)
{
    *shm_fd = shm_open(name, O_RDWR | O_CREAT, 0777);
    if (*shm_fd == -1) {
        return -1;
    }
    if (ftruncate(*shm_fd, size) == -1) {
        close(*shm_fd);
        *shm_fd = -1;
        return -1;
    }
    *shm_ptr = mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, *shm_fd, 0);
    if (*shm_ptr == MAP_FAILED) {
        close(*shm_fd);
        *shm_fd = -1;
        return -1;
    }
    return 0;
}
int sharedMemoryOpen(int* shm_fd,  void** shm_ptr, size_t size, const char *name)
{
    *shm_fd = shm_open(name, O_RDWR, 0777);
    if (*shm_fd == -1) {
        return -1;
    }
    *shm_ptr =  mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, *shm_fd, 0);
    if (*shm_ptr == MAP_FAILED) {
        close(*shm_fd);
        *shm_fd = -1;
        return -1;
    }
    return 0;
}

bool ReadHandle(void* shm_ptr, cudaIpcMemHandle_t* handle){
    if (shm_ptr == NULL)
        return false;
    // memcpy(handle, shm_ptr, sizeof(cudaIpcMemHandle_t));
    *handle = *static_cast<cudaIpcMemHandle_t*>(shm_ptr);
    return true;
}
// https://forums.developer.nvidia.com/t/gpu-inter-process-communications-ipc-question/35936/4
// 管道传递可跑通
int main(){
    cudaSetDevice(0);
    int *data;
    cudaIpcMemHandle_t my_handle;
    // 1. 读共享内存，获取handle
    std::string shmName = "flexgv_shm_cuda_ipc_handle";
    int shmFd = -1;
    void* shmPtr = NULL;
    sharedMemoryOpen(&shmFd, &shmPtr, sizeof(cudaIpcMemHandle_t), shmName.c_str());
    my_handle = *(static_cast<cudaIpcMemHandle_t*>(shmPtr));
 

    cudaIpcOpenMemHandle((void **)&data, my_handle, cudaIpcMemLazyEnablePeerAccess);
    cudaCheckErrors("IPC handle fail");

    // 2. 从GPU设备中读取数据
    int host_data;
    cudaMemcpy(&host_data, data, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("memcpy fail");

    // 3. 在主机上打印数据
    std::cout << "data is " << host_data << std::endl;
    cudaDeviceSynchronize();
    cudaCheckErrors("memset fail");
    return 0;
}
