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

int main() {
    // 0.初始化
    int *data;
    cudaIpcMemHandle_t my_handle;
    int shmFd = -1;
    void* shmPtr = NULL;
    std::string shmName = "flexgv_shm_cuda_ipc_handle";

    // 1.读共享内存，获取handle
    cudaSetDevice(0);    
    sharedMemoryOpen(&shmFd, &shmPtr, sizeof(cudaIpcMemHandle_t), shmName.c_str());
    my_handle = *(static_cast<cudaIpcMemHandle_t*>(shmPtr));
    cudaIpcOpenMemHandle((void **)&data, my_handle, cudaIpcMemLazyEnablePeerAccess);
    cudaCheckErrors("IPC handle fail");
    cudaPointerAttributes attributes;
    cudaPointerGetAttributes(&attributes, data);
    if (attributes.memoryType == cudaMemoryTypeDevice) {
            std::cout << "Device pointer points to device memory." << std::endl;
    } else {
            std::cout << "Device pointer points to host memory." << std::endl;
        }
    // 2.读取数据检验
    int *host_data;
    host_data = new int[4]; // 假设 SIZE 是你的数据大小
    cudaMemcpy(host_data, data, 4 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("memcpy fail");
    for (size_t i = 0; i < 4; i++)
    {
        if (i < 3)    
            printf("%d, ", host_data[i]);
        else
            printf("%d\n", host_data[i]);
    }
    return 0;
}
