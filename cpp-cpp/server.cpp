#include <iostream>
#include <unistd.h>
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

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
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
int main(){
    std::string shmName = "flexgv_shm_cuda_ipc_handle";
    int shmFd = -1;
    void* shmPtr = NULL;
    sharedMemoryCreate(&shmFd, &shmPtr, sizeof(cudaIpcMemHandle_t), shmName.c_str());

    int *data;
    cudaSetDevice(0);
    cudaMalloc((void**)&data, sizeof(int));
    cudaCheckErrors("malloc fail");
    cudaMemset(data, 0, sizeof(int));
    cudaCheckErrors("memset fail");

    // 将内存拷贝到主机内存中
    int host_data;
    cudaMemcpy(&host_data, data, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckErrors("memcpy fail");

    std::cout << "data is " << host_data << std::endl;
    cudaIpcMemHandle_t my_handle;
    cudaIpcGetMemHandle(&my_handle, data);
    // handle写共享内存
    memcpy(shmPtr, (void*)&my_handle, sizeof(cudaIpcMemHandle_t));
    while (1)
    {
    /* code */
    }
    return 0;
}
