import pycuda.driver as cuda
import numpy as np
import posix_ipc
import mmap

if __name__ == '__main__':
    # 1. 分配 CUDA 内存并获取句柄
    cuda.init()
    device = cuda.Device(0)  # 指定第一个 GPU 设备
    ctx = device.make_context()
    data = np.array([1,2,3,4], dtype=np.int32)
    print(f"data is:{data}, size is {data.nbytes} ")
    data_gpu = cuda.mem_alloc(data.nbytes)

    # 获取 CUDA IPC 内存句柄
    handle = cuda.mem_get_ipc_handle(data_gpu)
    print("type handle: ", type(handle))
    cuda.memcpy_htod(data_gpu, data)

    print("get handle :", len(handle))
    # 2. 将handle写入共享内存当中
    shm_obj = posix_ipc.SharedMemory(name=("/flexgv_shm_cuda_ipc_handle"), flags=posix_ipc.O_CREAT|posix_ipc.O_RDWR, size=64)
    shm_desc = mmap.mmap(shm_obj.fd, shm_obj.size)
    shm_desc.seek(0)
    shm_desc.write(handle)
    shm_obj.close_fd()
    ctx.pop()
    print("write handle in shm ")
    while True:
        pass

