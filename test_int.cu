#include <cuda_runtime.h>
#include <chrono>

#include <iostream>

#define N 2000000000  // 向量大小

// CUDA 核心函数：向量加法
__global__ void vectorAdd(int8_t *A, int8_t *B, int8_t *C, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        C[index] = A[index] + B[index];
    }
}

int main() {
    int8_t *h_A, *h_B, *h_C;  // 主机向量
    int8_t *d_A, *d_B, *d_C;  // 设备向量

    size_t bytes = N * sizeof(int8_t);

    // 分配主机内存
    cudaHostAlloc((void**)&h_A, bytes, cudaHostAllocMapped);
    cudaHostAlloc((void**)&h_B, bytes, cudaHostAllocMapped);
    cudaHostAlloc((void**)&h_C, bytes, cudaHostAllocMapped);

    // 初始化主机向量
    for (int i = 0; i < N; i++) {
        h_A[i] = static_cast<int8_t>(1);
        h_B[i] = static_cast<int8_t>(1);
    }

    // 分配设备内存
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 复制主机向量到设备
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    //warmup
    int threads = 1024;
    int runtimes = 500;
    int blocks = (N + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
    std::cout << "start" << std::endl;
    struct timespec startTimeSpec;
    clock_gettime(CLOCK_REALTIME, &startTimeSpec);

    // 启动 CUDA 核心
    for(int i=0;i<runtimes;i++){
      auto start = std::chrono::system_clock::now();
      vectorAdd<<<blocks, threads>>>(d_A, d_B, d_C, N);
      auto end = std::chrono::system_clock::now();
    //   std::cout << end - start << " ms" << std::endl;
    }

    struct timespec stopTimeSpec;
    clock_gettime(CLOCK_REALTIME, &stopTimeSpec);

    // 计算耗时
    //double clientTimeDelta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()/1000.0;
        double clientTimeDelta =
                        (double)stopTimeSpec.tv_sec +
                        (double)stopTimeSpec.tv_nsec / 1000000000.0 -
                        ((double)startTimeSpec.tv_sec +
                         (double)startTimeSpec.tv_nsec / 1000000000.0);
    // 计算总操作数（两个向量加法）
    double totalOperations = N*runtimes; // 每个加法操作算一次
    // 计算 TOPS
    float tops = (double)((unsigned long long int)runtimes *
                                 N) /
                        clientTimeDelta / 1000.0 / 1000.0 / 1000.0 / 1000.0 / 2; // 转换为 Tera Operations
    std::cout << "计算性能: " << tops << " TOPS" << std::endl;
    std::cout << "耗时:" << clientTimeDelta*1000 << std::endl;

    // 复制结果到主机
    return;
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < 100; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            correct = false;
            break;
        }
    }

    if (correct) {
        std::cout << "向量加法完成，结果正确！" << std::endl;
    } else {
        std::cout << "结果错误！" << std::endl;
    }

    // 清理内存
    //cudaFree(d_A);
    //cudaFree(d_B);
    //cudaFree(d_C);
    //free(h_A);
    //free(h_B);
    //free(h_C);

    return 0;
}

