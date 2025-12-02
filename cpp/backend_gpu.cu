// backend_gpu.cu
// GPU-accelerated matmul kernels for quantized matrix multiplication
// Supports CUDA for NVIDIA GPUs; seamlessly falls back to CPU if GPU unavailable

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>

// Control GPU usage
static bool USE_GPU = true;
static cublasHandle_t cublas_handle = nullptr;

extern "C" {

// Initialize GPU (if available)
int gpu_init() {
#ifdef ENABLE_CUDA
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "[GPU] CUDA device not available, falling back to CPU\n");
        USE_GPU = false;
        return -1;
    }
    
    cublasStatus_t stat = cublasCreate(&cublas_handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[GPU] Failed to create cuBLAS handle, falling back to CPU\n");
        USE_GPU = false;
        return -1;
    }
    fprintf(stderr, "[GPU] CUDA initialized successfully\n");
    return 0;
#else
    USE_GPU = false;
    return -1;
#endif
}

// Cleanup GPU resources
int gpu_cleanup() {
#ifdef ENABLE_CUDA
    if (cublas_handle != nullptr) {
        cublasDestroy(cublas_handle);
        cublas_handle = nullptr;
    }
    cudaDeviceSynchronize();
    return 0;
#else
    return 0;
#endif
}

// Query GPU availability
int gpu_available() {
#ifdef ENABLE_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
#else
    return 0;
#endif
}

#ifdef ENABLE_CUDA
// CUDA kernel for dequantizing integer quantized rows and computing matmul
// Each thread block computes one row of output
__global__ void matmul_q_kernel(
    const uint8_t* q_data,      // Input quantized data (out x in)
    const float* scales,         // Per-row scales (out,)
    const int* zero_points,      // Per-row zero points (out,) or NULL
    const float* in_vec,         // Input vector (in,)
    float* out_vec,              // Output vector (out,) - RESULT
    int out,
    int in,
    int bits
) {
    int row = blockIdx.x;
    if (row >= out) return;
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    float accum = 0.0f;
    float scale = scales[row];
    int zp = (zero_points != nullptr) ? zero_points[row] : 0;
    
    // Process input elements
    for (int col = tid; col < in; col += stride) {
        // Unquantize: (q[col] - zp) * scale * in_vec[col]
        uint8_t q_val = q_data[row * in + col];
        float q_float = (float)(q_val - zp) * scale;
        accum += q_float * in_vec[col];
    }
    
    // Reduce within block
    __shared__ float sdata[256];
    sdata[tid] = accum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        out_vec[row] = sdata[0];
    }
}

// CUDA kernel for packed 1-bit quantization with signed weights
__global__ void matmul_1bit_packed_kernel(
    const uint8_t* packed_weights,  // Packed 1-bit weights (out x ceil(in/8))
    const float* scales,             // Per-row scales (out,)
    const float* in_vec,             // Input vector (in,)
    float* out_vec,                  // Output (out,)
    int out,
    int in
) {
    int row = blockIdx.x;
    if (row >= out) return;
    
    int tid = threadIdx.x;
    int bytes_per_row = (in + 7) / 8;
    
    float accum = 0.0f;
    float scale = scales[row];
    const uint8_t* row_data = packed_weights + row * bytes_per_row;
    
    // Process bits in parallel
    for (int byte_idx = tid; byte_idx < bytes_per_row; byte_idx += blockDim.x) {
        uint8_t byte_val = row_data[byte_idx];
        for (int bit = 0; bit < 8; bit++) {
            int col = byte_idx * 8 + (7 - bit);
            if (col < in) {
                int bit_val = (byte_val >> (7 - bit)) & 1;
                float weight = bit_val ? 1.0f : -1.0f;
                accum += weight * in_vec[col];
            }
        }
    }
    
    // Reduce within block
    __shared__ float sdata[256];
    sdata[tid] = accum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        out_vec[row] = sdata[0] * scale;
    }
}
#endif

// GPU-accelerated matmul for integer quantized weights
int matmul_q_gpu(
    const void* q_ptr,
    int elem_bytes,
    const float* scales,
    const int* zero_points,
    const float* in_vec,
    float* out_vec,
    int out,
    int in,
    int bits,
    int threads
) {
#ifdef ENABLE_CUDA
    if (!USE_GPU || cublas_handle == nullptr) {
        return -1;  // Fall back to CPU
    }
    
    try {
        const uint8_t* q_data = reinterpret_cast<const uint8_t*>(q_ptr);
        
        // Allocate GPU memory
        uint8_t* d_q_data = nullptr;
        float* d_scales = nullptr;
        int* d_zero_points = nullptr;
        float* d_in_vec = nullptr;
        float* d_out_vec = nullptr;
        
        cudaMalloc(&d_q_data, out * in * elem_bytes);
        cudaMalloc(&d_scales, out * sizeof(float));
        if (zero_points) cudaMalloc(&d_zero_points, out * sizeof(int));
        cudaMalloc(&d_in_vec, in * sizeof(float));
        cudaMalloc(&d_out_vec, out * sizeof(float));
        
        // Copy data to GPU
        cudaMemcpy(d_q_data, q_data, out * in * elem_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_scales, scales, out * sizeof(float), cudaMemcpyHostToDevice);
        if (zero_points) {
            cudaMemcpy(d_zero_points, zero_points, out * sizeof(int), cudaMemcpyHostToDevice);
        }
        cudaMemcpy(d_in_vec, in_vec, in * sizeof(float), cudaMemcpyHostToDevice);
        
        // Launch kernel
        int block_size = 256;
        int grid_size = (out + block_size - 1) / block_size;
        matmul_q_kernel<<<grid_size, block_size>>>(
            d_q_data,
            d_scales,
            d_zero_points,
            d_in_vec,
            d_out_vec,
            out,
            in,
            bits
        );
        
        cudaError_t cudaErr = cudaGetLastError();
        if (cudaErr != cudaSuccess) {
            fprintf(stderr, "[GPU] Kernel error: %s\n", cudaGetErrorString(cudaErr));
            cudaFree(d_q_data);
            cudaFree(d_scales);
            if (d_zero_points) cudaFree(d_zero_points);
            cudaFree(d_in_vec);
            cudaFree(d_out_vec);
            return -1;
        }
        
        // Copy result back
        cudaMemcpy(out_vec, d_out_vec, out * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Cleanup
        cudaFree(d_q_data);
        cudaFree(d_scales);
        if (d_zero_points) cudaFree(d_zero_points);
        cudaFree(d_in_vec);
        cudaFree(d_out_vec);
        
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[GPU] Exception: %s\n", e.what());
        return -1;
    }
#else
    return -1;  // No CUDA support
#endif
}

// GPU-accelerated matmul for 1-bit packed weights
int matmul_1bit_gpu(
    const char* packed_file,
    const char* scales_file,
    const float* in_vec,
    float* out_vec,
    int out,
    int in,
    int threads
) {
#ifdef ENABLE_CUDA
    if (!USE_GPU || cublas_handle == nullptr) {
        return -1;  // Fall back to CPU
    }
    
    try {
        // Load packed weights
        std::ifstream f(packed_file, std::ios::binary);
        if (!f) return -1;
        f.seekg(0, std::ios::end);
        size_t size = f.tellg();
        f.seekg(0, std::ios::beg);
        std::vector<uint8_t> packed(size);
        f.read((char*)packed.data(), size);
        f.close();
        
        // Load scales
        std::vector<float> scales;
        std::ifstream sf(scales_file);
        if (!sf) return -2;
        float val;
        while (sf >> val) scales.push_back(val);
        sf.close();
        
        // Allocate GPU memory
        uint8_t* d_packed = nullptr;
        float* d_scales = nullptr;
        float* d_in_vec = nullptr;
        float* d_out_vec = nullptr;
        
        cudaMalloc(&d_packed, packed.size());
        cudaMalloc(&d_scales, scales.size() * sizeof(float));
        cudaMalloc(&d_in_vec, in * sizeof(float));
        cudaMalloc(&d_out_vec, out * sizeof(float));
        
        // Copy data
        cudaMemcpy(d_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_scales, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_in_vec, in_vec, in * sizeof(float), cudaMemcpyHostToDevice);
        
        // Launch kernel
        int block_size = 256;
        int grid_size = (out + block_size - 1) / block_size;
        matmul_1bit_packed_kernel<<<grid_size, block_size>>>(
            d_packed,
            d_scales,
            d_in_vec,
            d_out_vec,
            out,
            in
        );
        
        cudaError_t cudaErr = cudaGetLastError();
        if (cudaErr != cudaSuccess) {
            fprintf(stderr, "[GPU] Kernel error: %s\n", cudaGetErrorString(cudaErr));
            cudaFree(d_packed);
            cudaFree(d_scales);
            cudaFree(d_in_vec);
            cudaFree(d_out_vec);
            return -1;
        }
        
        // Copy result
        cudaMemcpy(out_vec, d_out_vec, out * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Cleanup
        cudaFree(d_packed);
        cudaFree(d_scales);
        cudaFree(d_in_vec);
        cudaFree(d_out_vec);
        
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "[GPU] Exception: %s\n", e.what());
        return -1;
    }
#else
    return -1;
#endif
}

}  // extern "C"
