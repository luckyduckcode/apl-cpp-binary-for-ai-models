// apl_ffi.cpp
// Tight FFI integration for APL - Native C++ wrapper for direct library calls
// Eliminates Python overhead for production inference

#include <bits/stdc++.h>
#include <cstring>

#ifdef _WIN32
#include <windows.h>
typedef HMODULE LibHandle;
typedef FARPROC FuncPtr;
#define LOAD_LIB(path) LoadLibraryA(path)
#define FREE_LIB(h) FreeLibrary(h)
#define GET_FUNC(h, name) GetProcAddress(h, name)
#else
#include <dlfcn.h>
typedef void* LibHandle;
typedef void* FuncPtr;
#define LOAD_LIB(path) dlopen(path, RTLD_LAZY)
#define FREE_LIB(h) dlclose(h)
#define GET_FUNC(h, name) dlsym(h, name)
#endif

// Callback function pointers from backend_1bit
typedef int (*matmul_1bit_fn)(const char*, const char*, const float*, float*, int, int, int, int);
typedef int (*matmul_q_in_mem_fn)(const void*, int, const float*, const int*, const float*, float*, int, int, int, int, int);
typedef int (*init_gpu_fn)();
typedef int (*cleanup_gpu_fn)();

// Global library handle
static LibHandle backend_lib = nullptr;
static matmul_1bit_fn g_matmul_1bit = nullptr;
static matmul_q_in_mem_fn g_matmul_q_in_mem = nullptr;
static init_gpu_fn g_init_gpu = nullptr;
static cleanup_gpu_fn g_cleanup_gpu = nullptr;

extern "C" {

// Initialize FFI: load backend library
int apl_ffi_init(const char* backend_path) {
    if (backend_lib != nullptr) {
        return 0;  // Already loaded
    }
    
    backend_lib = LOAD_LIB(backend_path);
    if (backend_lib == nullptr) {
        fprintf(stderr, "[APL_FFI] Failed to load backend: %s\n", backend_path);
        return -1;
    }
    
    // Load function pointers
    g_matmul_1bit = (matmul_1bit_fn)GET_FUNC(backend_lib, "matmul_1bit");
    g_matmul_q_in_mem = (matmul_q_in_mem_fn)GET_FUNC(backend_lib, "matmul_q_in_mem");
    g_init_gpu = (init_gpu_fn)GET_FUNC(backend_lib, "init_gpu");
    g_cleanup_gpu = (cleanup_gpu_fn)GET_FUNC(backend_lib, "cleanup_gpu");
    
    if (g_matmul_1bit == nullptr || g_matmul_q_in_mem == nullptr) {
        fprintf(stderr, "[APL_FFI] Required functions not found in backend\n");
        FREE_LIB(backend_lib);
        backend_lib = nullptr;
        return -2;
    }
    
    // Try to initialize GPU if available
    if (g_init_gpu != nullptr) {
        int gpu_status = g_init_gpu();
        if (gpu_status == 0) {
            fprintf(stderr, "[APL_FFI] GPU initialized successfully\n");
        } else {
            fprintf(stderr, "[APL_FFI] GPU not available, using CPU\n");
        }
    }
    
    return 0;
}

// Cleanup FFI: unload backend library
int apl_ffi_cleanup() {
    if (backend_lib == nullptr) {
        return 0;
    }
    
    if (g_cleanup_gpu != nullptr) {
        g_cleanup_gpu();
    }
    
    FREE_LIB(backend_lib);
    backend_lib = nullptr;
    g_matmul_1bit = nullptr;
    g_matmul_q_in_mem = nullptr;
    g_init_gpu = nullptr;
    g_cleanup_gpu = nullptr;
    
    return 0;
}

// Check if FFI is initialized
int apl_ffi_ready() {
    return (backend_lib != nullptr && g_matmul_1bit != nullptr && g_matmul_q_in_mem != nullptr) ? 1 : 0;
}

// Wrapper: 1-bit matmul
int apl_matmul_1bit(
    const char* packed_file,
    const char* scales_file,
    const float* in_vec,
    float* out_vec,
    int out_rows,
    int in_cols,
    int mode,
    int num_threads
) {
    if (!apl_ffi_ready()) {
        fprintf(stderr, "[APL_FFI] Backend not loaded\n");
        return -99;
    }
    
    return g_matmul_1bit(packed_file, scales_file, in_vec, out_vec, out_rows, in_cols, mode, num_threads);
}

// Wrapper: Integer quantized matmul
int apl_matmul_q(
    const uint8_t* q_data,
    int elem_bytes,
    const float* scales,
    const int* zero_points,
    const float* in_vec,
    float* out_vec,
    int out_rows,
    int in_cols,
    int bits,
    int num_threads
) {
    if (!apl_ffi_ready()) {
        fprintf(stderr, "[APL_FFI] Backend not loaded\n");
        return -99;
    }
    
    return g_matmul_q_in_mem(
        q_data, elem_bytes, scales, zero_points,
        in_vec, out_vec,
        out_rows, in_cols, bits,
        0,  // mode (unused)
        num_threads
    );
}

// Batch matmul: multiple matrix multiplications (optimized)
int apl_matmul_q_batch(
    const uint8_t* q_data,
    int elem_bytes,
    const float* scales,
    const int* zero_points,
    const float* in_vecs,      // Multiple vectors: (num_inputs x in_cols)
    float* out_vecs,            // Output vectors: (num_inputs x out_rows)
    int out_rows,
    int in_cols,
    int bits,
    int num_inputs,
    int num_threads
) {
    if (!apl_ffi_ready()) {
        fprintf(stderr, "[APL_FFI] Backend not loaded\n");
        return -99;
    }
    
    // Process each input vector
    for (int i = 0; i < num_inputs; i++) {
        const float* in_vec = in_vecs + i * in_cols;
        float* out_vec = out_vecs + i * out_rows;
        
        int status = g_matmul_q_in_mem(
            q_data, elem_bytes, scales, zero_points,
            in_vec, out_vec,
            out_rows, in_cols, bits,
            0,  // mode
            num_threads
        );
        
        if (status != 0) {
            return status;
        }
    }
    
    return 0;
}

// Get backend version/info
const char* apl_ffi_backend_info() {
    static const char* info = "APL_FFI v1.0 - Quantized Backend Integration";
    return info;
}

}  // extern "C"
