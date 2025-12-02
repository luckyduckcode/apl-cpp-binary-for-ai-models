#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>

extern "C" {

// Simplified C API to compute matmul with packed 1-bit weights and float activation
// packed_file: path to packed weights (out x ceil(in/8) bytes)
// scales_file: path to scales txt
// in_vec: pointer to float input vector
// out_vec: pointer to float output vector (preallocated with size out)
// out: number of output rows
// in: number of input columns
// mode: 0 -> float activation, 1 -> packed signbits
// threads: number of OpenMP threads
int matmul_1bit(const char* packed_file, const char* scales_file, const float* in_vec, float* out_vec, int out, int in, int mode, int threads){
    if(threads > 0) omp_set_num_threads(threads);
    // load packed
    std::ifstream f(packed_file, std::ios::binary);
    if(!f) return -1;
    f.seekg(0, std::ios::end);
    size_t size = f.tellg(); f.seekg(0, std::ios::beg);
    std::vector<uint8_t> packed(size);
    f.read((char*)packed.data(), size);
    f.close();

    // load scales
    std::vector<float> scales;
    std::ifstream sf(scales_file);
    if(!sf) return -2;
    float val;
    while(sf >> val) scales.push_back(val);
    sf.close();

    int bytes_per_row = (in + 7) / 8;
    if((int)packed.size() < out * bytes_per_row) return -3;

    if(mode == 1){
        // assume in_vec points to a packed signbits buffer instead of float
        const uint8_t* act_packed = reinterpret_cast<const uint8_t*>(in_vec);
        #pragma omp parallel for
        for(int r=0; r<out; ++r){
            const uint8_t* rowptr = packed.data() + r * bytes_per_row;
            int total_equal = 0;
            int b=0;
            for(; b+8 <= bytes_per_row; b+=8){
                uint64_t w = *((uint64_t*)(rowptr + b));
                uint64_t a = *((uint64_t*)(act_packed + b));
                uint64_t x = ~(w ^ a);
                total_equal += __builtin_popcountll(x);
            }
            for(; b < bytes_per_row; ++b){
                uint8_t wb = rowptr[b];
                uint8_t ab = act_packed[b];
                uint8_t x = ~(wb ^ ab);
                total_equal += __builtin_popcount((unsigned)x);
            }
            int valid_bits = in;
            float outv = (2.0f * (float)total_equal - (float)valid_bits);
            float s = (scales.size() == (size_t)out) ? scales[r] : ((scales.size() > 0) ? scales[0] : 1.0f);
            out_vec[r] = outv * s;
        }
    } else {
        // float activation: dequantize rows and dot with in_vec
        #pragma omp parallel for
        for(int r=0; r<out; ++r){
            const uint8_t* rowptr = packed.data() + r * bytes_per_row;
            float accum = 0.0f;
            for(int b=0; b<bytes_per_row; ++b){
                uint8_t wb = rowptr[b];
                for(int bit=0; bit<8; ++bit){
                    int id = b * 8 + (7-bit);
                    if(id >= in) break;
                    int bitval = (wb >> (7-bit)) & 1;
                    float sign = bitval ? 1.0f : -1.0f;
                    accum += sign * in_vec[id];
                }
            }
            float s = (scales.size() == (size_t)out) ? scales[r] : ((scales.size() > 0) ? scales[0] : 1.0f);
            out_vec[r] = accum * s;
        }
    }
    return 0;
}

// matmul for integer quantized weights in memory.
// q_ptr: pointer to quantized array contiguous row-major (out x in) of type uint8_t or uint16_t
// elem_bytes: 1 for uint8_t, 2 for uint16_t
// scales: per-row float32 scales (size out) or single scale
// zero_points: per-row int32 or single zero point (may be NULL)
// in_vec: float32 vector length in
// out_vec: float32 vector length out (output)
// bits: integer bit width (2,4,8)
// mode: reserved (unused); threads: omp threads
int matmul_q_in_mem(const void* q_ptr, int elem_bytes, const float* scales, const int* zero_points, const float* in_vec, float* out_vec, int out, int in, int bits, int mode, int threads){
    if(threads > 0) omp_set_num_threads(threads);
    if(bits <= 8 && elem_bytes == 1){
        const uint8_t* qdata = reinterpret_cast<const uint8_t*>(q_ptr);
        #pragma omp parallel for
        for(int r=0; r<out; ++r){
            const uint8_t* row = qdata + (size_t)r * in;
            float s = scales ? scales[r] : 1.0f;
            int zp = zero_points ? zero_points[r] : 0;
            double acc = 0.0;
            // Use AVX2 when available to compute dot product faster. We'll convert q values to float then perform vectorized dot.
            #if defined(__AVX2__)
            __m256 vsum = _mm256_setzero_ps();
            __m256i v_zp = _mm256_set1_epi32(zp);
            int c = 0;
            for(; c + 7 < in; c += 8){
                long long val64;
                memcpy(&val64, row + c, 8);
                __m128i v8 = _mm_cvtsi64_si128(val64);
                __m256i v_int = _mm256_cvtepu8_epi32(v8);
                __m256i v_q_minus_zp = _mm256_sub_epi32(v_int, v_zp);
                __m256 v_q_float = _mm256_cvtepi32_ps(v_q_minus_zp);
                __m256 v_in = _mm256_loadu_ps(in_vec + c);
                vsum = _mm256_fmadd_ps(v_q_float, v_in, vsum);
            }
            float tmp[8];
            _mm256_storeu_ps(tmp, vsum);
            for(int i=0; i<8; ++i) acc += tmp[i];
            for(; c<in; ++c){
                acc += (double)((int)row[c] - zp) * (double)in_vec[c];
            }
            acc *= s;
            #else
            for(int c=0; c<in; ++c){
                int qv = (int)row[c];
                acc += (double)(qv - zp) * (double)in_vec[c] * (double)s;
            }
            #endif
            out_vec[r] = (float)(acc);
        }
    } else if(elem_bytes == 2){
        const uint16_t* qdata = reinterpret_cast<const uint16_t*>(q_ptr);
        #pragma omp parallel for
        for(int r=0; r<out; ++r){
            const uint16_t* row = qdata + (size_t)r * in;
            int zp = zero_points ? zero_points[r] : 0;
            float s = scales ? scales[r] : 1.0f;
            double acc = 0.0;
            for(int c=0; c<in; ++c){
                int qv = (int)row[c];
                acc += (double)(qv - zp) * (double)in_vec[c] * (double)s;
            }
            out_vec[r] = (float)(acc);
        }
    } else {
        return -10; // unsupported element size
    }
    return 0;
}

// Specialized q4 matmul with optional packing/unpacking
// q4_ptr: pointer to q4 data, either packed (2 values per byte) or unpacked (1 value per byte)
// packed: if true, q4_ptr contains packed nibbles (2 per byte); if false, unpacked uint8
// scales, zero_points: per-row quantization parameters
// in_vec, out_vec: input/output float vectors
// out, in: matrix dimensions
// threads: OpenMP thread count
int matmul_q4_optimized(const void* q4_ptr, bool packed, const float* scales, const int* zero_points, const float* in_vec, float* out_vec, int out, int in, int threads){
    if(threads > 0) omp_set_num_threads(threads);
    
    if(packed){
        // Packed format: 2 q4 values per byte
        const uint8_t* qdata = reinterpret_cast<const uint8_t*>(q4_ptr);
        int bytes_per_row = (in + 1) / 2;
        
        #pragma omp parallel for
        for(int r=0; r<out; ++r){
            const uint8_t* row = qdata + (size_t)r * bytes_per_row;
            float s = scales ? scales[r] : 1.0f;
            int zp = zero_points ? zero_points[r] : 0;
            double acc = 0.0;
            
            #if defined(__AVX2__)
            __m256 vsum = _mm256_setzero_ps();
            __m256i v_zp = _mm256_set1_epi32(zp);
            __m256i mask_low = _mm256_set1_epi32(0x0F);
            
            int c = 0;
            // Process 16 q4 values (8 bytes) at a time
            for(; c + 15 < in; c += 16){
                // Load 8 bytes = 16 nibbles
                uint64_t val64;
                memcpy(&val64, row + c/2, 8);
                __m128i v8 = _mm_cvtsi64_si128(val64);
                
                // Unpack low nibbles (even indices)
                __m256i v_low = _mm256_cvtepu8_epi32(v8);
                v_low = _mm256_and_si256(v_low, mask_low);
                __m256i v_low_minus_zp = _mm256_sub_epi32(v_low, v_zp);
                __m256 v_low_float = _mm256_cvtepi32_ps(v_low_minus_zp);
                __m256 v_in_low = _mm256_loadu_ps(in_vec + c);
                vsum = _mm256_fmadd_ps(v_low_float, v_in_low, vsum);
                
                // Unpack high nibbles (odd indices) - shift right by 4
                __m128i v8_shifted = _mm_srli_epi16(v8, 4);
                __m256i v_high = _mm256_cvtepu8_epi32(v8_shifted);
                v_high = _mm256_and_si256(v_high, mask_low);
                __m256i v_high_minus_zp = _mm256_sub_epi32(v_high, v_zp);
                __m256 v_high_float = _mm256_cvtepi32_ps(v_high_minus_zp);
                __m256 v_in_high = _mm256_loadu_ps(in_vec + c + 8);
                vsum = _mm256_fmadd_ps(v_high_float, v_in_high, vsum);
            }
            
            float tmp[8];
            _mm256_storeu_ps(tmp, vsum);
            for(int i=0; i<8; ++i) acc += tmp[i];
            
            // Handle remaining elements
            for(; c<in; ++c){
                int byte_idx = c / 2;
                int nibble = (c % 2 == 0) ? (row[byte_idx] & 0x0F) : (row[byte_idx] >> 4);
                acc += (double)(nibble - zp) * (double)in_vec[c];
            }
            acc *= s;
            #else
            // Scalar fallback
            for(int c=0; c<in; ++c){
                int byte_idx = c / 2;
                int nibble = (c % 2 == 0) ? (row[byte_idx] & 0x0F) : (row[byte_idx] >> 4);
                acc += (double)(nibble - zp) * (double)in_vec[c] * (double)s;
            }
            #endif
            out_vec[r] = (float)acc;
        }
    } else {
        // Unpacked format: use existing matmul_q_in_mem
        return matmul_q_in_mem(q4_ptr, 1, scales, zero_points, in_vec, out_vec, out, in, 4, 0, threads);
    }
    return 0;
}

} // extern C
