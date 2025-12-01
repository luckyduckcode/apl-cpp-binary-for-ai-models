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

} // extern C
