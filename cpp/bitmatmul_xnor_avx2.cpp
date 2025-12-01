#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>

using namespace std;

static inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }

static vector<uint8_t> read_packed(const string& path) {
    ifstream f(path, ios::binary);
    if(!f) throw runtime_error("Failed to open packed: " + path);
    f.seekg(0, ios::end);
    size_t size = f.tellg(); f.seekg(0, ios::beg);
    vector<uint8_t> v(size);
    f.read((char*)v.data(), size);
    return v;
}

static vector<float> read_scales(const string& path) {
    ifstream f(path);
    if(!f) throw runtime_error("Failed to open scales: " + path);
    vector<float> s; float x;
    while(f >> x) s.push_back(x);
    return s;
}

int main(int argc, char** argv){
    if(argc < 7) { cerr << "Usage: bitmatmul_xnor_avx2 <packed> <scales> <out> <in> <input_vec> <mode> [threads]" << endl; return 1; }
    string packed_file = argv[1];
    string scales_file = argv[2];
    int out = atoi(argv[3]);
    int _in = atoi(argv[4]);
    string vec_file = argv[5];
    string mode = argv[6];
    int threads = 1;
    if(argc >= 8) threads = atoi(argv[7]);
    omp_set_num_threads(threads);

    auto packed = read_packed(packed_file);
    auto scales = read_scales(scales_file);

    // Read input vector
    vector<float> vec_f; vector<uint8_t> vec_b;
    bool vec_is_packed = false;
    if(mode == "binact"){
        ifstream vf(vec_file, ios::binary);
        if(!vf) throw runtime_error("Failed to open packed activation vector");
        vf.seekg(0, ios::end); size_t s = vf.tellg(); vf.seekg(0, ios::beg);
        vec_b.resize(s); vf.read((char*)vec_b.data(), s); vf.close(); vec_is_packed = true;
    } else {
        ifstream vf(vec_file);
        if(!vf) throw runtime_error("Failed to open activation text vector");
        vec_f.resize(_in); for(int i=0;i<_in;i++) vf >> vec_f[i]; vf.close();
    }

    int bytes_per_row = (_in + 7) / 8;
    vector<float> outv(out, 0.0f);

    // Precompute nibble popcount lookup table to vector registers
    alignas(32) uint8_t popcnt_nibble[32] = {0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4,
                                             0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};
    __m256i popcnt_tbl = _mm256_load_si256((const __m256i*)popcnt_nibble);
    // mask for low nibble
    __m256i low_mask = _mm256_set1_epi8(0x0F);

#pragma omp parallel for schedule(static)
    for(int r=0; r<out; ++r){
        const uint8_t* rowptr = packed.data() + r * bytes_per_row;
        if(vec_is_packed && mode == string("binact")){
            // process 32 bytes per vector (AVX2)
            int b = 0;
            uint32_t total = 0;
            // process blocks of 32 bytes
            for(; b + 32 <= bytes_per_row; b += 32){
                __m256i w = _mm256_loadu_si256((const __m256i*)(rowptr + b));
                __m256i a = _mm256_loadu_si256((const __m256i*)(vec_b.data() + b));
                __m256i x = _mm256_xor_si256(w, a); // XOR
                x = _mm256_xor_si256(x, _mm256_set1_epi8((char)0xFF)); // XNOR via NOT(XOR)
                // compute popcount per byte using nibble table
                __m256i lo = _mm256_and_si256(x, low_mask);
                __m256i hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), low_mask); // shift each byte right 4
                __m256i poplo = _mm256_shuffle_epi8(popcnt_tbl, lo);
                __m256i pophi = _mm256_shuffle_epi8(popcnt_tbl, hi);
                __m256i popsum = _mm256_adds_epu8(poplo, pophi);
                // sum across bytes in popsum
                // convert to 32-bit ints by zero extend 8-bit lanes and accumulate
                uint8_t tmp[32]; _mm256_storeu_si256((__m256i*)tmp, popsum);
                for(int i=0;i<32;i++) total += tmp[i];
            }
            // tail
            for(; b < bytes_per_row; ++b){
                uint8_t wb = rowptr[b]; uint8_t ab = vec_b[b]; uint8_t x = ~(wb ^ ab);
                total += __builtin_popcount((unsigned)x);
            }
            int valid_bits = _in; float val = 2.0f * (float)total - (float)valid_bits;
            float scale = (scales.size() == (size_t)out) ? scales[r] : ((scales.size()>0)?scales[0]:1.0f);
            outv[r] = val * scale;
        } else {
            // Float activation, dequantize per byte and accumulate
            float acc = 0.0f;
            for(int b = 0; b < bytes_per_row; ++b){
                uint8_t wb = rowptr[b];
                for(int bit=0; bit<8; ++bit){
                    int idx = b*8 + (7-bit);
                    if(idx >= _in) break;
                    int bitval = (wb >> (7-bit)) & 1;
                    float sign = bitval ? 1.0f : -1.0f;
                    acc += sign * vec_f[idx];
                }
            }
            float scale = (scales.size() == (size_t)out) ? scales[r] : ((scales.size()>0)?scales[0]:1.0f);
            outv[r] = acc * scale;
        }
    }

    for(int i = 0; i < min(out, 32); ++i) cout << outv[i] << " "; cout << endl;
    return 0;
}
