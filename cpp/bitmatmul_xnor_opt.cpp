#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>

using namespace std;
static inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }

static bool cpu_supports_avx512_vpopcntdq() {
#if defined(__x86_64__) || defined(_M_X64)
    // Simplified runtime detection via cpuid for AVX512 VPOPCNTDQ (bit is in ECX of leaf 7)
    unsigned int eax, ebx, ecx, edx;
    eax = 7; ecx = 0; ebx = ecx; edx = ecx;
    __asm__ volatile ("cpuid" : "=a" (eax), "=b" (ebx), "=c" (ecx), "=d" (edx) : "a" (7), "c" (0));
    // VPOPCNTDQ is EBX bit 14 per Intel docs? (This is more complicated; we'll do a simple fallback)
    // For simplicity: try hardware test using __builtin_cpu_supports if available
    #ifdef __GNUC__
    #if defined(__has_builtin)
    # if __has_builtin(__builtin_cpu_supports)
        if(__builtin_cpu_supports("avx512vpopcntdq")) return true;
    # endif
    #endif
    #endif
#endif
    return false;
}

// load binary file into vector
static vector<uint8_t> read_packed(const string &path) {
    ifstream f(path, ios::binary);
    if(!f) throw runtime_error("Failed to open packed file: " + path);
    f.seekg(0, ios::end);
    size_t size = f.tellg();
    f.seekg(0, ios::beg);
    vector<uint8_t> bytes(size);
    f.read((char*)bytes.data(), size);
    f.close();
    return bytes;
}

static vector<float> read_scales(const string& path) {
    ifstream sf(path);
    if(!sf) throw runtime_error("Failed to open scales: " + path);
    vector<float> s;
    float x;
    while(sf >> x) s.push_back(x);
    return s;
}

int main(int argc, char** argv){
    if(argc < 7) { cerr << "Usage: bitmatmul_xnor_opt <packed_file> <scales_txt> <out> <in> <input_vec_txt_or_bin> <mode> [threads]" << endl; return 1; }

    string packed_file = argv[1];
    string scales_file = argv[2];
    int out = atoi(argv[3]);
    int _in = atoi(argv[4]);
    string vec_file = argv[5];
    string mode = argv[6];
    int threads = 0; if(argc >= 8) threads = atoi(argv[7]);
    if(threads>0) omp_set_num_threads(threads);

    vector<uint8_t> packed = read_packed(packed_file);
    vector<float> scales = read_scales(scales_file);

    vector<float> vec; vector<uint8_t> vec_packed;
    bool vec_is_packed=false;
    if(mode == "random") {
        vec.resize(_in);
        for(int i=0;i<_in;i++) vec[i] = ((float)rand()/RAND_MAX)*2.0f - 1.0f;
    } else {
        if(mode == std::string("binact")){
            // read binary packed signbits
            ifstream vf(vec_file, ios::binary);
            if(!vf) throw runtime_error("Failed to open packed vector: " + vec_file);
            vf.seekg(0, ios::end);
            size_t s = vf.tellg(); vf.seekg(0, ios::beg);
            vec_packed.resize(s);
            vf.read((char*)vec_packed.data(), s);
            vec_is_packed = true;
            vf.close();
        } else {
            // floatact: read floats
            ifstream vtf(vec_file);
            if(!vtf) throw runtime_error("Failed to open input text vector: " + vec_file);
            vec.resize(_in);
            for(int i=0;i<_in;i++) vtf >> vec[i];
        }
    }

    int bytes_per_row = (_in + 7)/8;

    vector<float> outvec(out, 0.0f);
    bool use_avx512 = cpu_supports_avx512_vpopcntdq(); // best-case
    // fallback to AVX2-style 64-bit processing if AVX512 popcnt not available

#pragma omp parallel for schedule(static)
    for(int r=0;r<out;r++){
        const uint8_t* rowptr = packed.data() + r*bytes_per_row;
        if(mode == "binact" && vec_is_packed){
            // process 64-bit words, use AVX2 to accelerate loads but compute popcnt per 64-bit
            int total_equal = 0;
            int b = 0;
            for(; b + 8 <= bytes_per_row; b += 8){
                uint64_t w = *((uint64_t*)(rowptr + b));
                uint64_t a = *((uint64_t*)(vec_packed.data() + b));
                uint64_t x = ~(w ^ a);
                total_equal += popcount64(x);
            }
            for(; b < bytes_per_row; ++b){ uint8_t wb = rowptr[b], ab=vec_packed[b]; total_equal += __builtin_popcount((unsigned)(~(wb ^ ab))); }
            int valid_bits = _in;
            float val = 2.0f * (float) total_equal - (float) valid_bits;
            float s = (scales.size()==(size_t)out) ? scales[r] : ((scales.size()>0)?scales[0]:1.0f);
            outvec[r] = val * s;
        } else {
            float accum = 0.0f;
            for(int b=0;b<bytes_per_row;b++){
                uint8_t wb = rowptr[b];
                for(int bit=0; bit<8; ++bit){
                    int idx = b*8 + (7-bit); if(idx >= _in) break;
                    int bitval = (wb >> (7-bit)) & 1;
                    float sign = bitval ? 1.0f : -1.0f;
                    accum += sign * vec[idx];
                }
            }
            float s = (scales.size()==(size_t)out) ? scales[r] : ((scales.size()>0)?scales[0]:1.0f);
            outvec[r] = accum * s;
        }
    }

    for(int i=0;i<min(out, 32); ++i) cout << outvec[i] << " ";
    cout << endl;

    return 0;
}
