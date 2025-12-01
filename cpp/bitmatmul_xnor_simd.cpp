#include <bits/stdc++.h>
#include <immintrin.h>
#include <omp.h>

using namespace std;

static inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }

// Load binary file into vector<uint8_t>
static vector<uint8_t> load_bin(const string &path){
    ifstream f(path, ios::binary);
    if(!f) throw runtime_error("Failed to open: " + path);
    f.seekg(0, ios::end);
    size_t n = f.tellg(); f.seekg(0, ios::beg);
    vector<uint8_t> data(n);
    f.read((char*)data.data(), n);
    f.close();
    return data;
}

static vector<float> load_scales(const string &path){
    ifstream f(path);
    if(!f) throw runtime_error("Failed to open scales: " + path);
    vector<float> s;
    string line;
    while(getline(f, line)){
        if(line.empty()) continue;
        s.push_back(stof(line));
    }
    return s;
}

// Utility: get bit at position, but we process 64-bit chunks

int main(int argc, char** argv){
    if(argc < 7){
        cerr << "Usage: bitmatmul_xnor_simd <packed_file> <scales_txt> <out> <in> <input_vec_txt_or_bin> <mode> [threads]" << endl;
        cerr << "mode: floatact or binact" << endl;
        return 1;
    }
    string packed_file = argv[1];
    string scales_file = argv[2];
    int out = atoi(argv[3]);
    int _in = atoi(argv[4]);
    string vec_file = argv[5];
    string mode = argv[6];
    int threads = 0;
    if(argc >= 8) threads = atoi(argv[7]);
    if(threads > 0) omp_set_num_threads(threads);

    vector<uint8_t> packed = load_bin(packed_file);
    vector<float> scales = load_scales(scales_file);

    vector<float> vec; vector<uint8_t> vec_packed;
    if(mode == "random"){
        vec.resize(_in);
        for(int i=0;i<_in;i++) vec[i] = ((float)rand()/RAND_MAX)*2.0f - 1.0f;
    } else if(mode == "binact"){
        // load packed vector
        ifstream vf(vec_file, ios::binary);
        if(!vf) throw runtime_error("Failed to open sign bit vector: " + vec_file);
        vf.seekg(0, ios::end);
        size_t s = vf.tellg(); vf.seekg(0, ios::beg);
        vec_packed.resize(s);
        vf.read((char*)vec_packed.data(), s);
        vf.close();
    } else {
        // text floats
        ifstream vf(vec_file);
        if(!vf) throw runtime_error("Failed to open input file: " + vec_file);
        vec.resize(_in);
        for(int i=0;i<_in;i++) vf >> vec[i];
    }

    int bytes_per_row = (_in + 7)/8;

    vector<float> outvec(out, 0.0f);

    // Use AVX2 by processing 8x8 bytes -> 64-bit words
#ifdef _OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for(int r=0;r<out;r++){
        const uint8_t* rowptr = packed.data() + r*bytes_per_row;
        if(mode == "binact"){
            // binact: XNOR + popcount between rowptr and vec_packed
            int total_equal = 0;
            int maxb = bytes_per_row - ((bytes_per_row % 8));
            int b = 0;
            // process 8-byte blocks as 64-bit
            for(; b < maxb; b+=8){
                uint64_t w = *((uint64_t*)(rowptr + b));
                uint64_t a = *((uint64_t*)(vec_packed.data() + b));
                uint64_t x = ~(w ^ a);
                total_equal += popcount64(x);
            }
            // tail
            for(; b < bytes_per_row; ++b){
                uint8_t wb = rowptr[b];
                uint8_t ab = vec_packed[b];
                uint8_t x = ~(wb ^ ab);
                total_equal += __builtin_popcount((unsigned)x);
            }
            int valid_bits = _in;
            float val = 2.0f * (float)total_equal - (float)valid_bits;
            float s = (scales.size() == out) ? scales[r] : ((scales.size()>0)?scales[0]:1.0f);
            outvec[r] = val * s;
        } else { // floatact
            // dequantize and compute dot with float activation
            float accum = 0.0f;
            int idx = 0;
            int b=0;
            for(; b < bytes_per_row; ++b){
                uint8_t wb = rowptr[b];
                for(int bit=0; bit<8; ++bit){
                    int id = b*8 + (7-bit); // MSB-first
                    if(id >= _in) break;
                    int bitval = (wb >> (7-bit)) & 1;
                    float sign = bitval ? 1.0f : -1.0f;
                    accum += sign * vec[id];
                }
            }
            float s = (scales.size() == out) ? scales[r] : ((scales.size()>0)?scales[0]:1.0f);
            outvec[r] = accum * s;
        }
    }

    for(int i=0;i<min(out, 32); ++i) cout << outvec[i] << " ";
    cout << endl;

    return 0;
}
