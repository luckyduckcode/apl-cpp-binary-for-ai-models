#include <bits/stdc++.h>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

static inline int popcount64(uint64_t x) { return __builtin_popcountll(x); }

// Read packed bytes and return as vector<uint8_t>
static vector<uint8_t> read_packed(const string& path) {
    ifstream f(path, ios::binary);
    if(!f) throw runtime_error("Failed to open packed: " + path);
    f.seekg(0, ios::end);
    size_t sz = f.tellg(); f.seekg(0, ios::beg);
    vector<uint8_t> buff(sz);
    f.read((char*)buff.data(), sz);
    f.close();
    return buff;
}

static vector<float> read_scales_txt(const string& path) {
    ifstream f(path);
    if(!f) throw runtime_error("Failed to open scales: " + path);
    vector<float> scales;
    string line;
    while(getline(f, line)){
        if(line.size()==0) continue;
        scales.push_back(stof(line));
    }
    return scales;
}

static vector<float> read_input_txt(const string& path, int in) {
    ifstream f(path);
    vector<float> v(in);
    for(int i=0;i<in;i++) f >> v[i];
    return v;
}

// helper to pack a float vector to sign bits (msb-first packbits byte layout per 8 bits)
static vector<uint8_t> pack_signs(const vector<float>& v) {
    int n = v.size();
    int bytes = (n + 7) / 8;
    vector<uint8_t> out(bytes);
    int byte_index = 0; int bit_index = 0; uint8_t cur = 0;
    for(int i=0;i<n;i++){
        int bit = (v[i] >= 0.0f) ? 1 : 0; // 1 for positive sign
        cur = (cur << 1) | (bit & 1);
        bit_index++;
        if(bit_index==8){ out[byte_index++] = cur; cur = 0; bit_index = 0; }
    }
    if(bit_index != 0){ cur <<= (8 - bit_index); out[byte_index++] = cur; }
    return out;
}

int main(int argc, char** argv){
    if(argc < 7){
        cerr << "Usage: bitmatmul_xnor <packed_file> <scales_txt> <out> <in> <input_vec_txt_or_random> <mode> [threads]" << endl;
        cerr << "mode: floatact or binact" << endl;
        return 1;
    }
    string packed_file = argv[1];
    string scales_file = argv[2];
    int out = atoi(argv[3]);
    int _in = atoi(argv[4]);
    string vec_file = argv[5];
    string mode = argv[6];
    int threads = 1;
    if(argc >= 8) threads = atoi(argv[7]);
#ifdef _OPENMP
    if(threads > 0) omp_set_num_threads(threads);
#endif

    auto bytes = read_packed(packed_file);
    auto scales = read_scales_txt(scales_file);

    // read input
    vector<float> vec;
    vector<uint8_t> vec_packed;
    if(vec_file == string("random")){
        vec.resize(_in);
        for(int i=0;i<_in;i++) vec[i] = ((float)rand()/RAND_MAX)*2.0f - 1.0f;
    } else {
        // detect if it's a packed file (binary) or plain text
        if(vec_file.rfind(".bin") == vec_file.size()-4){
            ifstream f(vec_file, ios::binary);
            f.seekg(0, ios::end);
            size_t size = f.tellg(); f.seekg(0, ios::beg);
            vec_packed.resize(size);
            f.read((char*)vec_packed.data(), size);
            f.close();
        } else {
            vec = read_input_txt(vec_file, _in);
        }
    }

    int bytes_per_row = (_in + 7) / 8;
    if((int)bytes.size() < out * bytes_per_row){
        cerr << "Packed size smaller than expected. Will try with min rows." << endl;
    }

    vector<float> outvec(out, 0.0f);

    // mode: binact uses prepacked activation signbits in vec_packed; else floatact uses vec
    if(mode == "binact"){
        if(vec_packed.empty()){ cerr << "binact mode requires input as packed signbits file" << endl; return 2; }
        // compute per-row xnor popcount
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for(int r=0; r<out; ++r){
            const uint8_t* rowptr = bytes.data() + r*bytes_per_row;
            int total_equal = 0;
            for(int b=0;b<bytes_per_row;b++){
                uint8_t wb = rowptr[b];
                uint8_t ab = vec_packed[b];
                uint8_t x = ~(wb ^ ab); // xnor
                total_equal += __builtin_popcount((unsigned)x);
            }
            // total_equal counts equal bits; adjust for last partial byte
            int valid_bits = _in;
            float val = (2.0f * (float)total_equal - (float)valid_bits);
            float s = (scales.size() == out) ? scales[r] : ((scales.size() > 0) ? scales[0] : 1.0f);
            outvec[r] = val * s;
        }
    } else if(mode == "floatact"){
        // fallback to dequantized compute
#ifdef _OPENMP
        #pragma omp parallel for
#endif
        for(int r=0;r<out;r++){
            const uint8_t* rowptr = bytes.data() + r*bytes_per_row;
            float accum = 0.0f;
            for(int b=0;b<bytes_per_row;b++){
                uint8_t byte = rowptr[b];
                for(int bit=0;bit<8;bit++){
                    int idx = b*8 + bit;
                    if(idx >= _in) break;
                    int bitval = (byte >> (7-bit)) & 1;
                    float sign = bitval ? 1.0f : -1.0f;
                    accum += sign * vec[idx];
                }
            }
            float s = (scales.size() == out) ? scales[r] : ((scales.size() > 0) ? scales[0] : 1.0f);
            outvec[r] = accum * s;
        }
    } else {
        cerr << "Unknown mode: use binact or floatact" << endl; return 2;
    }

    // print first 32 outputs
    for(int i=0;i<min((int)out, 32); ++i) cout << outvec[i] << " ";
    cout << endl;

    return 0;
}
