#include <bits/stdc++.h>
using namespace std;

static inline int popcount8(uint8_t x){ return __builtin_popcount((unsigned)x); }

int main(int argc, char** argv){
    if(argc < 6){
        cerr << "Usage: bitmatmul <packed_file> <scales_txt> <out> <in> <input_vec_txt_or_random>" << endl;
        return 1;
    }
    string packed_file = argv[1];
    string scales_file = argv[2];
    int out = atoi(argv[3]);
    int _in = atoi(argv[4]);
    string vec_file = argv[5];

    // Read packed file: assume it's exactly out * ceil(_in / 8) bytes
    ifstream f(packed_file, ios::binary);
    if(!f){ cerr << "Failed to open packed file: " << packed_file << endl; return 2; }
    vector<uint8_t> bytes;
    f.seekg(0, ios::end);
    size_t size = f.tellg();
    f.seekg(0, ios::beg);
    bytes.resize(size);
    f.read((char*)bytes.data(), size);
    f.close();

    int bytes_per_row = ( _in + 7 ) / 8;
    if(size != (size_t)out * bytes_per_row){
        cerr << "Packed file length mismatch: expected" << out * bytes_per_row << " got " << size << endl;
        // do not exit; we will continue with as many rows we can
    }

    // Read scales
    vector<float> scales;
    {
        ifstream sf(scales_file);
        if(!sf){ cerr << "Failed to open scales file: " << scales_file << endl; return 3; }
        string line;
        while(getline(sf, line)){
            if(line.size() == 0) continue;
            float v = stof(line);
            scales.push_back(v);
        }
    }

    if((int)scales.size() != out && (int)scales.size() != _in){
        cerr << "Scales length mismatch: scales.txt length=" << scales.size()
             << " expected either out=" << out << " or in=" << _in << endl;
        // continue
    }

    // Read input vector
    vector<float> vec(_in);
    if(vec_file == string("random")){
        for(int i=0;i<_in;i++) vec[i] = ((float)rand()/RAND_MAX)*2.0f - 1.0f;
    } else {
        ifstream vf(vec_file);
        if(!vf){ cerr << "Failed to open input vector file: " << vec_file << "; defaulting to random" << endl; for(int i=0;i<_in;i++) vec[i] = ((float)rand()/RAND_MAX)*2.0f - 1.0f; }
        else{
            for(int i=0;i<_in;i++) vf >> vec[i];
            vf.close();
        }
    }

    // Compute output
    vector<float> outvec(out, 0.0f);
    for(int r=0;r<out;r++){
        const uint8_t* rowptr = bytes.data() + r * bytes_per_row;
        float accum = 0.0f;
        for(int b=0;b<bytes_per_row;b++){
            uint8_t byte = rowptr[b];
            for(int bit=0;bit<8;bit++){
                int idx = b*8 + bit;
                if(idx >= _in) break;
                int bitval = (byte >> (7 - bit)) & 1; // use MSB-first packbits
                float sign = bitval ? 1.0f : -1.0f;
                float s = (scales.size() == out) ? scales[r] : scales[idx];
                accum += sign * s * vec[idx];
            }
        }
        outvec[r] = accum;
    }

    // print first 16 outputs
    for(int i=0;i<std::min(out, 16); ++i){
        cout << outvec[i] << " ";
    }
    cout << endl;

    return 0;
}
