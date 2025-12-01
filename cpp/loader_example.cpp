// loader_example.cpp
// Demonstrates loading the quantized manifest and calling backend_1bit.so via dlopen/dlsym

#include <bits/stdc++.h>
#include <dlfcn.h>
using namespace std;

string read_file(const string &path) {
    ifstream f(path);
    if(!f) return "";
    string s((istreambuf_iterator<char>(f)), istreambuf_iterator<char>());
    f.close();
    return s;
}

string extract_nested_value(const string &json, const string &object_key, const string &nested_key) {
    // Find object_key (e.g., "fc.weight") and then find nested_key inside the following JSON object
    auto pos = json.find('"' + object_key + '"');
    if(pos == string::npos) return "";
    auto brace = json.find('{', pos);
    if(brace == string::npos) return "";
    // find matching closing brace by counting nested braces
    int depth = 0; size_t idx = brace;
    for(; idx < json.size(); ++idx){
        if(json[idx]=='{') depth++;
        else if(json[idx]=='}'){ depth--; if(depth==0) break; }
    }
    if(idx >= json.size()) return "";
    auto chunk = json.substr(brace, idx - brace + 1);
    // find nested key pair inside chunk
    auto p = chunk.find('"' + nested_key + '"');
    if(p == string::npos) return "";
    auto colon = chunk.find(':', p);
    if(colon == string::npos) return "";
    auto quote1 = chunk.find('"', colon+1);
    if(quote1 == string::npos) return "";
    auto quote2 = chunk.find('"', quote1+1);
    if(quote2 == string::npos) return "";
    return chunk.substr(quote1+1, quote2-quote1-1);
}

typedef int (*matmul_fn)(const char*, const char*, const float*, float*, int, int, int, int);

int main(int argc, char** argv){
    if(argc < 3){ cerr << "Usage: loader_example manifest.json backend_lib.so" << endl; return 1; }
    string manifest = argv[1];
    string libpath = argv[2];
    string json = read_file(manifest);
    if(json.empty()){ cerr << "Failed to read manifest" << endl; return 2; }
    string base = "fc.weight";
    string packed = extract_nested_value(json, base, "packed");
    // prefer text scale file for readability
    string scales = extract_nested_value(json, base, "scales_txt");
    if(scales.empty()) scales = extract_nested_value(json, base, "scales");
    // debug print chunk
    auto bpos = json.find('"' + base + '"');
    if(bpos != string::npos){
        auto brace = json.find('{', bpos);
        int depth = 0; size_t idx = brace;
        for(; idx < json.size(); ++idx){ if(json[idx]=='{') depth++; else if(json[idx]=='}'){ depth--; if(depth==0) break; }}
        cout << "Chunk: " << json.substr(brace, idx - brace + 1) << endl;
    }
    cout << "Packed: " << packed << " scales: " << scales << endl;

    // Print some architecture metadata if available for user feedback
    auto famPos = json.find("\"model\"");
    if(famPos != string::npos){
        // crude find for primary_family
        auto pfKey = json.find("\"primary_family\"", famPos);
        if(pfKey != string::npos){
            auto colon = json.find(':', pfKey);
            auto quote1 = json.find('"', colon+1);
            auto quote2 = json.find('"', quote1+1);
            if(quote1 != string::npos && quote2 != string::npos){
                cout << "Model family: " << json.substr(quote1+1, quote2-quote1-1) << endl;
            }
        }
    }

    // dlopen backend
    void* handle = dlopen(libpath.c_str(), RTLD_LAZY);
    if(!handle){ cerr << "dlopen failed: " << dlerror() << endl; return 3; }
    dlerror();
    matmul_fn fn = (matmul_fn) dlsym(handle, "matmul_1bit");
    const char* err = dlerror();
    if(err){ cerr << "dlsym failed: " << err << endl; return 4; }

    // Load scales to determine shape
    vector<float> sc;
    {
        ifstream sf(scales);
        float x;
        while(sf >> x) sc.push_back(x);
    }
    int out = (int) sc.size();
    // read shape from manifest or assume 64
    int in = 64;
    int threads = 0;
    vector<float> input(in);
    for(int i=0;i<in;i++) input[i] = (float) ((i%2==0) ? 1.0 : -1.0);
    vector<float> outv(out);
    int mode = 0; // float activation
    int ret = fn(packed.c_str(), scales.c_str(), input.data(), outv.data(), out, in, mode, threads);
    if(ret != 0) cerr << "matmul_1bit failed: " << ret << endl;
    cout << "Output[0..7]: ";
    for(int i=0;i<8 && i<out;i++) cout << outv[i] << " "; cout << endl;
    dlclose(handle);
    return 0;
}
