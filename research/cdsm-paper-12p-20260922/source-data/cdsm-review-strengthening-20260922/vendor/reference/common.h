#pragma once
#include <faiss/IndexHNSW.h>
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <omp.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;
inline double elapsed(Clock::time_point t) {
    return std::chrono::duration<double>(Clock::now()-t).count();
}
inline void need(bool b, const std::string& s) { if (!b) throw std::runtime_error(s); }
template<class T> struct Vectors { int d; size_t n; std::vector<T> x; };
template<class T> Vectors<T> readvec(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    need(bool(in), "Cannot read "+path);
    int d; in.read((char*)&d, 4); need(d>0 && d<100000, "Bad dimension");
    auto size = std::filesystem::file_size(path);
    need(size%(4+sizeof(T)*d)==0, "Bad fvecs/ivecs size");
    Vectors<T> v{d, size/(4+sizeof(T)*d), {}};
    v.x.resize(v.n*d); in.seekg(0);
    for (size_t i=0; i<v.n; ++i) {
        int row; in.read((char*)&row,4); need(row==d,"Mixed dimensions");
        in.read((char*)&v.x[i*d],sizeof(T)*d);
    }
    need(bool(in),"Short read"); return v;
}
template<class T> void writevec(const std::string& path, const std::vector<T>& x, int d) {
    need(x.size()%d==0,"Write shape");
    std::ofstream out(path,std::ios::binary);
    for(size_t i=0;i<x.size();i+=d){out.write((char*)&d,4);out.write((char*)&x[i],sizeof(T)*d);}
    need(bool(out),"Write failed "+path);
}
inline std::vector<uint> readattrs(size_t n) {
    std::vector<uint> a(n); std::ifstream in("data/attrs.u32",std::ios::binary);
    in.read((char*)a.data(),4*n); need(bool(in),"Read attributes"); return a;
}
inline std::unique_ptr<faiss::IndexHNSWFlat> loadindex() {
    faiss::Index* p=faiss::read_index("data/sift1m-M40-efC1000.faiss");
    auto* h=dynamic_cast<faiss::IndexHNSWFlat*>(p);
    need(h && h->ntotal==1000000 && h->d==128,"Wrong index");
    need(h->hnsw.efConstruction==1000 && h->hnsw.nb_neighbors(0)==80,"Wrong HNSW parameters");
    return std::unique_ptr<faiss::IndexHNSWFlat>(h);
}
