#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <vector>
namespace bio {
using Clock=std::chrono::steady_clock;
inline void need(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
inline long long ns(Clock::time_point t){return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-t).count();}
template<class T> struct Matrix {int n,d;std::vector<T> x;};
template<class T> Matrix<T> bin(const std::string& name){
  std::ifstream f(name,std::ios::binary);need(bool(f),"read "+name);Matrix<T> a;
  f.read((char*)&a.n,4);f.read((char*)&a.d,4);need(a.n>0&&a.d>0,"bin shape");
  a.x.resize(size_t(a.n)*a.d);f.read((char*)a.x.data(),a.x.size()*sizeof(T));need(bool(f),"short bin");return a;
}
template<class T> Matrix<T> vecs(const std::string& name){
  std::ifstream f(name,std::ios::binary);need(bool(f),"read "+name);int d;f.read((char*)&d,4);need(d>0&&d<1000,"vecs dimension");
  auto sz=std::filesystem::file_size(name);need(sz%(4+sizeof(T)*d)==0,"vecs size");Matrix<T> a{int(sz/(4+sizeof(T)*d)),d,{}};
  a.x.resize(size_t(a.n)*d);f.seekg(0);for(int i=0;i<a.n;i++){int dd;f.read((char*)&dd,4);need(dd==d,"vecs mixed dimension");f.read((char*)&a.x[size_t(i)*d],d*sizeof(T));}need(bool(f),"short vecs");return a;
}
template<class T> std::vector<T> raw(const std::string& name,int n){std::ifstream f(name,std::ios::binary);need(bool(f),"raw "+name);std::vector<T> a(n);f.read((char*)a.data(),n*sizeof(T));need(bool(f),"short raw");return a;}
inline std::vector<int> sortedIds(const std::vector<uint32_t>& attrs){std::vector<int> ids(attrs.size());std::iota(ids.begin(),ids.end(),0);std::sort(ids.begin(),ids.end(),[&](int a,int b){return std::make_pair(attrs[a],a)<std::make_pair(attrs[b],b);});return ids;}
inline std::pair<int,int> bounds(const std::vector<uint32_t>& sorted,uint32_t lo,uint32_t hi){return {int(std::lower_bound(sorted.begin(),sorted.end(),lo)-sorted.begin()),int(std::upper_bound(sorted.begin(),sorted.end(),hi)-sorted.begin())-1};}
inline int hits(const std::vector<int>& ids,const int* gt){int h=0;std::set<int> seen;for(int id:ids){need(seen.insert(id).second,"duplicate result");if(std::find(gt,gt+10,id)!=gt+10)h++;}return h;}
inline void ids(std::ostream& o,const std::vector<int>& a){bool first=true;for(int x:a){if(!first)o<<';';o<<x;first=false;}}
}
