// Filtered Faiss port of CrossMetricSearch.Workspace + CrossStudy.execute/scout.
// Native squared L2 replaces Lucene similarity. Graph, cache and cursor state
// are shared across routes; only the output collector applies the predicate.
#pragma once
#include "common.h"

namespace cdsm {
constexpr int K=10;
constexpr uint8_t BASE=1, COMPLETED=2, ENTRY=4, OPEN=8;
using Node=std::pair<float,int>;
constexpr float INF=std::numeric_limits<float>::infinity();

struct Top {
    std::array<Node,K> v;
    int size=0;
    void clear(){size=0;}
    float kth()const{return size<K?INF:v[K-1].first;}
    void collect(int n,float d){
        Node p{d,n};
        if(size==K && !(p<v[K-1]))return;
        int j=std::min(size,K-1);
        while(j>0 && p<v[j-1]){v[j]=v[j-1];--j;}
        v[j]=p;if(size<K)++size;
    }
};
struct MinHeap {
    std::vector<Node> v;
    bool empty()const{return v.empty();}
    Node top()const{return v.front();}
    void clear(){v.clear();}
    void add(int n,float d){v.emplace_back(d,n);std::push_heap(v.begin(),v.end(),std::greater<Node>());}
    int pop(){int n=v.front().second;std::pop_heap(v.begin(),v.end(),std::greater<Node>());v.pop_back();return n;}
};
struct Frontier {
    MinHeap active,deferred;
    int current=-1;
    void clear(){active.clear();deferred.clear();current=-1;}
};
struct JavaRandom {
    uint64_t state;
    explicit JavaRandom(uint64_t seed):state((seed^0x5deece66dULL)&((1ULL<<48)-1)){}
    uint32_t next(int bits){state=(state*0x5deece66dULL+11)&((1ULL<<48)-1);return uint32_t(state>>(48-bits));}
    int nextInt(int bound){
        need(bound>0,"positive bound");
        if((bound&-bound)==bound)return int((uint64_t(bound)*next(31))>>31);
        uint32_t bits,value;
        do{bits=next(31);value=bits%bound;}while(uint64_t(bits)-value+bound-1>=0x80000000ULL);
        return int(value);
    }
};
inline std::vector<int> entryTable(const faiss::HNSW& h){
    std::vector<int> pool,result;std::set<int> unique;
    for(int level=1;level<=h.max_level;++level)
        for(int n=0;n<int(h.levels.size());++n)if(h.levels[n]>level){pool.push_back(n);unique.insert(n);}
    JavaRandom rng(h.levels.size());std::set<int> used;
    size_t wanted=std::min(size_t(64),unique.size());
    while(result.size()<wanted){int n=pool[rng.nextInt(int(pool.size()))];if(used.insert(n).second)result.push_back(n);}
    return result;
}
struct Outcome {
    std::array<int,K> ids;
    std::array<float,K> ds;
    int dc=0,logical=0,edges=0,expansions=0,primary=0,stop=-1,scoutDc=0,scouts=0,upperDc=0;
    char selected='?';uint64_t sequence=0,primarySequence=0;
    int partialRemaining=0;
};

struct Workspace {
    const faiss::HNSW& h;
    std::unique_ptr<faiss::DistanceComputer> dis;
    std::vector<float> scores;
    std::vector<uint32_t> stamp,upperStamp;
    std::vector<int> pos,end,arena,touched,trace;
    std::vector<uint8_t> flags;
    uint32_t epoch=0,upperEpoch=0;
    const std::vector<uint8_t>* mask=nullptr;
    Frontier main,local;
    Top output,gate;
    int dc=0,logical=0,edges=0,expansions=0,scoutDc=0,scouts=0,upperDc=0;
    uint64_t sequence=0;
    bool diagnostic=false;

    explicit Workspace(const faiss::IndexHNSWFlat& index):h(index.hnsw),dis(index.storage->get_distance_computer()),
        scores(index.ntotal),stamp(index.ntotal),upperStamp(index.ntotal),pos(index.ntotal),end(index.ntotal),flags(index.ntotal){
        arena.reserve(131072);touched.reserve(16384);trace.reserve(16384);
    }
    bool scored(int n)const{return stamp[n]==epoch;}
    bool marked(int n,int flag)const{return scored(n) && (flags[n]&flag)!=0;}
    void mark(int n,int flag){flags[n]|=uint8_t(flag);}
    void reset(const float* q,const std::vector<uint8_t>& m){
        if(++epoch==0){std::fill(stamp.begin(),stamp.end(),0);epoch=1;}
        mask=&m;dis->set_query(q);arena.clear();touched.clear();trace.clear();
        dc=logical=edges=expansions=scoutDc=scouts=upperDc=0;
        sequence=0xcbf29ce484222325ULL;main.clear();local.clear();output.clear();gate.clear();
    }
    float score(int n){
        ++logical;if(scored(n))return scores[n];
        float d=(*dis)(n);++dc;sequence=(sequence^uint64_t(n))*0x100000001b3ULL;
        if(diagnostic)trace.push_back(n);
        stamp[n]=epoch;flags[n]=0;scores[n]=d;touched.push_back(n);
        if((*mask)[n])output.collect(n,d);
        return d;
    }
    int descend(int entry,int level,int cap){
        int before=dc;if(dc>=cap && !scored(entry))return -1;
        int ep=entry;float best=score(ep);
        if(++upperEpoch==0){std::fill(upperStamp.begin(),upperStamp.end(),0);upperEpoch=1;}
        for(int l=level;l>=1 && dc<cap;--l){
            bool changed=true;upperStamp[ep]=upperEpoch;
            while(changed && dc<cap){
                changed=false;size_t a,b;h.neighbor_range(ep,l,&a,&b);
                for(size_t j=a;j<b;++j){
                    int n=h.neighbors[j];if(n<0)break;++edges;
                    if(marked(n,BASE)||upperStamp[n]==upperEpoch)continue;
                    if(dc>=cap){upperDc+=dc-before;return ep;}
                    upperStamp[n]=upperEpoch;float d=score(n);
                    if(d<best){best=d;ep=n;changed=true;}
                }
            }
        }
        upperDc+=dc-before;return ep;
    }
    void seed(Frontier& f,int ep){
        if(ep<0 || marked(ep,BASE))return;
        float d=score(ep);mark(ep,BASE);gate.collect(ep,d);f.active.add(ep,d);
    }
    void initialize(int cap){int ep=descend(h.entry_point,h.max_level,cap);if(dc<cap)seed(main,ep);}
    void open(int n){
        if(marked(n,OPEN))return;
        size_t a,b;h.neighbor_range(n,0,&a,&b);pos[n]=int(arena.size());
        for(size_t j=a;j<b;++j){int v=h.neighbors[j];if(v<0)break;++edges;arena.push_back(v);}
        end[n]=int(arena.size());mark(n,OPEN);
    }
    bool complete(Frontier& f){
        int n=f.current;if(pos[n]!=end[n])return false;
        need(!marked(n,COMPLETED),"Duplicate adjacency completion");
        mark(n,COMPLETED);++expansions;f.current=-1;return true;
    }
    int advance(Frontier& f,int cap){
        while(dc<cap){
            if(f.current>=0){
                if(complete(f))continue;
                int n=arena[pos[f.current]];
                if(marked(n,BASE)){++pos[f.current];continue;}
                float tau=gate.kth(),d=score(n);++pos[f.current];mark(n,BASE);
                if(d<tau){f.active.add(n,d);gate.collect(n,d);}else f.deferred.add(n,d);
                continue;
            }
            float tau=gate.kth();
            while(!f.deferred.empty() && f.deferred.top().first<tau){
                int n=f.deferred.pop();if(!marked(n,COMPLETED))f.active.add(n,scores[n]);
            }
            while(!f.active.empty() && marked(f.active.top().second,COMPLETED))f.active.pop();
            if(f.active.empty())return f.deferred.empty()?2:1;
            if(f.active.top().first>tau)return 1;
            f.current=f.active.pop();open(f.current);
        }
        return 0;
    }
    int distanceAdvance(int cap,double gamma){
        Frontier& f=main;
        while(dc<cap){
            if(f.current>=0){
                if(complete(f))continue;
                int n=arena[pos[f.current]++];if(marked(n,BASE))continue;
                float d=score(n);mark(n,BASE);f.active.add(n,d);gate.collect(n,d);continue;
            }
            while(!f.deferred.empty()){
                int n=f.deferred.pop();if(!marked(n,COMPLETED))f.active.add(n,scores[n]);
            }
            while(!f.active.empty() && marked(f.active.top().second,COMPLETED))f.active.pop();
            if(f.active.empty())return 2;
            if(std::isfinite(gamma) && output.size==K && double(f.active.top().first)>=(1+gamma)*(1+gamma)*output.kth())return 1;
            f.current=f.active.pop();open(f.current);
        }
        return 0;
    }
    int nextEntry(const std::vector<int>& table)const{
        for(int n:table)if(!marked(n,ENTRY) && !marked(n,BASE))return n;
        return -1;
    }
    void scout(const std::vector<int>& table,int cap,int quota,bool injectOnly=false){
        int entry=nextEntry(table);if(entry<0 || dc>=cap || quota<=0)return;
        ++scouts;int before=dc,stop=std::min(cap,before+quota);
        local.clear();int ep=descend(entry,h.levels[entry]-1,stop);
        mark(entry,ENTRY);gate.clear();
        if(injectOnly){if(dc<stop)seed(main,ep);}
        else{
            if(dc<stop)seed(local,ep);advance(local,stop);
            for(auto p:local.active.v)main.active.add(p.second,p.first);
            for(auto p:local.deferred.v)main.deferred.add(p.second,p.first);
            if(local.current>=0)main.active.add(local.current,scores[local.current]);
        }
        scoutDc+=dc-before;
    }
    Outcome finish()const{
        need(dc==int(touched.size()),"Unique distance accounting");
        Outcome r;r.ids.fill(-1);r.ds.fill(INF);
        for(int j=0;j<output.size;++j){r.ids[j]=output.v[j].second;r.ds[j]=output.v[j].first;}
        r.dc=dc;r.logical=logical;r.edges=edges;r.expansions=expansions;r.scoutDc=scoutDc;r.scouts=scouts;r.upperDc=upperDc;r.sequence=sequence;
        r.partialRemaining=main.current<0?0:end[main.current]-pos[main.current];
        return r;
    }
    Outcome run(const float* q,const std::vector<uint8_t>& m,const std::vector<int>& table,char policy,int primaryCap=6400,int extraCap=6400){
        reset(q,m);initialize(primaryCap);
        int stop=distanceAdvance(primaryCap,.075),primary=dc;
        uint64_t primarySequence=sequence;
        char chosen=policy=='S'?(stop==0?'F':'C'):policy;
        if(chosen!='P'){
            int routes=(chosen=='F'||chosen=='M')?4:chosen=='O'?1:0;
            int quota=chosen=='O'?1600:400;
            for(int i=0;i<routes;++i)scout(table,primary+extraCap,quota,chosen=='M');
            distanceAdvance(primary+extraCap,std::numeric_limits<double>::infinity());
        }
        need(dc<=primary+extraCap,"Budget overshoot");
        Outcome r=finish();r.primary=primary;r.stop=stop;r.selected=chosen;r.primarySequence=primarySequence;
        return r;
    }
};
}
