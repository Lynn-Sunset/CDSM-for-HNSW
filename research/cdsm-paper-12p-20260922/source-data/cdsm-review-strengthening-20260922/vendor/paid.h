// Port of frozen BudgetEntryStudy FIX_*_800 + OnlineEntryStudy helpers.
// Raw Faiss squared L2 orders smaller values first. No ground truth or query ID.
#pragma once
#include "reference/cdsm.h"
#include <cstring>
#include <unordered_set>

namespace paid {
struct Candidate {
    int entry, endpoint, order, cost;
    float distance;
    std::vector<int> neighbors, unseen;
};
struct Run {
    cdsm::Outcome out;
    int cap=0, screenDc=0, featureEdges=0, attempted=0, partialEntry=-1, partialEndpoint=-1, partialLevel=0, partialDc=0;
    int screenEnd=0, scoutEnd=0;
    uint64_t screenSequence=0, scoutSequence=0;
    std::vector<Candidate> candidates;
    std::vector<int> planned, actualEntries, actualEndpoints, scoutCosts;
};
struct Descent { int endpoint; bool complete; int remainingLevel; };
inline Descent descend(cdsm::Workspace& w,int entry,int level,int cap){
    int before=w.dc;if(w.dc>=cap&&!w.scored(entry))return {-1,false,level};
    int ep=entry;float best=w.score(ep);
    if(++w.upperEpoch==0){std::fill(w.upperStamp.begin(),w.upperStamp.end(),0);w.upperEpoch=1;}
    int l=level;
    for(;l>=1;--l){
        if(w.dc>=cap){w.upperDc+=w.dc-before;return {ep,false,l};}
        bool changed=true;w.upperStamp[ep]=w.upperEpoch;
        while(changed){
            if(w.dc>=cap){w.upperDc+=w.dc-before;return {ep,false,l};}
            changed=false;size_t a,b;w.h.neighbor_range(ep,l,&a,&b);
            for(size_t j=a;j<b;++j){
                int n=w.h.neighbors[j];if(n<0)break;++w.edges;
                if(w.marked(n,cdsm::BASE)||w.upperStamp[n]==w.upperEpoch)continue;
                if(w.dc>=cap){w.upperDc+=w.dc-before;return {ep,false,l};}
                w.upperStamp[n]=w.upperEpoch;float d=w.score(n);
                if(d<best){best=d;ep=n;changed=true;}
            }
        }
    }
    w.upperDc+=w.dc-before;return {ep,true,0};
}
inline uint32_t javaFloatArrayHash(const float* q,int d){
    uint32_t hash=1;
    for(int i=0;i<d;++i){uint32_t bits;std::memcpy(&bits,q+i,4);if(std::isnan(q[i]))bits=0x7fc00000u;hash=31u*hash+bits;}
    return hash;
}
inline std::vector<int> randomized(const cdsm::Workspace& w,const std::vector<int>& table,const float* q,int d){
    std::vector<int> pool;for(int n:table)if(!w.marked(n,cdsm::BASE)&&!w.marked(n,cdsm::ENTRY))pool.push_back(n);
    cdsm::JavaRandom rng(0x51e17a20260918ULL ^ uint64_t(javaFloatArrayHash(q,d)));
    for(int i=int(pool.size())-1;i>0;--i)std::swap(pool[i],pool[rng.nextInt(i+1)]);
    return pool;
}
inline Candidate inspect(cdsm::Workspace& w,int entry,int ep,int order,int cost){
    Candidate c{entry,ep,order,cost,w.scores[ep],{}, {}};size_t a,b;w.h.neighbor_range(ep,0,&a,&b);
    for(size_t j=a;j<b;++j){int n=w.h.neighbors[j];if(n<0)break;++w.edges;c.neighbors.push_back(n);if(!w.marked(n,cdsm::BASE))c.unseen.push_back(n);}
    return c;
}
inline int novelty(const Candidate& c,const std::unordered_set<int>& covered){
    int n=0;for(int x:c.unseen)if(!covered.count(x))++n;return n;
}
inline bool better(const Candidate& a,const Candidate& b,const std::string& rule,int picked,const std::unordered_set<int>& covered){
    if(rule=="FIRST")return a.order<b.order;
    if(rule=="DIVERSE"&&picked>0){
        int64_t left=int64_t(novelty(a,covered))*std::max(size_t(1),b.neighbors.size());
        int64_t right=int64_t(novelty(b,covered))*std::max(size_t(1),a.neighbors.size());
        if(left!=right)return left>right;
    }
    if(a.distance!=b.distance)return a.distance<b.distance;return a.order<b.order;
}
inline std::vector<int> select(const std::vector<Candidate>& candidates,const std::string& rule){
    std::vector<int> chosen;std::unordered_set<int> landings,covered;
    while(chosen.size()<4){
        int best=-1;for(int i=0;i<int(candidates.size());++i)if(!landings.count(candidates[i].endpoint)&&(best<0||better(candidates[i],candidates[best],rule,int(chosen.size()),covered)))best=i;
        if(best<0)break;chosen.push_back(best);const auto& c=candidates[best];landings.insert(c.endpoint);covered.insert(c.endpoint);covered.insert(c.neighbors.begin(),c.neighbors.end());
    }
    return chosen;
}
inline int scout(cdsm::Workspace& w,int entry,int cap,int quota,int existingEndpoint){
    if(entry<0||w.dc>=cap)return -1;
    int before=w.dc,limit=std::min(cap,before+quota);++w.scouts;w.local.clear();
    int ep=existingEndpoint>=0?existingEndpoint:w.descend(entry,w.h.levels[entry]-1,limit);
    w.mark(entry,cdsm::ENTRY);w.gate.clear();
    if(w.dc<limit)w.seed(w.local,ep);w.advance(w.local,limit);
    w.scoutDc+=w.dc-before;
    for(auto p:w.local.active.v)w.main.active.add(p.second,p.first);
    for(auto p:w.local.deferred.v)w.main.deferred.add(p.second,p.first);
    if(w.local.current>=0)w.main.active.add(w.local.current,w.scores[w.local.current]);
    return ep;
}
inline Run search(cdsm::Workspace& w,const float* q,int d,const std::vector<uint8_t>& mask,const std::vector<int>& table,const std::string& rule,int primaryCap=6400,int extra=6400,int screenBudget=800,int absoluteCap=-1){
    if(absoluteCap>=0)primaryCap=std::min(primaryCap,absoluteCap);
    w.reset(q,mask);w.initialize(primaryCap);int stop=w.distanceAdvance(primaryCap,.075),primary=w.dc;
    uint64_t primarySequence=w.sequence;Run r;r.cap=absoluteCap>=0?absoluteCap:primary+extra;
    if(rule!="C"&&rule!="F"){
        auto pool=randomized(w,table,q,d);int limit=std::min(r.cap,primary+screenBudget);
        for(int i=0;i<int(pool.size())&&w.dc<limit;++i){
            int before=w.dc;++r.attempted;auto ds=descend(w,pool[i],w.h.levels[pool[i]]-1,limit);
            if(!ds.complete){r.partialEntry=pool[i];r.partialEndpoint=ds.endpoint;r.partialLevel=ds.remainingLevel;r.partialDc=w.dc-before;break;}
            auto c=inspect(w,pool[i],ds.endpoint,i,w.dc-before);r.featureEdges+=int(c.neighbors.size());r.candidates.push_back(std::move(c));
        }
    }
    r.screenDc=w.dc-primary;r.screenEnd=w.dc;r.screenSequence=w.sequence;
    r.planned=select(r.candidates,rule);
    if(rule!="C")for(int i=0;i<4;++i){
        const Candidate* c=i<int(r.planned.size())?&r.candidates[r.planned[i]]:nullptr;
        int entry=c?c->entry:w.nextEntry(table);if(entry<0)break;
        int before=w.dc,ep=scout(w,entry,r.cap,400,c?c->endpoint:-1);
        r.actualEntries.push_back(entry);r.actualEndpoints.push_back(ep);r.scoutCosts.push_back(w.dc-before);
    }
    r.scoutEnd=w.dc;r.scoutSequence=w.sequence;w.distanceAdvance(r.cap,INFINITY);
    r.out=w.finish();r.out.primary=primary;r.out.stop=stop;r.out.primarySequence=primarySequence;
    r.out.selected=rule=="C"?'C':rule=="F"?'F':'A';
    need(r.out.dc<=r.cap&&r.screenDc<=screenBudget&&r.out.scouts<=4&&r.out.scoutDc<=1600,"Paid phase budgets");
    need(r.out.dc==int(w.touched.size()),"Paid unique physical accounting");
    return r;
}
}
