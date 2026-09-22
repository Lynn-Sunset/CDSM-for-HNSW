#pragma once
#include "vendor/paid.h"

namespace r2 {
struct Audit {
    std::vector<int> entries,endpoints,costs,upper,seeded,complete,roots,skipped;
    int phaseEnd=0,primaryStop=-1;uint64_t phaseSequence=0;
};
struct Run {cdsm::Outcome out;Audit audit;};
inline void merge(cdsm::Workspace& w){
    for(auto p:w.local.active.v)w.main.active.add(p.second,p.first);
    for(auto p:w.local.deferred.v)w.main.active.add(p.second,p.first);
    if(w.local.current>=0)w.main.active.add(w.local.current,w.scores[w.local.current]);
}
inline std::vector<int> roots(const cdsm::Workspace& w){
    std::set<cdsm::Node> candidates;
    for(const auto* heap:{&w.main.active,&w.main.deferred})for(auto p:heap->v)
        if(p.second!=w.main.current && !w.marked(p.second,cdsm::COMPLETED))candidates.insert(p);
    std::vector<int> result;
    for(auto p:candidates){result.push_back(p.second);if(result.size()==4)break;}
    return result;
}
inline void frontierScouts(cdsm::Workspace& w,int cap,Audit& audit,bool capture){
    auto selected=roots(w);if(capture)audit.roots=selected;
    for(int root:selected){
        if(w.dc>=cap)break;
        if(w.marked(root,cdsm::COMPLETED)){if(capture)audit.skipped.push_back(root);continue;}
        need(root!=w.main.current && w.marked(root,cdsm::BASE),"pending scored root, exclude main partial current");
        int before=w.dc,limit=std::min(cap,w.dc+400);++w.scouts;
        w.local.clear();w.gate.clear();
        float d=w.score(root); // Already paid in the common prefix; real cache request, zero new DC.
        w.gate.collect(root,d);w.local.active.add(root,d);
        w.advance(w.local,limit);merge(w);w.scoutDc+=w.dc-before;
        if(capture){audit.entries.push_back(root);audit.endpoints.push_back(root);audit.costs.push_back(w.dc-before);
            audit.upper.push_back(0);audit.seeded.push_back(1);audit.complete.push_back(1);}
    }
}
inline Run search(cdsm::Workspace& w,const float* q,int d,const std::vector<uint8_t>& mask,
                  const std::vector<int>& table,const std::string& policy,int cap,bool capture=false,int primaryCap=6400){
    Run r;
    if(policy=="C"||policy=="F"){
        auto old=paid::search(w,q,d,mask,table,policy,primaryCap,6400,800,cap);
        r.out=old.out;r.audit.phaseEnd=old.scoutEnd;r.audit.phaseSequence=old.scoutSequence;r.audit.primaryStop=old.out.stop;
        if(capture){r.audit.entries=old.actualEntries;r.audit.endpoints=old.actualEndpoints;r.audit.costs=old.scoutCosts;}
    }else{
        int prefixLimit=std::min(primaryCap,cap);
        w.reset(q,mask);w.initialize(prefixLimit);int stop=w.distanceAdvance(prefixLimit,.075),primary=w.dc;
        uint64_t prefixSequence=w.sequence;r.audit.primaryStop=stop;
        if(policy.rfind("MEP",0)==0){
            int m=std::stoi(policy.substr(3));need(m==1||m==4||m==8||m==16,"MEP count");
            for(int i=0;i<m && w.dc<cap;++i){
                int entry=w.nextEntry(table);if(entry<0)break;
                int before=w.dc,up=w.upperDc;++w.scouts;w.local.clear();
                auto descent=paid::descend(w,entry,w.h.levels[entry]-1,cap);
                w.mark(entry,cdsm::ENTRY);w.gate.clear();
                bool seed=descent.complete && w.dc<cap && !w.marked(descent.endpoint,cdsm::BASE);
                if(seed)w.seed(w.main,descent.endpoint);
                w.scoutDc+=w.dc-before;
                if(capture){r.audit.entries.push_back(entry);r.audit.endpoints.push_back(descent.endpoint);
                    r.audit.costs.push_back(w.dc-before);r.audit.upper.push_back(w.upperDc-up);
                    r.audit.seeded.push_back(seed);r.audit.complete.push_back(descent.complete);}
            }
        }else if(policy=="FRONTIER")frontierScouts(w,cap,r.audit,capture);
        else{
            int quota=policy=="F200"?200:policy=="F800"?800:policy=="F400_CHECK"?400:-1;
            need(quota>0,"known policy");
            for(int i=0;i<4;++i){
                int entry=w.nextEntry(table);if(entry<0)break;
                int before=w.dc,up=w.upperDc;
                int ep=paid::scout(w,entry,cap,quota,-1);
                if(capture){r.audit.entries.push_back(entry);r.audit.endpoints.push_back(ep);r.audit.costs.push_back(w.dc-before);r.audit.upper.push_back(w.upperDc-up);}
            }
        }
        r.audit.phaseEnd=w.dc;r.audit.phaseSequence=w.sequence;
        w.distanceAdvance(cap,INFINITY);r.out=w.finish();r.out.primary=primary;r.out.primarySequence=prefixSequence;
        r.out.stop=stop;
    }
    need(r.out.dc>0 && r.out.dc<=cap,"hard physical cap");
    need(r.out.primary+r.out.scoutDc<=r.out.dc,"phase accounting");
    r.out.stop=r.out.dc==cap?0:2;
    return r;
}
inline void same(const cdsm::Outcome& a,const cdsm::Outcome& b){
    need(a.ids==b.ids&&a.ds==b.ds&&a.dc==b.dc&&a.sequence==b.sequence&&a.primarySequence==b.primarySequence&&
         a.primary==b.primary&&a.scoutDc==b.scoutDc&&a.upperDc==b.upperDc&&a.logical==b.logical&&
         a.edges==b.edges&&a.expansions==b.expansions&&a.scouts==b.scouts&&a.stop==b.stop,"path/counter/result parity");
}
}
