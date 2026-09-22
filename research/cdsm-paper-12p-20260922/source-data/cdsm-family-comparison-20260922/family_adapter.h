// Reuse the complete frozen executor, with only its CLI entrypoint renamed.
// vendor_sources.py changes that ONE declaration in a separate generated file;
// no macro is used, because main is also a workspace/Snapshot field name.
// vendor_sources.py copies every dependency byte-for-byte and records hashes.
// New production wrappers avoid archived diagnostic string assembly; old search
// primitives, cursor scheduling, gate rules, and distance accounting are reused.
#pragma once
#include "vendor/action-space-v10-20260920/space_body.inc"

namespace family {
struct Run {
    Answer answer;
    int prefixDc=-1,selectionDc=0,routeDc=0,routeUpper=0,routeStop=-1;
    int entry=-1,endpoint=-1,eligible=0;
    uint64_t selectionSeq=0;
};

inline Run fromEntranceResult(EntranceAnswer e){
    Run r;r.prefixDc=e.prefixDc;r.selectionDc=e.selectionDc;r.routeDc=e.routeDc;
    r.routeUpper=e.routeUpper;r.routeStop=e.routeStop;r.entry=e.entry;r.endpoint=e.endpoint;
    r.eligible=e.eligible;r.selectionSeq=e.selectionSeq;r.answer=std::move(e.a);return r;
}

// B = P + extra, where P is the actual initial 6400-cap primary search cost.
// A fresh invocation always pays its own primary search, preview, scout work,
// entrance selection, and final continuation. Nothing is replayed for free.
inline Run search(Loaded& l,const float* q,const std::string& method,int extra){
    if(extra!=3200 && extra!=6400)throw std::runtime_error("frozen family extras are 3200/6400");
    if(method=="C" || method=="F" || method=="SCORE" || method=="DIVERSE"){
        Run r;r.answer=execute(*l.h,*l.w,q,l.q.d,l.mask,l.table,method,extra);return r;
    }
    auto began=Clock::now();Run r;
    if(method=="KEEP"){
        // Exactly the original DFRONT_300 prefix, followed by the same merge and
        // main continuation. finishStop's route-observation strings are omitted.
        auto state=stopPrefix(l,q,extra);auto& w=*l.w;r.prefixDc=w.dc;
        mergePrefix(w,state);
        r.answer.tailStop=w.distanceAdvance(state.c.preview.cap,INFINITY);
        r.answer.out=w.finish();fillContext(r.answer,state.c);
        need(r.answer.out.dc==r.answer.cap,"KEEP exact physical budget");
    }else if(method=="E64_RAW"){
        need(l.table.size()==64,"E64 uses original frozen 64-node table");
        // The complete KEEP scout prefix runs first. The EXTRA action has an
        // 800-distance envelope, including all eligible entrance scores. RAW
        // starts at the selected upper node directly on L0, without descent.
        auto point=checkpoint(l,q,extra,false);
        auto screen=scorePool(*l.w,l.table,64,point.actionCap);
        r=fromEntranceResult(fromEntrance(l,point,screen,64,screen.nearest,"RAW"));
    }else throw std::runtime_error("unknown frozen family method: "+method);
    r.answer.ns=bio::ns(began);return r;
}

// Original complete dispatch for engineering comparison. Never call this from
// the measured production path. The SCORE baseline has no V10 grid dispatch;
// its original frozen execute() is the appropriate reference.
inline Run reference(Loaded& l,const float* q,const std::string& method,int extra){
    if(method=="C" || method=="F" || method=="SCORE" || method=="DIVERSE"){
        Run r;r.answer=execute(*l.h,*l.w,q,l.q.d,l.mask,l.table,method,extra);return r;
    }
    auto r=spaceSearch(l,l.table,q,method,extra);
    return fromEntranceResult(std::move(r.a.r.e));
}

inline void requireEquivalent(const Answer& a,const Answer& b){
    same(a,b,"family adapter matches archived output and physical trace hash");
    const auto& x=a.out;const auto& y=b.out;
    need(x.primary==y.primary && x.primarySequence==y.primarySequence &&
         x.stop==y.stop && x.scoutDc==y.scoutDc && x.scouts==y.scouts &&
         x.upperDc==y.upperDc && x.partialRemaining==y.partialRemaining &&
         a.cap==b.cap && a.preview==b.preview && a.previewSeq==b.previewSeq &&
         a.candidates==b.candidates && a.pool==b.pool && a.tailStop==b.tailStop,
         "family archived accounting, stops, and budget equality");
}

inline void requireEntranceEquivalent(const Run& a,const Run& b){
    need(a.prefixDc==b.prefixDc && a.selectionDc==b.selectionDc &&
         a.selectionSeq==b.selectionSeq && a.routeDc==b.routeDc &&
         a.routeUpper==b.routeUpper && a.routeStop==b.routeStop &&
         a.entry==b.entry && a.endpoint==b.endpoint && a.eligible==b.eligible,
         "E64 archived entrance selection and action equality");
}

// One query, one fixed extra budget: all six family methods, whole physical
// sequences, original KEEP/DFRONT alias, and diagnostic-off path equality.
// No truth/recall enters these checks or any search decision.
inline int engineeringCheck(Loaded& l,const float* q,int extra){
    auto& w=*l.w;bool diagnosticBefore=w.diagnostic;int comparisons=0;
    std::pair<int,uint64_t> primaryKey{-1,0};
    for(const std::string method:{"C","F","SCORE","DIVERSE","KEEP","E64_RAW"}){
        w.diagnostic=true;auto actual=search(l,q,method,extra);auto trace=w.trace;
        auto key=std::make_pair(actual.answer.out.primary,actual.answer.out.primarySequence);
        if(primaryKey.first<0)primaryKey=key;else need(primaryKey==key,"all family variants share actual primary sequence");
        need(actual.answer.out.dc==actual.answer.cap && actual.answer.cap==key.first+extra,"strict per-query B=P+E");
        auto archived=reference(l,q,method,extra);requireEquivalent(actual.answer,archived.answer);
        need(trace==w.trace,"family adapter archived complete physical scoring ID sequence");++comparisons;
        if(method=="E64_RAW")requireEntranceEquivalent(actual,archived);
        if(method=="KEEP"){
            auto alias=hybridSearch(l,q,"DFRONT_300",extra);
            requireEquivalent(actual.answer,alias.answer);need(trace==w.trace,"KEEP equals original DFRONT_300 full trace");++comparisons;
        }
        w.diagnostic=false;auto production=search(l,q,method,extra);
        requireEquivalent(actual.answer,production.answer);
        need(w.trace.empty(),"production records no diagnostic scoring trace");++comparisons;
    }
    w.diagnostic=diagnosticBefore;return comparisons;
}
} // namespace family
