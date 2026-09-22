// Frozen implementations are inherited without editing previous experiments.
#include "v3_reference.inc"

constexpr double RELAX_FACTOR=1.075*1.075;
bool stopAction(const std::string& m){return m=="KEEP"||m=="RELAX075"||m=="UNGATED"||m=="NEW800";}
float effectiveGate(const cdsm::Top& gate,double factor){
    if(factor==1)return gate.kth();
    return std::isfinite(factor)?float(double(gate.kth())*factor):INFINITY;
}
// Same program counter and admission rules as Workspace::advance; only tau changes.
int advanceGated(cdsm::Workspace& w,cdsm::Frontier& f,int cap,double factor){
    while(w.dc<cap){
        if(f.current>=0){
            if(w.complete(f))continue;
            int n=w.arena[w.pos[f.current]];
            if(w.marked(n,cdsm::BASE)){++w.pos[f.current];continue;}
            float tau=effectiveGate(w.gate,factor),d=w.score(n);++w.pos[f.current];w.mark(n,cdsm::BASE);
            if(d<tau){f.active.add(n,d);w.gate.collect(n,d);}else f.deferred.add(n,d);
            continue;
        }
        float tau=effectiveGate(w.gate,factor);
        while(!f.deferred.empty()&&f.deferred.top().first<tau){
            int n=f.deferred.pop();if(!w.marked(n,cdsm::COMPLETED))f.active.add(n,w.scores[n]);
        }
        while(!f.active.empty()&&w.marked(f.active.top().second,cdsm::COMPLETED))f.active.pop();
        if(f.active.empty())return f.deferred.empty()?2:1;
        if(f.active.top().first>tau)return 1;
        f.current=f.active.pop();w.open(f.current);
    }
    return 0;
}
void advanceRelaxed(ScoutCursor& s,cdsm::Workspace& w,int cap,int quota,double factor){
    need(quota>0,"positive relaxed segment");
    int before=w.dc,limit=std::min(cap,before+quota);auto old=w.output;double sum=distanceSum(old);
    if(!s.started){++w.scouts;s.started=true;}
    if(!s.upper.done)s.upper.advance(w,limit);
    if(s.upper.started)w.mark(s.upper.entry,cdsm::ENTRY);
    s.stop=0;
    if(!s.upper.done)++s.upperPauses;
    if(s.upper.done&&w.dc<limit){
        std::swap(w.gate,s.gate);
        if(!s.seeded){w.seed(s.frontier,s.upper.ep);s.seeded=true;}
        s.stop=advanceGated(w,s.frontier,limit,factor);
        std::swap(w.gate,s.gate);
    }
    int cost=w.dc-before;s.spent+=cost;w.scoutDc+=cost;++s.segments;
    s.gain=cost>0&&sum>1e-12?std::max(0.,(sum-distanceSum(w.output))/sum)*100./cost:0;
    for(int i=0;i<w.output.size;i++){
        int n=w.output.v[i].second;bool found=false;
        for(int k=0;k<old.size;k++)if(old.v[k].second==n)found=true;
        if(!found)++s.topEntries;
    }
    need(w.dc<=limit,"relaxed physical budget");
}
double retainedNearest(const ScoutCursor& s,const cdsm::Workspace& w){
    double d=INFINITY;
    if(s.frontier.current>=0&&!w.marked(s.frontier.current,cdsm::COMPLETED))d=w.scores[s.frontier.current];
    for(auto* heap:{&s.frontier.active,&s.frontier.deferred})for(auto p:heap->v)
        if(!w.marked(p.second,cdsm::COMPLETED))d=std::min(d,double(p.first));
    return d;
}
struct RouteObservation {
    int route=0,entry=-1,endpoint=-1,spent=0,stop=0,upperDone=0,seeded=0,gateSize=0;
    int rawActive=0,rawDeferred=0,current=-1,liveActive=0,liveDeferred=0,liveUnique=0,deadItems=0,currentRemaining=0,withinRelax=0,remainingPool=0;
    double kth=INFINITY,nearest=INFINITY,ratio=INFINITY;
    std::string reason,phase;
};
RouteObservation inspectRoute(const ScoutCursor& s,const cdsm::Workspace& w,int route,const std::string& phase){
    RouteObservation z;z.route=route;z.entry=s.upper.entry;z.endpoint=s.upper.ep;z.spent=s.spent;z.stop=s.stop;z.upperDone=s.upper.done;z.seeded=s.seeded;z.gateSize=s.gate.size;
    z.kth=s.gate.kth();z.rawActive=int(s.frontier.active.v.size());z.rawDeferred=int(s.frontier.deferred.v.size());z.current=s.frontier.current;z.phase=phase;z.remainingPool=1600-w.scoutDc;
    std::set<int> live,within;
    auto add=[&](int n,double d,int which){
        if(w.marked(n,cdsm::COMPLETED)){++z.deadItems;return;}
        if(which==0)++z.liveActive;if(which==1)++z.liveDeferred;
        live.insert(n);z.nearest=std::min(z.nearest,d);
        if(which==2||(which==0?d<=effectiveGate(s.gate,RELAX_FACTOR):d<effectiveGate(s.gate,RELAX_FACTOR)))within.insert(n);
    };
    for(auto p:s.frontier.active.v)add(p.second,p.first,0);
    for(auto p:s.frontier.deferred.v)add(p.second,p.first,1);
    if(z.current>=0){add(z.current,w.scores[z.current],2);if(w.marked(z.current,cdsm::OPEN))z.currentRemaining=w.end[z.current]-w.pos[z.current];}
    z.liveUnique=int(live.size());z.withinRelax=int(within.size());
    if(std::isfinite(z.nearest)&&std::isfinite(z.kth)&&z.kth>0)z.ratio=z.nearest/z.kth;
    if(!s.upper.done)z.reason="upper_budget_pause";
    else if(s.seeded&&s.gate.size==0)z.reason="seed_already_base";
    else if(s.stop==2){need(z.liveUnique==0,"empty stop has no live frontier");z.reason="frontier_empty";}
    else if(s.stop==1)z.reason=z.liveUnique?"gate_blocked":"stale_only";
    else if(s.spent>=800)z.reason="route_cap";
    else if(w.scoutDc>=1600)z.reason="pool_cap";
    else z.reason="quota_pause";
    return z;
}
struct StopState {Context c;std::vector<ScoutCursor> scouts;int cap=0;};
StopState stopPrefix(Loaded& l,const float* q,int extra){
    auto& w=*l.w;StopState s;s.c=primary(w,q,l.q.d,l.mask,extra);diversePreview(w,q,l.q.d,l.table,s.c);
    s.cap=std::min(s.c.preview.cap,w.dc+1600);s.scouts.reserve(5);
    for(int i=0;i<4&&w.dc<s.cap;i++){
        auto& p=s.c.preview;const paid::Candidate* candidate=i<int(p.planned.size())?&p.candidates[p.planned[i]]:nullptr;
        int entry=candidate?candidate->entry:w.nextEntry(l.table);if(entry<0)break;
        s.scouts.push_back(atPaidEndpoint(w,entry,candidate?candidate->endpoint:-1));s.scouts.back().advance(w,s.cap,300);
    }
    while(w.dc<s.cap){
        int best=-1;
        for(int i=0;i<int(s.scouts.size());i++)if(s.scouts[i].eligible()&&(best<0||preferred(s.scouts[i],s.scouts[best],w,"FRONTIER")))best=i;
        if(best<0)break;auto& cursor=s.scouts[best];int before=w.dc;
        cursor.advance(w,s.cap,std::min(100,800-cursor.spent));need(w.dc>before||cursor.stop!=0,"prefix makes progress");
    }
    return s;
}
std::string cursorValues(const std::vector<ScoutCursor>& scouts,const std::string& field){
    std::ostringstream out;out<<std::setprecision(17);
    for(size_t i=0;i<scouts.size();i++){
        if(i)out<<';';const auto& s=scouts[i];
        if(field=="spent")out<<s.spent;else if(field=="stops")out<<s.stop;else if(field=="gate")out<<s.gate.kth();else if(field=="entries")out<<s.upper.entry;else out<<s.upper.ep;
    }
    return out.str();
}
struct StopAnswer {
    Answer a;
    int prefixDc=-1,prefixScout=-1,prefixEdges=-1,prefixLogical=-1,prefixExpansions=-1,actionDc=0,actionUpper=0,actionRoutes=0,tailDc=0;
    uint64_t prefixSequence=0;
    std::string prefixSpent,prefixStops,prefixGate,spent,stops,entries,endpoints,decisions;
    std::vector<RouteObservation> observations;
};
StopAnswer finishStop(Loaded& l,StopState& s,const std::string& method,bool diagnostics){
    auto& w=*l.w;StopAnswer r;
    r.prefixDc=w.dc;r.prefixScout=w.scoutDc;r.prefixEdges=w.edges;r.prefixLogical=w.logical;r.prefixExpansions=w.expansions;r.prefixSequence=w.sequence;
    r.prefixSpent=cursorValues(s.scouts,"spent");r.prefixStops=cursorValues(s.scouts,"stops");r.prefixGate=cursorValues(s.scouts,"gate");
    int beforeUpper=w.upperDc,beforeRoutes=w.scouts;std::ostringstream actions;
    if(diagnostics&&method=="KEEP")for(int i=0;i<int(s.scouts.size());i++)r.observations.push_back(inspectRoute(s.scouts[i],w,i,"before"));
    if(method=="RELAX075"||method=="UNGATED"){
        double factor=method=="RELAX075"?RELAX_FACTOR:INFINITY;std::vector<bool> reopen(s.scouts.size(),false);
        if(w.dc<s.cap)for(size_t i=0;i<s.scouts.size();i++){
            auto& cursor=s.scouts[i];reopen[i]=cursor.stop==1&&cursor.spent<800&&std::isfinite(retainedNearest(cursor,w));
            if(reopen[i])cursor.stop=0;
        }
        while(w.dc<s.cap){
            int best=-1;double distance=INFINITY;
            for(int i=0;i<int(s.scouts.size());i++)if(reopen[i]&&s.scouts[i].eligible()){
                double d=retainedNearest(s.scouts[i],w);
                if(best<0||d<distance||(d==distance&&s.scouts[i].spent<s.scouts[best].spent)){best=i;distance=d;}
            }
            if(best<0)break;auto& cursor=s.scouts[best];int before=w.dc;
            advanceRelaxed(cursor,w,s.cap,std::min(100,800-cursor.spent),factor);
            if(actions.tellp()>0)actions<<';';actions<<best<<':'<<w.dc-before;
            need(w.dc>before||cursor.stop!=0,"relaxation makes progress");
        }
    }else if(method=="NEW800"){
        if(w.dc<s.cap){int entry=w.nextEntry(l.table);if(entry>=0){
            s.scouts.emplace_back(entry,w.h.levels[entry]-1);int before=w.dc;
            s.scouts.back().advance(w,s.cap,std::min(800,s.cap-w.dc));actions<<s.scouts.size()-1<<':'<<w.dc-before;
        }}
    }else need(method=="KEEP","known stop action");
    r.actionDc=w.dc-r.prefixDc;r.actionUpper=w.upperDc-beforeUpper;r.actionRoutes=w.scouts-beforeRoutes;r.decisions=actions.str();
    need(w.scoutDc<=1600&&r.actionDc<=1600-r.prefixScout,"action stays within original scout pool");
    for(auto& cursor:s.scouts)need(cursor.spent<=800,"per-route cap preserved");
    if(diagnostics&&method!="KEEP")for(int i=0;i<int(s.scouts.size());i++)r.observations.push_back(inspectRoute(s.scouts[i],w,i,"after"));
    r.spent=cursorValues(s.scouts,"spent");r.stops=cursorValues(s.scouts,"stops");r.entries=cursorValues(s.scouts,"entries");r.endpoints=cursorValues(s.scouts,"endpoints");
    for(auto& cursor:s.scouts)cursor.merge(w);
    if(!s.scouts.empty())w.gate=s.scouts.back().gate;
    int before=w.dc;r.a.tailStop=w.distanceAdvance(s.c.preview.cap,INFINITY);r.tailDc=w.dc-before;
    r.a.out=w.finish();fillContext(r.a,s.c);
    need(w.dc==s.c.preview.cap&&r.prefixDc+r.actionDc+r.tailDc==w.dc,"strict total real work");
    return r;
}
StopAnswer stopSearch(Loaded& l,const float* q,const std::string& method,int extra,bool diagnostics){
    if(!stopAction(method)){StopAnswer r;r.a=hybridSearch(l,q,method,extra).answer;return r;}
    auto began=Clock::now();auto state=stopPrefix(l,q,extra);auto result=finishStop(l,state,method,diagnostics);result.a.ns=bio::ns(began);return result;
}

void sameCursor(const ScoutCursor& a,const ScoutCursor& b){
    need(a.spent==b.spent&&a.stop==b.stop&&a.seeded==b.seeded,"cursor accounting and stop");
    need(a.frontier.active.v==b.frontier.active.v&&a.frontier.deferred.v==b.frontier.deferred.v&&a.frontier.current==b.frontier.current,"cursor frontier equality");
    need(a.gate.size==b.gate.size,"cursor gate size");for(int i=0;i<a.gate.size;i++)need(a.gate.v[i]==b.gate.v[i],"cursor gate contents");
}
void stopUnit(){
    hybridUnit();faiss::IndexHNSWFlat index(8,8);std::vector<float> data(4096*8);
    for(int i=0;i<4096;i++)for(int j=0;j<8;j++)data[i*8+j]=float((i*31+j*137)%4093)/4093;
    index.add(4096,data.data());cdsm::Workspace w(index);w.diagnostic=true;std::vector<uint8_t> mask(4096,1);auto table=cdsm::entryTable(index.hnsw);
    for(int qi:{13,71,309}){
        auto* q=data.data()+qi*8;w.reset(q,mask);w.initialize(64);w.distanceAdvance(64,INFINITY);int entry=w.nextEntry(table);need(entry>=0,"unit entrance");Snapshot prefix(w);
        for(double factor:{1.,RELAX_FACTOR,double(INFINITY)})for(int total:{1,17,100,400}){
            prefix.restore(w,q,mask);int cap=w.dc+total;ScoutCursor full(entry,w.h.levels[entry]-1);
            if(factor==1)full.advance(w,cap,total);else advanceRelaxed(full,w,cap,total,factor);
            Answer expected;expected.out=w.finish();auto trace=w.trace;
            for(int chunk:{1,17,100}){
                prefix.restore(w,q,mask);ScoutCursor partial(entry,w.h.levels[entry]-1);
                while(partial.stop==0&&partial.spent<total)advanceRelaxed(partial,w,cap,std::min(chunk,total-partial.spent),factor);
                Answer actual;actual.out=w.finish();same(actual,expected,"factor one / relaxed split exact replay");sameCursor(partial,full);
                need(actual.out.scoutDc==expected.out.scoutDc&&actual.out.upperDc==expected.out.upperDc&&w.trace==trace,"complete physical trace and phase work equality");
            }
        }
    }
    // stop=1 alone is not proof that a live frontier remains.
    w.reset(data.data(),mask);w.score(0);w.mark(0,cdsm::BASE|cdsm::COMPLETED);ScoutCursor stale(0,0);stale.upper.done=true;stale.seeded=true;stale.stop=1;stale.gate.collect(0,w.scores[0]);stale.frontier.deferred.add(0,w.scores[0]);
    need(inspectRoute(stale,w,0,"unit").reason=="stale_only","stale deferred classification");
    stale.frontier.clear();stale.gate.clear();stale.stop=2;need(inspectRoute(stale,w,0,"unit").reason=="seed_already_base","skipped seed classification");
    std::cout<<"STOP_UNIT_COMPLETE factor1 parity, relaxed split full trace, stop classification"<<std::endl;
}
void checkpointCheck(Loaded& l,const float* q,int extra){
    auto state=stopPrefix(l,q,extra);Snapshot checkpoint(*l.w);
    for(const std::string method:{"KEEP","RELAX075","UNGATED","NEW800"}){
        checkpoint.restore(*l.w,q,l.mask);auto branch=state;auto resumed=finishStop(l,branch,method,false);auto trace=l.w->trace;
        auto replayed=stopSearch(l,q,method,extra,false);same(resumed.a,replayed.a,"snapshot versus independently paid prefix");
        need(trace==l.w->trace&&resumed.spent==replayed.spent&&resumed.stops==replayed.stops,"checkpoint full trace and cursor state equality");
    }
}
std::string prefixKey(const StopAnswer& r){
    std::ostringstream s;s<<r.prefixDc<<'|'<<r.prefixScout<<'|'<<r.prefixSequence<<'|'<<r.prefixEdges<<'|'<<r.prefixLogical<<'|'<<r.prefixExpansions<<'|'<<r.prefixSpent<<'|'<<r.prefixStops<<'|'<<r.prefixGate;return s.str();
}
void stopEvaluate(const Args& a){
    Loaded l(a);auto truth=bio::vecs<int>(a.s("gt"));need(truth.n==l.q.n&&truth.d==11,"truth shape");
    auto methods=split(a.s("methods"));need(methods.size()==8,"fixed methods");bool timing=a.s("mode")=="timing",engineering=a.s("mode")=="engineering";l.w->diagnostic=engineering;
    auto destination=a.s("out");need(!std::filesystem::exists(destination),"preserve raw output");std::ofstream f(destination);f<<std::setprecision(17);
    std::ofstream detail;
    if(!timing){need(!std::filesystem::exists(destination+".routes.csv"),"preserve route output");detail.open(destination+".routes.csv");detail<<std::setprecision(17);
        detail<<"cohort,graph,qid,extra,method,phase,route,entry,endpoint,spent,stop_code,reason,upper_done,seeded,gate_size,gate_kth,raw_active,raw_deferred,current,live_active,live_deferred,live_unique,dead_items,current_remaining,nearest_live,nearest_over_gate,live_within_relax,remaining_pool\n";
    }
    f<<"cohort,graph,qid,extra,method,rep,hits,dc,cap,primary,stop,preview,scout,scouts,upper_dc,edges,expansions,logical,sequence,primary_sequence,preview_sequence,ns,prefix_dc,prefix_scout,prefix_sequence,prefix_edges,prefix_logical,prefix_expansions,prefix_spent,prefix_stops,prefix_gate,action_dc,action_upper,action_routes,tail_dc,spent,stops,entries,endpoints,decisions,ids\n";
    auto begun=Clock::now();int done=0;
    for(int extra:{3200,6400}){
        std::map<std::pair<int,std::string>,StopAnswer> signatures;
        for(int rep=timing?-2:0;rep<(timing?5:1);rep++)for(int qi:l.ids){
            const float* q=l.q.x.data()+size_t(qi)*l.q.d;auto order=methods;
            if(timing){std::mt19937 rng(20260924+qi*17+(rep+2)*127+extra);std::shuffle(order.begin(),order.end(),rng);}
            std::pair<int,uint64_t> commonPrimary{-1,0},commonPreview{-1,0};std::string commonPrefix;
            std::map<std::string,Answer> compare;
            for(const auto& method:order){
                auto r=stopSearch(l,q,method,extra,!timing);auto& z=r.a;auto& o=z.out;
                need(o.dc==z.cap&&o.dc==o.primary+extra,"exact same physical budget");
                auto primary=std::make_pair(o.primary,o.primarySequence);
                if(commonPrimary.first<0)commonPrimary=primary;else need(commonPrimary==primary,"common primary");
                if(method!="C"&&method!="F"){
                    auto preview=std::make_pair(z.preview,z.previewSeq);
                    if(commonPreview.first<0)commonPreview=preview;else need(commonPreview==preview,"common paid preview");
                }
                if(stopAction(method)){
                    auto key=prefixKey(r);if(commonPrefix.empty())commonPrefix=key;else need(commonPrefix==key,"identical intervention checkpoint");
                    if(method=="KEEP")need(r.actionDc==0,"keep uses no intervention budget");
                }
                if(method=="KEEP"||method=="DFRONT_300")compare[method]=z;
                if(compare.size()==2)same(compare.at("KEEP"),compare.at("DFRONT_300"),"diagnostic KEEP reproduces frozen DFRONT");
                if(timing){auto key=std::make_pair(qi,method);if(rep==-2)signatures[key]=r;else{
                    auto& old=signatures.at(key);same(z,old.a,"timing repeat results");need(prefixKey(r)==prefixKey(old)&&r.spent==old.spent&&r.stops==old.stops&&r.decisions==old.decisions,"timing repeat prefix and intervention");
                }}
                if(rep>=0){
                    std::vector<int> ids;for(int n:o.ids)if(n>=0)ids.push_back(l.mapping.empty()?n:l.mapping.at(n));int hits=bio::hits(ids,truth.x.data()+size_t(qi)*11);
                    f<<a.s("cohort")<<','<<a.s("name")<<','<<qi<<','<<extra<<','<<method<<','<<rep<<','<<hits<<','<<o.dc<<','<<z.cap<<','<<o.primary<<','<<o.stop<<','<<z.preview<<','<<o.scoutDc<<','<<o.scouts<<','<<o.upperDc<<','<<o.edges<<','<<o.expansions<<','<<o.logical<<','<<o.sequence<<','<<o.primarySequence<<','<<z.previewSeq<<','<<z.ns<<','<<r.prefixDc<<','<<r.prefixScout<<','<<r.prefixSequence<<','<<r.prefixEdges<<','<<r.prefixLogical<<','<<r.prefixExpansions<<','<<r.prefixSpent<<','<<r.prefixStops<<','<<r.prefixGate<<','<<r.actionDc<<','<<r.actionUpper<<','<<r.actionRoutes<<','<<r.tailDc<<','<<r.spent<<','<<r.stops<<','<<r.entries<<','<<r.endpoints<<','<<r.decisions<<',';bio::ids(f,ids);f<<'\n';
                    for(const auto& d:r.observations)detail<<a.s("cohort")<<','<<a.s("name")<<','<<qi<<','<<extra<<','<<method<<','<<d.phase<<','<<d.route<<','<<d.entry<<','<<d.endpoint<<','<<d.spent<<','<<d.stop<<','<<d.reason<<','<<d.upperDone<<','<<d.seeded<<','<<d.gateSize<<','<<d.kth<<','<<d.rawActive<<','<<d.rawDeferred<<','<<d.current<<','<<d.liveActive<<','<<d.liveDeferred<<','<<d.liveUnique<<','<<d.deadItems<<','<<d.currentRemaining<<','<<d.nearest<<','<<d.ratio<<','<<d.withinRelax<<','<<d.remainingPool<<'\n';
                }
            }
            if(engineering&&qi==l.ids.front())checkpointCheck(l,q,extra);
            if(++done%80==0){f.flush();if(detail.is_open())detail.flush();std::cout<<"SEARCH "<<done<<" seconds="<<elapsed(begun)<<std::endl;}
        }
    }
    need(bool(f)&&(!detail.is_open()||bool(detail)),"output writes");std::cout<<"STOP_EVALUATION_COMPLETE seconds="<<elapsed(begun)<<std::endl;
}
int main(int n,char** v){try{
    Args a(n,v);omp_set_num_threads(1);
    if(a.i("cpu",-1)>=0)need(SetThreadAffinityMask(GetCurrentThread(),DWORD_PTR(1)<<a.i("cpu"))!=0,"CPU affinity");
    if(a.s("mode")=="unit")stopUnit();else stopEvaluate(a);return 0;
}catch(const std::exception& e){std::cerr<<"ERROR "<<e.what()<<std::endl;return 1;}}
