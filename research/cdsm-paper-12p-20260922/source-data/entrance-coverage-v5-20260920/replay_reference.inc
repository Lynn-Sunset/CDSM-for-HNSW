// Prior experiments are included unchanged except for their main function name.
#include "v4_reference.inc"

std::vector<int> expandedTable(const faiss::HNSW& h,const std::vector<int>& original){
    need(original.size()==64,"original 64 entrance IDs");
    std::set<int> old(original.begin(),original.end());need(old.size()==64,"distinct original entries");
    std::vector<int> reservoir;cdsm::JavaRandom random(20260920);int seen=0;
    for(int n=0;n<int(h.levels.size());n++)if(h.levels[n]>1&&!old.count(n)){
        ++seen;if(reservoir.size()<192)reservoir.push_back(n);
        else {int j=random.nextInt(seen);if(j<192)reservoir[j]=n;}
    }
    need(reservoir.size()==192,"enough upper nodes");
    auto all=original;all.insert(all.end(),reservoir.begin(),reservoir.end());
    need(std::set<int>(all.begin(),all.end()).size()==256,"expanded distinct entries");return all;
}
void mergePrefix(cdsm::Workspace& w,const StopState& s){
    for(const auto& cursor:s.scouts)cursor.merge(w);
    if(!s.scouts.empty())w.gate=s.scouts.back().gate;
}
struct EntranceCheckpoint {
    Context c;int prefixDc=0,startDc=0,actionCap=0;uint64_t prefixSeq=0,startSeq=0;
};
EntranceCheckpoint checkpoint(Loaded& l,const float* q,int extra,bool late){
    auto state=stopPrefix(l,q,extra);auto& w=*l.w;EntranceCheckpoint p;
    p.c=state.c;p.prefixDc=w.dc;p.prefixSeq=w.sequence;mergePrefix(w,state);
    need(w.dc<=p.c.preview.cap-800,"original scouts leave protected tail");
    if(late){w.distanceAdvance(p.c.preview.cap-800,INFINITY);need(w.dc==p.c.preview.cap-800,"protected main reaches checkpoint");}
    p.startDc=w.dc;p.startSeq=w.sequence;p.actionCap=w.dc+800;return p;
}
struct Screen {
    int cost=0,nearest=-1;uint64_t sequence=0;
    std::vector<int> entries,slots;std::vector<float> distances;
};
Screen scorePool(cdsm::Workspace& w,const std::vector<int>& table,int count,int cap){
    Screen s;int before=w.dc;
    for(int i=0;i<count;i++){
        int n=table[i];if(w.marked(n,cdsm::BASE)||w.marked(n,cdsm::ENTRY))continue;
        need(w.dc<cap||w.scored(n),"candidate selection must be paid");
        float d=w.score(n);int j=int(s.entries.size());s.entries.push_back(n);s.slots.push_back(i);s.distances.push_back(d);
        if(s.nearest<0||d<s.distances[s.nearest])s.nearest=j;
    }
    s.cost=w.dc-before;s.sequence=w.sequence;need(s.cost<=count,"paid candidate costs");return s;
}
struct EntranceAnswer {
    Answer a;int prefixDc=-1,startDc=-1,protectedMain=0,pool=0,eligible=0,selectionDc=0;
    int entry=-1,slot=-1,endpoint=-1,routeDc=0,routeUpper=0,routeStop=-1,upperDone=0,seeded=0,tailDc=0;
    float entryDistance=INFINITY;uint64_t prefixSeq=0,startSeq=0,selectionSeq=0;
};
EntranceAnswer fromEntrance(Loaded& l,const EntranceCheckpoint& p,const Screen& s,int pool,int candidate,const std::string& mode){
    auto& w=*l.w;EntranceAnswer r;r.prefixDc=p.prefixDc;r.prefixSeq=p.prefixSeq;r.startDc=p.startDc;r.startSeq=p.startSeq;
    r.protectedMain=p.startDc-p.prefixDc;r.pool=pool;r.eligible=int(s.entries.size());r.selectionDc=s.cost;r.selectionSeq=s.sequence;
    if(mode!="CONT"&&candidate>=0){
        need(mode=="RAW"||mode=="DESC","known entrance route");
        r.entry=s.entries.at(candidate);r.slot=s.slots.at(candidate);r.entryDistance=s.distances.at(candidate);
        need(w.scored(r.entry)&&!w.marked(r.entry,cdsm::BASE|cdsm::ENTRY),"scored eligible entry");
        ScoutCursor cursor=atPaidEndpoint(w,r.entry,mode=="RAW"?r.entry:-1);int upper=w.upperDc,before=w.dc;
        if(w.dc<p.actionCap)cursor.advance(w,p.actionCap,p.actionCap-w.dc);
        r.routeDc=w.dc-before;r.routeUpper=w.upperDc-upper;r.endpoint=cursor.upper.ep;
        r.routeStop=cursor.stop;r.upperDone=cursor.upper.done;r.seeded=cursor.seeded;
        cursor.merge(w);w.gate=cursor.gate;
    }
    int before=w.dc;r.a.tailStop=w.distanceAdvance(p.c.preview.cap,INFINITY);r.tailDc=w.dc-before;
    r.a.out=w.finish();fillContext(r.a,p.c);
    need(r.a.out.dc==r.a.cap&&r.startDc+r.selectionDc+r.routeDc+r.tailDc==r.a.cap,"exact total work including all selection");
    need(r.selectionDc+r.routeDc<=800&&r.routeUpper<=r.routeDc,"bounded entrance intervention");return r;
}
EntranceAnswer entranceSearch(Loaded& l,const std::vector<int>& table,const float* q,const std::string& method,int extra){
    if(method=="C"||method=="F"||method=="DIVERSE"||method=="KEEP"){EntranceAnswer r;r.a=stopSearch(l,q,method,extra,false).a;return r;}
    auto began=Clock::now();auto parts=split(method,'_');need(parts.size()==2,"entrance method name");
    int count=std::stoi(parts[0].substr(1));need(count==64||count==256,"pool size");
    auto p=checkpoint(l,q,extra,true);auto s=scorePool(*l.w,table,count,p.actionCap);
    auto r=fromEntrance(l,p,s,count,s.nearest,parts[1]);r.a.ns=bio::ns(began);return r;
}
std::string stateKey(const EntranceAnswer& r){
    std::ostringstream o;o<<r.prefixDc<<':'<<r.prefixSeq<<':'<<r.startDc<<':'<<r.startSeq;return o.str();
}
void entranceChecks(Loaded& l,const std::vector<int>& table,const float* q,int extra){
    auto& w=*l.w;need(w.diagnostic,"full trace check enabled");
    auto p=checkpoint(l,q,extra,true);Snapshot start(w);w.distanceAdvance(p.c.preview.cap,INFINITY);
    Answer splitAnswer;splitAnswer.out=w.finish();fillContext(splitAnswer,p.c);auto wholeTrace=w.trace;
    auto frozen=stopSearch(l,q,"KEEP",extra,false);same(splitAnswer,frozen.a,"protected main split exactly replays KEEP");need(w.trace==wholeTrace,"KEEP complete distance ID trace");
    for(int count:{64,256}){
        start.restore(w,q,l.mask);auto screen=scorePool(w,table,count,p.actionCap);Snapshot scored(w);
        for(const std::string mode:{"CONT","DESC","RAW"}){
            scored.restore(w,q,l.mask);auto r=fromEntrance(l,p,screen,count,screen.nearest,mode);auto trace=w.trace;
            std::string method=(mode=="CONT"?"S":"L")+std::to_string(count)+"_"+mode;
            auto full=entranceSearch(l,table,q,method,extra);same(r.a,full.a,"branch snapshot versus independently paid execution");
            need(trace==w.trace&&stateKey(r)==stateKey(full)&&r.selectionSeq==full.selectionSeq&&r.entry==full.entry&&r.endpoint==full.endpoint,"complete branch trace and paid state parity");
        }
    }
}
void writePool(const std::string& path,const std::vector<int>& table,const faiss::HNSW& h){
    need(!std::filesystem::exists(path),"preserve pool output");std::ofstream f(path);f<<"slot,entry,highest_level\n";
    for(int i=0;i<int(table.size());i++)f<<i<<','<<table[i]<<','<<h.levels[table[i]]-1<<'\n';need(bool(f),"write pool");
}
std::string commonHeader(){return "cohort,graph,qid,extra,method,rep,hits,dc,cap,primary,stop,preview,scout,scouts,upper_dc,edges,expansions,logical,sequence,primary_sequence,preview_sequence,ns,prefix_dc,prefix_sequence,start_dc,start_sequence,protected_main,pool,eligible,selection_dc,selection_sequence,entry,slot,entry_distance,endpoint,route_dc,route_upper,route_stop,upper_done,seeded,tail_dc,ids";}
void emitEntrance(std::ostream& f,const Args& args,const EntranceAnswer& r,int qi,int extra,const std::string& method,int rep,const std::vector<int>& mapping,const int* truth){
    const auto& z=r.a;const auto& o=z.out;std::vector<int> ids;for(int n:o.ids)if(n>=0)ids.push_back(mapping.empty()?n:mapping.at(n));
    int hits=bio::hits(ids,truth);
    f<<args.s("cohort")<<','<<args.s("name")<<','<<qi<<','<<extra<<','<<method<<','<<rep<<','<<hits<<','<<o.dc<<','<<z.cap<<','<<o.primary<<','<<o.stop<<','<<z.preview<<','<<o.scoutDc<<','<<o.scouts<<','<<o.upperDc<<','<<o.edges<<','<<o.expansions<<','<<o.logical<<','<<o.sequence<<','<<o.primarySequence<<','<<z.previewSeq<<','<<z.ns<<','<<r.prefixDc<<','<<r.prefixSeq<<','<<r.startDc<<','<<r.startSeq<<','<<r.protectedMain<<','<<r.pool<<','<<r.eligible<<','<<r.selectionDc<<','<<r.selectionSeq<<','<<r.entry<<','<<r.slot<<','<<r.entryDistance<<','<<r.endpoint<<','<<r.routeDc<<','<<r.routeUpper<<','<<r.routeStop<<','<<r.upperDone<<','<<r.seeded<<','<<r.tailDc<<',';bio::ids(f,ids);
}
void entranceEvaluate(const Args& a){
    Loaded l(a);auto table=expandedTable(l.h->hnsw,l.table);auto truth=bio::vecs<int>(a.s("gt"));need(truth.n==l.q.n&&truth.d==11,"truth shape");
    auto methods=split(a.s("methods"));need(methods.size()==10,"fixed ten methods");bool timing=a.s("mode")=="timing",engineering=a.s("mode")=="engineering";l.w->diagnostic=engineering;
    auto path=a.s("out");need(!std::filesystem::exists(path),"preserve raw results");writePool(path+".pool.csv",table,l.h->hnsw);
    std::ofstream f(path);f<<std::setprecision(17)<<commonHeader()<<'\n';auto began=Clock::now();int done=0;
    for(int extra:{3200,6400}){
        std::map<std::pair<int,std::string>,EntranceAnswer> signatures;
        for(int rep=timing?-2:0;rep<(timing?5:1);rep++)for(int qi:l.ids){
            auto* q=l.q.x.data()+size_t(qi)*l.q.d;auto order=methods;
            if(timing){std::mt19937 random(20260925+qi*17+(rep+2)*127+extra);std::shuffle(order.begin(),order.end(),random);}
            std::pair<int,uint64_t> primaryKey{-1,0};std::string checkpointKey;std::map<int,std::pair<int,uint64_t>> screens;
            for(const auto& method:order){
                auto r=entranceSearch(l,table,q,method,extra);const auto& z=r.a;
                need(z.out.dc==z.cap&&z.cap==z.out.primary+extra,"equal actual work");auto key=std::make_pair(z.out.primary,z.out.primarySequence);
                if(primaryKey.first<0)primaryKey=key;else need(key==primaryKey,"common primary trace");
                if(r.pool){
                    need(r.startDc==z.cap-800,"preserved main budget");auto current=stateKey(r);
                    if(checkpointKey.empty())checkpointKey=current;else need(checkpointKey==current,"same protected checkpoint");
                    auto sk=std::make_pair(r.selectionDc,r.selectionSeq);if(screens.count(r.pool))need(screens[r.pool]==sk,"same paid candidate scoring");else screens[r.pool]=sk;
                }
                if(timing){auto sk=std::make_pair(qi,method);if(rep==-2)signatures[sk]=r;else{
                    auto& old=signatures.at(sk);same(r.a,old.a,"timing exact repeat");need(stateKey(r)==stateKey(old)&&r.entry==old.entry&&r.endpoint==old.endpoint&&r.selectionSeq==old.selectionSeq,"timing selector invariant");
                }}
                if(rep>=0){emitEntrance(f,a,r,qi,extra,method,rep,l.mapping,truth.x.data()+size_t(qi)*11);f<<'\n';}
            }
            if(engineering&&qi==l.ids.front())entranceChecks(l,table,q,extra);
            if(++done%80==0){f.flush();std::cout<<"SEARCH "<<done<<" seconds="<<elapsed(began)<<std::endl;}
        }
    }
    need(bool(f),"write results");std::cout<<"ENTRANCE_EVALUATION_COMPLETE seconds="<<elapsed(began)<<std::endl;
}
void coverageAudit(const Args& a){
    Loaded l(a);auto& w=*l.w;auto table=expandedTable(l.h->hnsw,l.table);auto truth=bio::vecs<int>(a.s("gt"));
    auto path=a.s("out");need(!std::filesystem::exists(path),"preserve audit output");writePool(path+".pool.csv",table,l.h->hnsw);
    std::ofstream f(path),previewOut(path+".preview.csv");f<<std::setprecision(17)<<commonHeader()<<",phase,chosen_nearest\n";
    previewOut<<std::setprecision(17)<<"cohort,graph,qid,extra,attempted,complete,planned_count,order,entry,endpoint,distance,descent_dc,planned,neighbors,unseen,neighbor_ids\n";
    std::ifstream cases(a.s("cases"));need(bool(cases),"diagnostic case list");int qi,extra,done=0;auto began=Clock::now();
    while(cases>>qi>>extra){
        auto* q=l.q.x.data()+size_t(qi)*l.q.d;
        for(const std::string phase:{"EARLY","LATE"}){
            auto p=checkpoint(l,q,extra,phase=="LATE");Snapshot beforeScreen(w);
            if(phase=="EARLY"){
                const auto& preview=p.c.preview;
                for(int i=0;i<int(preview.candidates.size());i++){
                    const auto& c=preview.candidates[i];bool planned=std::find(preview.planned.begin(),preview.planned.end(),i)!=preview.planned.end();
                    previewOut<<a.s("cohort")<<','<<a.s("name")<<','<<qi<<','<<extra<<','<<preview.attempted<<','<<preview.candidates.size()<<','<<preview.planned.size()<<','<<c.order<<','<<c.entry<<','<<c.endpoint<<','<<c.distance<<','<<c.cost<<','<<planned<<','<<c.neighbors.size()<<','<<c.unseen.size()<<',';bio::ids(previewOut,c.neighbors);previewOut<<'\n';
                }
            }
            for(int count:{64,256}){
                beforeScreen.restore(w,q,l.mask);auto s=scorePool(w,table,count,p.actionCap);Snapshot scored(w);
                auto cont=fromEntrance(l,p,s,count,-1,"CONT");emitEntrance(f,a,cont,qi,extra,"CONT",0,l.mapping,truth.x.data()+size_t(qi)*11);f<<','<<phase<<",0\n";
                for(int i=0;i<int(s.entries.size());i++)for(const std::string mode:{"DESC","RAW"}){
                    scored.restore(w,q,l.mask);auto r=fromEntrance(l,p,s,count,i,mode);
                    emitEntrance(f,a,r,qi,extra,mode,0,l.mapping,truth.x.data()+size_t(qi)*11);f<<','<<phase<<','<<(i==s.nearest)<<'\n';
                }
            }
        }
        ++done;f.flush();previewOut.flush();std::cout<<"CASE "<<done<<" qid="<<qi<<" E="<<extra<<" seconds="<<elapsed(began)<<std::endl;
    }
    need(done>0&&bool(f)&&bool(previewOut),"complete coverage write");std::cout<<"COVERAGE_AUDIT_COMPLETE cases="<<done<<" seconds="<<elapsed(began)<<std::endl;
}
int frozen_coverage_v5_main(int n,char** v){try{
    Args a(n,v);omp_set_num_threads(1);
    if(a.i("cpu",-1)>=0)need(SetThreadAffinityMask(GetCurrentThread(),DWORD_PTR(1)<<a.i("cpu"))!=0,"CPU affinity");
    if(a.s("mode")=="unit"){stopUnit();std::cout<<"V5_INHERITED_UNITS_COMPLETE"<<std::endl;}
    else if(a.s("mode")=="coverage")coverageAudit(a);else entranceEvaluate(a);return 0;
}catch(const std::exception& e){std::cerr<<"ERROR "<<e.what()<<std::endl;return 1;}}
