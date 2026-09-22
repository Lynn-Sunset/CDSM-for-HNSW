#include "v9_reference.inc"

bool gridMethod(const std::string& m){return m.starts_with("E64_")||m.starts_with("E256_")||m.starts_with("L64_")||m.starts_with("L256_");}
ValueAnswer spaceSearch(Loaded& l,const std::vector<int>& table,const float* q,const std::string& method,int extra){
    if(!gridMethod(method))return valueSearch(l,table,q,method,extra);
    auto began=Clock::now();auto parts=split(method,'_');int count=std::stoi(parts[0].substr(1));
    auto p=checkpoint(l,q,extra,method[0]=='L');auto s=scorePool(*l.w,table,count,p.actionCap);
    ValueAnswer r;r.a.r.e=fromEntrance(l,p,s,count,s.nearest,parts[1]);r.a.r.e.a.ns=bio::ns(began);return r;
}
void splitEntranceCheck(Loaded& l,const float* q,const EntranceCheckpoint& p,const Screen& s,int candidate,const std::string& mode){
    if(candidate<0)return;auto& w=*l.w;Snapshot paid(w);int entry=s.entries.at(candidate);
    auto direct=atPaidEndpoint(w,entry,mode=="RAW"?entry:-1);direct.advance(w,p.actionCap,p.actionCap-w.dc);Snapshot expected(w);
    for(int chunk:{1,17,64}){
        paid.restore(w,q,l.mask);auto partial=atPaidEndpoint(w,entry,mode=="RAW"?entry:-1);
        while(w.dc<p.actionCap&&partial.eligible()){
            int before=w.dc;partial.advance(w,p.actionCap,std::min(chunk,800-partial.spent));need(w.dc>before||!partial.eligible(),"segmented entrance progresses");
        }
        Answer a,b;a.out=w.finish();b.out=expected.counts;same(a,b,"upper and local segmented entrance parity");sameCursor(direct,partial);need(w.trace==expected.trace,"entire segmented entrance scoring trace");
    }
}
void spaceChecks(Loaded& l,const std::vector<int>& table,const float* q,int extra){
    auto& w=*l.w;need(w.diagnostic,"complete engineering distance traces");
    auto old=valueSearch(l,table,q,"L64_DESC",extra);auto oldTrace=w.trace;auto now=spaceSearch(l,table,q,"L64_DESC",extra);
    sameValue(old,now);need(oldTrace==w.trace,"L64_DESC exact inherited implementation");
    for(bool late:{false,true}){
        auto p=checkpoint(l,q,extra,late);Snapshot before(w);w.distanceAdvance(p.c.preview.cap,INFINITY);Answer keep;keep.out=w.finish();fillContext(keep,p.c);auto trace=w.trace;
        auto full=valueSearch(l,table,q,"KEEP",extra);same(keep,full.a.r.e.a,"early or late segmentation equals KEEP");need(trace==w.trace,"KEEP split entire trace");
        for(int count:{64,256}){
            before.restore(w,q,l.mask);auto screen=scorePool(w,table,count,p.actionCap);Snapshot paid(w);
            for(const std::string mode:{"CONT","RAW","DESC"}){
                paid.restore(w,q,l.mask);auto branch=fromEntrance(l,p,screen,count,screen.nearest,mode);auto trace=w.trace;
                auto method=std::string(late?"L":"E")+std::to_string(count)+"_"+mode;auto full=spaceSearch(l,table,q,method,extra);
                same(branch.a,full.a.r.e.a,"saved paid screen equals full online method");need(trace==w.trace,"checkpoint replay exact complete scoring trace");
                if(mode!="CONT"){paid.restore(w,q,l.mask);splitEntranceCheck(l,q,p,screen,screen.nearest,mode);}
            }
        }
    }
    valueChecks(l,table,q,extra);
}
void spaceEvaluate(const Args& a){
    Loaded l(a);auto table=expandedTable(l.h->hnsw,l.table);auto truth=bio::vecs<int>(a.s("gt"));need(truth.n==l.q.n&&truth.d==11,"truth shape");
    auto methods=split(a.s("methods"));need(methods.size()==16,"sixteen frozen online methods");bool timing=a.s("mode")=="timing",engineering=a.s("mode")=="engineering";l.w->diagnostic=engineering;
    auto path=a.s("out");need(!std::filesystem::exists(path),"preserve raw results");writePool(path+".pool.csv",table,l.h->hnsw);
    std::ofstream f(path);f<<std::setprecision(17)<<valueHeader()<<'\n';auto began=Clock::now();int done=0;
    for(int extra:{3200,6400}){
        std::map<std::pair<int,std::string>,ValueAnswer> signatures;
        for(int rep=timing?-2:0;rep<(timing?5:1);rep++)for(int qi:l.ids){
            const float* q=l.q.x.data()+size_t(qi)*l.q.d;auto order=methods;
            if(timing){std::mt19937 random(20260930+qi*17+(rep+2)*127+extra);std::shuffle(order.begin(),order.end(),random);}
            std::pair<int,uint64_t> pk{-1,0};std::map<std::string,ValueAnswer> answers;
            for(const auto& method:order){
                auto r=spaceSearch(l,table,q,method,extra);const auto& e=r.a.r.e;const auto& z=e.a;
                need(z.out.dc==z.cap&&z.cap==z.out.primary+extra,"exact per query actual paid budget");auto key=std::make_pair(z.out.primary,z.out.primarySequence);
                if(pk.first<0)pk=key;else need(pk==key,"shared initial main distance sequence");
                if(gridMethod(method))need(e.startDc==(method[0]=='L'?z.cap-800:e.prefixDc),"declared intervention timing");
                answers[method]=r;
                if(timing){auto sk=std::make_pair(qi,method);if(rep==-2)signatures[sk]=r;else sameValue(r,signatures.at(sk));}
                if(rep>=0)emitValue(f,a,r,qi,extra,method,rep,l,truth.x.data()+size_t(qi)*11);
            }
            for(const std::string m:{"G4_COMMIT","G4_VALUE"})if(!answers.at(m).a.r.fired)same(answers.at(m).a.r.e.a,answers.at("KEEP").a.r.e.a,"gate off exact KEEP");
            if(engineering)spaceChecks(l,table,q,extra);
            if(++done%80==0){f.flush();std::cout<<"SEARCH "<<done<<" seconds="<<elapsed(began)<<std::endl;}
        }
    }
    need(bool(f),"complete result write");std::cout<<"V10_EVALUATION_COMPLETE seconds="<<elapsed(began)<<std::endl;
}
void spaceAudit(const Args& a){
    Loaded l(a);auto& w=*l.w;auto table=expandedTable(l.h->hnsw,l.table);auto truth=bio::vecs<int>(a.s("gt"));
    auto path=a.s("out");need(!std::filesystem::exists(path)&&!std::filesystem::exists(path+".top4.csv"),"preserve audit evidence");writePool(path+".pool.csv",table,l.h->hnsw);
    std::ofstream f(path),top(path+".top4.csv");f<<std::setprecision(17)<<commonHeader()<<",phase,distance_rank,chosen_nearest\n";
    top<<std::setprecision(17)<<valueHeader()<<",action,action_eligible,original_allowed,original_reason\n";
    std::ifstream cases(a.s("cases"));need(bool(cases),"preselected audit cases");int qi,extra,done=0;auto began=Clock::now();
    while(cases>>qi>>extra){
        const float* q=l.q.x.data()+size_t(qi)*l.q.d;const int* gt=truth.x.data()+size_t(qi)*11;
        for(const std::string phase:{"EARLY","LATE"}){
            auto p=checkpoint(l,q,extra,phase=="LATE");Snapshot start(w);
            for(int count:{64,256}){
                start.restore(w,q,l.mask);auto s=scorePool(w,table,count,p.actionCap);Snapshot paid(w);
                std::vector<int> order(s.entries.size()),rank(s.entries.size());std::iota(order.begin(),order.end(),0);
                std::stable_sort(order.begin(),order.end(),[&](int x,int y){return s.distances[x]<s.distances[y];});
                for(int i=0;i<int(order.size());i++)rank[order[i]]=i+1;
                auto cont=fromEntrance(l,p,s,count,-1,"CONT");emitEntrance(f,a,cont,qi,extra,"CONT",0,l.mapping,gt);f<<','<<phase<<",0,0\n";
                for(int i=0;i<int(s.entries.size());i++)for(const std::string mode:{"RAW","DESC"}){
                    paid.restore(w,q,l.mask);auto r=fromEntrance(l,p,s,count,i,mode);emitEntrance(f,a,r,qi,extra,mode,0,l.mapping,gt);f<<','<<phase<<','<<rank[i]<<','<<(i==s.nearest)<<'\n';
                }
            }
        }
        auto z=fastObservedTail(l,q,extra);bool allowed=z.allowed;auto reason=z.reason;z.allowed=1;if(!allowed)z.reason="audit_forced";
        auto state=prepareValue(l,table,z,extra);Snapshot paid(w);
        for(int action=-1;action<4;action++){
            bool eligible=action==-1||(action<int(state.allocation.scouts.size())&&state.allocation.scouts[action].eligible());if(!eligible)continue;
            paid.restore(w,q,l.mask);ValueAnswer r;r.prepared=1;r.features=state.features;r.valueReads=state.reads;r.eligibleMask=state.eligibleMask;r.nearestChoice=state.nearestChoice;r.choice=action;
            r.a=finishCommit(l,state.allocation,action);std::ostringstream line;line<<std::setprecision(17);emitValue(line,a,r,qi,extra,"FORCED_TOP4",0,l,gt);auto s=line.str();s.pop_back();top<<s<<','<<action<<",1,"<<allowed<<','<<reason<<'\n';
        }
        // The original gated selectors are replayed too, for exact archived parity.
        for(const std::string method:{"G4_COMMIT","G4_VALUE"}){
            auto r=valueSearch(l,table,q,method,extra);std::ostringstream line;line<<std::setprecision(17);emitValue(line,a,r,qi,extra,method,0,l,gt);auto s=line.str();s.pop_back();top<<s<<','<<r.choice<<",1,"<<allowed<<','<<reason<<'\n';
        }
        ++done;f.flush();top.flush();std::cout<<"CASE "<<done<<" qid="<<qi<<" E="<<extra<<" seconds="<<elapsed(began)<<std::endl;
    }
    need(done>0&&bool(f)&&bool(top),"all audit cases written");std::cout<<"V10_AUDIT_COMPLETE cases="<<done<<" seconds="<<elapsed(began)<<std::endl;
}
int main(int n,char** v){try{
    Args a(n,v);omp_set_num_threads(1);if(a.i("cpu",-1)>=0)need(SetThreadAffinityMask(GetCurrentThread(),DWORD_PTR(1)<<a.i("cpu"))!=0,"CPU affinity");
    if(a.s("mode")=="unit"){allocationUnit();std::cout<<"V10_UNIT_COMPLETE"<<std::endl;}
    else if(a.s("mode")=="audit")spaceAudit(a);else spaceEvaluate(a);return 0;
}catch(const std::exception& e){std::cerr<<"ERROR "<<e.what()<<std::endl;return 1;}}
