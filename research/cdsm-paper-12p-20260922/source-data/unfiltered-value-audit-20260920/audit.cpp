// Frozen CDSM is included unchanged. New controls and diagnostic replay live here.
using uint = unsigned int;
#include "paid.h"
#include "bench_io.h"
#include <Windows.h>
#include <map>
#include <sstream>

struct Args {
    std::map<std::string,std::string> v;
    Args(int n,char** a){need(n%2==1,"--key value pairs");for(int i=1;i<n;i+=2)need(v.emplace(a[i]+2,a[i+1]).second,"duplicate option");}
    std::string s(std::string k,std::string f="")const{auto p=v.find(k);return p==v.end()?f:p->second;}
    int i(std::string k,int f=0)const{auto x=s(k);return x.empty()?f:std::stoi(x);}
};
std::vector<std::string> split(std::string s,char c=','){std::vector<std::string> a;std::istringstream f(s);std::string t;while(std::getline(f,t,c))if(!t.empty())a.push_back(t);return a;}
std::vector<int> readids(std::string path){std::ifstream f(path);need(bool(f),"ID file");std::vector<int>a;int i;while(f>>i)a.push_back(i);return a;}

// Copy only touched records. Upper descent epochs remain monotone; every descent
// starts a fresh upper epoch, so old upper visitation marks cannot leak branches.
struct Snapshot {
    std::vector<int> touched,pos,end,arena,trace;std::vector<float> scores;std::vector<uint8_t> flags;
    cdsm::Frontier main,local;cdsm::Top output,gate;cdsm::Outcome counts;
    explicit Snapshot(const cdsm::Workspace& w):touched(w.touched),arena(w.arena),trace(w.trace),main(w.main),local(w.local),output(w.output),gate(w.gate),counts(w.finish()){
        for(int n:touched){scores.push_back(w.scores[n]);flags.push_back(w.flags[n]);pos.push_back(w.pos[n]);end.push_back(w.end[n]);}
    }
    void restore(cdsm::Workspace& w,const float* q,const std::vector<uint8_t>& mask)const{
        w.reset(q,mask);w.touched=touched;w.arena=arena;w.trace=trace;w.main=main;w.local=local;w.output=output;w.gate=gate;
        for(size_t i=0;i<touched.size();i++){int n=touched[i];w.stamp[n]=w.epoch;w.scores[n]=scores[i];w.flags[n]=flags[i];w.pos[n]=pos[i];w.end[n]=end[i];}
        w.dc=counts.dc;w.logical=counts.logical;w.edges=counts.edges;w.expansions=counts.expansions;
        w.scoutDc=counts.scoutDc;w.scouts=counts.scouts;w.upperDc=counts.upperDc;w.sequence=counts.sequence;
    }
};
struct Context {
    int primary=0,stop=0;uint64_t primarySeq=0,previewSeq=0;
    paid::Run preview;std::vector<int> pool;
};
Context primary(cdsm::Workspace& w,const float* q,int d,const std::vector<uint8_t>& mask,int extra){
    w.reset(q,mask);w.initialize(6400);Context c;c.stop=w.distanceAdvance(6400,.075);c.primary=w.dc;c.primarySeq=w.sequence;c.preview.cap=w.dc+extra;return c;
}
void preview(cdsm::Workspace& w,const float* q,int d,const std::vector<int>& table,Context& c){
    auto& r=c.preview;auto pool=paid::randomized(w,table,q,d);int limit=std::min(r.cap,c.primary+800);
    for(int i=0;i<int(pool.size())&&w.dc<limit;i++){
        int before=w.dc;++r.attempted;auto ds=paid::descend(w,pool[i],w.h.levels[pool[i]]-1,limit);
        if(!ds.complete){r.partialEntry=pool[i];r.partialEndpoint=ds.endpoint;r.partialLevel=ds.remainingLevel;r.partialDc=w.dc-before;break;}
        auto item=paid::inspect(w,pool[i],ds.endpoint,i,w.dc-before);r.featureEdges+=int(item.neighbors.size());r.candidates.push_back(std::move(item));
    }
    r.screenDc=w.dc-c.primary;r.screenEnd=w.dc;r.screenSequence=w.sequence;c.previewSeq=w.sequence;
    std::set<int> seen;
    auto add=[&](int i){if(c.pool.size()<8&&seen.insert(r.candidates[i].endpoint).second)c.pool.push_back(i);};
    for(auto rule:{"SCORE","DIVERSE"})for(int i:paid::select(r.candidates,rule))add(i);
    for(int i=0;i<int(r.candidates.size());i++)add(i);
}

// NSW §4.2 adaptation: shared queue/visited/results across starts, global k-th
// threshold and strict > stopping. Retain the stopping candidate for next start.
int nswAdvance(cdsm::Workspace& w,int cap){
    auto& f=w.main;
    while(w.dc<cap){
        if(f.current>=0){if(w.complete(f))continue;int n=w.arena[w.pos[f.current]++];if(w.marked(n,cdsm::BASE))continue;
            float d=w.score(n);w.mark(n,cdsm::BASE);f.active.add(n,d);w.gate.collect(n,d);continue;}
        while(!f.deferred.empty()){int n=f.deferred.pop();if(!w.marked(n,cdsm::COMPLETED))f.active.add(n,w.scores[n]);}
        while(!f.active.empty()&&w.marked(f.active.top().second,cdsm::COMPLETED))f.active.pop();
        if(f.active.empty())return 2;
        if(w.output.size==10&&f.active.top().first>w.output.kth())return 1;
        f.current=f.active.pop();w.open(f.current);
    }return 0;
}
void nswStarts(cdsm::Workspace& w,const float* q,int d,int starts,int cap){
    cdsm::JavaRandom rng(0x4e535720260920ULL^uint64_t(paid::javaFloatArrayHash(q,d)));
    for(int i=0;i<starts&&w.dc<cap;i++){
        int ep=-1;for(int tries=0;tries<int(w.scores.size());tries++){int x=rng.nextInt(int(w.scores.size()));if(!w.marked(x,cdsm::BASE)){ep=x;break;}}
        if(ep<0)break;w.seed(w.main,ep);++w.scouts;nswAdvance(w,cap);
    }
}
struct Answer {
    cdsm::Outcome out;int cap=0,preview=0,candidates=0,pool=0,entry=-1,endpoint=-1,available=1,tailStop=-1;
    uint64_t previewSeq=0;long long ns=0;std::string poolIds;
};
void fillContext(Answer& a,const Context& c){
    a.cap=c.preview.cap;a.preview=c.preview.screenDc;a.candidates=int(c.preview.candidates.size());a.pool=int(c.pool.size());
    a.out.primary=c.primary;a.out.stop=c.stop;a.out.primarySequence=c.primarySeq;a.previewSeq=c.previewSeq;
    std::ostringstream ids;for(int i:c.pool){if(ids.tellp()>0)ids<<';';ids<<c.preview.candidates[i].endpoint;}a.poolIds=ids.str();
}
bool usesPreview(const std::string& m){return m=="P_CONT"||m.starts_with("A")&&m!="ABS";}
Answer fromState(cdsm::Workspace& w,const float* q,int d,const std::vector<int>& table,const std::string& m,const Context& c){
    Answer a;int cap=c.preview.cap;
    if(m=="MEP"){
        // Same preselected upper-node table as F, four descent/injection attempts.
        // No private scout expansion; all endpoints go to the one global queue.
        for(int i=0;i<4;i++)w.scout(table,cap,400,true);
    }else if(m=="NSW4_CONT"||m=="NSW16_CONT")nswStarts(w,q,d,m=="NSW4_CONT"?4:16,cap);
    else if(m.starts_with("A")){
        auto fields=split(m,'_');need(fields.size()==2,"action name");int slot=std::stoi(fields[0].substr(1)),quota=std::stoi(fields[1]);
        if(slot<int(c.pool.size())){auto& candidate=c.preview.candidates[c.pool[slot]];a.entry=candidate.entry;a.endpoint=candidate.endpoint;paid::scout(w,a.entry,cap,quota,a.endpoint);}else a.available=0;
    }else need(m=="P_CONT","known from-state method");
    a.tailStop=w.distanceAdvance(cap,INFINITY);a.out=w.finish();fillContext(a,c);return a;
}
Answer execute(faiss::IndexHNSWFlat& index,cdsm::Workspace& w,const float* q,int d,const std::vector<uint8_t>& mask,const std::vector<int>& table,const std::string& method,int extra){
    auto started=Clock::now();Answer a;auto fields=split(method,':');const auto& m=fields[0];
    if(m=="NATIVE"){
        int ef=std::stoi(fields.at(1));index.hnsw.efSearch=ef;faiss::hnsw_stats.reset();std::array<faiss::idx_t,10> ids;index.search(1,q,10,a.out.ds.data(),ids.data());
        for(int j=0;j<10;j++)a.out.ids[j]=int(ids[j]);a.out.dc=int(faiss::hnsw_stats.ndis);a.cap=-1;
    }else if(m=="ABS"||m=="NSW_SEARCH"){
        w.reset(q,mask);int cap=100000;
        if(m=="ABS"){w.initialize(cap);a.tailStop=w.distanceAdvance(cap,std::stod(fields.at(1)));}
        else {nswStarts(w,q,d,std::stoi(fields.at(1)),cap);a.tailStop=w.dc==cap?0:1;}
        a.out=w.finish();a.cap=cap;
    }else if(m=="C"||m=="F"||m=="SCORE"||m=="DIVERSE"){
        auto r=paid::search(w,q,d,mask,table,m,6400,extra,800);a.out=r.out;a.cap=r.cap;a.preview=r.screenDc;a.candidates=int(r.candidates.size());a.previewSeq=r.screenSequence;
    }else{
        auto c=primary(w,q,d,mask,extra);if(usesPreview(m))preview(w,q,d,table,c);a=fromState(w,q,d,table,m,c);
    }
    a.ns=bio::ns(started);need(a.cap<0||a.out.dc<=a.cap,"distance cap");return a;
}
std::vector<std::string> strictMethods(){return {"C","F","SCORE","DIVERSE","MEP","NSW4_CONT","NSW16_CONT"};}
std::vector<std::string> actions(){std::vector<std::string> v{"P_CONT"};for(int i=0;i<8;i++)for(int d:{100,400,1600})v.push_back("A"+std::to_string(i)+"_"+std::to_string(d));return v;}
std::vector<std::string> naturals(){std::vector<std::string> v;for(int e:{32,64,128,256,512,1024})v.push_back("NATIVE:"+std::to_string(e));for(auto g:{"0","0.025","0.05","0.075","0.15","0.3"})v.push_back("ABS:"+std::string(g));for(int m:{1,4,16,64})v.push_back("NSW_SEARCH:"+std::to_string(m));return v;}
void same(const Answer& a,const Answer& b,const std::string& what){
    need(a.out.ids==b.out.ids&&a.out.ds==b.out.ds&&a.out.dc==b.out.dc&&a.out.sequence==b.out.sequence&&a.out.edges==b.out.edges&&a.out.expansions==b.out.expansions&&a.out.logical==b.out.logical,what);
}
void record(std::ostream& out,const Answer& a,const std::string& graph,const std::string& cohort,int qi,int extra,const std::string& m,int rep,const std::vector<int>& mapping,const int* gt){
    std::vector<int> ids;for(int id:a.out.ids)if(id>=0)ids.push_back(mapping.empty()?id:mapping.at(id));int hits=bio::hits(ids,gt);
    out<<graph<<','<<cohort<<','<<qi<<','<<extra<<','<<m<<','<<rep<<','<<hits<<','<<a.out.dc<<','<<a.cap<<','<<a.out.primary<<','<<a.out.stop<<','<<a.preview<<','<<a.out.scoutDc<<','<<a.out.scouts<<','<<a.out.edges<<','<<a.out.expansions<<','<<a.out.logical<<','<<a.out.sequence<<','<<a.out.primarySequence<<','<<a.previewSeq<<','<<a.candidates<<','<<a.pool<<','<<a.entry<<','<<a.endpoint<<','<<a.available<<','<<a.tailStop<<','<<a.ns<<',';bio::ids(out,ids);out<<','<<a.poolIds<<'\n';
}
void unitTests(){
    faiss::IndexHNSWFlat h(4,4);std::vector<float>x(128*4);for(int i=0;i<128;i++)for(int j=0;j<4;j++)x[i*4+j]=float((i*17+j*31)%113)/113;
    h.add(128,x.data());
    // A known connected ring removes random construction connectivity from the test.
    for(int n=0;n<128;n++){size_t a,b;h.hnsw.neighbor_range(n,0,&a,&b);std::fill(h.hnsw.neighbors.begin()+a,h.hnsw.neighbors.begin()+b,-1);int j=0;for(int delta:{1,127,7,121})h.hnsw.neighbors[a+j++]=(n+delta)%128;}
    cdsm::Workspace w(h);std::vector<uint8_t> mask(128,1);auto table=cdsm::entryTable(h.hnsw);auto q=x.data()+9*4;
    w.diagnostic=true;w.reset(q,mask);float d=w.score(5);auto seq=w.sequence;need(d==w.score(5)&&w.dc==1&&w.sequence==seq&&w.logical==2,"cache physical accounting");
    w.reset(q,mask);w.seed(w.main,0);w.distanceAdvance(7,INFINITY);Snapshot s(w);w.distanceAdvance(50,INFINITY);auto a=w.finish();auto trace=w.trace;
    s.restore(w,q,mask);w.distanceAdvance(50,INFINITY);auto b=w.finish();need(a.ids==b.ids&&a.dc==b.dc&&a.sequence==b.sequence&&w.trace==trace,"partial adjacency resume snapshot");
    need(w.dc==50&&std::set<int>(w.trace.begin(),w.trace.end()).size()==50,"exact unique budget");
    for(int cap:{1,2,3,9,17,31,63,100}){w.reset(q,mask);w.initialize(cap);w.distanceAdvance(cap,INFINITY);need(w.dc<=cap,"small cap no overrun");}
    w.reset(q,mask);w.initialize(20);w.distanceAdvance(20,INFINITY);Snapshot before(w);nswStarts(w,q,4,4,80);w.distanceAdvance(80,INFINITY);a=w.finish();
    before.restore(w,q,mask);nswStarts(w,q,4,4,80);w.distanceAdvance(80,INFINITY);b=w.finish();need(a.ids==b.ids&&a.sequence==b.sequence&&a.dc==b.dc,"random starts branch isolation");
    std::cout<<"UNIT_TESTS passed cache, tiny caps, interrupted adjacency, snapshot isolation, random starts"<<std::endl;
}
void run(const Args& a){
    omp_set_num_threads(1);if(a.i("cpu",-1)>=0)need(SetThreadAffinityMask(GetCurrentThread(),DWORD_PTR(1)<<a.i("cpu"))!=0,"CPU affinity");
    auto q=bio::bin<float>(a.s("queries"));auto gt=bio::vecs<int>(a.s("gt"));need(q.n==gt.n&&gt.d==11,"GT top11 shape");auto ids=readids(a.s("ids-file"));
    std::unique_ptr<faiss::Index> own(faiss::read_index(a.s("index").c_str()));auto* h=dynamic_cast<faiss::IndexHNSWFlat*>(own.get());need(h&&h->d==q.d&&h->metric_type==faiss::METRIC_L2,"unfiltered L2 HNSW");
    auto mapping=a.s("map").empty()?std::vector<int>{}:bio::raw<int>(a.s("map"),int(h->ntotal));
    auto table=a.s("table").empty()?cdsm::entryTable(h->hnsw):bio::raw<int>(a.s("table"),64);
    if(!a.s("base").empty()){
        auto base=bio::bin<float>(a.s("base"));need(base.n==h->ntotal&&base.d==h->d,"base shape");std::vector<float> row(h->d);
        for(int i=0;i<128;i++){int n=int(uint64_t(i)*7919%h->ntotal),canonical=mapping.empty()?n:mapping[n];h->storage->reconstruct(n,row.data());need(std::memcmp(row.data(),base.x.data()+size_t(canonical)*h->d,h->d*4)==0,"index mapping and base vector identity");}
    }
    cdsm::Workspace w(*h);std::vector<uint8_t> mask(h->ntotal,1);auto mode=a.s("mode","quality"),outpath=a.s("out");need(!std::filesystem::exists(outpath),"preserve result");std::ofstream out(outpath);need(bool(out),"output open");
    out<<"graph,cohort,qid,extra,method,rep,hits,dc,cap,primary,primary_stop,preview,scout,scouts,edges,expansions,logical,sequence,primary_sequence,preview_sequence,candidates,pool,entry,endpoint,available,tail_stop,ns,ids,pool_ids\n";
    auto emit=[&](const Answer& z,int qi,int e,const std::string&m,int rep){record(out,z,a.s("name"),a.s("cohort"),qi,e,m,rep,mapping,gt.x.data()+size_t(qi)*11);};
    auto start=Clock::now();int done=0;
    if(mode=="timing"){
        auto methods=split(a.s("methods"));int extra=a.i("extra");need(!methods.empty(),"timing methods");std::map<std::pair<int,std::string>,Answer> signature;
        for(int rep=-2;rep<5;rep++)for(int qi:ids){auto order=methods;std::mt19937 rng(20260920+qi*7+(rep+2)*9173);std::shuffle(order.begin(),order.end(),rng);
            for(const auto& m:order){auto z=execute(*h,w,q.x.data()+size_t(qi)*q.d,q.d,mask,table,m,extra);auto key=std::make_pair(qi,m);if(rep==-2)signature[key]=z;else same(z,signature.at(key),"timing repeat equality");if(rep>=0)emit(z,qi,extra,m,rep);}
            if(++done%80==0){out.flush();std::cout<<"TIMING "<<done<<'/'<<7*ids.size()<<" seconds="<<elapsed(start)<<std::endl;}}
    }else{
        for(int qi:ids){need(qi>=0&&qi<q.n,"query range");const float* x=q.x.data()+size_t(qi)*q.d;
            for(int extra:{3200,6400}){
                auto context=primary(w,x,q.d,mask,extra);Snapshot ps(w);preview(w,x,q.d,table,context);Snapshot ss(w);
                for(auto m:strictMethods()){
                    auto z=execute(*h,w,x,q.d,mask,table,m,extra);need(z.out.primary==context.primary&&z.out.primarySequence==context.primarySeq,"common primary equality");
                    if(m=="SCORE"||m=="DIVERSE")need(z.preview==context.preview.screenDc&&z.previewSeq==context.previewSeq&&z.candidates==context.preview.candidates.size(),"paid preview equality");
                    if(mode=="engineering"&&(m=="MEP"||m.starts_with("NSW"))){ps.restore(w,x,mask);auto t=fromState(w,x,q.d,table,m,context);same(z,t,"primary snapshot replay");}
                    emit(z,qi,extra,m,0);
                }
                for(auto m:actions()){
                    ss.restore(w,x,mask);auto z=fromState(w,x,q.d,table,m,context);
                    if(mode=="engineering"){auto t=execute(*h,w,x,q.d,mask,table,m,extra);same(z,t,"paid action snapshot vs full replay");}
                    emit(z,qi,extra,m,0);
                }
                if(mode=="engineering"){
                    auto frozen=w.run(x,mask,table,'F',6400,extra);auto z=execute(*h,w,x,q.d,mask,table,"F",extra);need(frozen.ids==z.out.ids&&frozen.sequence==z.out.sequence&&frozen.dc==z.out.dc,"original F reference vs paid F");
                    frozen=w.run(x,mask,table,'C',6400,extra);z=execute(*h,w,x,q.d,mask,table,"C",extra);need(frozen.ids==z.out.ids&&frozen.sequence==z.out.sequence&&frozen.dc==z.out.dc,"original C reference vs paid C");
                }
            }
            for(auto m:naturals())emit(execute(*h,w,x,q.d,mask,table,m,0),qi,0,m,0);
            if(++done%16==0){out.flush();std::cout<<"QUALITY "<<done<<'/'<<ids.size()<<" seconds="<<elapsed(start)<<std::endl;}
        }
    }
    out.flush();need(bool(out),"result write");std::cout<<"COMPLETE "<<outpath<<" seconds="<<elapsed(start)<<std::endl;
}
int main(int n,char**v){try{Args a(n,v);omp_set_num_threads(1);if(a.s("mode")=="unit")unitTests();else run(a);return 0;}catch(const std::exception&e){std::cerr<<"ERROR "<<e.what()<<std::endl;return 1;}}
