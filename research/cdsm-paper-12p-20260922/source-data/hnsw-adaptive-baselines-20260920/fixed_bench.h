// Query-only ports on read-only HNSW topology. CDSM reference headers are unchanged.
using uint = unsigned int;
#include "../fannbench-t2i-20260919/paid.h"
#include "../fannbench-t2i-20260919/bench_io.h"
#include "../ann-systematic-comparison-20260919/platform_port.h"
#include <iomanip>
#include <map>
#include <sstream>
#include <tuple>

struct Args {
    std::map<std::string,std::string> v;
    Args(int n,char** a){need(n%2==1,"--key value arguments");for(int i=1;i<n;i+=2){need(std::string(a[i]).starts_with("--"),"argument key");need(v.emplace(a[i]+2,a[i+1]).second,"duplicate argument");}}
    std::string s(std::string k,std::string fallback="")const{auto i=v.find(k);return i==v.end()?fallback:i->second;}
    int i(std::string k,int fallback=0)const{auto x=s(k);return x.empty()?fallback:std::stoi(x);}
};
struct Config {std::string policy;int id,ef;double gamma;int patience;double saturation;int cap;};
std::vector<int> ints(std::string s){std::replace(s.begin(),s.end(),',',' ');std::istringstream in(s);std::vector<int> r;int x;while(in>>x)r.push_back(x);return r;}
size_t peakMemory(){return peakResidentBytes();}

// PiP paper section 3: overlap between consecutive base-layer top-k sets,
// evaluated after each complete expansion. This is different from Lucene
// main's cumulative successful-collection ratio; it is not labeled native Lucene.
struct Patience {
    cdsm::Top previous;
    int stable=0;
    bool update(const cdsm::Top& current,double threshold,int patience){
        if(current.size<cdsm::K)return false;
        int overlap=0;
        for(int i=0;i<current.size;++i)for(int j=0;j<previous.size;++j)
            if(current.v[i].second==previous.v[j].second)++overlap;
        stable=double(overlap)/cdsm::K>=threshold?stable+1:0;
        previous=current;
        return stable>=patience;
    }
};

cdsm::Outcome beam(cdsm::Workspace& w,const float* q,const std::vector<uint8_t>& mask,const Config& c){
    w.reset(q,mask);w.initialize(c.cap);
    need(!w.main.active.empty(),"beam entry");
    std::priority_queue<cdsm::Node> beamBest;
    auto seed=w.main.active.top();beamBest.push(seed);
    cdsm::Top baseTop;baseTop.collect(seed.second,seed.first);
    Patience patience;int stop=2;
    while(w.dc<c.cap && !w.main.active.empty()){
        if(int(beamBest.size())==c.ef && w.main.active.top().first>beamBest.top().first){stop=1;break;}
        w.main.current=w.main.active.pop();w.open(w.main.current);
        while(w.dc<c.cap && w.pos[w.main.current]<w.end[w.main.current]){
            int n=w.arena[w.pos[w.main.current]++];if(w.marked(n,cdsm::BASE))continue;
            float d=w.score(n);w.mark(n,cdsm::BASE);baseTop.collect(n,d);
            if(int(beamBest.size())<c.ef || d<beamBest.top().first){
                w.main.active.add(n,d);beamBest.emplace(d,n);
                if(int(beamBest.size())>c.ef)beamBest.pop();
            }
        }
        bool completed=w.complete(w.main);
        if(completed && c.policy=="PIP" && patience.update(baseTop,c.saturation,c.patience)){stop=3;break;}
    }
    if(w.dc>=c.cap)stop=0;
    auto out=w.finish();out.stop=stop;return out;
}

// Independent simple reference for the lower layer, used only in engineering.
// It uses an ordered set and separate state, not the production heap/cursors.
cdsm::Outcome reference(cdsm::Workspace& w,const float* q,const std::vector<uint8_t>& mask,const Config& c){
    w.reset(q,mask);w.initialize(c.cap);need(!w.main.active.empty(),"reference entry");
    auto seed=w.main.active.top();std::set<cdsm::Node> candidates{seed},best{seed};
    cdsm::Top baseTop;baseTop.collect(seed.second,seed.first);
    std::set<int> previous;int stable=0,stop=2;
    while(w.dc<c.cap && !candidates.empty()){
        auto p=*candidates.begin();
        if(c.policy=="ABS" || c.policy=="ABS_CAP"){
            if(w.output.size==10 && double(p.first)>=(1+c.gamma)*(1+c.gamma)*w.output.kth()){stop=1;break;}
        }else if(int(best.size())==c.ef && p.first>best.rbegin()->first){stop=1;break;}
        candidates.erase(candidates.begin());size_t a,b;w.h.neighbor_range(p.second,0,&a,&b);
        for(size_t j=a;j<b && w.dc<c.cap;++j){
            int n=w.h.neighbors[j];if(n<0)break;if(w.marked(n,cdsm::BASE))continue;
            float d=w.score(n);w.mark(n,cdsm::BASE);baseTop.collect(n,d);
            if(c.policy=="ABS" || c.policy=="ABS_CAP" || int(best.size())<c.ef || d<best.rbegin()->first){
                candidates.emplace(d,n);best.emplace(d,n);
                if(c.policy!="ABS" && c.policy!="ABS_CAP" && int(best.size())>c.ef)best.erase(std::prev(best.end()));
            }
        }
        if(c.policy=="PIP" && baseTop.size==10){
            std::set<int> current;for(int i=0;i<10;++i)current.insert(baseTop.v[i].second);
            int overlap=0;for(int n:current)overlap+=previous.count(n);
            stable=double(overlap)/10>=c.saturation?stable+1:0;previous=current;
            if(stable>=c.patience){stop=3;break;}
        }
    }
    if(w.dc>=c.cap)stop=0;
    auto r=w.finish();r.stop=stop;return r;
}

cdsm::Outcome execute(cdsm::Workspace& w,const float* q,int dim,const std::vector<uint8_t>& mask,const std::vector<int>& table,const Config& c,bool check){
    cdsm::Outcome r;
    if(c.policy=="C" || c.policy=="F"){
        r=paid::search(w,q,dim,mask,table,c.policy,6400,6400,800,c.cap).out;
        r.stop=r.dc>=c.cap?0:2;
    }
    else if(c.policy=="ABS" || c.policy=="ABS_CAP"){
        w.reset(q,mask);w.initialize(c.cap);int stop=w.distanceAdvance(c.cap,c.gamma);r=w.finish();r.stop=stop;
    }else r=beam(w,q,mask,c);
    need(r.dc>0 && r.dc<=c.cap,"physical score bound");
    if(check && (c.policy=="ABS" || c.policy=="ABS_CAP" || c.policy=="PIP" || c.policy=="BEAM")){
        auto z=reference(w,q,mask,c);
        need(r.ids==z.ids && r.ds==z.ds && r.dc==z.dc && r.sequence==z.sequence && r.stop==z.stop,"independent traversal parity");
        if(c.policy=="PIP"){
            Config disabled=c;disabled.patience=std::numeric_limits<int>::max();auto a=beam(w,q,mask,disabled);
            disabled.policy="BEAM";auto b=beam(w,q,mask,disabled);
            need(a.ids==b.ids && a.dc==b.dc && a.sequence==b.sequence,"PiP disabled equals beam");
            need(r.dc<=b.dc,"PiP is an early stop of its beam control");
        }
    }
    return r;
}

void testRules(){
    cdsm::Top t;for(int j=0;j<10;++j)t.collect(j,float(j));
    Patience p;need(!p.update(t,.95,3),"initial window");need(!p.update(t,.95,3),"first stable window");
    need(!p.update(t,.95,3),"second stable window");need(p.update(t,.95,3),"third stable window");
    cdsm::Top changed;for(int j=0;j<9;++j)changed.collect(j,float(j));changed.collect(10,9);
    need(!p.update(changed,.95,3) && p.stable==0,"one replacement resets 95 percent at k10");
    need(1.15<(1.075*1.075) && 1.16>=(1.075*1.075),"squared L2 factor");
}

void query(const Args& a){
    omp_set_num_threads(1);testRules();
    need(a.s("mode","query")=="query","query-only driver; graph construction is unavailable");
    auto queries=bio::bin<float>(a.s("queries"));auto gt=bio::vecs<int>(a.s("gt"));
    std::ifstream qfile(a.s("ids-file"));need(bool(qfile),"query IDs");std::vector<int> qids;int qi;while(qfile>>qi)qids.push_back(qi);
    need(!qids.empty() && queries.n==gt.n && gt.d==10,"input shapes");
    std::unique_ptr<faiss::Index> index(faiss::read_index(a.s("index").c_str()));
    auto* h=dynamic_cast<faiss::IndexHNSWFlat*>(index.get());need(h && h->d==queries.d && h->metric_type==faiss::METRIC_L2,"fixed HNSW L2 index");
    std::vector<int> mapping;if(!a.s("map").empty())mapping=bio::raw<int>(a.s("map"),int(h->ntotal));
    std::vector<int> table=a.s("table").empty()?cdsm::entryTable(h->hnsw):bio::raw<int>(a.s("table"),64);
    std::vector<uint8_t> mask(h->ntotal,1);std::unique_ptr<cdsm::Workspace> workspace;
    if(a.s("policy")!="NATIVE")workspace=std::make_unique<cdsm::Workspace>(*h);
    std::ifstream cf(a.s("configs"));need(bool(cf),"configuration file");std::string line;std::getline(cf,line);
    std::map<int,Config> configs;Config c;while(cf>>c.policy>>c.id>>c.ef>>c.gamma>>c.patience>>c.saturation>>c.cap){
        if(c.policy==a.s("policy"))need(configs.emplace(c.id,c).second,"unique configuration");
    }
    auto params=ints(a.s("params"));need(!params.empty(),"config IDs");for(int id:params)need(configs.count(id),"known config");
    bool check=a.i("check",0)!=0;int rep=a.i("rep-offset",0);
    std::string path=a.s("out");need(!path.empty()&&!std::filesystem::exists(path),"preserve output");
    std::ofstream out(path);out<<std::setprecision(12);
    out<<"method,policy,param,rep,qid,ns,hits,dc,primary,scout,upper_dc,logical,edges,expansions,stop,sequence,ids,distances\n";
    auto run=[&](const Config& cfg,int qid){
        need(qid>=0&&qid<queries.n,"query range");const float* q=queries.x.data()+size_t(qid)*queries.d;
        auto started=Clock::now();cdsm::Outcome r;
        if(cfg.policy=="NATIVE"){
            h->hnsw.efSearch=cfg.ef;faiss::hnsw_stats.reset();faiss::idx_t ids[10];float ds[10];
            h->search(1,q,10,ds,ids);for(int j=0;j<10;++j){r.ids[j]=int(ids[j]);r.ds[j]=ds[j];}r.dc=int(faiss::hnsw_stats.ndis);r.stop=-1;
        }else r=execute(*workspace,q,queries.d,mask,table,cfg,check);
        auto ns=bio::ns(started);
        std::vector<int> result;for(int& id:r.ids){if(id>=0){if(!mapping.empty())id=mapping.at(id);result.push_back(id);}}
        int hits=bio::hits(result,gt.x.data()+size_t(qid)*10);
        return std::tuple(r,ns,hits);
    };
    std::mt19937 rng(20260920+rep);std::shuffle(params.begin(),params.end(),rng);
    for(int param:params){
        const auto& cfg=configs.at(param);
        if(!check)for(int j=0;j<std::min(10,int(qids.size()));++j)run(cfg,qids[j]);
        for(int qid:qids){auto [r,ns,hits]=run(cfg,qid);
            out<<a.s("name")<<','<<cfg.policy<<','<<param<<','<<rep<<','<<qid<<','<<ns<<','<<hits<<','<<r.dc<<','<<r.primary<<','<<r.scoutDc<<','<<r.upperDc<<','<<r.logical<<','<<r.edges<<','<<r.expansions<<','<<r.stop<<','<<r.sequence<<',';
            for(int j=0;j<10;++j){if(j)out<<';';out<<r.ids[j];}out<<',';
            for(int j=0;j<10;++j){if(j)out<<';';out<<r.ds[j];}out<<'\n';
        }
        out.flush();need(bool(out),"result write");std::cout<<"QUERY_PROGRESS "<<a.s("name")<<" param="<<param<<" queries="<<qids.size()<<std::endl;
    }
    std::ofstream meta(path+".json");meta<<"{\"queries\":"<<qids.size()<<",\"parameters\":"<<params.size()<<",\"rep\":"<<rep<<",\"peak_process_bytes\":"<<peakMemory()<<",\"index_bytes\":"<<std::filesystem::file_size(a.s("index"))<<",\"engineering_reference_check\":"<<(check?"true":"false")<<"}\n";
    need(bool(meta),"metadata write");
}
int main(int argc,char**argv)try{query(Args(argc,argv));return 0;}catch(const std::exception& e){std::cerr<<"ERROR "<<e.what()<<std::endl;return 1;}
