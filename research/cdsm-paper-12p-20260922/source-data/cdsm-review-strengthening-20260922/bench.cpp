using uint=unsigned int;
#include "strengthen.h"
#include "vendor/bench_io.h"
#include <map>
#include <sstream>
#include <sys/resource.h>

struct Args {
    std::map<std::string,std::string> a;
    Args(int n,char**v){need(n%2==1,"key value arguments");for(int i=1;i<n;i+=2)need(a.emplace(std::string(v[i]).substr(2),v[i+1]).second,"unique argument");}
    std::string s(std::string k,std::string fallback="")const{auto i=a.find(k);return i==a.end()?fallback:i->second;}
    int i(std::string k,int fallback=0)const{auto v=s(k);return v.empty()?fallback:std::stoi(v);}
};
std::vector<int> ints(std::string s){std::replace(s.begin(),s.end(),',',' ');std::istringstream f(s);std::vector<int> a;int x;while(f>>x)a.push_back(x);return a;}
template<class T>void seq(std::ostream& out,const T& values){bool first=true;for(auto v:values){if(!first)out<<';';out<<v;first=false;}}
int main(int argc,char**argv)try{
    Args a(argc,argv);omp_set_num_threads(1);
    std::unique_ptr<faiss::Index> index(faiss::read_index(a.s("index").c_str()));
    auto* h=dynamic_cast<faiss::IndexHNSWFlat*>(index.get());need(h&&h->metric_type==faiss::METRIC_L2,"HNSW L2 index");
    auto queries=bio::bin<float>(a.s("queries"));auto gt=bio::vecs<int>(a.s("gt"));
    need(queries.d==h->d&&queries.n==gt.n&&gt.d==10,"query truth dimensions");
    std::ifstream input(a.s("ids-file"));need(bool(input),"query IDs");std::vector<int> qids;int v;while(input>>v)qids.push_back(v);
    need(!qids.empty()&&std::set<int>(qids.begin(),qids.end()).size()==qids.size(),"unique nonempty query IDs");
    std::vector<int> mapping;if(!a.s("map").empty())mapping=bio::raw<int>(a.s("map"),int(h->ntotal));
    auto table=a.s("table").empty()?cdsm::entryTable(h->hnsw):bio::raw<int>(a.s("table"),64);
    std::vector<uint8_t> mask(h->ntotal,1);cdsm::Workspace w(*h);
    std::string policy=a.s("policy"),stage=a.s("stage");auto caps=ints(a.s("params"));need(!caps.empty(),"caps");
    bool check=a.i("check")!=0,capture=stage!="timing";int rep=a.i("rep-offset");w.diagnostic=check;
    auto execute=[&](int cap,int qid){
        need(qid>=0&&qid<queries.n,"query range");const float* q=queries.x.data()+size_t(qid)*queries.d;
        auto start=Clock::now();auto r=r2::search(w,q,h->d,mask,table,policy,cap,capture);auto ns=bio::ns(start);
        if(check){
            auto trace=w.trace;auto again=r2::search(w,q,h->d,mask,table,policy,cap,false);
            r2::same(r.out,again.out);need(trace==w.trace,"capture leaves full scoring sequence unchanged");
            if(policy=="F"){
                auto other=r2::search(w,q,h->d,mask,table,"F400_CHECK",cap,true);
                r2::same(r.out,other.out);need(trace==w.trace,"parameterized 400 path equals frozen F");
            }
        }
        for(int& id:r.out.ids)if(id>=0&&!mapping.empty())id=mapping.at(id);
        std::vector<int> result;for(int id:r.out.ids)if(id>=0)result.push_back(id);
        int hits=bio::hits(result,gt.x.data()+size_t(qid)*10);
        return std::tuple(r,ns,hits);
    };
    std::string path=a.s("out");need(!path.empty()&&!std::filesystem::exists(path),"preserve output");std::ofstream out(path);out<<std::setprecision(12);
    out<<"method,policy,param,rep,qid,ns,hits,dc,primary,scout,upper_dc,logical,edges,expansions,stop,sequence,ids,distances,primary_sequence,primary_stop,phase_end,phase_sequence,scouts,entries,endpoints,route_dc,route_upper_dc,seeded,descent_complete,roots,skipped_roots\n";
    std::mt19937 rng(20260922+rep);std::shuffle(caps.begin(),caps.end(),rng);
    for(int cap:caps){
        if(!check)for(int pass=0;pass<2;++pass)for(int j=0;j<std::min(32,int(qids.size()));++j)execute(cap,qids[j]);
        for(int qid:qids){auto [r,ns,hits]=execute(cap,qid);const auto& o=r.out;const auto& d=r.audit;
            out<<a.s("name")<<','<<policy<<','<<cap<<','<<rep<<','<<qid<<','<<ns<<','<<hits<<','<<o.dc<<','<<o.primary<<','<<o.scoutDc<<','<<o.upperDc<<','<<o.logical<<','<<o.edges<<','<<o.expansions<<','<<o.stop<<','<<o.sequence<<',';
            seq(out,o.ids);out<<',';seq(out,o.ds);out<<','<<o.primarySequence<<','<<d.primaryStop<<','<<d.phaseEnd<<','<<d.phaseSequence<<','<<o.scouts<<',';
            seq(out,d.entries);out<<',';seq(out,d.endpoints);out<<',';seq(out,d.costs);out<<',';seq(out,d.upper);out<<',';seq(out,d.seeded);out<<',';seq(out,d.complete);out<<',';seq(out,d.roots);out<<',';seq(out,d.skipped);out<<'\n';
        }
        out.flush();need(bool(out),"write results");std::cout<<"PROGRESS "<<a.s("name")<<" cap="<<cap<<" queries="<<qids.size()<<std::endl;
    }
    rusage usage{};getrusage(RUSAGE_SELF,&usage);
    std::ofstream meta(path+".json");meta<<"{\"queries\":"<<qids.size()<<",\"parameters\":"<<caps.size()<<",\"rep\":"<<rep<<",\"peak_process_bytes\":"<<uint64_t(usage.ru_maxrss)*1024<<",\"capture_route_audit\":"<<(capture?"true":"false")<<",\"engineering_check\":"<<(check?"true":"false")<<",\"warmup_passes\":"<<(check?0:2)<<"}\n";
    need(bool(meta),"metadata");return 0;
}catch(const std::exception& e){std::cerr<<"ERROR "<<e.what()<<std::endl;return 1;}
