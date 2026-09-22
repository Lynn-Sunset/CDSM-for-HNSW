#include "NativeContinuation.h"
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
using namespace native_study;
namespace fs=std::filesystem;
constexpr int PRIMARY=512,EXTRA=512,SCOUT=16,EF=1000;
const std::array<std::string,5> ARMS={"NATIVE","C","KEEP","DROP","GLOBAL"};
struct Route{int number,entry=-1,seed=-1,skip=0,upper_dc=0,base_dc=0,steps=0,stop=0,active=0;};
struct Result{
    std::array<faiss::idx_t,10> ids{};std::array<float,10> dis{};
    std::vector<Score> trace;std::vector<Route> routes;
    int primary_steps=0,primary_dc=0,scout_steps=0,scout_dc=0,scout_base_dc=0,post_steps=0,post_dc=0;
    int64_t upper_hops=0;int stop=0;int64_t native_ndis=0,native_nhops=0;
};
inline bool answers_equal(const Result& a,const Result& b){
    if(a.ids!=b.ids)return false;for(int i=0;i<10;i++)if(std::bit_cast<uint32_t>(a.dis[i])!=std::bit_cast<uint32_t>(b.dis[i]))return false;return true;
}
Result native_reference(const faiss::HNSW& h,faiss::DistanceComputer& raw,size_t n){
    CountDC dc(raw);Answers a;faiss::VisitedTableVector vt(n);faiss::SearchParametersHNSW par;par.efSearch=EF;par.bounded_queue=true;par.check_relative_distance=true;
    auto s=h.search(dc,nullptr,a.handler,vt,&par);Result r;r.ids=a.sorted_ids();r.dis=a.sorted_dis();r.trace=std::move(dc.trace);r.native_ndis=s.ndis;r.native_nhops=s.nhops;return r;
}
Result execute(const faiss::HNSW& h,faiss::DistanceComputer& raw,size_t n,const std::vector<int>& entries,const std::string& arm,bool uncapped=false,bool unsplit=false){
    CountDC dc(raw);Answers a;faiss::VisitedTableVector vt(n);Frontier main(EF);Result r;
    float d;int nearest=descend(h,dc,h.entry_point,h.max_level,d,r.upper_hops);main.candidates.push(nearest,d);
    if(uncapped){advance(h,dc,a,main,vt,INT_MAX);r.primary_steps=main.steps;r.primary_dc=int(dc.trace.size());}
    else if(unsplit){advance(h,dc,a,main,vt,PRIMARY+EXTRA);r.primary_steps=main.steps;r.primary_dc=int(dc.trace.size());}
    else{
        advance(h,dc,a,main,vt,PRIMARY);r.primary_steps=main.steps;r.primary_dc=int(dc.trace.size());
        if(arm!="C"){
            std::unordered_set<int> used;
            for(int route=0;route<4;route++){
                dc.phase=route+1;Route rr;rr.number=route;
                for(int e:entries)if(!used.count(e)&&!vt.get(e)){rr.entry=e;used.insert(e);break;}
                if(rr.entry<0){rr.skip=1;r.routes.push_back(rr);continue;}
                size_t before=dc.trace.size();float sd;rr.seed=descend(h,dc,rr.entry,h.levels[rr.entry]-1,sd,r.upper_hops);rr.upper_dc=int(dc.trace.size()-before);
                if(vt.get(rr.seed)){rr.skip=2;r.routes.push_back(rr);continue;}
                Answers local;Frontier scout(10);scout.candidates.push(rr.seed,sd);
                int allowance=std::min(SCOUT,EXTRA-r.scout_steps);
                rr.steps=advance(h,dc,local,scout,vt,allowance,&a,arm=="GLOBAL");rr.base_dc=int(scout.distances);rr.stop=scout.stop;rr.active=scout.candidates.size();
                r.scout_steps+=rr.steps;r.scout_base_dc+=rr.base_dc;
                if(arm!="DROP")merge(main,scout);
                r.routes.push_back(rr);
            }
        }
        r.scout_dc=int(dc.trace.size())-r.primary_dc;dc.phase=5;auto begin=dc.trace.size();
        r.post_steps=advance(h,dc,a,main,vt,EXTRA-r.scout_steps);r.post_dc=int(dc.trace.size()-begin);
        require(r.primary_steps+r.scout_steps+r.post_steps<=r.primary_steps+EXTRA,"base expansion cap violated");
    }
    r.stop=main.stop;r.ids=a.sorted_ids();r.dis=a.sorted_dis();r.trace=std::move(dc.trace);return r;
}
std::vector<float> read_queries(const std::string& path,int& nq,int& dim){
    std::ifstream f(path,std::ios::binary);require(bool(f),"query file missing");f.read(reinterpret_cast<char*>(&nq),4);f.read(reinterpret_cast<char*>(&dim),4);
    require(nq==7999&&dim==200,"unexpected query shape");std::vector<float> q(size_t(nq)*dim);f.read(reinterpret_cast<char*>(q.data()),q.size()*4);require(bool(f),"query read incomplete");
    for(float v:q)require(std::isfinite(v),"nonfinite query");return q;
}
std::vector<int> read_ints(const std::string& path){std::ifstream f(path);require(bool(f),"integer file missing");std::vector<int> v;int x;while(f>>x)v.push_back(x);return v;}
std::vector<int> make_entries(const faiss::HNSW& h){
    std::vector<int> pool;
    for(int level=1;level<=h.max_level;level++)for(size_t i=0;i<h.levels.size();i++)if(h.levels[i]>level)pool.push_back(int(i));
    require(pool.size()>64,"upper membership too small");faiss::RandomGenerator rng(int64_t(h.levels.size()));std::vector<int> result;std::unordered_set<int> used;
    while(result.size()<64){int v=pool[rng.rand_int(int(pool.size()))];if(used.insert(v).second)result.push_back(v);}return result;
}
void save_trace(const fs::path& p,const std::vector<Score>& v){
    static_assert(sizeof(Score)==12);require(!fs::exists(p),"trace already exists");std::ofstream f(p,std::ios::binary);uint64_t n=v.size();f.write(reinterpret_cast<const char*>(&n),8);f.write(reinterpret_cast<const char*>(v.data()),v.size()*sizeof(Score));require(bool(f),"trace write failed");
}
std::string ids_text(const Result& r){std::ostringstream o;for(int i=0;i<10;i++){if(i)o<<';';o<<r.ids[i];}return o.str();}
std::string bits_text(const Result& r){std::ostringstream o;for(int i=0;i<10;i++){if(i)o<<';';o<<std::bit_cast<uint32_t>(r.dis[i]);}return o.str();}
std::string ids_values(const std::vector<int>& ids){std::ostringstream o;for(size_t i=0;i<ids.size();i++){if(i)o<<';';o<<ids[i];}return o.str();}
struct ToyDC:faiss::DistanceComputer{
    std::vector<float> values;explicit ToyDC(std::vector<float> v):values(std::move(v)){}
    void set_query(const float*)override{}
    float operator()(faiss::idx_t i)override{return values.at(size_t(i));}
    float symmetric_dis(faiss::idx_t,faiss::idx_t)override{return 0;}
};
void selftest(const fs::path& out){
    require(!fs::exists(out),"selftest output exists");std::mt19937 rng(2026091308);int cases=0,checks=0;
    for(int rep=0;rep<360;rep++){
        int n=25+int(rng()%100);faiss::HNSW h(8);h.levels.assign(n,1);h.entry_point=0;h.max_level=0;h.offsets.resize(n+1);h.neighbors.resize(size_t(n)*16);
        for(int i=0;i<=n;i++)h.offsets[i]=size_t(i)*16;
        std::fill(h.neighbors.begin(),h.neighbors.end(),-1);
        for(int i=0;i<n;i++){
            if(rep%6==0){if(i+1<n)h.neighbors[size_t(i)*16]=i+1;}
            else for(int j=0;j<8;j++)h.neighbors[size_t(i)*16+j]=int(rng()%n);
        }
        std::vector<float> scores(n);for(float& x:scores)x=rep%6==0?0.5f:float(rng()%257)/256.0f;ToyDC raw(scores);
        auto ref=native_reference(h,raw,n);auto full=execute(h,raw,n,{},"C",true);require(answers_equal(ref,full)&&same_trace(ref.trace,full.trace),"synthetic upstream mismatch");
        require(ref.trace.size()==size_t(ref.native_ndis+1)&&ref.native_nhops==full.primary_steps+full.upper_hops,"synthetic native counters mismatch");checks+=2;
        for(int cut:{0,1,2,7,16,31}){
            CountDC dc(raw);Answers a;faiss::VisitedTableVector vt(n);Frontier f(EF);f.candidates.push(0,dc(0));advance(h,dc,a,f,vt,cut);advance(h,dc,a,f,vt,INT_MAX);
            Result split;split.ids=a.sorted_ids();split.dis=a.sorted_dis();split.trace=std::move(dc.trace);require(answers_equal(ref,split)&&same_trace(ref.trace,split.trace),"synthetic continuation mismatch");checks++;
        }
        // Native width-ten tied frontier has implicit strict admission; it is not Java A_LOCAL.
        if(rep==0){CountDC dc(raw);Answers a;faiss::VisitedTableVector vt(n);Frontier f(10);f.candidates.push(0,dc(0));advance(h,dc,a,f,vt,INT_MAX);require(dc.trace.size()==11,"native tie frontier boundary changed");checks++;}
        cases++;
    }
    {
        faiss::HNSW h(8);h.levels.assign(32,1);h.entry_point=0;h.max_level=0;h.offsets.resize(33);h.neighbors.resize(32*16);
        for(int i=0;i<=32;i++)h.offsets[i]=size_t(i)*16;std::fill(h.neighbors.begin(),h.neighbors.end(),-1);
        for(int i=20;i<31;i++)h.neighbors[size_t(i)*16]=i+1;
        std::vector<float> values(32,0);for(int i=20;i<32;i++)values[i]=float(i);ToyDC raw(values);
        for(bool global:{false,true}){
            Answers main,local;for(int i=0;i<10;i++)main.handler.add_result(values[i],i);CountDC dc(raw);faiss::VisitedTableVector vt(32);Frontier f(10);f.candidates.push(20,dc(20));
            int steps=advance(h,dc,local,f,vt,4,&main,global);
            require(steps==(global?0:4)&&dc.trace.size()==size_t(global?1:5),"synthetic global/local stop mismatch");checks++;
        }
    }
    std::ofstream f(out);f<<"{\"status\":\"passed\",\"synthetic_graphs\":"<<cases<<",\"checks\":"<<checks<<",\"native_tied_width10_scored\":11,\"split_boundaries_per_graph\":6,\"known_global_stop_fixture\":true}\n";
    std::cout<<"Native synthetic checks passed: "<<cases<<" graphs, "<<checks<<" checks."<<std::endl;
}
struct Inputs{
    std::unique_ptr<faiss::Index> own;faiss::IndexHNSW* index;faiss::IndexFlat* flat;
    int nq=0,dim=0;std::vector<float> queries;std::vector<int> ids,entries;
    Inputs(const char* ix,const char* qs,const char* qi):own(faiss::read_index(ix)),index(dynamic_cast<faiss::IndexHNSW*>(own.get())),flat(index?dynamic_cast<faiss::IndexFlat*>(index->storage):nullptr){
        require(index&&flat&&index->ntotal==1000000&&index->d==200&&index->metric_type==faiss::METRIC_L2,"wrong native index");
        require(!index->hnsw.is_similarity&&!index->hnsw.is_panorama&&index->hnsw.nb_neighbors(0)==32,"unsupported native graph flags");
        index->hnsw.efSearch=EF;index->hnsw.check_relative_distance=true;index->hnsw.search_bounded_queue=true;index->hnsw.use_visited_hashset=false;
        queries=read_queries(qs,nq,dim);ids=read_ints(qi);require(ids.size()==size_t(nq),"wrong ids length");entries=make_entries(index->hnsw);
    }
    std::unique_ptr<faiss::DistanceComputer> distance(int pos){auto d=std::unique_ptr<faiss::DistanceComputer>(flat->get_distance_computer());d->set_query(queries.data()+size_t(pos)*dim);return d;}
};
void preflight(Inputs& x,const fs::path& out){
    require(!fs::exists(out),"preflight output exists");fs::create_directory(out);std::ofstream entries(out/"entries.txt");for(int e:x.entries)entries<<e<<'\n';entries.close();
    int checked=0;faiss::distance_compute_blas_threshold=INT_MAX;
    for(int p:{0,137,3999,7998}){
        auto d=x.distance(p);auto ref=native_reference(x.index->hnsw,*d,x.index->ntotal);auto full=execute(x.index->hnsw,*d,x.index->ntotal,x.entries,"C",true);
        require(answers_equal(ref,full)&&same_trace(ref.trace,full.trace),"real native full mismatch");require(ref.trace.size()==size_t(ref.native_ndis+1)&&ref.native_nhops==full.primary_steps+full.upper_hops,"real native counters mismatch");
        std::array<float,10> ds;std::array<faiss::idx_t,10> is;x.index->search(1,x.queries.data()+size_t(p)*x.dim,10,ds.data(),is.data());
        require(is==ref.ids,"public native ids mismatch");for(int k=0;k<10;k++)require(std::bit_cast<uint32_t>(ds[k])==std::bit_cast<uint32_t>(ref.dis[k]),"public native score bits mismatch");
        auto c=execute(x.index->hnsw,*d,x.index->ntotal,x.entries,"C");auto one=execute(x.index->hnsw,*d,x.index->ntotal,x.entries,"C",false,true);
        require(answers_equal(c,one)&&same_trace(c.trace,one.trace),"real split C mismatch");
        auto keep=execute(x.index->hnsw,*d,x.index->ntotal,x.entries,"KEEP"),drop=execute(x.index->hnsw,*d,x.index->ntotal,x.entries,"DROP");
        auto global=execute(x.index->hnsw,*d,x.index->ntotal,x.entries,"GLOBAL");
        require(global.primary_dc==c.primary_dc&&std::equal(c.trace.begin(),c.trace.begin()+c.primary_dc,global.trace.begin(),[](Score a,Score b){return a.id==b.id&&a.bits==b.bits;}),"GLOBAL primary mismatch");
        size_t prefix=size_t(keep.primary_dc+keep.scout_dc);require(prefix==size_t(drop.primary_dc+drop.scout_dc),"KEEP/DROP prefix length mismatch");
        require(std::equal(keep.trace.begin(),keep.trace.begin()+prefix,drop.trace.begin(),[](Score a,Score b){return a.id==b.id&&a.bits==b.bits&&a.phase==b.phase;}),"KEEP/DROP scout prefix mismatch");
        std::array<float,50> gd;std::array<faiss::idx_t,50> gi;x.flat->search(1,x.queries.data()+size_t(p)*x.dim,50,gd.data(),gi.data());
        for(int j=0;j<50;j++)require(std::bit_cast<uint32_t>(gd[j])==std::bit_cast<uint32_t>((*d)(gi[j])),"flat truth and native scorer bit mismatch");
        save_trace(out/("native-"+std::to_string(p)+".bin"),ref.trace);save_trace(out/("keep-"+std::to_string(p)+".bin"),keep.trace);save_trace(out/("global-"+std::to_string(p)+".bin"),global.trace);checked++;
    }
    double minnorm=1e30,maxnorm=0;for(int p=0;p<x.nq;p++){double norm=0;for(int j=0;j<x.dim;j++){double v=x.queries[size_t(p)*x.dim+j];norm+=v*v;}minnorm=std::min(minnorm,norm);maxnorm=std::max(maxnorm,norm);}
    std::ofstream f(out/"COMPLETE.json");f<<std::setprecision(17)<<"{\"status\":\"passed\",\"positions\":[0,137,3999,7998],\"cases\":"<<checked<<",\"native_public_api_and_trace_matched\":true,\"flat_scorer_bits_matched\":true,\"query_squared_norm_min\":"<<minnorm<<",\"query_squared_norm_max\":"<<maxnorm<<"}\n";
    std::cout<<"Native preflight passed."<<std::endl;
}
void groundtruth(Inputs& x,const fs::path& out){
    require(!fs::exists(out),"truth output exists");fs::create_directory(out);faiss::distance_compute_blas_threshold=INT_MAX;
    std::ofstream f(out/"top50.bin",std::ios::binary);int32_t header[]={0x4e464731,x.nq,50,x.dim};f.write(reinterpret_cast<char*>(header),16);int ties=0;
    for(int p=0;p<x.nq;p+=8){
        int n=std::min(8,x.nq-p);std::vector<float> ds(size_t(n)*50);std::vector<faiss::idx_t> is(size_t(n)*50);x.flat->search(n,x.queries.data()+size_t(p)*x.dim,50,ds.data(),is.data());
        for(int i=0;i<n;i++){auto dc=x.distance(p+i);if(ds[i*50+9]==ds[i*50+10])ties++;
            for(int j=0;j<50;j++){int32_t id=int32_t(is[i*50+j]);uint32_t bits=std::bit_cast<uint32_t>(ds[i*50+j]);require(bits==std::bit_cast<uint32_t>((*dc)(id)),"GT native scoring bits mismatch");f.write(reinterpret_cast<char*>(&id),4);f.write(reinterpret_cast<char*>(&bits),4);}
        }
        if(p%128==0)std::cout<<"groundtruth "<<p+n<<'/'<<x.nq<<std::endl;
    }
    f.close();std::ofstream meta(out/"COMPLETE.json");meta<<"{\"status\":\"passed\",\"queries\":"<<x.nq<<",\"k\":50,\"rank10_rank11_ties\":"<<ties<<",\"all_returned_score_bits_native_checked\":true,\"blas_disabled\":true}\n";std::cout<<"groundtruth completed"<<std::endl;
}
struct GT{std::vector<std::array<int,10>> top;explicit GT(const fs::path& path){std::ifstream f(path,std::ios::binary);int32_t h[4];f.read(reinterpret_cast<char*>(h),16);require(h[0]==0x4e464731&&h[1]==7999&&h[2]==50&&h[3]==200,"bad native truth");top.resize(7999);for(int i=0;i<7999;i++)for(int j=0;j<50;j++){int32_t id;uint32_t bits;f.read(reinterpret_cast<char*>(&id),4);f.read(reinterpret_cast<char*>(&bits),4);if(j<10)top[i][j]=id;}require(bool(f),"incomplete truth");}};
int hits(const Result& r,const std::array<int,10>& gt){int n=0;for(auto id:r.ids)n+=std::find(gt.begin(),gt.end(),id)!=gt.end();return n;}
void effect(Inputs& x,const fs::path& truth,const fs::path& out){
    require(!fs::exists(out),"effect output exists");fs::create_directory(out);fs::create_directory(out/"traces");fs::create_directory(out/"witnesses");GT gt(truth);
    std::ofstream f(out/"effect.csv"),routes(out/"routes.csv"),u(out/"unions.csv"),checks(out/"anchors.csv");
    f<<"position,qid,arm,hits,returned_ids,score_bits,dc,primary_dc,scout_dc,scout_base_dc,post_dc,primary_steps,scout_steps,post_steps,upper_hops,stop,trace_hash,primary_hash,prescout_hash\n";
    routes<<"position,qid,arm,route,entry,seed,skip,upper_dc,base_dc,steps,stop,active\n";
    u<<"position,qid,arm,c_hits,arm_hits,union_hits,gt_coverage,union_size,replay_scoring_calls,union_top10,new_gt_after_scout,post_gt_ids\n";
    checks<<"position,qid,native_full_equal,native_counter_equal,C_split_equal,KEEP_DROP_prefix_equal,primary_paths_equal,score_bits_stationary\n";
    for(int p=0;p<x.nq;p++){
        auto dc=x.distance(p);auto ref=native_reference(x.index->hnsw,*dc,x.index->ntotal);auto full=execute(x.index->hnsw,*dc,x.index->ntotal,x.entries,"C",true);
        require(answers_equal(ref,full)&&same_trace(ref.trace,full.trace),"full native anchor mismatch");require(ref.trace.size()==size_t(ref.native_ndis+1)&&ref.native_nhops==full.primary_steps+full.upper_hops,"native stats anchor mismatch");
        ref.primary_steps=full.primary_steps;ref.primary_dc=int(ref.trace.size());ref.upper_hops=full.upper_hops;ref.stop=full.stop;
        std::vector<Result> rs;rs.push_back(std::move(ref));for(int a=1;a<5;a++)rs.push_back(execute(x.index->hnsw,*dc,x.index->ntotal,x.entries,ARMS[a]));
        auto one=execute(x.index->hnsw,*dc,x.index->ntotal,x.entries,"C",false,true);require(answers_equal(one,rs[1])&&same_trace(one.trace,rs[1].trace),"C segmented anchor mismatch");
        auto& keep=rs[2];auto& drop=rs[3];size_t prefix=size_t(keep.primary_dc+keep.scout_dc);
        require(prefix==size_t(drop.primary_dc+drop.scout_dc)&&std::equal(keep.trace.begin(),keep.trace.begin()+prefix,drop.trace.begin(),[](Score a,Score b){return a.id==b.id&&a.bits==b.bits&&a.phase==b.phase;}),"KEEP DROP prefix mismatch");
        for(int a=2;a<5;a++)require(rs[a].primary_dc==rs[1].primary_dc&&std::equal(rs[1].trace.begin(),rs[1].trace.begin()+rs[1].primary_dc,rs[a].trace.begin(),[](Score a,Score b){return a.id==b.id&&a.bits==b.bits;}),"primary anchor mismatch");
        std::unordered_map<int,uint32_t> all_scores;for(const auto& r:rs)for(auto s:r.trace){auto [it,fresh]=all_scores.emplace(s.id,s.bits);require(fresh||it->second==s.bits,"nonstationary native score bits");}
        for(int a=0;a<5;a++){
            const auto& r=rs[a];require(std::unordered_set<faiss::idx_t>(r.ids.begin(),r.ids.end()).size()==10,"duplicate output ids");
            f<<p<<','<<x.ids[p]<<','<<ARMS[a]<<','<<hits(r,gt.top[p])<<','<<ids_text(r)<<','<<bits_text(r)<<','<<r.trace.size()<<','<<r.primary_dc<<','<<r.scout_dc<<','<<r.scout_base_dc<<','<<r.post_dc<<','<<r.primary_steps<<','<<r.scout_steps<<','<<r.post_steps<<','<<r.upper_hops<<','<<r.stop<<','<<trace_hash(r.trace)<<','<<trace_hash(r.trace,0,r.primary_dc)<<','<<trace_hash(r.trace,0,r.primary_dc+r.scout_dc)<<'\n';
            for(auto rr:r.routes)routes<<p<<','<<x.ids[p]<<','<<ARMS[a]<<','<<rr.number<<','<<rr.entry<<','<<rr.seed<<','<<rr.skip<<','<<rr.upper_dc<<','<<rr.base_dc<<','<<rr.steps<<','<<rr.stop<<','<<rr.active<<'\n';
            if(p==0||p==137||p==3999||p==7998)save_trace(out/"traces"/(std::to_string(p)+"-"+ARMS[a]+".bin"),r.trace);
        }
        for(int a:{2,4}){
            const auto& r=rs[a];std::unordered_set<int> set;for(auto s:rs[1].trace)set.insert(s.id);size_t boundary=size_t(r.primary_dc+r.scout_dc);for(size_t i=0;i<boundary;i++)set.insert(r.trace[i].id);
            std::vector<std::pair<float,int>> ordered;for(int id:set)ordered.emplace_back(std::bit_cast<float>(all_scores.at(id)),id);std::sort(ordered.begin(),ordered.end());std::vector<int> top;for(int j=0;j<10;j++)top.push_back(ordered[j].second);
            int uh=0,cov=0;std::vector<int> later;for(int id:gt.top[p]){uh+=std::find(top.begin(),top.end(),id)!=top.end();cov+=set.count(id);if(std::find(r.ids.begin(),r.ids.end(),id)!=r.ids.end()&&!set.count(id))later.push_back(id);}
            u<<p<<','<<x.ids[p]<<','<<ARMS[a]<<','<<hits(rs[1],gt.top[p])<<','<<hits(r,gt.top[p])<<','<<uh<<','<<cov<<','<<set.size()<<','<<rs[1].trace.size()+r.scout_dc<<','<<ids_values(top)<<','<<later.size()<<','<<ids_values(later)<<'\n';
            if(hits(rs[1],gt.top[p])<5&&hits(r,gt.top[p])>=5&&cov<5){
                for(int w:{1,a}){auto path=out/"witnesses"/(std::to_string(p)+"-"+ARMS[w]+".bin");if(!fs::exists(path))save_trace(path,rs[w].trace);}
            }
        }
        if((hits(keep,gt.top[p])<5)!=(hits(drop,gt.top[p])<5))for(int w:{1,2,3}){
            auto path=out/"witnesses"/(std::to_string(p)+"-"+ARMS[w]+".bin");if(!fs::exists(path))save_trace(path,rs[w].trace);
        }
        checks<<p<<','<<x.ids[p]<<",1,1,1,1,1,1\n";
        if(p%100==0){f.flush();routes.flush();u.flush();checks.flush();std::cout<<"effect "<<p+1<<'/'<<x.nq<<std::endl;}
    }
    f.close();routes.close();u.close();checks.close();std::ofstream meta(out/"COMPLETE.json");meta<<"{\"status\":\"passed\",\"queries\":7999,\"effect_rows\":39995,\"extra_anchor_replays\":15998,\"route_rows\":95988,\"union_rows\":15998,\"complete_native_paths_equal\":true,\"C_pause_paths_equal\":true,\"KEEP_DROP_prefix_equal\":true,\"native_scores_stationary\":true}\n";std::cout<<"effect completed"<<std::endl;
}
int main(int argc,char** argv){
    try{
        require(argc>=3,"mode and output required");std::string mode=argv[1];omp_set_num_threads(mode=="groundtruth"?4:1);
        if(mode=="selftest"){selftest(argv[2]);return 0;}
        require(argc>=6,"need index queries ids output");Inputs x(argv[2],argv[3],argv[4]);
        if(mode=="preflight")preflight(x,argv[5]);
        else if(mode=="groundtruth")groundtruth(x,argv[5]);
        else if(mode=="effect"){require(argc==7,"need truth and output");effect(x,argv[5],argv[6]);}
        else throw std::runtime_error("unknown mode");
        return 0;
    }catch(const std::exception& e){std::cerr<<"FAILED: "<<e.what()<<std::endl;return 1;}
}
