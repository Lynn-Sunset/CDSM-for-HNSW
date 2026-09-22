using uint=unsigned int;
#include "fanng_search.h"

namespace {
void setRows(faiss::IndexHNSWFlat& index,const std::vector<std::vector<int>>& rows){
    need(rows.size()==size_t(index.ntotal),"test rows shape");
    for(int n=0;n<int(index.ntotal);++n){
        size_t a,b;index.hnsw.neighbor_range(n,0,&a,&b);
        need(rows[n].size()<=b-a,"test row capacity");
        std::fill(index.hnsw.neighbors.begin()+a,index.hnsw.neighbors.begin()+b,-1);
        std::copy(rows[n].begin(),rows[n].end(),index.hnsw.neighbors.begin()+a);
    }
    index.hnsw.entry_point=0;index.hnsw.max_level=0;
}
void preserveGraph(const faiss::IndexHNSWFlat& index,const fanng::PreparedGraph& graph){
    uint64_t count=0;
    for(int n=0;n<int(index.ntotal);++n){
        size_t a,b;index.hnsw.neighbor_range(n,0,&a,&b);
        std::vector<int> original,ordered;
        for(size_t j=a;j<b && index.hnsw.neighbors[j]>=0;++j)original.push_back(index.hnsw.neighbors[j]);
        ordered.assign(graph.neighbors.begin()+graph.offsets[n],graph.neighbors.begin()+graph.offsets[n+1]);
        count+=original.size();std::sort(original.begin(),original.end());std::sort(ordered.begin(),ordered.end());
        need(original==ordered,"sorted adjacency preserves per-source edge multiset");
    }
    need(count==graph.directedEdges && count==graph.distanceComputations,"preprocessing physical count");
}
int parityChecks=0;
void checkQuery(cdsm::Workspace& w,const fanng::PreparedGraph& graph,const float* q,
        const std::vector<uint8_t>& mask,int cap){
    w.diagnostic=true;
    auto a=fanng::search(w,q,mask,graph,cap);auto trace=w.trace;
    auto b=fanng::reference(w,q,mask,graph,cap);
    fanng::requireParity(a,b);need(trace==w.trace,"exact physical score trace parity");++parityChecks;
}
void edgeInterleaving(){
    const std::vector<float> x{10,9,-10,8,0};
    faiss::IndexHNSWFlat index(1,4,faiss::METRIC_L2);index.add(x.size(),x.data());
    setRows(index,{{1,2},{3},{0},{4},{}});
    auto old=index.hnsw.neighbors;
    fanng::PreparedGraph graph(index);preserveGraph(index,graph);
    need(old==index.hnsw.neighbors,"preprocessing leaves source index unchanged");
    cdsm::Workspace w(index);w.diagnostic=true;std::vector<uint8_t> mask(x.size(),1);float q=0;
    auto r=fanng::search(w,&q,mask,graph,4);
    need(w.trace==std::vector<int>({0,1,3,4}),"per-edge greediness reaches deeper node before distant sibling");
    need(r.ids[0]==4 && r.dc==4 && r.stop==0,"interleaving result and hard cap");
    w.reset(&q,mask);w.initialize(4);w.distanceAdvance(4,INFINITY);
    need(w.trace==std::vector<int>({0,1,2,3}),"full expansion counterexample differs");
    for(int cap=1;cap<=8;++cap)checkQuery(w,graph,&q,mask,cap);
    auto exhausted=fanng::search(w,&q,mask,graph,8);
    need(exhausted.dc==5 && exhausted.stop==2,"empty queue reports true underfill without fabricated work");
    setRows(index,{{2,1},{3},{0},{4},{}});
    fanng::PreparedGraph reordered(index);preserveGraph(index,reordered);
    need(reordered.edge(0,0)==1 && reordered.edge(0,1)==2,"offline node-distance edge ordering");
}
void upperCacheIsNotVisited(){
    // Find two nodes with allocated upper levels in a deterministic small index.
    constexpr int n=256,d=3;
    std::vector<float> x(n*d);for(int i=0;i<n;++i){x[i*d]=float(i+1);x[i*d+1]=.5f;x[i*d+2]=0;}
    faiss::IndexHNSWFlat index(d,4,faiss::METRIC_L2);index.add(n,x.data());
    std::vector<int> upper;
    for(int i=0;i<n;++i)if(index.hnsw.levels[i]>1)upper.push_back(i);
    need(upper.size()>=2,"test upper nodes exist");
    int seed=upper[0],cached=upper[1],target=-1;
    for(int i=0;i<n;++i)if(i!=seed && i!=cached){target=i;break;}
    std::vector<std::vector<int>> rows(n);rows[seed]={cached};rows[cached]={target};
    setRows(index,rows);index.hnsw.entry_point=seed;index.hnsw.max_level=1;
    for(int v:upper){size_t a,b;index.hnsw.neighbor_range(v,1,&a,&b);std::fill(index.hnsw.neighbors.begin()+a,index.hnsw.neighbors.begin()+b,-1);}
    size_t a,b;index.hnsw.neighbor_range(seed,1,&a,&b);need(a<b,"upper capacity");index.hnsw.neighbors[a]=cached;
    // A query equal to seed makes cached strictly worse, so descent stays seed.
    const float* q=x.data()+seed*d;
    fanng::PreparedGraph graph(index);cdsm::Workspace w(index);std::vector<uint8_t> mask(n,1);
    w.diagnostic=true;auto result=fanng::search(w,q,mask,graph,10);
    need(result.upperDc==2 && result.dc==3,"upper cached target discovered without repeat distance");
    need(w.trace==std::vector<int>({seed,cached,target}),"upper-cached node remains traversable at L0");
    for(int cap=1;cap<=6;++cap)checkQuery(w,graph,q,mask,cap);
}
void randomAndTies(){
    constexpr int n=61,d=5;
    std::mt19937 rng(20260922);std::vector<float> x(n*d);
    // Quantized values exercise repeated distances and deterministic tie breaks.
    for(float& v:x)v=float(int(rng()%9)-4);
    faiss::IndexHNSWFlat index(d,8,faiss::METRIC_L2);index.add(n,x.data());
    std::vector<std::vector<int>> rows(n);
    for(int v=0;v<n;++v){rows[v].push_back((v+1)%n);for(int j=0;j<7;++j)rows[v].push_back(int(rng()%n));}
    // Includes self edges and duplicate edges: both are scanned, never rescored.
    setRows(index,rows);auto original=index.hnsw.neighbors;
    fanng::PreparedGraph graph(index);preserveGraph(index,graph);
    need(original==index.hnsw.neighbors,"random graph remains read only");
    auto cache=std::filesystem::temp_directory_path()/
        ("fanng-engineering-"+std::to_string(Clock::now().time_since_epoch().count())+".csr");
    graph.save(cache.string());auto restored=fanng::PreparedGraph::load(cache.string(),index);
    need(restored.offsets==graph.offsets && restored.neighbors==graph.neighbors &&
         restored.sourceTopologyFingerprint==graph.sourceTopologyFingerprint &&
         restored.directedEdges==graph.directedEdges && restored.buildSeconds==graph.buildSeconds,"CSR cache round trip");
    std::filesystem::remove(cache);
    cdsm::Workspace w(index);std::vector<uint8_t> mask(n,1);
    for(int query=0;query<12;++query){
        std::array<float,d> q;for(float& v:q)v=float(int(rng()%9)-4);
        for(int cap=1;cap<=n+3;++cap)checkQuery(w,graph,q.data(),mask,cap);
        auto full=fanng::search(w,q.data(),mask,graph,n+3);
        need(full.dc==n && full.stop==2,"connected graph exhausts all unique points");
        cdsm::Top exact;std::unique_ptr<faiss::DistanceComputer> distance(index.storage->get_distance_computer());distance->set_query(q.data());
        for(int v=0;v<n;++v)exact.collect(v,(*distance)(v));
        for(int j=0;j<10;++j)need(full.ids[j]==exact.v[j].second && full.ds[j]==exact.v[j].first,"exhaustion equals exact top10");
    }
    // A collector mask changes answers only; it does not truncate traversal.
    for(int v=0;v<n;++v)mask[v]=uint8_t(v%3==0);
    checkQuery(w,graph,x.data(),mask,19);
    auto filtered=fanng::search(w,x.data(),mask,graph,n+3);need(filtered.dc==n,"collector mask leaves traversal unchanged");
}
}
int main()try{
    omp_set_num_threads(1);edgeInterleaving();upperCacheIsNotVisited();randomAndTies();
    std::cout<<"FANNG_ENGINEERING_PASS parity_cases="<<parityChecks
        <<" checks=per_edge_interleaving,ordered_edges,unchanged_topology,hard_cap,empty_queue,upper_cache,duplicates,ties,exact_exhaustion"<<std::endl;
    return 0;
}catch(const std::exception& e){std::cerr<<"FANNG_ENGINEERING_FAIL "<<e.what()<<std::endl;return 1;}
