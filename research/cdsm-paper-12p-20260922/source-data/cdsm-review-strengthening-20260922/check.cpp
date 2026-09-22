using uint=unsigned int;
#include "strengthen.h"

void rows(faiss::IndexHNSWFlat& index,const std::vector<std::vector<int>>& adjacency){
    for(int n=0;n<index.ntotal;++n){size_t a,b;index.hnsw.neighbor_range(n,0,&a,&b);need(adjacency[n].size()<=b-a,"test capacity");
        std::fill(index.hnsw.neighbors.begin()+a,index.hnsw.neighbors.begin()+b,-1);
        std::copy(adjacency[n].begin(),adjacency[n].end(),index.hnsw.neighbors.begin()+a);}
}
int main()try{
    omp_set_num_threads(1);constexpr int n=256,d=5;
    std::mt19937 rng(20260922);std::vector<float> x(n*d);for(float& v:x)v=float(int(rng()%13)-6);
    faiss::IndexHNSWFlat index(d,4,faiss::METRIC_L2);index.add(n,x.data());
    std::vector<std::vector<int>> adjacency(n);
    for(int i=0;i<n;++i)adjacency[i]={(i+1)%n,i,(i+1)%n,int(rng()%n),int(rng()%n)};
    rows(index,adjacency);auto original=index.hnsw.neighbors;auto table=cdsm::entryTable(index.hnsw);
    std::vector<uint8_t> mask(n,1);cdsm::Workspace w(index);w.diagnostic=true;
    int count=0,frontierRoutes=0;
    for(int qid=0;qid<4;++qid){std::array<float,d> q;for(float& v:q)v=float(int(rng()%13)-6);
        for(int prefix:{1,8,32,64,256})for(int cap:{1,2,7,16,31,65,127,255,256,271}){
            for(std::string p:{"C","F","MEP1","MEP4","MEP8","MEP16","FRONTIER","F200","F800"}){
                auto a=r2::search(w,q.data(),d,mask,table,p,cap,true,prefix);auto trace=w.trace;
                need(int(trace.size())==a.out.dc && std::set<int>(trace.begin(),trace.end()).size()==trace.size(),"exact unique physical evaluations");
                for(int node:w.touched)if(w.marked(node,cdsm::COMPLETED))need(w.pos[node]==w.end[node],"completed cursor exhausted");
                auto b=r2::search(w,q.data(),d,mask,table,p,cap,false,prefix);r2::same(a.out,b.out);need(trace==w.trace,"audit-free timing path parity");
                if(p=="F"){auto c=r2::search(w,q.data(),d,mask,table,"F400_CHECK",cap,true,prefix);r2::same(a.out,c.out);need(trace==w.trace,"original F unchanged");}
                if(p=="FRONTIER")frontierRoutes+=a.out.scouts;
                if(cap>=n&&prefix>=n){need(a.out.dc==n,"connected graph exhausted without padding");cdsm::Top exact;auto dis=std::unique_ptr<faiss::DistanceComputer>(index.storage->get_distance_computer());dis->set_query(q.data());for(int j=0;j<n;++j)exact.collect(j,(*dis)(j));for(int j=0;j<10;++j)need(a.out.ids[j]==exact.v[j].second&&a.out.ds[j]==exact.v[j].first,"exact top10 at exhaustion");}
                ++count;
            }
        }
    }
    need(frontierRoutes>0,"frontier takeover exercised");need(original==index.hnsw.neighbors,"read-only graph");
    std::cout<<"R2_ENGINEERING_PASS cases="<<count<<" frontier_routes="<<frontierRoutes<<" checks=hard_cap,unique_scores,shared_cursors,frontier_takeover,ties,duplicate_edges,empty_frontier,exact_exhaustion,audit_timing_parity,F400_parity,unchanged_graph"<<std::endl;
    return 0;
}catch(const std::exception& e){std::cerr<<"R2_ENGINEERING_FAIL "<<e.what()<<std::endl;return 1;}
