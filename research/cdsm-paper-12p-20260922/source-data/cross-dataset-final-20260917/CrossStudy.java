package phase0;

import java.io.*;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.*;
import org.apache.lucene.util.hnsw.*;

/** Final-policy adapter. Only the scorer and distance conversion depend on metric. */
public final class CrossStudy {
  static long checks;
  static void need(boolean ok,String why){if(!ok)throw new AssertionError(why);checks++;}
  record Policy(String name,String family,double parameter,int cap){}
  record Result(int[] hits,int dc,int logical,int edges,int expansions,int primary,int stop,
                String selected,int scoutDc,int scouts,long sequence,long nanos,int[] trace){}
  static final class Data implements AutoCloseable {
    Directory directory; DirectoryReader reader;
    PhaseT2IEndToEnd.Leaf leaf;
    VectorSimilarityFunction metric;
    Data(Properties p)throws Exception{
      metric=VectorSimilarityFunction.valueOf(p.getProperty("metric","EUCLIDEAN"));
      if(p.getProperty("kind").equals("native")){
        directory=FSDirectory.open(Path.of(p.getProperty("graph")),NoLockFactory.INSTANCE);
        reader=DirectoryReader.open(directory);need(reader.leaves().size()==1&&!reader.hasDeletions(),"single undeleted native index");
        leaf=NativeAlternatives.leaf(reader);
        var field=reader.leaves().get(0).reader().getFieldInfos().fieldInfo("vec");
        need(field.getVectorSimilarityFunction()==metric,"native metric metadata");
        need(field.getVectorDimension()==Integer.parseInt(p.getProperty("dimension")),"native dimension metadata");
      }else leaf=PaperSearch.arrayLeaf(Path.of(p.getProperty("graph")),Path.of(p.getProperty("base")));
      need(leaf.size==1000000,"fixed million-vector graph");
      for(int i=0;i<leaf.size;i++)need(leaf.toDoc(i)==i,"ordinal/document identity");
    }
    public void close()throws Exception{if(reader!=null)reader.close();if(directory!=null)directory.close();}
  }
  static List<Policy> policies(Path path)throws Exception{
    var result=new ArrayList<Policy>();var names=new HashSet<String>();var lines=Files.readAllLines(path);
    need(lines.get(0).equals("name,family,parameter,cap"),"policy schema");
    for(String line:lines.subList(1,lines.size())){
      var a=line.split(",");need(a.length==4&&names.add(a[0]),"unique policy names");
      var p=new Policy(a[0],a[1],Double.parseDouble(a[2]),Integer.parseInt(a[3]));
      need(Set.of("C","F","O","S","native","dabs","ungated").contains(p.family)&&p.cap==12800,"fixed arm/cap");result.add(p);
    }return result;
  }
  // Mirrors DepthAllocationExperiments.scout(..., rebuild=false), including partial state.
  static void scout(CrossMetricSearch.Workspace w,int[] table,int cap,int quota)throws Exception{
    int entry=-1;for(int n:table)if(!w.marked(n,CrossMetricSearch.ENTRY)&&!w.marked(n,CrossMetricSearch.BASE)){entry=n;break;}
    if(entry<0||w.dc>=cap||quota<=0)return;
    w.scoutCount++;int before=w.dc,stop=Math.min(cap,before+quota);
    w.local.reset();int ep=w.descend(entry,w.lf.nodeLevel[entry],stop);
    w.mark(entry,CrossMetricSearch.ENTRY);w.freshGate(10);
    if(w.dc<stop)w.seed(w.local,ep);w.advance(w.local,stop,0);
    for(int n:w.local.active.nodes())w.main.active.add(n,w.scores[n]);
    for(int n:w.local.deferred.nodes())w.main.deferred.add(n,w.scores[n]);
    if(w.local.current>=0)w.main.active.add(w.local.current,w.scores[w.local.current]);
    w.scoutDc+=w.dc-before;
  }
  static Result execute(CrossMetricSearch.Workspace w,float[] q,int[] table,Policy p,boolean diagnostic)throws Exception{
    w.setDiagnostic(diagnostic);
    if(Set.of("C","F","O","S").contains(p.family)){
      long begin=System.nanoTime();
      w.reset(AblationSearch.scorer(w.lf,q,w.similarity),10);w.initialize(6400);
      int stop=w.distanceAdvance(6400,.075),primary=w.dc;
      String chosen=p.family.equals("S")?(stop==0?"F":"C"):p.family;
      int routes=chosen.equals("F")?4:chosen.equals("O")?1:0,quota=chosen.equals("F")?400:1600;
      for(int i=0;i<routes;i++)scout(w,table,primary+6400,quota);
      w.distanceAdvance(primary+6400,Double.POSITIVE_INFINITY);
      var docs=w.output.topDocs().scoreDocs;int[] hits=new int[docs.length];for(int i=0;i<hits.length;i++)hits[i]=docs[i].doc;
      long nanos=System.nanoTime()-begin;
      need(w.dc==w.touchedCount&&w.dc<=primary+6400,"phase budget and cache accounting");
      return new Result(hits,w.dc,w.logical,w.edgeReads,w.expansions,primary,stop,chosen,w.scoutDc,w.scoutCount,w.sequence,nanos,
                        diagnostic?Arrays.copyOf(w.diagnosticTrace,w.traceSize):new int[0]);
    }
    long begin=System.nanoTime();var r=CrossMetricSearch.policy(w,q,new CrossMetricSearch.Policy(p.name,p.family,p.parameter,p.cap),table);
    long nanos=System.nanoTime()-begin;
    return new Result(r.hits(),r.dc(),r.logical(),r.edges(),r.expansions(),-1,-1,p.name,r.scoutDc(),r.scouts(),r.sequence(),nanos,r.trace());
  }
  static String stable(Result r){return Arrays.toString(r.hits)+":"+r.dc+":"+r.logical+":"+r.edges+":"+r.expansions+":"+
    r.primary+":"+r.stop+":"+r.selected+":"+r.scoutDc+":"+r.scouts+":"+r.sequence;}
  record Truth(int[][] ids,float[][] scores){}
  static Truth truth(Path path,int offset,int n)throws Exception{
    try(var in=new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))){
      int count=in.readInt(),k=in.readInt();need(k==50&&count>=offset+n&&Files.size(path)==8L+400L*count,"native top50 shape");
      in.skipNBytes(400L*offset);int[][] ids=new int[n][50];float[][] scores=new float[n][50];
      for(int i=0;i<n;i++)for(int j=0;j<50;j++){ids[i][j]=in.readInt();scores[i][j]=in.readFloat();if(j>0)need(scores[i][j-1]>=scores[i][j],"truth descending score");}
      return new Truth(ids,scores);
    }
  }
  static int hits(int[] results,int[] truth){int h=0;for(int r:results)for(int i=0;i<10;i++)if(r==truth[i]){h++;break;}return h;}
  static int[] parseInts(String s){return Arrays.stream(s.split(",")).mapToInt(Integer::parseInt).toArray();}
  static String ints(int[] x){return DecisionExperiments.ints(x);}
  static final String HEADER="graph,position,qid,policy,family,dc,logical,edges,expansions,primary_dc,primary_stop,selected,scout_dc,scouts,sequence,hits,hit_count,tie_hits,boundary_tie,score_bits,nanos,rep,arm_order";
  static void emit(PrintWriter out,String graph,int pos,int qid,Policy p,Result r,Truth gt,PhaseT2IEndToEnd.Leaf leaf,
                   float[] q,VectorSimilarityFunction metric,int rep,int order)throws Exception{
    int[] bits=new int[r.hits.length];int tie=0;
    var scorer=AblationSearch.scorer(leaf,q,metric);
    for(int i=0;i<bits.length;i++){float s=scorer.score(r.hits[i]);bits[i]=Float.floatToRawIntBits(s);if(s>=gt.scores[pos][9])tie++;}
    out.printf(Locale.ROOT,"%s,%d,%d,%s,%s,%d,%d,%s,%s,%s,%s,%s,%d,%d,%d,%s,%d,%d,%s,%s,%d,%d,%d%n",
      graph,pos,qid,p.name,p.family,r.dc,r.logical,r.edges<0?"":r.edges,r.expansions<0?"":r.expansions,
      r.primary<0?"":r.primary,r.stop<0?"":r.stop,r.selected,r.scoutDc,r.scouts,r.sequence,ints(r.hits),hits(r.hits,gt.ids[pos]),tie,
      gt.scores[pos][9]==gt.scores[pos][10],ints(bits),r.nanos,rep,order);
  }
  static void compareSnapshot(FastSearch.Workspace a,CrossMetricSearch.Workspace b){
    need(a.dc==b.dc&&a.logical==b.logical&&a.upperDc==b.upperDc&&a.edgeReads==b.edgeReads&&a.expansions==b.expansions&&
      a.scoutDc==b.scoutDc&&a.scoutCount==b.scoutCount&&a.sequence==b.sequence,"cosine counters equal");
    need(a.main.current==b.main.current&&a.local.current==b.local.current,"cosine partial frontier equal");
    need(Arrays.equals(a.main.active.nodes(),b.main.active.nodes())&&Arrays.equals(a.main.deferred.nodes(),b.main.deferred.nodes())&&
      Arrays.equals(a.local.active.nodes(),b.local.active.nodes())&&Arrays.equals(a.local.deferred.nodes(),b.local.deferred.nodes()),"cosine frontier heaps equal");
    need(a.touchedCount==b.touchedCount&&a.arenaSize==b.arenaSize,"cosine touched and arena sizes");
    for(int i=0;i<a.arenaSize;i++)need(a.arena[i]==b.arena[i],"cosine adjacency arena");
    for(int i=0;i<a.touchedCount;i++){
      int n=a.touched[i];need(n==b.touched[i]&&a.flags[n]==b.flags[n]&&Float.floatToRawIntBits(a.scores[n])==Float.floatToRawIntBits(b.scores[n])&&
        a.cursorPos[n]==b.cursorPos[n]&&a.cursorEnd[n]==b.cursorEnd[n],"cosine per-node owned state");
    }
  }
  static void cosine(Data data,float[][] qs,int[] table,List<Policy> ps,Path out)throws Exception{
    var old=new FastSearch.Workspace(data.leaf);var run=new OnlineSelection.Run(old);var w=new CrossMetricSearch.Workspace(data.leaf,data.metric);
    int rows=0;for(int pos:new int[]{0,137,3999,7998})for(var p:ps){
      var result=execute(w,qs[pos],table,p,true);old.setDiagnostic(true);int[] ids,trace;
      if(Set.of("C","F","O","S").contains(p.family)){
        run.execute(qs[pos],table,OnlineSelection.Arm.valueOf(p.family));ids=run.s.hits;trace=Arrays.copyOf(old.diagnosticTrace,old.traceSize);
        need(result.primary==run.s.primaryDc&&result.stop==run.s.primaryStop&&result.selected.equals(OnlineSelection.actionName(run.selected)),"cosine primary boundary and action equal");
      }else{
        var r=FastSearch.policy(old,qs[pos],new FastSearch.Policy(p.name,p.family,p.parameter,p.cap),table);ids=r.hits();trace=r.trace();
        need(result.dc==r.dc()&&result.logical==r.logical()&&result.edges==r.edges()&&result.expansions==r.expansions(),"cosine baseline counters equal");
      }
      need(Arrays.equals(ids,result.hits)&&Arrays.equals(trace,result.trace),"cosine final IDs and full score sequence equal");
      if(!p.family.equals("native"))compareSnapshot(old,w);rows++;
    }
    OnlineSelection.write(out.resolve("COMPLETE.json"),OnlineSelection.map("status","passed","rows",rows,"checks",checks,"metric","COSINE","full_trace_and_owned_state_parity",true));
  }
  static float[] sourceVector(Path path,int row,int d)throws Exception{
    try(var f=new RandomAccessFile(path.toFile(),"r")){
      f.seek((long)row*(d+1)*4);byte[] bytes=new byte[(d+1)*4];f.readFully(bytes);var b=ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
      need(b.getInt()==d,"source dimension");float[] v=new float[d];for(int i=0;i<d;i++)v[i]=b.getFloat();return v;
    }
  }
  static void audit(Data data,Properties p,float[][] qs,Truth gt,int[] table,List<Policy> ps,Path out)throws Exception{
    var leaf=data.leaf;int[] positions=parseInts(p.getProperty("audit"));int dimension=Integer.parseInt(p.getProperty("dimension"));
    // Native vector identity and metric conversion are checked independently of search.
    for(int ord:new int[]{0,137,499999,999999}){
      int row=ord;if(data.reader!=null){var doc=data.reader.storedFields().document(leaf.toDoc(ord));var field=doc.getField("row");need(field!=null,"native source-row metadata");row=field.numericValue().intValue();}
      var expected=sourceVector(Path.of(p.getProperty("base")),row,dimension);var actual=((RandomAccessVectorValues.Floats)leaf.rav).vectorValue(ord);
      need(Arrays.equals(expected,actual),"source/native vector bit identity");
      for(int pos:positions){double d2=0;for(int j=0;j<dimension;j++){double v=(double)expected[j]-qs[pos][j];d2+=v*v;}
        float score=AblationSearch.scorer(leaf,qs[pos],data.metric).score(ord);double restored=1.0/score-1.0;
        need(Math.abs(restored-d2)<=Math.max(1e-8,Math.abs(d2)*3e-6),"EUCLIDEAN score inverse agrees with direct coordinates");
      }
    }
    var w=new CrossMetricSearch.Workspace(leaf,data.metric);var fresh=new CrossMetricSearch.Workspace(leaf,data.metric);
    var records=new ArrayList<Map<String,Object>>();
    for(int pos:positions){
      var scorer=AblationSearch.scorer(leaf,qs[pos],data.metric);var exact=new TopKnnCollector(50,Integer.MAX_VALUE);
      for(int ord=0;ord<leaf.size;ord++)exact.collect(leaf.toDoc(ord),scorer.score(ord));
      var docs=exact.topDocs().scoreDocs;
      for(int j=0;j<50;j++)need(docs[j].doc==gt.ids[pos][j]&&Float.floatToRawIntBits(docs[j].score)==Float.floatToRawIntBits(gt.scores[pos][j]),"reused exact native IDs and score bits");
      for(var policy:ps){
        var a=execute(w,qs[pos],table,policy,false);var b=execute(w,qs[pos],table,policy,true);var c=execute(fresh,qs[pos],table,policy,true);
        need(stable(a).equals(stable(b))&&stable(b).equals(stable(c))&&Arrays.equals(b.trace,c.trace),"ordinary/instrumented and independent workspace parity");
        need(b.trace.length==b.dc,"complete physical trace accounting");
        if(policy.family.equals("C")){
          var u=CrossMetricSearch.policy(fresh,qs[pos],new CrossMetricSearch.Policy("matched-U","ungated",0,b.primary+6400),table);
          need(Arrays.equals(b.hits,u.hits())&&Arrays.equals(b.trace,u.trace())&&b.edges==u.edges()&&b.expansions==u.expansions(),"pause/resume C equals uninterrupted ungated at same actual cap");
        }
        records.add(OnlineSelection.map("position",pos,"policy",policy.name,"status","passed","dc",b.dc,"primary",b.primary,"stop",b.stop));
      }
      var old=PaperSearch.run(leaf,scorer,new PaperSearch.Config("single-w0","single",0,0,0,0,0,0,"upper","fixed",false,true),6400,table);
      var modern=CrossMetricSearch.policy(fresh,qs[pos],new CrossMetricSearch.Policy("old-cap-U","ungated",0,6400),table);
      need(Arrays.equals(Arrays.copyOf(old.hits(),10),modern.hits())&&old.state().dc==modern.dc()&&old.state().sequence==modern.sequence(),"historical 6400 ungated bridge");
      System.out.println("AUDIT query "+pos+" passed");
    }
    OnlineSelection.write(out.resolve("COMPLETE.json"),OnlineSelection.map("status","passed","checks",checks,"records",records,"native_exact_queries",positions.length,
      "metric","EUCLIDEAN","vector_identity",true,"old_ungated_bridge",true,"continuation_uninterrupted_parity",true));
  }
  static void study(String mode,Data data,Properties p,float[][] qs,Truth gt,int[] table,List<Policy> ps,Path out)throws Exception{
    var w=new CrossMetricSearch.Workspace(data.leaf,data.metric);int rows=0,offset=Integer.parseInt(p.getProperty("offset"));
    String graph=p.getProperty("name");
    try(var csv=new PrintWriter(Files.newBufferedWriter(out.resolve(mode+".csv"),StandardCharsets.UTF_8,StandardOpenOption.CREATE_NEW))){
      csv.println(HEADER);
      if(mode.equals("effect")){
        for(int pos=0;pos<qs.length;pos++){
          var results=new HashMap<String,Result>();
          for(int order=0;order<ps.size();order++){
            var policy=ps.get(order);var r=execute(w,qs[pos],table,policy,false);results.put(policy.name,r);
            need(r.hits.length==10&&Arrays.stream(r.hits).distinct().count()==10&&r.dc<=12800,"final Top10 and physical cap");
            emit(csv,graph,pos,pos+offset,policy,r,gt,data.leaf,qs[pos],data.metric,-1,order);rows++;
          }
          if(results.containsKey("C")&&results.containsKey("S")){
            var s=results.get("S");var chosen=results.get(s.selected);need(stable(s).equals(stable(chosen)),"S exactly reproduces selected C/F per execution");
          }
          if((pos+1)%100==0){csv.flush();need(!csv.checkError(),"effect output");System.out.println("EFFECT "+graph+" queries="+(pos+1)+"/"+qs.length+" rows="+rows);}
        }
      }else{
        var byName=new HashMap<String,Policy>();for(var policy:ps)byName.put(policy.name,policy);
        var reference=new HashMap<String,String>();var lines=Files.readAllLines(Path.of(p.getProperty("schedule")));int lastRep=-99;
        for(String line:lines.subList(1,lines.size())){
          var a=line.split(",");int rep=Integer.parseInt(a[0]),pos=Integer.parseInt(a[1]),qid=Integer.parseInt(a[2]),order=Integer.parseInt(a[3]);
          need(qid==pos+offset,"schedule source identity");var policy=byName.get(a[4]);need(policy!=null,"scheduled policy exists");
          var r=execute(w,qs[pos],table,policy,false);String key=pos+":"+policy.name,old=reference.putIfAbsent(key,stable(r));
          need(old==null||old.equals(stable(r)),"quality and work stable across repetitions");
          emit(csv,graph,pos,qid,policy,r,gt,data.leaf,qs[pos],data.metric,rep,order);rows++;
          if(rep!=lastRep){csv.flush();System.out.println("TIMING "+graph+" rep="+rep+" rows="+rows);lastRep=rep;}
        }
      }
      need(!csv.checkError(),"completed CSV write");
    }
    OnlineSelection.write(out.resolve("COMPLETE.json"),OnlineSelection.map("status","passed","mode",mode,"graph",graph,"rows",rows,"queries",qs.length,
       "checks",checks,"metric",data.metric.name(),"workspace_fixed_array_bytes",w.fixedArrayBytes()));
  }
  public static void main(String[] args)throws Exception{
    need(args.length==3,"MODE PROPERTIES OUTDIR");String mode=args[0];Properties p=new Properties();
    try(var r=Files.newBufferedReader(Path.of(args[1]),StandardCharsets.UTF_8)){p.load(r);}
    Path out=Path.of(args[2]);Files.createDirectory(out);
    try(var data=new Data(p)){
      int offset=Integer.parseInt(p.getProperty("offset")),n=Integer.parseInt(p.getProperty("n"));
      var qs=AblationSearch.queries(Path.of(p.getProperty("query")),offset,n);var table=PersistentSearch.entryTable(data.leaf);need(table.length==64,"64 unchanged entries");
      if(mode.equals("exact")){
        AblationSearch.exact(data.leaf,qs,data.metric,Path.of(p.getProperty("gt")));
        OnlineSelection.write(out.resolve("COMPLETE.json"),OnlineSelection.map("status","passed","queries",n,"native_scores",(long)n*data.leaf.size));return;
      }
      var ps=policies(Path.of(p.getProperty("configs")));
      if(mode.equals("cosine")){cosine(data,qs,table,ps,out);return;}
      var gt=truth(Path.of(p.getProperty("gt")),Integer.parseInt(p.getProperty("gt_offset","0")),n);
      if(mode.equals("audit"))audit(data,p,qs,gt,table,ps,out);
      else {need(mode.equals("effect")||mode.equals("timing"),"known mode");study(mode,data,p,qs,gt,table,ps,out);}
    }catch(Throwable t){try(var w=new PrintWriter(Files.newBufferedWriter(out.resolve("FAILURE.txt"),StandardOpenOption.CREATE_NEW))){t.printStackTrace(w);}throw t;}
  }
}
