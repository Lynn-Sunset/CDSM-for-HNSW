"""Bounded Linux run: parallel quality, isolated serial timing, load backoff."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes, datetime, fcntl, hashlib, json, os, random, resource
import signal, subprocess, sys, threading, time, traceback

R=Path(__file__).resolve().parent
ROOT=R.parent.parent
CPUS=list(range(16,24))
TIMING_CPU=22
LOCK=threading.Lock()
ACTIVE={}
QUALITY_DONE=0
TIMING_DONE=0
HZ=os.sysconf('SC_CLK_TCK')
ENV=os.environ.copy()
ENV.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
           PARLAY_NUM_THREADS='1',PARLAY_ELASTIC_PARALLELISM='0')
ENV['LD_LIBRARY_PATH']='/opt/rh/gcc-toolset-13/root/usr/lib64:'+ENV.get('LD_LIBRARY_PATH','')

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def atomic(path,data):
    tmp=Path(str(path)+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));tmp.replace(path)
def state(phase,**extra):
    with LOCK:
        atomic(R/'STATE.json',dict(utc=now(),pid=os.getpid(),phase=phase,
               active=dict(ACTIVE),quality_completed=QUALITY_DONE,quality_total=66,
               timing_completed=TIMING_DONE,timing_total=330,**extra))
def memory_available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
def cpu_snapshot():
    values={}
    for line in Path('/proc/stat').read_text().splitlines():
        fields=line.split()
        if fields[0].startswith('cpu') and fields[0][3:].isdigit():
            nums=list(map(int,fields[1:9]));values[int(fields[0][3:])]=sum(nums)-nums[3]-nums[4]
    return time.monotonic(),values
def cpu_ticks(pid):
    try:
        tail=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        ticks=sum(map(int,tail[11:15]))
        children=Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
        for child in children:
            value=cpu_ticks(int(child))
            if value is not None:ticks+=value
        return ticks
    except FileNotFoundError:return None
def usage(before,after,owned_cpus):
    dt=after[0]-before[0]
    ratios={c:max(0,after[1][c]-before[1].get(c,0))/HZ/dt for c in after[1]}
    return sum(v for c,v in ratios.items()if c not in owned_cpus),ratios
def wait_quiet(timing=False):
    quiet=0;before=cpu_snapshot()
    while quiet<2:
        time.sleep(.25 if timing else 2);after=cpu_snapshot();outside,ratios=usage(before,after,CPUS);before=after
        busy=outside>.75 or memory_available()<64*2**30
        busy=busy or any(ratios.get(c+28,0)>.08 for c in CPUS)
        if timing:busy=busy or sum(ratios[c]for c in [TIMING_CPU,TIMING_CPU+28])>.15
        quiet=0 if busy else quiet+1
        if busy:
            state('waiting_for_idle_resources',external_cpu_equivalents=round(outside,2))
            time.sleep(2)

def verify_inputs():
    records=json.loads((R/'TRANSFER.json').read_text())
    for item in records:
        p=ROOT/item['path']
        assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],item['path']
    return len(records)
def freeze_sources():
    paths=[]
    for base in [R,R.parent/'native-faiss-mechanism-20260913/upstream/faiss-1.15.0',
                 R.parent/'fannbench-expanded-baselines-20260919/vendor/RangeFilteredANN/ParlayANN',
                 R.parent/'fannbench-t2i-20260919']:
        for p in base.rglob('*'):
            if p.is_file() and p.suffix in ['.py','.cpp','.h','.hpp','.cmake','.sh'] and not any(x in p.parts for x in ['build','__pycache__']):paths.append(p)
    for name in ['PLAN.json','EXECUTION-PLAN.json','TRANSFER.json','PORTABILITY.json']:
        paths.append(R/name)
    return {p.relative_to(ROOT).as_posix():sha(p)for p in paths}

def profile_command(plan,p,group,out,rep=0):
    data=plan['data'];cmd=['runtime/vamana_bench.exe'if p.get('vamana')else'runtime/ann_bench.exe']
    if not p.get('vamana'):cmd+=['--mode','query','--name',p['name'],'--policy',p.get('policy','NATIVE')]
    cmd+=['--index',p['index'],'--queries',data['queries'],'--gt',data['gt'],
          '--ids-file',f'queries/{group}.txt','--params',','.join(map(str,p['params'])),
          '--out',out,'--reps','1','--rep-offset',str(rep)]
    for key in ['map','table']:
        if key in p:cmd+=['--'+key,p[key]]
    if p.get('vamana'):cmd+=['--base',data['base'],'--attrs','../fannbench-t2i-20260919/data/attrs.u32']
    if p.get('rerank'):cmd+=['--rerank',str(p['rerank']),'--base',data['base']]
    return cmd

def stage(name,command,cpu=20,timing=False,final=None,outputs=None):
    global QUALITY_DONE,TIMING_DONE
    done_path=R/'logs'/f'{name}.DONE.json'
    if done_path.exists():
        rec=json.loads(done_path.read_text());assert rec['exit_code']==0
        for path,digest in rec['outputs'].items():assert sha(R/path)==digest
        return rec
    for retry in range(20):
        if timing:wait_quiet(True)
        attempt=1+len(list((R/'logs').glob(name+'-a*.process.json')))
        stem=f'{name}-a{attempt:03d}'
        cmd=command.copy()
        attempt_out=None
        if final:
            attempt_out=f'attempts/{stem}.csv';cmd[cmd.index('--out')+1]=attempt_out
        child_command=['taskset','-c',str(cpu)]+cmd
        prefix=R/'logs'/stem
        rec=dict(started_utc=now(),command=child_command,timing=timing,clock_gaps=[],
                 resource_events=[],cpu=cpu,nice=os.getpriority(os.PRIO_PROCESS,0))
        before=cpu_snapshot();before_ticks=None;paused=False;bad=0;invalid=False
        continuity=time.clock_gettime(time.CLOCK_BOOTTIME)-time.monotonic()
        with Path(str(prefix)+'.stdout.log').open('xb')as out,Path(str(prefix)+'.stderr.log').open('xb')as err:
            proc=subprocess.Popen(child_command,cwd=R,env=ENV,stdout=out,stderr=err,start_new_session=True)
            rec['pid']=proc.pid;atomic(Path(str(prefix)+'.process.json'),rec)
            with LOCK:ACTIVE[name]={'pid':proc.pid,'cpu':cpu,'attempt':attempt}
            state('timing'if timing else'quality'if name.startswith('quality-')else'engineering')
            last=time.monotonic()
            while proc.poll()is None:
                try:proc.wait(timeout=2)
                except subprocess.TimeoutExpired:pass
                if proc.poll()is not None:break
                after=cpu_snapshot();outside,ratios=usage(before,after,[cpu]if timing else CPUS)
                ticks=cpu_ticks(proc.pid);competition=0
                if ticks is not None and before_ticks is not None:
                    own=(ticks-before_ticks)/HZ/(after[0]-before[0]);competition=max(0,ratios.get(cpu,0)-own)
                low_memory=memory_available()<64*2**30
                busy=outside>.75 or low_memory or ratios.get(cpu+28,0)>.08 or competition>.15
                if timing:busy=busy or ratios.get(cpu+28,0)>.05 or competition>.15
                before,before_ticks=after,ticks
                gap=(time.clock_gettime(time.CLOCK_BOOTTIME)-time.monotonic())-continuity
                if timing and abs(gap)>2:rec['clock_gaps'].append({'utc':now(),'seconds':gap});busy=True;bad=2
                bad=bad+1 if busy else 0
                if timing and bad>=2:
                    rec['resource_events'].append({'utc':now(),'event':'invalidate_and_retry','external_cpu_equivalents':outside,'core_competition':competition})
                    invalid=True;os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30);break
                if not timing:
                    if busy and not paused:
                        os.killpg(proc.pid,signal.SIGSTOP);paused=True
                        rec['resource_events'].append({'utc':now(),'event':'yield_to_other_work'})
                    elif not busy and paused:
                        os.killpg(proc.pid,signal.SIGCONT);paused=False
                        rec['resource_events'].append({'utc':now(),'event':'resume'})
                if time.monotonic()-last>20:
                    state('timing'if timing else'quality',resource_wait=paused);last=time.monotonic()
            rec.update(finished_utc=now(),exit_code=proc.returncode,accepted=not invalid)
        with LOCK:ACTIVE.pop(name,None)
        atomic(Path(str(prefix)+'.process.json'),rec)
        if invalid:continue
        if proc.returncode:raise RuntimeError(f'{name} failed: {proc.returncode}; inspect {prefix}')
        if final:
            for suffix in ['', '.json']:
                src=R/(attempt_out+suffix);dest=R/(final+suffix)
                assert not dest.exists();src.rename(dest)
            output_paths=[final,final+'.json']
        else:output_paths=outputs or []
        rec['outputs']={path:sha(R/path)for path in output_paths}
        atomic(done_path,rec)
        with LOCK:
            QUALITY_DONE+=name.startswith('quality-');TIMING_DONE+=name.startswith('timing-')
        return rec
    raise RuntimeError(f'{name}: repeated external load; inspect preserved attempts')

def main():
    global QUALITY_DONE,TIMING_DONE
    os.chdir(R);os.sched_setaffinity(0,CPUS)
    if os.getpriority(os.PRIO_PROCESS,0)<10:os.nice(10-os.getpriority(os.PRIO_PROCESS,0))
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    lock=(R/'SERVER.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for folder in ['logs','results','attempts','engineering']:(R/folder).mkdir(exist_ok=True)
    while not (R/'runtime/vamana_bench.exe').exists():
        state('waiting_for_build');time.sleep(10)
    assert 'LINUX_BUILD_COMPLETE' in (ROOT/'incoming/build.log').read_text()
    sources=freeze_sources();binary={p:sha(R/p)for p in ['runtime/ann_bench.exe','runtime/vamana_bench.exe']}
    frozen_path=R/'SERVER-FROZEN.json'
    frozen={'sources':sources,'binaries':binary}
    if frozen_path.exists():assert json.loads(frozen_path.read_text())==frozen
    else:atomic(frozen_path,frozen)
    atomic(R/'SERVER-ENVIRONMENT.json',dict(utc=now(),host=os.uname().nodename,
           cpu_info=Path('/proc/cpuinfo').read_text().split('\n\n')[0],
           quality_cpus=CPUS,timing_cpu=TIMING_CPU,nice=10,io_class='idle',
           address_space_limit_gib=8,minimum_available_memory_gib=64,
           external_cpu_backoff_threshold=1.5,
           build_costs='Source Windows builds; indexes reused byte-for-byte',
           python=sys.version,query_threads=1,quality_parallelism=4))
    wait_quiet()
    stage('engineering-linux',[sys.executable,'-B','engineering.py','linux'],outputs=['engineering/linux/PASS.json'])
    stage('analysis-check-linux',[sys.executable,'-B','check_analysis.py'],outputs=['engineering/analysis-check/PASS.json'])
    while not (ROOT/'INPUTS-READY.json').exists():
        state('waiting_for_transferred_inputs',synthetic_engineering_passed=True);time.sleep(10)
    state('verifying_transferred_inputs');n=verify_inputs()
    atomic(R/'SERVER-INPUTS-VERIFIED.json',dict(utc=now(),files=n,byte_identical=True))
    stage('full-smoke-linux',[sys.executable,'-B','full_smoke.py','linux'],outputs=['engineering/full-linux/PASS.json'])
    stage('vamana-smoke-linux',[sys.executable,'-B','check_vamana.py','linux'],outputs=['engineering/vamana-linux-PASS.json'])
    plan=json.loads((R/'PLAN.json').read_text());ex=json.loads((R/'EXECUTION-PLAN.json').read_text())
    QUALITY_DONE=len(list((R/'logs').glob('quality-*.DONE.json')))
    TIMING_DONE=len(list((R/'logs').glob('timing-*.DONE.json')))
    order=ex['profiles'].copy();random.Random(20260919).shuffle(order)
    def quality_lane(lane):
        for p in order[lane::len(CPUS)]:
            for group in ['development','evaluation']:
                name=f"quality-{group}-{p['name']}";out=f'results/{name}.csv'
                stage(name,profile_command(plan,p,group,out),cpu=CPUS[lane],final=out)
    with ThreadPoolExecutor(max_workers=len(CPUS))as pool:
        for future in as_completed([pool.submit(quality_lane,i)for i in range(len(CPUS))]):future.result()
    state('quality_complete_cooling_before_timing')
    time.sleep(30)
    for rep in range(plan['timing_repetitions']):
        order=ex['profiles'].copy();random.Random(20260919+rep).shuffle(order)
        for p in order:
            for group in ['timing-development','timing-evaluation']:
                name=f"{group}-r{rep}-{p['name']}";out=f'results/{name}.csv'
                stage(name,profile_command(plan,p,group,out,rep),cpu=TIMING_CPU,timing=True,final=out)
    state('final_input_and_source_verification')
    assert freeze_sources()==sources
    for p,h in binary.items():assert sha(R/p)==h
    verify_inputs()
    stage('analyze-linux',[sys.executable,'-B','analyze.py'],outputs=['results/SUMMARY.json','results/AUDIT.json','REPORT.zh-CN.md'])
    state('complete',report='REPORT.zh-CN.md',visual_inspection_pending=True)
    atomic(R/'COMPLETION.json',dict(utc=now(),passed=True,host='10.26.6.117',report='REPORT.zh-CN.md'))

if __name__=='__main__':
    try:main()
    except Exception as exc:
        state('failed',error=str(exc),traceback=traceback.format_exc())
        raise
