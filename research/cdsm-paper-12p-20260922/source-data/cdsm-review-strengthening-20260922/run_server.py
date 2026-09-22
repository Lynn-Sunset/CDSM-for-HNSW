from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import fcntl,importlib.util,json,os,random,re,resource,subprocess,sys,time,traceback

R=Path(__file__).resolve().parent;ROOT=R.parent.parent
P=json.loads((R/'PLAN.json').read_text());ENGINE=R.parent/'hnsw-adaptive-baselines-20260920/resource_runner.py'
spec=importlib.util.spec_from_file_location('r2_resource_engine',ENGINE);engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
engine.R=R;engine.ROOT=ROOT;engine.CPUS=P['quality_cpus'];engine.TIMING_CPU=P['timing_cpu']
ANALYSIS=str(R.parent/'hnsw-adaptive-baselines-20260920/venv/bin/python')
def state(phase,**extra):
    with engine.LOCK:engine.atomic(R/'STATE.json',dict(utc=engine.now(),pid=os.getpid(),phase=phase,active=dict(engine.ACTIVE),
        quality_completed=engine.QUALITY_DONE,quality_total=P['quality_jobs'],timing_completed=engine.TIMING_DONE,timing_total=P['timing_jobs'],
        selection_frozen=(R/'SELECTION-FROZEN.json').exists(),**extra))
engine.state=state
def hashes():
    files=[R/x for x in ['bench.cpp','strengthen.h','check.cpp','compile.sh','run_server.py','preflight.py','analyze.py','PLAN.json','PROTOCOL.zh-CN.md','DEPENDENCIES.json','runtime/r2_bench','runtime/r2_check']]
    files+=list((R/'vendor').rglob('*'))+list((R/'queries').glob('*.txt'))
    files+=[ENGINE,R.parent/'ann-systematic-comparison-20260919/build/faiss/libfaiss_avx2.a']
    return {str(p.relative_to(ROOT)):engine.sha(p)for p in files if p.is_file()}
def verify():
    for rec in P['inputs']:
        p=ROOT/rec['path'];assert p.stat().st_size==rec['bytes']and engine.sha(p)==rec['sha256'],p
    for rec in json.loads((R/'DEPENDENCIES.json').read_text()):assert engine.sha(R/rec['path'])==rec['sha256'],rec
    dev=set((R/'queries/development.txt').read_text().split());ev=set((R/'queries/evaluation.txt').read_text().split())
    assert len(dev)==1000 and len(ev)==8000 and not dev&ev
    assert set((R/'queries/timing-evaluation.txt').read_text().split())<=ev
def command(g,policy,group,out,rep=0,check=False):
    caps=P['sensitivity_caps']if policy in P['sensitivity']else P['caps']
    args=[str(R/'runtime/r2_bench'),'--index',str(ROOT/g['index']),'--name',g['name']+'-'+policy,'--policy',policy,
        '--queries',str(ROOT/P['data']['queries']),'--gt',str(ROOT/P['data']['gt']),
        '--ids-file',str(R/'queries'/f'{group}.txt'),'--params',','.join(map(str,caps)),'--out',out,
        '--rep-offset',str(rep),'--stage','timing'if group=='timing-evaluation'else'quality','--check',str(int(check))]
    for key in ['map','table']:
        if key in g:args+=['--'+key,str(ROOT/g[key])]
    return args
def lane(g,i,phase):
    for policy in P['policies']+P['sensitivity']:
        name=g['name']+'-'+policy
        if phase=='engineering':
            out='engineering/'+name+'.csv'
            engine.stage('engineering-'+name,command(g,policy,'engineering',out,check=True),cpu=P['quality_cpus'][i],outputs=[out,out+'.json'])
        else:
            out=f'results/quality-{phase}-{name}.csv'
            engine.stage(f'quality-{phase}-{name}',command(g,policy,phase,out),cpu=P['quality_cpus'][i],final=out)
def parallel(phase):
    state(phase)
    with ThreadPoolExecutor(max_workers=4)as pool:
        futures=[pool.submit(lane,g,i,phase)for i,g in enumerate(P['graphs'])]
        for future in futures:future.result()
def take_locks():
    held=[]
    names=[R/'SERVER.lock',ROOT/'research/CDSM-SERIAL-EXPERIMENT.lock']
    for study in ['fanng-search-comparison-20260922','cdsm-family-comparison-20260922','cdsm-crossmodal-supplement-20260922','hnsw-adaptive-baselines-20260920']:
        p=R.parent/study/'SERVER.lock'
        if p.exists():names.append(p)
    for p in names:
        f=p.open('r'if p.exists()else'a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);held.append(f)
    return held
def reject_other_runs():
    for p in Path('/proc').iterdir():
        if not p.name.isdigit()or int(p.name)==os.getpid():continue
        try:
            cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace');cwd=(p/'cwd').resolve()
        except OSError:continue
        if ROOT not in cwd.parents and cwd!=ROOT and str(ROOT)not in cmd:continue
        if re.search(r'(?:run_server|resource_runner)\.py|(?:r2|family|fanng|ann)_bench(?:\.exe)?(?:\s|$)',cmd):
            raise RuntimeError('Concurrent project experiment PID '+p.name)
def main():
    os.chdir(R)
    for name in ['logs','results','engineering','attempts']:(R/name).mkdir(exist_ok=True)
    held=take_locks();reject_other_runs()
    os.sched_setaffinity(0,set(P['quality_cpus']+[P['timing_cpu']]))
    if os.getpriority(os.PRIO_PROCESS,0)<19:os.nice(19-os.getpriority(os.PRIO_PROCESS,0))
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    engine.QUALITY_DONE=len(list((R/'logs').glob('quality-*.DONE.json')));engine.TIMING_DONE=len(list((R/'logs').glob('timing-*.DONE.json')))
    state('verifying_inputs');verify();engine.atomic(R/'INPUTS-VERIFIED.json',{'passed':True,'utc':engine.now()})
    engine.stage('synthetic-current-source',[str(R/'runtime/r2_check')],cpu=16)
    engine.stage('analysis-selftest',[ANALYSIS,'-B','analyze.py','selftest'],cpu=22)
    parallel('engineering')
    engine.stage('preflight',[sys.executable,'-B','preflight.py','engineering'],cpu=22,outputs=['engineering/PASS.json'])
    frozen={'sources':hashes()}
    if(R/'FROZEN.json').exists():assert json.loads((R/'FROZEN.json').read_text())==frozen
    else:engine.atomic(R/'FROZEN.json',frozen)
    engine.atomic(R/'ENVIRONMENT.json',{'utc':engine.now(),'cpu':Path('/proc/cpuinfo').read_text().split('\n\n')[0],
        'compiler':subprocess.check_output(['/opt/rh/gcc-toolset-13/root/usr/bin/g++','--version'],text=True),
        'threads_per_query':1,'quality_cpus':P['quality_cpus'],'timing_cpu':22,'nice':19,'timing_rounds':5,
        'historical_query_exposure':True,'memory_limit_gib_per_process':8,'global_experiment_lock':True,
        'QPS_definition':'inverse mean query median ns over five repeats','p99_definition':'query median latency percentile; not raw service latency'})
    parallel('development')
    engine.stage('development-selection',[ANALYSIS,'-B','analyze.py','select'],cpu=22,outputs=['SELECTION.json'])
    selected={'sha256':engine.sha(R/'SELECTION.json'),'analysis_sha256':engine.sha(R/'analyze.py')}
    if(R/'SELECTION-FROZEN.json').exists():assert json.loads((R/'SELECTION-FROZEN.json').read_text())==selected
    else:engine.atomic(R/'SELECTION-FROZEN.json',selected)
    parallel('evaluation')
    engine.stage('legacy-parity',[sys.executable,'-B','preflight.py','evaluation'],cpu=22,outputs=['engineering/LEGACY-PARITY.json'])
    state('cooling_before_serial_timing');time.sleep(5)
    profiles=[(g,p)for g in P['graphs']for p in P['policies']+P['sensitivity']]
    for rep in range(P['timing_rounds']):
        order=profiles.copy();random.Random(20260922+rep).shuffle(order)
        for g,p in order:
            name=g['name']+'-'+p;out=f'results/timing-{name}-r{rep}.csv'
            engine.stage(f'timing-{name}-r{rep}',command(g,p,'timing-evaluation',out,rep),cpu=22,timing=True,final=out)
    state('final_verification');assert hashes()==frozen['sources'];verify()
    assert engine.sha(R/'SELECTION.json')==selected['sha256']
    engine.stage('final-analysis',[ANALYSIS,'-B','analyze.py','final'],cpu=22,outputs=['SUMMARY.json','AUDIT.json','REPORT.zh-CN.md'])
    engine.atomic(R/'COMPLETION.json',{'passed':True,'utc':engine.now(),'development_records':92000,'evaluation_records':736000,'timing_records':368000})
    state('complete');print('R2_EXPERIMENT_COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:state('failed',error=str(e),traceback=traceback.format_exc());raise
