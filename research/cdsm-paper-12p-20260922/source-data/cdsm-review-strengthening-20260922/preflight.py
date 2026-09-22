from pathlib import Path
import csv,json,sys
R=Path(__file__).resolve().parent;P=json.loads((R/'PLAN.json').read_text());OLD=R.parent/'fanng-search-comparison-20260922'
FIELDS=['hits','dc','primary','scout','upper_dc','logical','edges','expansions','stop','sequence','ids','distances']
def rows(path):
    with path.open()as f:return {(int(x['param']),int(x['qid'])):x for x in csv.DictReader(f)}
def main():
    phase=sys.argv[1];compared=0;engineering=0;counts=[]
    for g in P['graphs']:
        if phase=='engineering':
            allrows={}
            for policy in P['policies']+P['sensitivity']:
                path=R/'engineering'/f'{g["name"]}-{policy}.csv';r=rows(path)
                caps=P['sensitivity_caps']if policy in P['sensitivity']else P['caps']
                assert set(r)=={(b,q)for b in caps for q in map(int,(R/'queries/engineering.txt').read_text().split())}
                allrows[policy]=r;engineering+=len(r)
            for policy,r in allrows.items():
                for key,v in r.items():
                    base=allrows['C'][key]
                    assert all(v[k]==base[k]for k in ['primary','primary_sequence','primary_stop']),('common prefix',g['name'],policy,key)
                    assert 0<int(v['dc'])<=key[0]
            counts.extend(int(v['dc'])==key[0]for r in allrows.values()for key,v in r.items())
        for policy in ['C','F']:
            if phase=='engineering':current=R/'engineering'/f'{g["name"]}-{policy}.csv';reference=OLD/'engineering'/current.name
            else:current=R/'results'/f'quality-evaluation-{g["name"]}-{policy}.csv';reference=OLD/'results'/f'quality-{g["name"]}-{policy}.csv'
            a=rows(current);b=rows(reference);assert set(a)==set(b),(current,reference)
            for key,r in a.items():
                for field in FIELDS:assert r[field]==b[key][field],(g['name'],policy,key,field,r[field],b[key][field])
                compared+=1
    result={'passed':True,'phase':phase,'legacy_CF_identical_records':compared,'legacy_fields':FIELDS,
            'engineering_records':engineering,'engineering_all_actual_dc_equal_cap':all(counts)}
    path=R/'engineering'/('PASS.json'if phase=='engineering'else'LEGACY-PARITY.json')
    text=json.dumps(result,indent=2)+'\n'
    if path.exists():assert path.read_text()==text
    else:path.write_text(text)
    print(json.dumps(result))
if __name__=='__main__':main()
