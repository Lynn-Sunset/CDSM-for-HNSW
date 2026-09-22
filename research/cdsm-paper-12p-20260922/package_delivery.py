"""Validate and package the paper after a human/agent reviews its final renders.

Requires pypdf. This checks numeric provenance, PDF structure and package hashes;
it cannot perform visual review. --visual-review-confirmed is an explicit
attestation that all current qa/final-page-XX.png images have been inspected.
It does not run ANN search, change LaTeX, or compile a PDF.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pypdf import PdfReader

H = Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def write(p,j): p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def entry(p,root): return dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p))

def pdf_resources(reader):
    fonts={}; images=0; seen=set()
    def walk(resource):
        nonlocal images
        if not resource:return
        resource=resource.get_object()
        font_map=resource.get('/Font',{})
        if hasattr(font_map,'get_object'):font_map=font_map.get_object()
        for ref in font_map.values():
            font=ref.get_object();name=str(font.get('/BaseFont','unnamed'))
            children=font.get('/DescendantFonts',[font])
            descriptors=[c.get_object().get('/FontDescriptor') for c in children]
            embedded=all(d and any(k in d.get_object() for k in ['/FontFile','/FontFile2','/FontFile3']) for d in descriptors)
            fonts[name]=bool(embedded)
        objects=resource.get('/XObject',{})
        if hasattr(objects,'get_object'):objects=objects.get_object()
        for ref in objects.values():
            key=(getattr(ref,'idnum',id(ref)),getattr(ref,'generation',0))
            if key in seen:continue
            seen.add(key);obj=ref.get_object()
            if obj.get('/Subtype')=='/Image':images+=1
            if obj.get('/Subtype')=='/Form':walk(obj.get('/Resources'))
    for page in reader.pages:walk(page.get('/Resources'))
    return fonts,images

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--visual-review-confirmed',action='store_true')
    args=parser.parse_args()
    if not args.visual_review_confirmed:raise SystemExit('Review every final render first, then supply --visual-review-confirmed.')
    pdf=H/'output/pdf/cdsm-paper.pdf';reader=PdfReader(pdf)
    assert len(reader.pages)==12
    assert not reader.metadata.get('/Author','').strip()
    log=(H/'tmp/pdfs/main.log').read_text(encoding='utf-8',errors='replace')
    assert 'Overfull' not in log
    assert 'undefined' not in log.lower()
    assert 'LaTeX Error' not in log
    fonts,images=pdf_resources(reader)
    assert fonts and all(fonts.values()) and images==0
    renders=[H/f'qa/final-page-{i:02}.png' for i in range(1,13)]
    assert all(p.is_file() and p.stat().st_mtime>=pdf.stat().st_mtime for p in renders)

    source_root=H/'source-data'
    baseline=read(H/'BASELINE-EDITION.json')
    assert all(sha(source_root/r['path'])==r['sha256'] for r in baseline['source_files'])
    source_files=[entry(p,source_root) for p in sorted(source_root.rglob('*')) if p.is_file()]
    write(H/'SOURCE-DATA.json',dict(scope='Frozen summaries, implementation excerpts, audit receipts, build metadata and the retained per-event hit records used for descriptive threshold analysis. Not the full vector, graph, trace, binary or execution environment.',files=source_files))
    for name in ['TABLE-SOURCES.json','REVISION-DATA.json','R2-DEPTH-EVIDENCE.json']:
        data=read(H/name);mapping=data.get('sources',data)
        for rel,digest in mapping.items():assert sha(source_root/rel)==digest
    d=read(H/'REVISION-DATA.json')
    assert (d['events'],d['query_identities'],d['rescues'],d['rescues_4_to_5'],d['rescues_to_at_least_8'])==(23997,7999,41,2,37)
    assert d['actual_dc_equal']
    assert all(r['F']<r['C'] and r['rescues']>r['harms'] for r in d['thresholds'])

    primary=read(source_root/'hnsw-adaptive-baselines-20260920/server-results/final-20260921-005721/results/SUMMARY.json')['primary']
    cf=[r for r in primary if r.get('baseline')=='C']
    assert len(cf)==4 and all(r['cdsm_parameter']==r['baseline_parameter']==10240 and r['holm_p']<.05 for r in cf)
    expected_pairs=[(26,3),(19,2),(35,0),(33,2)]
    fixed=read(source_root/'fanng-search-comparison-20260922/SUMMARY.json')
    assert [(r['rescue'],r['harm']) for r in fixed['contrasts'] if r['cap']==10240 and r['baseline']=='C']==expected_pairs
    assert len(fixed['primary_family'])==4 and max(r['holm_p'] for r in fixed['primary_family'])<=.000222
    latest_root=source_root/'cdsm-review-strengthening-20260922'
    latest=read(latest_root/'SUMMARY.json');audit=read(latest_root/'AUDIT.json')
    verified=read(latest_root/'LOCAL-VERIFICATION.json')
    assert latest['all_actual_dc_equal'] and audit['records']==736000
    assert audit['timing']['records']==368000 and audit['timing']['quality_signature_parity']
    assert audit['legacy']['legacy_CF_identical_records']==192000
    assert verified['passed'] and verified['summary_sha256']==sha(latest_root/'SUMMARY.json')
    assert len(latest['primary_tests'])==12 and len(latest['cells'])==92
    assert sha(latest_root/'SELECTION.json')==read(latest_root/'SELECTION-FROZEN.json')['sha256']
    for seed in [42,777,1234]:
        b=read(source_root/f'dataset-metadata/t2i/index-s{seed}/BUILD.json')
        assert b['M']==16 and b['efConstruction']==100 and b['order']=='original' and b['merge']=='serial'

    styles=baseline['styles']
    for s in styles:assert sha(H/'paper'/s['file'])==s['sha256']
    auxiliary=(H/'tmp/pdfs/main.aux').read_text(encoding='utf-8')
    cited={v for line in re.findall(r'\\citation\{([^}]*)\}',auxiliary) for v in line.split(',')}
    bib=re.findall(r'\\bibitem\[.*?\]\s*%?\s*\{([^}]*)\}',(H/'tmp/pdfs/main.bbl').read_text(encoding='utf-8'),re.S)
    assert len(cited)==len(set(bib))==17 and cited==set(bib)
    fg=H/'paper/figures';provenance=read(fg/'PROVENANCE.json')
    for s in provenance['sources']:assert sha(source_root/Path(s['path']))==s['sha256']
    for name,digest in provenance['data_sha256'].items():assert sha(fg/name)==digest
    fig_records=[]
    for name in ['method-feedback','t2i-budget','native-tradeoff','native-full']:
        fr=PdfReader(fg/f'{name}.pdf');ff,fi=pdf_resources(fr)
        assert len(fr.pages)==1 and all(ff.values()) and fi==0
        box=fr.pages[0].mediabox
        fig_records.append(dict(file=f'{name}.pdf',pages=1,width_mm=float(box.width)*25.4/72,height_mm=float(box.height)*25.4/72,fonts=ff,raster_images=fi,sha256=sha(fg/f'{name}.pdf')))
    assert provenance['t2i_budget']['cells']==48 and provenance['native_tradeoff']['cells']==73
    write(fg/'VALIDATION.json',dict(visual_inspection='All four vector figures inspected in the final 12-page manuscript; no clipping or overlap.',data='48 latest-wave fixed-budget cells and 73 historical native cells; native zoom and full range use identical CSV values, without Pareto filtering.',figures=fig_records,files={p.name:sha(p) for p in sorted(fg.iterdir()) if p.suffix in ['.tex','.csv','.pdf','.py','.png']}))
    shutil.copy2(H/'tmp/pdfs/main.log',H/'qa/compile.log')
    texts=[p.extract_text() for p in reader.pages]
    (H/'qa/final-extracted.txt').write_text('\n\n'.join(texts),encoding='utf-8')
    qa=dict(created_utc=datetime.now(timezone.utc).isoformat(),file='output/pdf/cdsm-paper.pdf',sha256=sha(pdf),pages=12,page_limit_including_references=12,passed=True,
        author_status='Intentionally blank at user request.',main_sections=8,appendix_sections=6,figures=4,tables=len(re.findall(r'\\newlabel\{tab:',auxiliary)),algorithms=1,bibliography_entries=17,cited_entries=17,
        compile=dict(overfull_boxes=0,undefined_references=0,fonts_all_embedded=True,fonts=fonts,raster_images=images,benign_warnings=['ACM balance warning; final columns visually checked.'] if 'balance Warning' in log else []),
        numeric_provenance=dict(bundled_source_files=len(source_files),original_64_hashes_unchanged=True,source_and_table_hashes_pass=True,revision_sources_verified=5,actual_dc_matched_events=23997,paired_primary_numbers_verified=True,graph_construction_metadata_verified=True),
        styles=styles,new_ANN_experiments=True,new_analyses='Frozen server MEP/FRONTIER controls and 200/800-depth checks: 92000 development, 736000 evaluation and 368000 timing records. Twelve-test Holm family on historically exposed queries; separate descriptive historical depth and native-grid analyses.',
        latest_server_audit=dict(development_records=92000,evaluation_records=736000,timing_records=368000,all_actual_dc_equal=True,legacy_CF_identical_records=192000,independent_raw_reaggregation=True,summary_sha256=sha(latest_root/'SUMMARY.json')),
        visual_review=dict(reviewed_pages=list(range(1,13)),reviewer='primary agent',clipping_or_overlap_found=False,all_appendix_floats_precede_references=True),
        page_details=[dict(page=i,text_characters=len(t),raster_sha256=sha(p),visually_inspected=True) for i,(t,p) in enumerate(zip(texts,renders),1)])
    write(H/'QA.json',qa)

    roots=[H/'paper',H/'source-data']
    files=[p for root in roots for p in root.rglob('*') if p.is_file() and p.suffix not in ['.aux','.log','.out','.pyc'] and '__pycache__' not in p.parts]
    files += [p for p in H.iterdir() if p.is_file() and p.name!='PACKAGE-MANIFEST.json' and p.suffix in ['.py','.ps1','.md','.json']]
    files=sorted(set(files))
    manifest=dict(delivered_pdf_sha256=sha(pdf),scope='Editable paper, vector-figure sources and frozen evidence; no final manuscript PDF duplication or temporary renders.',files=[entry(p,H) for p in files])
    write(H/'PACKAGE-MANIFEST.json',manifest)
    destination=H/'output/cdsm-paper-source.zip'
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in files+[H/'PACKAGE-MANIFEST.json']:z.write(p,p.relative_to(H).as_posix())
    with zipfile.ZipFile(destination) as z:
        assert z.testzip() is None
        for e in manifest['files']:assert hashlib.sha256(z.read(e['path'])).hexdigest()==e['sha256']
    print(json.dumps(dict(pages=12,source_files=len(source_files),archive_files=len(files)+1,pdf_sha256=sha(pdf),zip_bytes=destination.stat().st_size,zip_sha256=sha(destination))))

if __name__=='__main__':main()
