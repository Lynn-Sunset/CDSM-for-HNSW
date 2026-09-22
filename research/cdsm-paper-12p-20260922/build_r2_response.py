"""Readable revision record from verified server results; never runs ANN."""
from pathlib import Path
import hashlib,json

R=Path(__file__).resolve().parent;D=R/'source-data'
E=D/'cdsm-review-strengthening-20260922'
S=json.loads((E/'SUMMARY.json').read_text(encoding='utf-8'))
V=json.loads((E/'LOCAL-VERIFICATION.json').read_text(encoding='utf-8'))
assert V['passed'] and S['all_actual_dc_equal']
G=['Faiss-M32','Imported-s42','Imported-s777','Imported-s1234']
M={(x['graph'],x['policy'],x['cap']):x for x in S['cells']}
names={'Faiss-M32':'Faiss M32','Imported-s42':'Lucene seed 42','Imported-s777':'Lucene seed 777','Imported-s1234':'Lucene seed 1234'}
def delta(g,a,b):return 100*(M[g,a,10240]['qps']/M[g,b,10240]['qps']-1)
def prange(values,precision=2):return f'{min(values):+.{precision}f}% 至 {max(values):+.{precision}f}%'
lines=['# 第二轮审稿意见补强：结果与论文修改','',
    '2026-09-23。ANN 搜索、工程检查和正式计时全部在服务器完成。原始 CDSM-F 未调参，作者留空，论文继续按搜索方法组织；正文、附录与参考文献共 12 页。',
    '', '本轮把“直接接入更多入口”和“把原前沿分成局部路线”提升为正文主要对照。CDSM 的新增价值体现在如何把计算分给局部探索并恢复后续搜索，而不是入口数量本身。',
    '', '## 1. 同样计算量下，与合理多入口的比较','',
    '**C** 是原路直接续搜；**F** 是原 CDSM，每次最多四个 scout、每路最多 400 次新距离。**MEP** 从备用入口下降后直接加入共同队列，不先单独探索。MEP-DEV 是开发集选好的入口数量，不是根据评价查询选出的最佳答案。',
    '', '以下每张图使用同一组 8,000 条评价查询，每条查询都实际花费 10,240 次距离计算。严重失败是只找到真实前十近邻中的 0 至 4 个。',
    '', '|图|开发选择的入口数|MEP-DEV → F 平均召回 %|MEP-DEV → F 严重失败|F 相对 MEP-DEV 的 QPS|',
    '|---|---:|---:|---:|---:|']
for g in G:
    p=S['selection'][g+'-10240'];a=M[g,p,10240];f=M[g,'F',10240]
    lines.append(f"|{names[g]}|{p[3:]}|{a['recall']*100:.5f} → {f['recall']*100:.5f}|{a['severe']} → {f['severe']}|{delta(g,'F',p):+.2f}%|")
lines += ['',
    '在主预算上，F 的平均召回高于四张图上全部 1/4/8/16 入口 MEP。对开发选择的 MEP，平均召回和严重失败点估计都改善；严重失败经过十二项 Holm 校正后，只在 seed 777 上显著。对固定 MEP-4，seed 777 与 1234 显著。没有把重复的 MEP-4/MEP-DEV 比较算成独立证据。',
    '',
    '**必须保留的取舍：** MEP-16 在 Faiss M32 和 seed 42 的严重失败为 6、18，少于 F 的 9、22，但平均召回也更低。8,192 预算下，开发选择的 MEP 在 seed 42/1234 上平均召回更高；12,800 下，它在 M32 上的严重失败为 4，F 为 5。因此，本文支持具体区间内的改进，不宣称所有多入口配置和指标均被 F 超过。',
    '', '## 2. 从已有前沿启动 scout 不能重现 F 的质量收益','',
    'FRONTIER 从主搜索已经发现、尚未展开完的候选中选最多四个根，使用 F 的局部搜索、共享进度和回流规则。它不再去备用上层入口开辟路线；这是明确实现的自建对照，不是原生 iQAN。',
    '',
    '在 10,240 预算下，四图严重失败从 29/35/46/45 降到 F 的 9/22/14/20；四项比较均通过 Holm 校正，平均召回增加 0.195–0.391 个百分点。F 相对 FRONTIER 的 QPS 差为 '+prange([delta(g,'F','FRONTIER')for g in G])+'。计时包括 FRONTIER 的候选集合构建。实际局部投入与后续访问也会变化，所以这是整体搜索策略的比较，不能单独认定“只换上层入口”就是全部原因。',
    '', '## 3. 相邻深度及原生工作点','',
    '仅改变每路 scout 的额度，其他规则和总预算不变。下表是 10,240 预算下的描述性检查；原 F 继续固定为 400，并未按本轮排名更换。',
    '', '|每路上限|相对 C 的平均召回增量（百分点，四图范围）|严重失败（M32/42/777/1234）|相对 C 的 QPS 范围|',
    '|---:|---:|---|---:|']
for policy,quota in [('F200',200),('F',400),('F800',800)]:
    changes=[100*(M[g,policy,10240]['recall']-M[g,'C',10240]['recall'])for g in G]
    severe='/'.join(str(M[g,policy,10240]['severe'])for g in G)
    lines.append(f'|{quota}|{min(changes):+.5f} 至 {max(changes):+.5f}|{severe}|{prange([delta(g,policy,"C")for g in G])}|')
joint=all(M[g,p,10240]['recall']>M[g,'C',10240]['recall'] and M[g,p,10240]['severe']<M[g,'C',10240]['severe'] for g in G for p in ['F200','F','F800'])
lines += ['',
    ('三个深度在四图上都保持了相对 C 的平均召回和严重失败改善，说明收益并非只出现在 400 这一单点。'if joint else'邻近深度的完整取舍见表，不以某个指标排名替换原 F。')+
    '这只支持当前 T2I、图和主预算附近的稳定性。旧 Java 开发记录另列附录：四路每路 100 时约 9% 路线在底层播种前耗尽额度；400 增至 1,600 只多 0.01 个百分点最终召回。旧实验的 COSINE 与 p+6,400 合同没有混入当前 C++ 表。',
    '',
    '原生对照采用完整双向工作点匹配：每个参照点寻找平均召回不低、严重失败不多的最快对方点，保留所有未匹配情形。在既有 seed 777 网格中，原生 ef=1,280 与 F@11,264 的平均召回为 98.97375% → 99.00500%，严重失败为 31 → 14，QPS 为 217.18 → 231.04（+6.38%），p99 查询中位延迟为 6.193 → 5.411 ms。这支持一个局部联合优势；旧的开发时间目标所选工作点上，F 相对原生约 10% 的吞吐代价也继续保留。这里不使用等计算量或全网格占优的说法。',
    '', '## 4. 时间、统计与复现口径','',
    '- '+f"本轮 F 相对 C 的 QPS 差为 {prange([delta(g,'F','C')for g in G])}。所有最新对照共同重测，不拼接旧波次 QPS；1.5% 是实用容差，不是统计等效检验。",
    '- 正式时间采用 800 条查询、五轮，先取每条查询延迟中位数，再以平均值的倒数计算 QPS。p95/p99 是这些中位数的分位数。',
    '- 原 F 的归档返回结构维护仍在计时内；新策略的可选逐路诊断关闭。时间结论针对已交付实现，不能全部归因于数学规则。',
    '- 开发 92,000、评价 736,000、计时 368,000 条记录均完成。全部评价记录实际用满对应预算，无填充工作或删除未满记录。192,000 条 C/F 记录复现旧评分顺序哈希、计数及返回结果。',
    '- 本地独立复算 828,000 条开发/评价记录的真值交集、开发选择、92 个结果单元、全部计时汇总及十二项 McNemar/Holm 检验；bootstrap 区间使用冻结分析实现，不声称有第二套区间复算。',
    '- 开发与评价查询身份不重叠，但已在历史中使用。本轮是固定策略补充比较，不是新查询独立确认；四图汇总按查询 ID 成簇。',
    '- LAION、WebVid 的负迁移及 F → SCORE → DIVERSE → KEEP → E64_RAW 全系列保留。没有增加选择器或继续找数据直到得到正结果。',
    '', '## 5. 已落实到论文的修改','',
    '正文主图现在展示 C/F/MEP-DEV/FRONTIER 的完整三预算曲线；主表展示四档 MEP 和 200/400/800 深度；另表列出救援、伤害与校正 p 值。方法和设计段落补全入口资格、前沿根选择、选参和收费规则。正文增加原生联合优势工作点，附录加入旧深度表。原始 F、12 页上限、空署名和外部负结果均保留。',
    '',
    '论文的主张继续是：CDSM 是一种通过有限局部探索及状态回流改善预算使用的方法，已有严格同实际距离的正向竞争证据。本文没有宣称入口和保存状态本身首创，也没有把 iQAN 的文献区别当作实测性能胜利。',
    '',
    '完整最新数值见 `source-data/cdsm-review-strengthening-20260922/SUMMARY.json`；协议、源码、选参、审计与术语说明同目录保留。单独的服务器原始归档身份见 `SERVER-BUNDLE.json`。旧稿在 `revisions/pre-r2-strengthening/` 保存，正式交付页数和文件哈希以 `QA.json` 为准。',
]
(R/'R2-REVISION-RESPONSE.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Wrote readable R2 revision response from verified server summary.')
