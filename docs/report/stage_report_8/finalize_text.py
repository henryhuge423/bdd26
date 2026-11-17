from pathlib import Path
p=Path(__file__).resolve().parent
replacements={
'RESULTS_ZH.md': [('# 细胞核中心检测：第八阶段报告：细胞核中心检测（P27–P31）','# 第八阶段报告：细胞核中心检测（P27–P31）'),('十一组实验分别改变了什么','十二组实验分别改变了什么'),('本阶段十一组固定方案','这十二组固定方案'),('| P30，中间十二层适配 | +7／+15 | +648／+516 | −215／−254 |','| P30，中间十二层适配 | +7／+15 | +648／+516 | −215／−254 |\n| P31，检测头分组归一化 | +8／+2 | −140／+6 | −307／−89 |')],
'RESULTS_EN.md': [('# Nucleus-center detection: evidence from P20–P31 and the P31 experiment','# Stage report 8: nucleus-center detection (P27–P31)'),('The eleven comparisons','The twelve comparisons'),('these eleven fixed comparisons','these twelve fixed comparisons')],
'READER_ZH.md': [('十一组实验的目标核变化','十二组实验的目标核变化')]}
for name, changes in replacements.items():
 f=p/name; s=f.read_text(encoding='utf-8')
 for old,new in changes:
  assert old in s, (name,old)
  s=s.replace(old,new,1)
 f.write_text(s,encoding='utf-8')
