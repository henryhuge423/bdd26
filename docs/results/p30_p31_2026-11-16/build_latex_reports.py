"""Compile the verified P20--P30 Markdown reports through XeLaTeX and PGFPlots.

Run from any directory: python build_latex_reports.py --root PATH
Requires existing pandoc and xelatex; no package installation or dependency changes.
"""
from pathlib import Path
import argparse,csv,json,hashlib,re,subprocess,math

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
args=p.parse_args();root=args.root.resolve();out=root/'reports';out.mkdir(exist_ok=True)
fig=out/'tex_figures';fig.mkdir(exist_ok=True)
analysis=json.loads((root/'p30_verified_analysis.json').read_text(encoding='utf-8'))
stage=list(csv.DictReader((root/'stage_pairs.csv').open(encoding='utf-8')))
colors={'C':'blue!65!black','M':'orange!85!black'}

def esc(s):
    return ''.join({'\\':r'\textbackslash{}','_':r'\_','%':r'\%','&':r'\&','#':r'\#','{':r'\{','}':r'\}'}.get(c,c) for c in str(s))

def coords(rows):
    for x,y in rows:assert math.isfinite(float(x)) and math.isfinite(float(y))
    return ' '.join('(%s,%s)'%(format(x,'.12g'),format(y,'.12g')) for x,y in rows)

native={}
for key,title in [('net_dead','Matched interior Dead nuclei'),('net_all','All matched nuclei'),('net_extra','Unmatched predictions')]:
    lines=[r'\begin{tikzpicture}',r'\begin{axis}[width=15.5cm,height=11.5cm,xlabel={Intervention minus same-round control: count},ylabel={Experiment},title={'+title+r'},ytick={0,1,2,3,4,5,6,7,8,9,10},yticklabels={P20,P21,P22,P23,P24,P25,P26,P27,P28,P29,P30},ymin=-.6,ymax=10.6,y dir=reverse,grid=major,grid style={gray!18},legend style={at={(.5,-.13)},anchor=north,legend columns=2},axis line style={gray!70},tick label style={font=\small},label style={font=\small},title style={font=\normalsize}]']
    for seed,mark,col,shift in ((19,'*',colors['C'],-.13),(1,'diamond*',colors['M'],.13)):
        rows=[r for r in stage if r['pair'].endswith('_'+str(seed))];assert len(rows)==11
        lines.append(r'\addplot[only marks,mark='+mark+',mark size=2.7pt,color='+col+'] coordinates {'+coords([(int(r[key]),i+shift) for i,r in enumerate(rows)])+'};')
        lines.append(r'\addlegendentry{Seed '+str(seed)+'}')
    lines += [r'\end{axis}',r'\end{tikzpicture}']
    native['stage_'+key]='\n'.join(lines)

for seed in (19,1):
    changes=analysis['pairs'][str(seed)]['dead_images'];assert len(changes)==11 and sum(r['dead_total'] for r in changes)==145
    labels=','.join('{'+esc(f"{r['tissue']} {r['image']} ({r['dead_total']})")+'}' for r in changes)
    lines=[r'\begin{tikzpicture}',r'\begin{axis}[xbar,width=15.5cm,height=11.5cm,xlabel={Middle twelve minus last twelve: matched nuclei},title={All eleven interior-Dead development images, seed '+str(seed)+r'},ytick={0,1,2,3,4,5,6,7,8,9,10},yticklabels={'+labels+r'},ymin=-.6,ymax=10.6,y dir=reverse,bar width=8pt,grid=major,grid style={gray!18},tick label style={font=\small},label style={font=\small},title style={font=\normalsize}]',r'\addplot[fill=blue!60!black,draw=none] coordinates {'+coords([(r['net'],i) for i,r in enumerate(changes)])+'};',r'\end{axis}',r'\end{tikzpicture}']
    native[f'p30_dead_by_image_seed_{seed}']='\n'.join(lines)
    lines=[r'\begin{tikzpicture}',r'\begin{groupplot}[group style={group size=1 by 2,vertical sep=2.5cm},width=15cm,height=6.5cm,grid=major,grid style={gray!18},legend style={font=\small},tick label style={font=\small},label style={font=\small},title style={font=\normalsize}]',r'\nextgroupplot[xlabel={Epoch},ylabel={Training focal loss},ymode=log,title={Full training trajectory, seed '+str(seed)+'}]']
    for arm,label in [('C','Last twelve blocks'),('M','Middle twelve blocks')]:
        run=analysis['runs'][f'{arm}_{seed}'];assert len(run['loss_curve'])==130
        lines += [r'\addplot[thick,color='+colors[arm]+'] coordinates {'+coords(list(enumerate(run['loss_curve'],1)))+'};',r'\addlegendentry{'+label+'}']
    lines += [r'\nextgroupplot[xlabel={Image role},ylabel={All-nucleus center recall},title={Last-checkpoint training and development coverage},xtick={0,1},xticklabels={Training,Development},xmin=-.15,xmax=1.15,ymin=.5,ymax=1.01]']
    for arm,label in [('C','Last twelve blocks'),('M','Middle twelve blocks')]:
        run=analysis['runs'][f'{arm}_{seed}'];lines += [r'\addplot[thick,mark=*,color='+colors[arm]+'] coordinates {'+coords([(0,run['fit']['recall']),(1,run['audit']['recall'])])+'};',r'\addlegendentry{'+label+'}']
    lines += [r'\end{groupplot}',r'\end{tikzpicture}'];native[f'p30_training_seed_{seed}']='\n'.join(lines)

for name,text in native.items():(fig/(name+'.tex')).write_text(text+'\n',encoding='utf-8')
header=r'''\usepackage{float}
\usepackage{caption}
\usepackage{tikz,pgfplots}
\usepackage{hyperref}
\pgfplotsset{compat=1.18}
\usepgfplotslibrary{groupplots}
\hypersetup{colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black}
\setlength{\emergencystretch}{3em}
\setlength{\parskip}{0.4em}
\pagestyle{plain}
'''
(out/'report_header.tex').write_text(header,encoding='utf-8')
input_paths=[root/'p30_verified_analysis.json',root/'stage_pairs.csv',*[root/(name+'.md') for name in ('RESULTS_EN','RESULTS_ZH','READER_ZH')],*sorted((root/'figures').glob('*_image_*.png'))]
provenance={x.relative_to(root).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in input_paths}
reports=[]
for name in ('RESULTS_EN','RESULTS_ZH','READER_ZH'):
    source=(root/(name+'.md')).read_text(encoding='utf-8');first,*rest=source.splitlines();assert first.startswith('# ')
    title=first[2:];text='\n'.join(rest)
    if name in ('RESULTS_EN','RESULTS_ZH'):
        headings=['Run','Interior Dead','Small Dead','All matches','Predictions','Recall','Precision'] if name=='RESULTS_EN' else ['运行','内部 Dead','小 Dead','总体匹配','预测点','召回率','精确率']
        rows=[]
        for arm,seed in [('C',19),('M',19),('C',1),('M',1)]:
            r=analysis['runs'][f'{arm}_{seed}']['audit']
            rows.append(' & '.join([f'{arm}{seed}',f"{r['dead_hit']}/145",f"{r['small_hit']}/32",f"{r['matched']:,}",f"{r['candidates']:,}",f"{100*r['recall']:.3f}\\%",f"{100*r['precision']:.3f}\\%"])+r' \\')
        caption='P30 final development results: C updates the last twelve blocks; M updates the middle twelve. Suffixes denote training seeds.' if name=='RESULTS_EN' else 'P30 最后检查点开发评估：C 训练最后十二层，M 训练中间十二层，数字为随机种子。'
        table='\n'+r'\begin{table}[H]\centering\small'+'\n'+r'\begin{tabular}{lrrrrrr}\hline'+'\n'+' & '.join(headings)+r' \\\hline'+'\n'+'\n'.join(rows)+'\n'+r'\hline\end{tabular}'+'\n'+r'\caption{'+caption+'}\n'+r'\end{table}'+'\n'
        text,count=re.subn(r'\| (?:Run|运行) \|[^\n]+\n(?:\|[^\n]+\n){5}',lambda m:table,text)
        assert count==1,(name,count)
    # Keep each microscopy pair with its first object-level explanation as one box.
    def microscopy(m):
        caption,path,paragraph=m.groups();assert (root/path).is_file()
        return '\n\n'+r'\noindent\begin{minipage}{\linewidth}\centering'+'\n'+r'\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{../'+path+'}\n'+r'\captionof{figure}{'+esc(caption)+'}\n'+r'\raggedright'+'\n\n'+paragraph+'\n\n'+r'\end{minipage}'+'\n\n'
    text=re.sub(r'!\[([^\]]*)\]\((figures/p(?:22|25|27|29|30)_image_[^)]+\.png)\)\n\n(.+?)(?=\n\n|\Z)',microscopy,text,flags=re.S)
    def replace(m):
        caption,path=m.groups();stem=Path(path).stem
        if stem in native:
            return '\n'+r'\begin{figure}[H]\centering'+'\n'+r'\input{tex_figures/'+stem+'.tex}\n'+r'\caption{'+esc(caption)+'}\n'+r'\end{figure}'+'\n'
        # Real tissue-image overlays use the verified existing image bytes.
        assert path.startswith('figures/'),path
        return '\n'+r'\begin{figure}[H]\centering'+'\n'+r'\includegraphics[width=\linewidth,height=0.40\textheight,keepaspectratio]{../'+path+'}\n'+r'\caption{'+esc(caption)+'}\n'+r'\end{figure}'+'\n'
    text=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',replace,text)
    # Links to scientific payloads are relative to the original report root.
    text=re.sub(r'(?<!!)\[([^\]]+)\]\(([^)]+)\)',lambda m:'['+m.group(1)+']('+('../'+m.group(2) if not re.match(r'[a-z]+:',m.group(2)) and not m.group(2).startswith('#') else m.group(2))+')',text)
    md=out/(name+'.build.md');md.write_text(text,encoding='utf-8')
    tex=out/(name+'.tex')
    class_options=['--metadata','documentclass=article'] if name=='RESULTS_EN' else ['--metadata','documentclass=ctexart','--metadata','classoption=fontset=fandol']
    command=['pandoc',str(md),'--standalone','--from=markdown+raw_tex','--to=latex',*class_options,'--metadata','fontsize=11pt','--metadata','geometry:margin=2cm','--metadata','title='+title,'--metadata','date=2026-11-16','--include-in-header',str(out/'report_header.tex'),'--toc','--toc-depth=2','--output',str(tex)]
    subprocess.run(command,check=True,capture_output=True,text=True)
    for iteration in (1,2):
        result=subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','-file-line-error',tex.name],cwd=out,capture_output=True,text=True,encoding='utf-8',errors='replace')
        (out/(name+f'.xelatex{iteration}.stdout')).write_text(result.stdout,encoding='utf-8')
        if result.returncode:raise RuntimeError(name+' XeLaTeX failed: '+result.stdout[-4000:])
    pdf=out/(name+'.pdf');assert pdf.stat().st_size>30000
    reports.append(dict(name=name,pdf_bytes=pdf.stat().st_size,pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),tex_sha256=hashlib.sha256(tex.read_bytes()).hexdigest()))
(out/'BUILD_MANIFEST.json').write_text(json.dumps(dict(reports=reports,input_sha256=provenance,native_PGFPlots_figures=sorted(native),real_image_figures='existing SHA-verified microscopy overlays',compiler='XeLaTeX, two passes; Pandoc only produces TeX source'),indent=2),encoding='utf-8')
print(json.dumps(reports,indent=2))
