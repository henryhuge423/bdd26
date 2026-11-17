from pathlib import Path
import argparse,json,re,hashlib
import fitz
from PIL import Image,ImageDraw

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();reports=root/'reports';qa=root/'pdf_qa';qa.mkdir(exist_ok=True)
results=[]
for name in ('RESULTS_EN','RESULTS_ZH','READER_ZH'):
    doc=fitz.open(reports/(name+'.pdf'));images=[];texts=[]
    for i,page in enumerate(doc):
        text=page.get_text();assert text.strip(),(name,i+1);texts.append(text)
        pix=page.get_pixmap(matrix=fitz.Matrix(1.3,1.3));target=qa/f'{name}_page_{i+1:02}.png';pix.save(target)
        im=Image.open(target).convert('RGB');im.thumbnail((420,610));images.append(im.copy())
    for group in range(0,len(images),6):
        selected=images[group:group+6];sheet=Image.new('RGB',(3*440,2*650),'#ddd');d=ImageDraw.Draw(sheet)
        for n,im in enumerate(selected):
            x=(n%3)*440+10;y=(n//3)*650+25;sheet.paste(im,(x,y));d.text((x,y-18),f'{name}: page {group+n+1}',fill='black')
        sheet.save(qa/f'{name}_contact_{group//6+1}.png')
    combined='\n'.join(texts);(qa/(name+'.txt')).write_text(combined,encoding='utf-8')
    if name!='READER_ZH':
        assert all(x in combined for x in ('P27','P28','P29','P30','P31','145'))
    if name.endswith('ZH'):assert any('\u4e00'<=c<='\u9fff' for c in combined)
    log=(reports/(name+'.log')).read_text(encoding='utf-8',errors='replace');assert 'Missing character:' not in log
    over=[float(x) for x in re.findall(r'Overfull \\hbox \(([0-9.]+)pt too wide\)',log)]
    assert max(over,default=0)<4,(name,over)
    results.append(dict(name=name,pages=len(doc),all_pages_rendered=True,nonempty_text_every_page=True,missing_glyphs=0,max_overfull_hbox_pt=max(over,default=0),pdf_sha256=hashlib.sha256((reports/(name+'.pdf')).read_bytes()).hexdigest()))
(qa/'QA.json').write_text(json.dumps(results,indent=2),encoding='utf-8');print(json.dumps(results,indent=2))
