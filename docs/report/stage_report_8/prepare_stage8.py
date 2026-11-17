from pathlib import Path
import csv, json, math, re
from PIL import Image, ImageDraw, ImageFont

root = Path(__file__).resolve().parent
fig = root / 'figures'
states = {(True,True): '#fffbe3', (False,True): '#65ffaf', (True,False): '#ffb947', (False,False): '#ff6191'}
for seed, image in [(19,1037),(19,776),(1,1025),(1,776)]:
    rec=[]
    for arm in ('C','G'):
        d=root/'p31_files'/f'A_{arm}_s{seed}'/'point_examples_audit'
        r=next(x for x in json.loads((d/'coordinates.json').read_text(encoding='utf-8'))['records'] if x['image']==image)
        rec.append((d,r))
    assert rec[0][1]['gt']==rec[1][1]['gt']
    pc,pg=[dict(r['unique_pairs']) for _,r in rec]
    canvas=Image.new('RGB',(1024,512),'white')
    for panel,(d,r) in enumerate(rec):
        with Image.open(d/r['png']) as im: base=im.crop((0,42,256,298)).convert('RGB').resize((512,512))
        draw=ImageDraw.Draw(base)
        for y,x in r['points']:
            x,y=2*x,2*y; draw.line((x-5,y,x+5,y),fill='#6deaff',width=2);draw.line((x,y-5,x,y+5),fill='#6deaff',width=2)
        for i,(klass,(y,x),interior,area) in enumerate(r['gt']):
            if klass!=4 or not interior: continue
            x,y=2*x,2*y; color=states[(i in pc,i in pg)]
            draw.ellipse((x-8,y-8,x+8,y+8),outline=color,width=3)
            if (i in pc)!=(i in pg):
                font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',16)
                draw.rectangle((x+7,y-19,x+35,y+1),fill='#202020');draw.text((x+8,y-19),str(i),font=font,fill=color)
        canvas.paste(base,(panel*512,0))
    canvas.save(fig/f'p31_image_{image}_seed_{seed}.png')

stage=root/'stage_pairs.csv'
rows=list(csv.DictReader(stage.open(encoding='utf-8-sig')))
if not any(r['pair']=='P31_19' for r in rows):
    rows += [dict(pair='P31_19',net_dead=8,net_all=-140,net_extra=-307),dict(pair='P31_1',net_dead=2,net_all=6,net_extra=-89)]
    with stage.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['pair','net_dead','net_all','net_extra']);w.writeheader();w.writerows(rows)

zh = '''## P31：在检测头中加入分组归一化

P31 将八个分组归一化模块加入细胞核中心检测头；这种方法在每张图像中按通道分组计算均值和方差，再使用可学习的缩放与偏移调整特征。对照使用原检测头，两组都训练编码器最后十二层，新增参数 2,816 个。模型、数据、训练轮数和中心匹配规则与同期对照相同。四次训练均完成 130 轮，以下使用最后一轮检查点。

| 随机种子与方案 | 内部死亡核匹配 | 小死亡核匹配 | 全部核匹配 | 全部预测点 | 精确率 |
|---|---:|---:|---:|---:|---:|
| 19，对照 | 65/145 | 7/32 | 7,274 | 10,536 | 69.04% |
| 19，分组归一化 | 73/145 | 6/32 | 7,134 | 10,089 | 70.71% |
| 1，对照 | 68/145 | 9/32 | 7,256 | 10,437 | 69.52% |
| 1，分组归一化 | 70/145 | 9/32 | 7,262 | 10,354 | 70.14% |

两个种子分别多找到 8 和 2 个内部死亡核，全部核匹配分别减少 140 和增加 6；未匹配预测点分别减少 307 和 89。训练集全部核召回率从 97.59%／97.81% 提升至 99.56%／99.73%。开发评估集死亡核召回率变化为 +5.52／+1.38 个百分点，按图像配对、组织分层的 95% 自助抽样区间分别为 [+3.49,+9.38] 和 [−4.30,+4.32] 个百分点。原定双种子判定综合目标召回、精确率及配对区间，两种子的完整结果均未达到该条件。

![P31 肺组织图像 1037，种子 19：左为对照，右为分组归一化。青色十字为预测中心；白、绿、橙、粉色圆圈分别表示共同找到、新方案找到、对照找到、共同漏检的内部死亡核；数字为标注对象编号。](figures/p31_image_1037_seed_19.png)

这张肺组织图像从 8/20 增至 9/20。对象 56 的标注中心在（行 16，列 74），最近预测点距离由 21.21 降至 2.24 像素；对象 66（91，155）由 19.65 降至 1.00。对象 63（67，158）则由 2.24 增至 20.52 像素，原有匹配消失。图中可观察到预测点出现和消失；距离按已保存的预测坐标计算。

![P31 结肠组织图像 776，种子 19：颜色和左右顺序同上一图。](figures/p31_image_776_seed_19.png)

该图两组均找到 4/7 个目标，但对象发生交换：对象 9（205，201）由 4.12 移至 2.00 像素，进入匹配范围；对象 12（238，110）由 2.00 增至 61.40 像素，附近预测消失。种子 1 中这两个对象均新增匹配，图像由 4/7 增至 6/7。相同的图像总数可以包含不同对象的变化。

![P31 肺组织图像 1025，种子 1：颜色和左右顺序同上一图。](figures/p31_image_1025_seed_1.png)

这张图保持 18/25：面积 31 像素的对象 14（43，212）从最近点距离 41.98 降至 1.00，对象 13 从 3.16 降至 2.24；对象 27 和 35 则分别从 2.24、1.41 增至 21.59、41.01。原始图像和预测坐标显示新增与丢失，造成这些变化的模型内部过程需要另行测量。
'''
en = '''## P31: group normalization in the center detector

P31 adds eight group-normalization modules to the center detector. For each image, group normalization calculates feature statistics within groups of channels and applies learned scaling and offsets. The control retains the original detector; both models update the last twelve encoder blocks. The added modules have 2,816 parameters. Data, training schedule and point matching remain shared with the paired control. All four runs completed 130 epochs and the results below use the last checkpoint.

| Seed and model | Interior Dead matched | Small Dead matched | All matched | Predictions | Precision |
|---|---:|---:|---:|---:|---:|
| 19, control | 65/145 | 7/32 | 7,274 | 10,536 | 69.04% |
| 19, group normalization | 73/145 | 6/32 | 7,134 | 10,089 | 70.71% |
| 1, control | 68/145 | 9/32 | 7,256 | 10,437 | 69.52% |
| 1, group normalization | 70/145 | 9/32 | 7,262 | 10,354 | 70.14% |

The intervention added 8 and 2 interior Dead matches, changed all matches by −140 and +6, and reduced unmatched predictions by 307 and 89. Training-set recall increased from 97.59%/97.81% to 99.56%/99.73%. Development Dead recall increased by 5.52/1.38 percentage points; the paired, tissue-stratified image-bootstrap 95% intervals were [+3.49,+9.38] and [−4.30,+4.32] points. The original combined two-seed criterion was not met by either seed.

![P31 lung image 1037, seed 19: control left, group normalization right. Cyan crosses are predicted centers; white, green, orange and pink circles indicate matched by both, intervention only, control only and missed by both. Numbers identify annotated objects.](figures/p31_image_1037_seed_19.png)

The image increased from 8/20 to 9/20. Object 56 at row 16, column 74 changed its nearest prediction distance from 21.21 to 2.24 pixels; object 66 at (91,155) changed from 19.65 to 1.00. Object 63 at (67,158) changed from 2.24 to 20.52 pixels and lost its match. The saved coordinates show the appearance and disappearance of nearby predictions.

![P31 colon image 776, seed 19; colors and panel order are as above.](figures/p31_image_776_seed_19.png)

Both models matched 4/7 targets, but exchanged objects. Object 9 at (205,201) moved from 4.12 to 2.00 pixels from the closest prediction; object 12 at (238,110) changed from 2.00 to 61.40 pixels. Both objects were gained in seed 1, where the image rose from 4/7 to 6/7.

![P31 lung image 1025, seed 1; colors and panel order are as above.](figures/p31_image_1025_seed_1.png)

The image remained at 18/25. Object 14, with area 31 pixels, gained a prediction at distance 1.00 instead of 41.98; object 13 moved from 3.16 to 2.24. Objects 27 and 35 lost their matches as nearest distances increased from 2.24 and 1.41 to 21.59 and 41.01. The saved images and coordinates describe these changes; measuring their internal causes requires additional model observations.
'''
reader='''## P31：增加归一化后找到了哪些核？

最新实验在检测细胞核中心的部分加入分组归一化：模型先按通道组调整每张图像的特征，再学习合适的缩放和偏移。两个随机种子的死亡核分别从 65 增至 73、从 68 增至 70；全部核匹配分别减少 140 和增加 6。分组归一化模型在训练集找到约 99.6%／99.7% 的核，在开发评估集中找到约 58.2%／59.3% 的全部标注核。

![P31 结肠组织图像 776，种子 19：左为对照，右为分组归一化；青色十字为预测，绿色圆圈表示新找到，橙色表示原来找到而新方案漏掉，白色表示共同找到，粉色表示共同漏掉。](figures/p31_image_776_seed_19.png)

两幅图各找到 7 个死亡核中的 4 个，但绿色的对象 9 被找到，橙色的对象 12 却丢失了附近的预测。种子 1 的同一张图从 4 个增加到 6 个，两个对象都被找到。这说明总体数字和实际找到的是哪些细胞核需要一起看。
'''
p28zh='''## P28：交错训练十二层的结果

P28 在 24 层编码器中隔一层训练一层，保持与最后十二层对照相同的可训练参数量。种子 19 的内部死亡核为 68→70/145，全部核为 7,177→7,705；种子 1 分别为 72→71/145 和 7,361→7,634。精确率从 69.23%／69.07% 升至 71.24%／72.44%。肺组织目标核变化 +6／0，结肠为 −4／−1。

![P28 肺组织图像 1037，种子 19：左为最后十二层对照，右为交错层；青色十字为预测，绿色、橙色、白色、粉色圈依次表示新找到、丢失、共同找到和共同漏检。](figures/p28_image_1037_seed_19.png)

此图从 12/20 降至 10/20。对象 71（170，23）的最近预测由 21.02 降至 2.24 像素，而对象 56（16，74）由 2.24 增至 22.02；对象 62 从 2.24 移至 3.16，跨过三像素匹配边界。这些具体变化与总体图像数量一起解释了交错层结果。
'''
p28en='''## P28: updating alternating encoder blocks

P28 updates every other one of the 24 encoder blocks, with the same number of trainable encoder parameters as the last-twelve control. For seed 19, interior Dead matches changed from 68 to 70/145 and all matches from 7,177 to 7,705; for seed 1 the corresponding changes were 72 to 71/145 and 7,361 to 7,634. Precision rose from 69.23%/69.07% to 71.24%/72.44%. Lung target changes were +6/0 and colon changes −4/−1.

![P28 lung image 1037, seed 19: last-twelve control on the left and alternating blocks on the right. Cyan crosses are predictions; green, orange, white and pink circles denote gained, lost, matched by both and missed by both.](figures/p28_image_1037_seed_19.png)

This image changed from 12/20 to 10/20. Object 71 at row 170, column 23 changed its nearest distance from 21.02 to 2.24 pixels; object 56 at (16,74) changed from 2.24 to 22.02, and object 62 moved from 2.24 to 3.16 across the three-pixel matching boundary.
'''
for name,new,p28 in [('RESULTS_ZH',zh,p28zh),('RESULTS_EN',en,p28en),('READER_ZH',reader,None)]:
    p=root/(name+'.md'); text=p.read_text(encoding='utf-8')
    text=text.replace('P20–P30 阶段结果与 P31 实验','第八阶段报告：细胞核中心检测（P27–P31）').replace('P20–P30 中文导读','第八阶段中文导读：细胞核中心检测').replace('P20–P30','P20–P31')
    lines=text.splitlines(); lines[1:3]=['','本报告以阶段报告 7 为起点，重点介绍 P27 至 P31，并回顾与这些实验直接相关的 P20 至 P26。' if name!='RESULTS_EN' else 'Stage report 8 covers P27 through P31, with P20–P26 as context for the subsequent experiments.'];text='\n'.join(lines)+'\n'
    if name=='RESULTS_ZH': text=re.sub(r'## P31：.*?(?=## 结果保存)',lambda m:new+'\n',text,flags=re.S)
    elif name=='RESULTS_EN': text=re.sub(r'## P31[^\n]*\n.*?(?=## Results, reproducibility)',lambda m:new+'\n',text,flags=re.S)
    else: text=re.sub(r'## 下一轮正在检验什么.*',lambda m:new,text,flags=re.S)
    if p28:
        marker='## P30：' if name=='RESULTS_ZH' else '## P30:'
        assert marker in text, (name,marker)
        text=text.replace(marker,p28+'\n'+marker,1)
    if name=='RESULTS_ZH':
        text=re.sub(r'P31 的隔离部署封存.*?。', 'P31 四次运行的小型结果与原始保存清单逐项核对。',text)
        text=re.sub(r'P31 完成后需依次.*?汇总。','P31 的 92 个小型结果文件已核对 SHA256；大型检查点保留原清单和文件大小记录。',text,flags=re.S)
    elif name=='RESULTS_EN':
        text=re.sub(r'Completion of P31 requires.*?those checks\.', 'All 92 small P31 result files were checked against their preservation manifests; large checkpoint sizes and original preservation records are retained.',text,flags=re.S)
    p.write_text(text,encoding='utf-8')
