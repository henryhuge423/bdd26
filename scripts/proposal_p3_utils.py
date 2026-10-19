"""P3 cohort, synchronized augmentation and unique proposal scoring."""
import hashlib
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import maximum_filter

def split_indices(tissue, dead):
    fit,audit=[],[]
    groups={}
    for i,(t,d) in enumerate(zip(tissue,dead)):
        groups.setdefault((str(t),bool(d)),[]).append(i)
    for key,ids in sorted(groups.items()):
        ids=sorted(ids,key=lambda i:hashlib.sha256(f'p3-20261018:{i}'.encode()).hexdigest())
        n=max(1,int(.2*len(ids))) if len(ids)>=2 else 0
        audit.extend(ids[:n]);fit.extend(ids[n:])
    return sorted(fit),sorted(audit)

def gt_objects(inst,typ):
    ring=set(np.concatenate([inst[0],inst[-1],inst[:,0],inst[:,-1]]).tolist())
    out=[]
    for oid in np.unique(inst):
        if not oid: continue
        yy,xx=np.nonzero(inst==oid)
        votes=np.bincount(typ[yy,xx].astype(int),minlength=6)
        if not votes[1:].sum(): raise ValueError('GT object without type')
        j=np.argmin((yy-yy.mean())**2+(xx-xx.mean())**2)
        out.append((int(votes[1:].argmax()+1),(int(yy[j]),int(xx[j])),oid not in ring,len(yy)))
    return out

def gaussian_target(gt,height=256,width=256):
    out=np.zeros((height,width),np.float32)
    d=np.arange(-3,4);y,x=np.meshgrid(d,d,indexing='ij');kernel=np.exp(-(x*x+y*y)/2).astype(np.float32)
    for _,(y,x),_,_ in gt:
        y0,y1=max(0,y-3),min(height,y+4);x0,x1=max(0,x-3),min(width,x+4)
        out[y0:y1,x0:x1]=np.maximum(out[y0:y1,x0:x1],kernel[y0-y+3:y1-y+3,x0-x+3:x1-x+3])
    return out

def augment_pair(image,target,rng,photometric=True):
    k=int(rng.integers(4));image=torch.rot90(image,k,(-2,-1));target=torch.rot90(target,k,(-2,-1))
    if rng.random()<.5: image=image.flip(-1);target=target.flip(-1)
    if photometric:
        image=image*float(rng.uniform(.8,1.2));mean=image.mean((-2,-1),keepdim=True)
        image=((image-mean)*float(rng.uniform(.8,1.2))+mean).clamp(0,1)
    return image.contiguous(),target.contiguous()

def accumulation_denominator(start,total,effective_batch):
    return min(effective_batch,total-start)

def focal_numerator(logits,target):
    import torch.nn.functional as F
    logits=logits.float();p=logits.sigmoid()
    return -(F.logsigmoid(logits)*(1-p).square()*(target==1)+F.logsigmoid(-logits)*p.square()*(1-target).pow(4)*(target<1)).sum()

def decode_peaks(p,threshold=.5,cap=512):
    p=np.asarray(p,dtype=np.float64)
    ids=np.flatnonzero((p>=threshold)&(p==maximum_filter(p,size=3,mode='constant',cval=-1)))
    ids=ids[np.lexsort((ids,-p.ravel()[ids]))[:cap]]
    return [(int(v//p.shape[1]),int(v%p.shape[1])) for v in ids]

def score_points(points,gt):
    # GT order is ascending instance ID; candidates are decoder score/row-major order.
    # Dummy cost dominates every possible distance sum (<512*9), enforcing max cardinality.
    hit=[]
    if len(points) and len(gt):
        pos=np.asarray([g[1] for g in gt],float)
        dist=((pos[:,None,:]-np.asarray(points,float)[None,:,:])**2).sum(-1)
        cost=np.full((len(gt),len(points)+len(gt)),1e6)
        cost[:,:len(points)]=np.where(dist<=9,dist,2e6)
        rows,cols=linear_sum_assignment(cost)
        hit=[int(r) for r,c in zip(rows,cols) if c<len(points) and dist[r,c]<=9]
    dead=[i for i,g in enumerate(gt) if g[0]==4 and g[2]]
    small=[i for i in dead if gt[i][3]<67]
    return dict(matched=len(hit),total=len(gt),candidates=len(points),dead_hit=len(set(hit)&set(dead)),dead_total=len(dead),small_hit=len(set(hit)&set(small)),small_total=len(small))

def aggregate(rows):
    def ratio(a,b):
        den=sum(r[b] for r in rows)
        return sum(r[a] for r in rows)/den if den else None
    positive=[r for r in rows if r['dead_total']]
    return dict(recall=ratio('matched','total'),precision=ratio('matched','candidates'),dead_recall=ratio('dead_hit','dead_total'),dead_images=len(positive),dead_objects=sum(r['dead_total'] for r in rows),nondeg=sum(r['candidates']>0 for r in positive)/len(positive) if positive else None)

def admission(data):
    details={}
    for seed in ('19','1'):
        if seed not in data: return {'pass':False,'reason':'missing_seed','details':details}
        b,t=data[seed]['F1'],data[seed]['T1']
        assert [x['image'] for x in b]==[x['image'] for x in t]
        assert [x['dead_total'] for x in b]==[x['dead_total'] for x in t]
        ba,ta=aggregate(b),aggregate(t)
        groups={}
        for i,r in enumerate(b):
            if r['dead_total']: groups.setdefault(str(r['tissue']),[]).append(i)
        ci=[None,None];delta=None
        if groups:
            rng=np.random.default_rng(20261018);draws=[]
            for _ in range(1000):
                ids=np.concatenate([rng.choice(v,len(v),replace=True) for _,v in sorted(groups.items())])
                denom=sum(b[i]['dead_total'] for i in ids)
                draws.append(sum(t[i]['dead_hit']-b[i]['dead_hit'] for i in ids)/denom)
            ci=np.percentile(draws,[2.5,97.5]).tolist()
            delta=ta['dead_recall']-ba['dead_recall']
        support=ta['dead_images']>=10 and ta['dead_objects']>=100
        passed=bool(support and delta is not None and ta['precision'] is not None and ta['recall'] is not None and ba['precision'] is not None and ba['recall'] is not None and ta['dead_recall']>=.60 and delta>=.05 and ci[0]>0 and ta['precision']>=.75 and ta['precision']>=ba['precision']-.02 and ta['recall']>=ba['recall']-.02 and ta['nondeg']>=.95)
        details[seed]=dict(pass_seed=passed,support=support,baseline=ba,trainable=ta,delta=delta,ci95=ci)
    return {'pass':all(v['pass_seed'] for v in details.values()),'details':details}
