"""E6 legal targets, sampled Hungarian costs and complete native-mask losses."""
import math

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as F


def make_targets(inst,typ,capacity=512):
    if inst.ndim!=3 or inst.shape!=typ.shape or capacity<1:
        raise ValueError('Expected matching BHW instance/type maps and positive capacity')
    ims=inst.detach().cpu().numpy();tps=typ.detach().cpu().numpy()
    result=[]
    for image,types in zip(ims,tps):
        if (image<0).any() or (types<0).any() or (types>5).any():
            raise ValueError('Invalid instance or type labels')
        ids=np.unique(image);ids=ids[ids>0]
        if len(ids)>capacity:
            raise ValueError(f'{len(ids)} objects exceed query capacity{capacity}')
        classes=[];points=[]
        for oid in ids:
            yy,xx=np.nonzero(image==oid)
            votes=np.bincount(types[yy,xx].astype(np.int64),minlength=6)
            if votes[1:].sum()==0:
                raise ValueError('Instance has no foreground class votes')
            classes.append(int(votes[1:].argmax()+1))
            distance=(yy-yy.mean())**2+(xx-xx.mean())**2
            j=int(distance.argmin())
            points.append((int(yy[j]),int(xx[j])))
        result.append(dict(ids=torch.as_tensor(ids.astype(np.int64)),
                           classes=torch.tensor(classes,dtype=torch.long),
                           points=torch.tensor(points,dtype=torch.long).reshape(-1,2),
                           inst=torch.as_tensor(np.array(image,dtype=np.int64))))
    return result


def matching_points(target,height,width):
    y,x=torch.meshgrid(torch.arange(4,height,8),torch.arange(4,width,8),indexing='ij')
    grid=torch.stack((y.flatten(),x.flatten()),dim=1)
    return torch.unique(torch.cat((grid,target['points']),dim=0),dim=0,sorted=True)


@torch.no_grad()
def match_queries(class_logits,mask_embed,mask_features,targets):
    assignments=[]
    for b,target in enumerate(targets):
        count=len(target['ids'])
        if count>class_logits.shape[1]:
            raise ValueError('Ground truth exceeds query capacity')
        if not count:
            assignments.append((torch.empty(0,dtype=torch.long),torch.empty(0,dtype=torch.long)))
            continue
        h,w=mask_features.shape[-2:]
        point=matching_points(target,h,w)
        y,x=point[:,0].to(mask_features.device),point[:,1].to(mask_features.device)
        sample=mask_features[b,:,y,x].float()
        logits=mask_embed[b].float()@sample/math.sqrt(mask_embed.shape[-1])
        truth=(target['inst'][point[:,0],point[:,1]][None,:]==target['ids'][:,None]).to(logits.device).float()
        bce=F.softplus(logits).mean(1)[:,None]-logits@truth.T/logits.shape[1]
        prob=logits.sigmoid()
        dice=1-(2*(prob@truth.T)+1)/(prob.sum(1)[:,None]+truth.sum(1)[None,:]+1)
        classes=target['classes'].to(class_logits.device)
        cost=-2*class_logits[b].float().softmax(-1)[:,classes]+5*bce+5*dice
        if not torch.isfinite(cost).all():
            raise ValueError('Nonfinite matching cost')
        qi,ti=linear_sum_assignment(cost.cpu().numpy())
        assignments.append((torch.from_numpy(qi).long(),torch.from_numpy(ti).long()))
    return assignments


def setmask_loss(pred,batch):
    classes=pred['class_logits'].float()
    embed=pred['mask_embed'].float();features=pred['mask_features'].float()
    targets=make_targets(batch['inst_map'],batch['tp_map'],capacity=classes.shape[1])
    assignments=match_queries(classes,embed,features,targets)
    labels=torch.zeros(classes.shape[:2],dtype=torch.long,device=classes.device)
    zero=embed.sum()*0+features.sum()*0
    bce_sum=zero;dice_sum=zero;count=0
    for b,((qi,ti),target) in enumerate(zip(assignments,targets)):
        labels[b,qi.to(labels.device)]=target['classes'][ti].to(labels.device)
        for start in range(0,len(qi),16):
            q=qi[start:start+16].to(embed.device);t=ti[start:start+16]
            logits=torch.einsum('qc,chw->qhw',embed[b,q],features[b])/math.sqrt(embed.shape[-1])
            truth=(target['inst'][None,:,:]==target['ids'][t,None,None]).to(logits.device).float()
            bce_sum=bce_sum+F.binary_cross_entropy_with_logits(logits,truth,reduction='none').mean((1,2)).sum()
            p=logits.sigmoid().flatten(1);gt=truth.flatten(1)
            dice_sum=dice_sum+(1-(2*(p*gt).sum(1)+1)/(p.sum(1)+gt.sum(1)+1)).sum()
            count+=len(q)
    weight=classes.new_tensor([.1,1,1,1,1,1])
    terms={
        'query_ce':2*F.cross_entropy(classes.flatten(0,1),labels.flatten(),weight=weight),
        'mask_bce':5*bce_sum/max(count,1),
        'mask_dice':5*dice_sum/max(count,1),
        'tissue_ce':.1*F.cross_entropy(pred['tissue'].float(),batch['tissue'].to(classes.device)),
    }
    loss=sum(terms.values())
    return loss,{k:float(v.detach()) for k,v in terms.items()}
