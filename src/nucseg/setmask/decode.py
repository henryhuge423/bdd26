"""Deterministic original-resolution query decoding with bounded mask chunks."""
import math

import numpy as np
import torch


@torch.no_grad()
def decode_set(pred,class_threshold=.5,mask_threshold=.5,chunk_size=16):
    logits,embed,features=(pred[k] for k in ('class_logits','mask_embed','mask_features'))
    if (logits.ndim!=3 or logits.shape[-1]!=6 or embed.ndim!=3 or features.ndim!=4
            or logits.shape[:2]!=embed.shape[:2] or logits.shape[0]!=features.shape[0]
            or embed.shape[-1]!=features.shape[1] or chunk_size<1
            or not 0<class_threshold<1 or not 0<mask_threshold<1):
        raise ValueError('Invalid set-mask prediction or decoding parameters')
    if not all(torch.isfinite(t).all() for t in (logits,embed,features)):
        raise ValueError('Nonfinite set-mask prediction')
    prob=logits.float().softmax(-1);score,classes=prob.max(-1)
    batch,height,width=features.shape[0],features.shape[-2],features.shape[-1]
    result_inst=np.zeros((batch,height,width),np.int32)
    result_type=np.zeros((batch,height,width),np.uint8)
    for b in range(batch):
        accepted=torch.nonzero((classes[b]>0)&(score[b]>=class_threshold),as_tuple=False).flatten()
        best=torch.full((height,width),float('-inf'),device=features.device)
        ids=torch.zeros((height,width),dtype=torch.int32,device=features.device)
        types=torch.zeros((height,width),dtype=torch.uint8,device=features.device)
        native=features[b].float()
        for start in range(0,len(accepted),chunk_size):
            q=accepted[start:start+chunk_size]
            masks=(torch.einsum('qc,chw->qhw',embed[b,q].float(),native)/math.sqrt(embed.shape[-1])).sigmoid()
            weighted=(masks*score[b,q,None,None]).masked_fill(masks<mask_threshold,float('-inf'))
            top,which=weighted.max(0)
            replace=top>best
            winner=q[which]
            ids[replace]=(winner[replace]+1).to(torch.int32)
            types[replace]=classes[b,winner[replace]].to(torch.uint8)
            best=torch.maximum(best,top)
        result_inst[b]=ids.cpu().numpy();result_type[b]=types.cpu().numpy()
    return result_inst,result_type
