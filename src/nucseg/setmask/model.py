"""Minimal E6 set representation sharing the original plain UNI pixel decoder."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from ..cellvit.model import CellViTUNI


def spatial_position(height,width,dim,device,dtype):
    y,x=torch.meshgrid(torch.arange(height,device=device,dtype=torch.float32),
                       torch.arange(width,device=device,dtype=torch.float32),indexing='ij')
    freq=10000**(-torch.arange(dim//4,device=device,dtype=torch.float32)/(dim//4))
    y=y.flatten()[:,None]*freq;x=x.flatten()[:,None]*freq
    return torch.cat((y.sin(),y.cos(),x.sin(),x.cos()),dim=-1).to(dtype)[None]


class QueryLayer(nn.Module):
    def __init__(self,dim,heads):
        super().__init__()
        self.cross=nn.MultiheadAttention(dim,heads,dropout=.1,batch_first=True)
        self.self_attention=nn.MultiheadAttention(dim,heads,dropout=.1,batch_first=True)
        self.norm1=nn.LayerNorm(dim);self.norm2=nn.LayerNorm(dim);self.norm3=nn.LayerNorm(dim)
        self.ff=nn.Sequential(nn.Linear(dim,4*dim),nn.ReLU(),nn.Dropout(.1),nn.Linear(4*dim,dim))
        self.drop=nn.Dropout(.1)

    def forward(self,q,qpos,memory,mpos):
        t=self.norm1(q)
        q=q+self.drop(self.cross(t+qpos,memory+mpos,memory,need_weights=False)[0])
        t=self.norm2(q)
        q=q+self.drop(self.self_attention(t+qpos,t+qpos,t,need_weights=False)[0])
        return q+self.drop(self.ff(self.norm3(q)))


class QueryMaskHead(nn.Module):
    def __init__(self,channels=64,dim=256,mask_dim=128,queries=512,layers=3,heads=8):
        super().__init__()
        if min(channels,dim,mask_dim,queries,layers,heads)<1 or dim%4 or dim%heads:
            raise ValueError('Positive dimensions, dim divisible by4 and heads required')
        self.dim=dim
        self.mask_projection=nn.Conv2d(channels,mask_dim,1)
        self.memory_projection=nn.Conv2d(channels,dim,1)
        self.query_content=nn.Embedding(queries,dim);self.query_position=nn.Embedding(queries,dim)
        self.layers=nn.ModuleList(QueryLayer(dim,heads) for _ in range(layers))
        self.class_head=nn.Linear(dim,6)
        self.mask_head=nn.Sequential(nn.Linear(dim,dim),nn.ReLU(),nn.Linear(dim,dim),nn.ReLU(),nn.Linear(dim,mask_dim))

    def forward(self,features):
        if features.ndim!=4 or min(features.shape[-2:])<8:
            raise ValueError('Native feature map must be BCHW with each spatial axis>=8')
        memory=self.memory_projection(F.avg_pool2d(features,8))
        mpos=spatial_position(*memory.shape[-2:],self.dim,memory.device,memory.dtype)
        memory=memory.flatten(2).transpose(1,2)
        q=self.query_content.weight[None].expand(features.shape[0],-1,-1)
        qpos=self.query_position.weight[None]
        for layer in self.layers:
            q=layer(q,qpos,memory,mpos)
        return dict(class_logits=self.class_head(q),mask_embed=self.mask_head(q),
                    mask_features=self.mask_projection(features))


class SetMaskUNI(CellViTUNI):
    def __init__(self,uni_ckpt=None):
        # Exact common initialization order is inherited before new query parameters consume RNG.
        super().__init__(uni_ckpt=uni_ckpt,fusion='plain')
        del self.np_branch,self.hv_branch,self.np_head,self.hv_head,self.tp_head
        self.query_head=QueryMaskHead()

    def forward(self,x):
        if x.shape[-2:]!=(256,256):
            raise ValueError('E6 native input is frozen at256x256')
        cls,zs=self.encoder(x)
        hp,wp=x.shape[-2]//16,x.shape[-1]//16
        z1,z2,z3,z4=(z[:,self.encoder.num_prefix_tokens:].transpose(1,2).reshape(-1,1024,hp,wp) for z in zs)
        skips=(self.skip0(x),self.skip1(z1),self.skip2(z2),self.skip3(z3),z4)
        out=self.query_head(self._decode(skips,self.tp_branch))
        out['tissue']=self.tissue_head(cls)
        return out
