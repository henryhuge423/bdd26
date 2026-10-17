"""Stateless epoch/draw augmentation shared by both E6 representations."""
from contextlib import contextmanager
import hashlib
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from ..cellvit.data import cellvit_train_augs, normalize, small_nuclei, tissue_ids
from ..constants import NUM_CLASSES
from ..data.pannuke import PanNukeFold
from ..hovernet.data import hv_targets


def stable_seed(*parts, bits=32):
    raw=':'.join(map(str,parts)).encode('ascii')
    return int.from_bytes(hashlib.blake2b(raw,digest_size=8).digest(),'little') % (1 << bits)


def sample_keys(weights,seed,epoch):
    weights=torch.as_tensor(weights,dtype=torch.float64,device='cpu')
    if weights.ndim!=1 or not len(weights) or not torch.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:
        raise ValueError('Positive finite sampling support required')
    if epoch<0:
        raise ValueError('Negative epoch')
    g=torch.Generator().manual_seed(stable_seed('0',seed,epoch,bits=63))
    ids=torch.multinomial(weights,len(weights),replacement=True,generator=g).tolist()
    return [(int(i),int(epoch),int(draw)) for draw,i in enumerate(ids)]


@contextmanager
def augmentation_rng(seed):
    old_python,old_numpy=random.getstate(),np.random.get_state()
    try:
        random.seed(seed);np.random.seed(seed)
        yield
    finally:
        random.setstate(old_python);np.random.set_state(old_numpy)


class PairedDataset(Dataset):
    def __init__(self,fold,train,seed,root=None):
        if fold not in (1,2) or (train and fold!=1):
            raise ValueError('E6 only permits fold1 training and fold1/fold2 read-only data')
        if seed not in (19,1):
            raise ValueError('E6 seeds are frozen at19/1')
        self.fold=PanNukeFold(fold,root=root)
        self.train,self.seed=bool(train),int(seed)
        self.augs=cellvit_train_augs() if train else None
        self._tissues=tissue_ids(self.fold.tissue)

    def __len__(self):
        return len(self.fold)

    def tissue_labels(self):
        return self._tissues.copy()

    def class_presence(self):
        return np.array([[(self.fold.type[i]==c).any() for c in range(1,NUM_CLASSES+1)]
                         for i in range(len(self))],dtype=bool)

    def __getitem__(self,key):
        if isinstance(key,(int,np.integer)):
            if self.train:
                raise ValueError('Training access requires(index,epoch,draw)')
            i,epoch,draw=int(key),0,int(key)
        else:
            i,epoch,draw=map(int,key)
        if not 0<=i<len(self) or epoch<0 or draw<0:
            raise ValueError('Invalid sample key')
        image=np.array(self.fold.images[i])
        inst=np.array(self.fold.inst[i],dtype=np.int32)
        typ=np.array(self.fold.type[i],dtype=np.int32)
        if self.augs is not None:
            with augmentation_rng(stable_seed(self.seed,epoch,draw,i)):
                result=self.augs(image=image,mask=np.stack((inst,typ),axis=-1))
            image=result['image'];inst=result['mask'][...,0];typ=result['mask'][...,1]
        h=hashlib.sha256(f'{self.seed}:{epoch}:{draw}:{i}'.encode())
        for array in (image,inst,typ):
            h.update(np.ascontiguousarray(array).tobytes())
        return {
            'img':torch.from_numpy(normalize(image)),
            'inst_map':torch.from_numpy(np.ascontiguousarray(inst,dtype=np.int64)),
            'np_map':torch.from_numpy((inst>0).astype(np.int64)),
            'hv_map':torch.from_numpy(hv_targets(inst,min_size=30)),
            'tp_map':torch.from_numpy(typ.astype(np.int64)),
            'small_map':torch.from_numpy(small_nuclei(inst,100)),
            'tissue':int(self._tissues[i]),'index':i,
            'sample_key':torch.tensor((i,epoch,draw),dtype=torch.int64),
            'digest':h.hexdigest(),
        }
