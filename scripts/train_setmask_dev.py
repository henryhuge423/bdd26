#!/usr/bin/env python
"""Train one approved E6 arm/seed; only fold1→fold2, explicit epoch-boundary resume."""
import argparse
from nucseg.setmask.engine import SetMaskConfig,train_dev


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=('B','Q'),required=True);p.add_argument('--seed',type=int,choices=(19,1),required=True)
    p.add_argument('--out',required=True);p.add_argument('--uni-ckpt',default='weights/MahmoodLab/UNI/pytorch_model.bin')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    train_dev(SetMaskConfig(a.arm,a.seed,a.out,uni_ckpt=a.uni_ckpt),resume=a.resume)


if __name__=='__main__':main()
