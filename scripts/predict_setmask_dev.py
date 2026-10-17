#!/usr/bin/env python
"""Predict final E6 checkpoint on fold2; evaluate in a separate existing evaluator process."""
import argparse
from nucseg.setmask.engine import predict_dev


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--fold',type=int,choices=(2,),default=2)
    a=p.parse_args();print(predict_dev(a.run,fold=a.fold),flush=True)


if __name__=='__main__':main()
