#!/usr/bin/env python3.7
import sys
from hashlib import sha1

import mindspore as ms


def read_ckpt(filename):
    param_dict = ms.train.serialization.load_checkpoint(filename)
    for idx, (k, p) in enumerate(param_dict.items()):
        x = p.asnumpy()
        h = sha1(x.tobytes()).hexdigest()
        meta = '%s%-20s' % (x.dtype, x.shape)
        stat = '[%f, %f] ~ %f' % (x.min(), x.max(), x.mean())
        print('[%3d]    %s    %-24s: %s %s' % (idx, h, k, meta, stat))


def main(filenames):
    if len(filenames) == 0:
        print('no files given!')

    for filename in filenames:
        print(filename)
        read_ckpt(filename)
        print()


main(sys.argv[1:])
