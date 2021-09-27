# step -> cluster size

max_step = 200
# max_step = 2

static_schedule = {
    # empty
    max_step: 0,  # stop
}

elastic_schedule = {
    # 25: 4,
    # 200: 3,
    # 300: 4,
    # 400: 1,
    # 500: 3,
    # 600: 4,
    # 700: 2,
    # 800: 4,
    # 900: 1,
    max_step: 0,  # stop
}

import os
import time

import mindspore as ms
from kungfu.python import current_rank, propose_new_size


def ckpt(es):
    return 'progress-%010d.log' % (es._progress)


def read_step(es):
    with open(ckpt(es)) as f:
        return int(f.read().strip())


def save_step(es, step):
    with open(ckpt(es), 'w') as f:
        f.write('%d\n' % (step))


class ElasticScheduleCallback(ms.train.callback.Callback):
    def __init__(self, es, schedule):
        self._es = es
        self._schedule = schedule
        self._rank = current_rank()
        self._step = 0
        if self._es._progress > 0:
            # all ranks should read
            self._step = read_step(self._es)

        if self._rank == 0:
            print('starting from step %d' % (self._step))

        self._proc_start = int(os.getenv('KUNGFU_PROC_START_TIMESTAMP'))
        self._local_step = 0

    def begin(self, run_context):
        pass

    def epoch_begin(self, run_context):
        pass

    def epoch_end(self, run_context):
        pass

    def step_begin(self, run_context):
        if self._rank == 0:
            print('running step %d' % (self._step))

        if self._rank == 0 and self._local_step == 0:
            d = time.time() - self._proc_start
            print('first step BEGIN after reload took %.fs' % (d))

        self._step_begin_ts = time.time()

    def step_end(self, run_context):
        step_took = time.time() - self._step_begin_ts

        self._step += 1
        self._local_step += 1
        if self._rank == 0:
            if self._local_step == 1:
                d = time.time() - self._proc_start
                print('first step END after reload took %.fs' % (d))
            print('local step %d took %.fs' % (self._local_step, step_took))

        if self._step in self._schedule:
            if current_rank() == 0:
                new_size = self._schedule[self._step]
                propose_new_size(new_size)

    def end(self, run_context):
        if self._rank == 0:
            # only save from 0
            save_step(self._es, self._step)
            print('stopping at step %d' % (self._step))
