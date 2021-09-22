# step -> cluster size

schedule = {
    10: 2,
    20: 3,
    30: 4,
    40: 1,
    50: 3,
    60: 4,
    70: 2,
    80: 4,
    90: 1,
    100: 0,
}

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
        if self._rank == 0 and self._es._progress > 0:
            self._step = read_step(self._es)

    def begin(self, run_context):
        pass

    def epoch_begin(self, run_context):
        pass

    def epoch_end(self, run_context):
        pass

    def step_begin(self, run_context):
        pass

    def step_end(self, run_context):
        self._step += 1

        if self._step in self._schedule:
            if current_rank() == 0:
                new_size = self._schedule[self._step]
                propose_new_size(new_size)

    def end(self, run_context):
        if self._rank == 0:
            save_step(self._es, self._step)

        pass