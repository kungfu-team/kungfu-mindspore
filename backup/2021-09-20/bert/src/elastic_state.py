import os
import sys
import time

from kungfu._utils import show_duration
from kungfu.python.elastic_state import ElasticState, ElasticContext
import mindspore as ms

__all__ = [
    'ElasticState',
    'ElasticContext',
    'ElasticCallback',
]


def estimate_remain(p, d):
    if p == 0:
        return 1e10
    return (1 - p) * d / p


class ElasticCallback(ms.train.callback.Callback):
    def __init__(self, elastic_state, global_batch_size):
        self._elastic_state = elastic_state
        self._global_batch_size = global_batch_size
        self._job_start = int(os.getenv('KUNGFU_JOB_START_TIMESTAMP'))

    def begin(self, run_context):
        pass
        print('ElasticCallback::begin')

    def epoch_begin(self, run_context):
        pass
        print('ElasticCallback::epoch_begin')

    def epoch_end(self, run_context):
        pass
        print('ElasticCallback::epoch_end')

    def step_begin(self, run_context):
        print('ElasticCallback::step_begin')
        should_sync = self._elastic_state.begin()
        if should_sync:
            print(
                'TODO: sync state to %d, no need to sync dataset state in reload mode'
                % (self._elastic_state._progress))
            #print('resetting dataset')
            #self._dataset.reset()

        duration = time.time() - self._job_start
        p = (float(self._elastic_state._progress) /
             float(self._elastic_state._max_progress))

        print('progress: %d/%d, took %s, remain: %s' % (
            self._elastic_state._progress,
            self._elastic_state._max_progress,
            show_duration(duration),
            show_duration(estimate_remain(p, duration)),
        ))

    def step_end(self, run_context):
        print('ElasticCallback::step_end')
        self._elastic_state.end(self._global_batch_size)
        if self._elastic_state.stopped():
            print('_elastic_state stopped, requesting run_context to stop')
            run_context.request_stop()

            d = self._elastic_state.get_duration_since_resize()
            print('from resize start to after request stop: %dms' % (d / 1e6))

    def end(self, run_context):
        pass
        print('StopCallback::end')


# Old code

# class ElasticState:
#     def __init__(self, max_progress=None):
#         self._progress = 0
#         self._max_progress = max_progress
#         self._synced = False
#         self._stop_reason = None

#         import pystdml as ml
#         # print('creating new ElasticState, must be a singleton, pid=%d, sys.argv=%s' % (os.getpid(), sys.argv))
#         # print('creating new ElasticState, must be a singleton, pid=%d' % (os.getpid()))
#         self._sess = ml.init_elastic()

#     def begin(self):
#         should_sync = not self._synced
#         if should_sync:
#             new_progress = self._sess.all_reduce_max(self._progress)
#             self._progress = new_progress
#             self._synced = True
#         return should_sync

#     def end(self, progress=1):
#         self._progress += progress
#         if self._max_progress:
#             if self._progress >= self._max_progress:
#                 self._stop_reason = 'finished'
#                 return

#         result = self._sess.resize()
#         if result.changed:
#             if result.detached:
#                 self._stop_reason = 'detached'
#                 return
#             self._synced = False

#     def stopped(self):
#         return self._stop_reason is not None

#     def stop_reason(self):
#         return self._stop_reason

# class ElasticContext:
#     def __init__(self, elastic_state):
#         self._elastic_state = elastic_state

#     def __enter__(self):
#         return self._elastic_state.begin()

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self._elastic_state.end()

# class ElasticCallback(ms.train.callback.Callback):
#     def __init__(self, elastic_state, dataset):
#         self._elastic_state = elastic_state
#         self._dataset = dataset

#     def begin(self, run_context):
#         pass
#         print('ElasticCallback::begin')

#     def epoch_begin(self, run_context):
#         pass
#         print('ElasticCallback::epoch_begin')

#     def epoch_end(self, run_context):
#         pass
#         print('ElasticCallback::epoch_end')

#     def step_begin(self, run_context):
#         print('ElasticCallback::step_begin')
#         should_sync = self._elastic_state.begin()
#         if should_sync:
#             print('TODO: sync state to %d' % (self._elastic_state._progress))
#             #print('resetting dataset')
#             #self._dataset.reset()

#         print('progress: %d' % (self._elastic_state._progress))

#     def step_end(self, run_context):
#         print('ElasticCallback::step_end')
#         self._elastic_state.end()
#         if self._elastic_state.stopped():
#             print('_elastic_state stopped, requesting run_context to stop')
#             run_context.request_stop()

#     def end(self, run_context):
#         pass
#         print('StopCallback::end')
