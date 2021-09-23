'''
Bert finetune and evaluation script.
'''
import os
import argparse
import sys
from src.bert_for_finetune import BertSquadCell, BertSquad
from src.finetune_eval_config import optimizer_cfg, bert_net_cfg
from src.dataset import create_squad_dataset
from src.utils import make_directory, LossCallBack, LoadNewestCkpt, BertLearningRate
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.nn.optim import AdamWeightDecay, Lamb, Momentum
from mindspore.train.model import Model
from mindspore.train.callback import (CheckpointConfig, ModelCheckpoint,
                                      TimeMonitor, SummaryCollector,
                                      LossMonitor)
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.train.serialization import save_checkpoint
from mindspore.common import set_seed
from src.kungfu_mindspore_optimizer import KungFuLamb
from src.parse_env import parse_kungfu_env
from src.elastic_state import ElasticState, ElasticCallback

from kungfu.python.elastic import create_tf_records
from kungfu.python import _init as kungfu_init
from kungfu.python import current_rank, current_cluster_size, propose_new_size
from mindspore._c_expression import kungfu_nccl_finalize, kungfu_nccl_init
from schedule import schedule, ElasticScheduleCallback


def log_pid(msg=''):
    pid = os.getpid()
    ppid = os.getppid()
    print('%s pid=%d/ppid=%d' % (msg, pid, ppid))


kf_env = parse_kungfu_env()
_cur_dir = os.getcwd()


def do_train(elastic_callbacks,
             dataset=None,
             network=None,
             load_checkpoint_path="",
             save_checkpoint_path="",
             epoch_num=1,
             distributed=False):
    if load_checkpoint_path == "":
        raise ValueError(
            "Pretrain model missed, finetune task must load pretrain model!")
    steps_per_epoch = dataset.get_dataset_size()
    # optimizer
    if optimizer_cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = BertLearningRate(
            learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
            end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
            warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
            decay_steps=steps_per_epoch * epoch_num,
            power=optimizer_cfg.AdamWeightDecay.power)
        params = network.trainable_params()
        decay_params = list(
            filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(
            filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x),
                   params))
        group_params = [{
            'params':
            decay_params,
            'weight_decay':
            optimizer_cfg.AdamWeightDecay.weight_decay
        }, {
            'params': other_params,
            'weight_decay': 0.0
        }]

        optimizer = AdamWeightDecay(group_params,
                                    lr_schedule,
                                    eps=optimizer_cfg.AdamWeightDecay.eps)
    elif optimizer_cfg.optimizer == 'Lamb':
        lr_schedule = BertLearningRate(
            learning_rate=optimizer_cfg.Lamb.learning_rate,
            end_learning_rate=optimizer_cfg.Lamb.end_learning_rate,
            warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
            decay_steps=steps_per_epoch * epoch_num,
            power=optimizer_cfg.Lamb.power)
        # optimizer = Lamb(network.trainable_params(), learning_rate=lr_schedule)
        optimizer = KungFuLamb(network.trainable_params(),
                               learning_rate=lr_schedule)
    elif optimizer_cfg.optimizer == 'Momentum':
        optimizer = Momentum(
            network.trainable_params(),
            learning_rate=optimizer_cfg.Momentum.learning_rate,
            momentum=optimizer_cfg.Momentum.momentum)
    else:
        raise Exception(
            "Optimizer not supported. support: [AdamWeightDecay, Lamb, Momentum]"
        )

    # ckpt_config = CheckpointConfig(
    #     save_checkpoint_steps=250,
    #     keep_checkpoint_max=10,
    # )
    # ckpoint_cb = ModelCheckpoint(
    #     prefix="squad",
    #     directory=None if save_checkpoint_path == "" else save_checkpoint_path,
    #     config=ckpt_config,
    # )

    # load checkpoint into network
    print('loading checkpoint from %s' % (load_checkpoint_path))
    param_dict = load_checkpoint(load_checkpoint_path)
    load_param_into_net(network, param_dict)

    # print('using optimizer: %s' % (optimizer))
    update_cell = DynamicLossScaleUpdateCell(
        loss_scale_value=2**32,
        scale_factor=2,
        scale_window=1000,
    )
    netwithgrads = BertSquadCell(
        network,
        optimizer=optimizer,
        scale_update_cell=update_cell,
    )
    model = Model(netwithgrads)
    callbacks = [
        TimeMonitor(dataset.get_dataset_size()),
        LossCallBack(dataset.get_dataset_size()),
        # ckpoint_cb,
    ]

    if distributed:
        # rank = kfops.kungfu_current_rank()
        rank = kf_env['rank']
        summary_path = "./summary_{}".format(rank)
    else:
        summary_path = "./summary"
    callbacks.append(SummaryCollector(summary_path))
    callbacks.append(LossMonitor())
    callbacks.extend(elastic_callbacks)

    print('before model.train')
    model.train(epoch_num,
                dataset,
                callbacks=callbacks,
                dataset_sink_mode=False)
    print('after model.train')


def run_squad():
    """run squad task"""
    parser = argparse.ArgumentParser(description="run squad")
    parser.add_argument("--device_target",
                        type=str,
                        default="Ascend",
                        choices=["Ascend", "GPU"],
                        help="Device type, default is Ascend")
    parser.add_argument("--distribute",
                        type=str,
                        default="false",
                        choices=["true", "false"],
                        help="Run distribute, default is false.")
    parser.add_argument("--do_train",
                        type=str,
                        default="false",
                        choices=["true", "false"],
                        help="Eable train, default is false")
    parser.add_argument("--do_eval",
                        type=str,
                        default="false",
                        choices=["true", "false"],
                        help="Eable eval, default is false")
    parser.add_argument("--device_id",
                        type=int,
                        default=0,
                        help="Device id, default is 0.")
    parser.add_argument("--epoch_num",
                        type=int,
                        default=3,
                        help="Epoch number, default is 1.")
    parser.add_argument("--num_class",
                        type=int,
                        default=2,
                        help="The number of class, default is 2.")
    parser.add_argument("--train_data_shuffle",
                        type=str,
                        default="true",
                        choices=["true", "false"],
                        help="Enable train data shuffle, default is true")
    parser.add_argument("--eval_data_shuffle",
                        type=str,
                        default="false",
                        choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    # parser.add_argument("--train_batch_size", type=int, default=32, help="Train batch size, default is 32")
    parser.add_argument("--eval_batch_size",
                        type=int,
                        default=1,
                        help="Eval batch size, default is 1")
    parser.add_argument("--vocab_file_path",
                        type=str,
                        default="",
                        help="Vocab file path")
    parser.add_argument("--eval_json_path",
                        type=str,
                        default="",
                        help="Evaluation json file path, can be eval.json")
    parser.add_argument("--save_finetune_checkpoint_path",
                        type=str,
                        default="",
                        help="Save checkpoint path")
    parser.add_argument("--load_pretrain_checkpoint_path",
                        type=str,
                        default="",
                        help="Load checkpoint file path")
    parser.add_argument("--load_finetune_checkpoint_path",
                        type=str,
                        default="",
                        help="Load checkpoint file path")
    parser.add_argument("--train_data_file_path",
                        type=str,
                        default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--schema_file_path",
                        type=str,
                        default="",
                        help="Schema path, it is better to use absolute path")

    parser.add_argument('--max-progress', type=int, default=10)
    parser.add_argument('--global-batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--index-file', type=str, default='')
    parser.add_argument('--reload', action='store_true', default=True)

    args_opt = parser.parse_args()
    epoch_num = args_opt.epoch_num
    save_finetune_checkpoint_path = args_opt.save_finetune_checkpoint_path

    if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower(
    ) == "false":
        raise ValueError(
            "At least one of 'do_train' or 'do_eval' must be true")
    if args_opt.do_train.lower(
    ) == "true" and args_opt.train_data_file_path == "":
        raise ValueError(
            "'train_data_file_path' must be set when do finetune task")
    if args_opt.do_eval.lower() == "true":
        if args_opt.vocab_file_path == "":
            raise ValueError(
                "'vocab_file_path' must be set when do evaluation task")
        if args_opt.eval_json_path == "":
            raise ValueError(
                "'tokenization_file_path' must be set when do evaluation task")
    if args_opt.distribute.lower() == "true":
        distributed = True
    else:
        distributed = False
    if distributed:
        # kfops.init(args_opt.device_target)
        #device_num = kfops.kungfu_current_cluster_size()
        #rank = kfops.kungfu_current_rank()
        device_num = kf_env['size']
        rank = kf_env['rank']
        print("kungfu rank={}, size={}".format(rank, device_num))

        save_finetune_checkpoint_path = os.path.join(
            save_finetune_checkpoint_path, "ckpt_" + str(rank))
    else:
        device_num = 1
        rank = 0

    kungfu_init()
    kungfu_nccl_init()
    shard = create_tf_records(args_opt.index_file, args_opt.seed,
                              args_opt.global_batch_size)
    filenames = shard['filenames']
    batch_size, steps = shard['batch_sizes'][0]
    dropped = shard['dropped']
    print(shard)
    print('local batch size: %d, dropped %d' % (batch_size, dropped))
    es = ElasticState(args_opt.max_progress - dropped, args_opt.reload)
    schedule_cb = ElasticScheduleCallback(es, schedule)
    elastic_callbacks = [
        ElasticCallback(es, args_opt.global_batch_size),
        schedule_cb,
    ]

    progress = es._progress
    rank = current_rank()
    size = current_cluster_size()

    if progress == 0:
        load_pretrain_checkpoint_path = args_opt.load_pretrain_checkpoint_path
        if rank == 0:
            propose_new_size(size)  # init config server
    else:
        print('progress: %d, step: %d' % (progress, schedule_cb._step))
        load_pretrain_checkpoint_path = "step-%d.ckpt" % (schedule_cb._step)

    target = args_opt.device_target
    if target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE,
                            device_target="Ascend",
                            device_id=args_opt.device_id)
    elif target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
        if bert_net_cfg.compute_type != mstype.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            bert_net_cfg.compute_type = mstype.float32
    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    netwithloss = BertSquad(bert_net_cfg, True, 2, dropout_prob=0.1)

    if args_opt.do_train.lower() == "true":
        print("do train ...")
        do_shuffle = args_opt.train_data_shuffle.lower() == "true"
        print("do_shuffle: %s" % (do_shuffle))
        ds = create_squad_dataset(batch_size=batch_size,
                                  repeat_count=1,
                                  data_file_path=filenames,
                                  schema_file_path=args_opt.schema_file_path,
                                  do_shuffle=do_shuffle,
                                  device_num=device_num,
                                  rank=rank)
        print('ds = %s' % (ds))
        do_train(
            elastic_callbacks,
            ds,
            netwithloss,
            load_pretrain_checkpoint_path,
            save_finetune_checkpoint_path,
            epoch_num,
            distributed,
        )
        print('do_train finished')

        if args_opt.do_eval.lower() == "true":
            if save_finetune_checkpoint_path == "":
                load_finetune_checkpoint_dir = _cur_dir
            else:
                load_finetune_checkpoint_dir = make_directory(
                    save_finetune_checkpoint_path)
            load_finetune_checkpoint_path = LoadNewestCkpt(
                load_finetune_checkpoint_dir, ds.get_dataset_size(), epoch_num,
                "squad")

    if distributed:
        pass
        # kfops.finalize(args_opt.device_target)
        # print('kfops.finalize done')

    kungfu_nccl_finalize()
    print('Train Finished at step %d' % (schedule_cb._step))
    if rank == 0:
        save_checkpoint(netwithloss, "step-%d.ckpt" % (schedule_cb._step))


if __name__ == "__main__":
    log_pid(__file__ + ' started')
    set_seed(1)
    run_squad()
    print('%s finished' % (__file__))
    print('%s finished' % (__file__), file=sys.stderr)
