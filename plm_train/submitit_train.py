# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import uuid
from pathlib import Path

import main_train
import submitit


DEFAULT_JOB_DIR = "/scratch/07861/jozhang/logs/plm/%j"

def parse_args():
    train_parser = main_train.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for MAE finetune", parents=[train_parser], conflict_handler='resolve')
    parser.add_argument("--account", default='CGAI24022', type=str)
    parser.add_argument("--ngpus", default=1, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=32, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=2880, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default=DEFAULT_JOB_DIR, type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--partition", default="gh", type=str, help="Partition where to submit")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")
    return parser.parse_known_args()


def get_shared_folder() -> Path:
    return Path('/scratch/07861/jozhang/submitit')


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args, override_args):
        self.args = args
        self.override_args = override_args

    def __call__(self):
        import main_train

        self._setup_gpu_args()
        main_train.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args, self.override_args)
        empty_trainer = type(self)(self.args, self.override_args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.log_dir = self.args.output_dir
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        self.args.pretrain = False
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args, override_args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}

    exclude_nodes = 'c612-001,c612-002,c612-011,c612-012,c612-021,c612-022,c612-031,c612-032,c612-041,c612-042,c612-051,c612-052,c612-061,c612-062,c612-071,c612-072,c612-081,c612-082,c612-091,c612-092,c612-101,c612-102,c612-111,c612-112,c612-121,c612-122,c612-131,c612-132,c612-141,c612-142,c612-151,c612-152,c613-001,c613-002,c613-011,c613-012,c613-021,c613-022,c613-031,c613-032,c613-041,c613-042,c613-051,c613-052,c613-061,c613-062,c613-071,c613-072,c613-081,c613-082,c613-091,c613-092,c613-101,c613-102,c613-111,c613-112,c613-121,c613-122,c613-131,c613-132,c613-141,c613-142,c613-151,c613-152,c619-001,c619-002,c619-011,c619-012,c619-021,c619-022,c619-031,c619-032,c619-041,c619-042,c619-051,c619-052,c619-061,c619-062,c619-071,c619-072,c619-081,c619-082,c619-091,c619-092,c619-101,c619-102,c619-111,c619-112,c619-121,c619-122,c619-131,c619-132,c619-141,c619-142,c619-151,c619-152,c620-001,c620-002,c620-011,c620-012,c620-021,c620-022,c620-031,c620-032,c620-041,c620-042,c620-051,c620-052,c620-061,c620-062,c620-071,c620-072,c620-081,c620-082,c620-091,c620-092,c620-101,c620-102,c620-111,c620-112,c620-121,c620-122,c620-131,c620-132,c620-141,c620-142,c620-151,c620-152,c621-001,c621-002,c621-011,c621-012,c621-021,c621-022,c621-031,c621-032,c621-041,c621-042,c621-051,c621-052,c621-061,c621-062,c621-071,c621-072,c621-081,c621-082,c621-091,c621-092,c621-101,c621-102,c621-111,c621-112,c621-121,c621-122,c621-131,c621-132,c621-141,c621-142,c621-151,c621-152'
    exclude_nodes = 'c613-001,c613-002,c613-011,c613-012,c613-021,c613-022,c613-031,c613-032,c613-041,c613-042,c613-051,c613-052,c613-061,c613-062,c613-071,c613-072,c613-081,c613-082,c613-091,c613-092,c613-101,c613-102,c613-111,c613-112,c613-121,c613-122,c613-131,c613-132,c613-141,c613-142,c613-151,c613-152,c619-001,c619-002,c619-011,c619-012,c619-021,c619-022,c619-031,c619-032,c619-041,c619-042,c619-051,c619-052,c619-061,c619-062,c619-071,c619-072,c619-081,c619-082,c619-091,c619-092,c619-101,c619-102,c619-111,c619-112,c619-121,c619-122,c619-131,c619-132,c619-141,c619-142,c619-151,c619-152,c620-001,c620-002,c620-011,c620-012,c620-021,c620-022,c620-031,c620-032,c620-041,c620-042,c620-051,c620-052,c620-061,c620-062,c620-071,c620-072,c620-081,c620-082,c620-091,c620-092,c620-101,c620-102,c620-111,c620-112,c620-121,c620-122,c620-131,c620-132,c620-141,c620-142,c620-151,c620-152,c621-001,c621-002,c621-011,c621-012,c621-021,c621-022,c621-031,c621-032,c621-041,c621-042,c621-051,c621-052,c621-061,c621-062,c621-071,c621-072,c621-081,c621-082,c621-091,c621-092,c621-101,c621-102,c621-111,c621-112,c621-121,c621-122,c621-131,c621-132,c621-141,c621-142,c621-151,c621-152'
    exclude_nodes = 'c621-001,c621-002,c621-011,c621-012,c621-021,c621-022,c621-031,c621-032,c621-041,c621-042,c621-051,c621-052,c621-061,c621-062,c621-071,c621-072,c621-081,c621-082,c621-091,c621-092,c621-101,c621-102,c621-111,c621-112,c621-121,c621-122,c621-131,c621-132,c621-141,c621-142,c621-151,c621-152'
    exclude_nodes = 'c613-022,c610-001,c620-022'
    exclude_nodes = 'c620-091,c620-101,c621-031,c621-022,c611-081,c619-101,c620-131,c621-042,c621-072,c639-012,c621-002,c640-012,c621-021,c621-011,c621-032,c620-142,c620-122,c620-141'

    executor.update_parameters(
        # mem_gb=100 * num_gpus_per_node,
        # gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node, # one task per GPU
        #cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        slurm_additional_parameters={
            'account': args.account,
        },
        slurm_exclude=exclude_nodes,
        **kwargs
    )

    executor.update_parameters(name="plm")

    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args, override_args)
    job = executor.submit(trainer)

    print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()

