import subprocess
import os
from pathlib import Path
import argparse
import torch 
import traceback

MPI_HOME = os.getenv("MPI_HOME", "/usr/local/mpi")
CUDA_HOME = os.getenv("CUDA_HOME", "/usr/local/cuda")
NCCL_HOME = os.getenv("NCCL_HOME", "/usr")
TEST_META = {
    "allreduce": {"cmd": "mpirun --hostfile /job/hostfile ./build/all_reduce_perf -b 8 -e 2G -f 2", "ngpus": "local"},
    "allreduce_single_node": {"cmd": "./build/all_reduce_perf -b 8 -e 2G -f 2 -g 8", "ngpus": "all"},
}

def get_num_gpus():
    assert Path('/job/hostfile').exists()
    return sum([int(line.strip().split("slots=")[-1]) for line in open('/job/hostfile').readlines()])
    

parser = argparse.ArgumentParser()
parser.add_argument(
    "-o",
    "--outpath",
    type=str,
    help="A prefix path specifying where results of the tests will be saved as a .txt file. \
                                                       The general save format will be: {prefix}_{n_gpus}_gpus_{test_name}.txt",
    default="",
)
parser.add_argument(
    "-t",
    "--tests",
    type=str,
    nargs="+",
    help="Which tests to run. Default is to run allreduce on a single node, and across all nodes",
    default=["allreduce_single_node", "allreduce"],
)
args = parser.parse_args()

# validate arguments
for x in args.tests:
    assert x in TEST_META, f"test {x} not recognized."
outpath = Path(args.outpath)

# clone gpt-neox if it doesn't exist
if not (Path.home() / "gpt-neox").exists():
    subprocess.run(
        f"cd ~; git clone https://github.com/EleutherAI/gpt-neox", shell=True
    )

# assert a hostfile exists at /job/hostfile and /job/hosts
assert Path("/job/hostfile").exists(), "/job/hostfile does not exist"
assert Path("/job/hosts").exists(), "/job/hostsdoes not exist"

# clone tests repo in parallel across all nodes
subprocess.run(
    f"cd ~; bash ~/gpt-neox/tools/sync_cmd.sh 'git clone https://github.com/NVIDIA/nccl-tests/'", shell=True
)

# run makefile
subprocess.run(
    f"cd ~/nccl-tests; make -j8 MPI=1 MPI_HOME={str(MPI_HOME)} CUDA_HOME={str(CUDA_HOME)} NCCL_HOME={str(NCCL_HOME)}", shell=True
)

# sync build folder
subprocess.run(f"cd ~/nccl-tests; bash ~/gpt-neox/tools/syncdir.sh build", shell=True)

# make output path parent if it doesn't exists
outpath.parent.mkdir(exist_ok=True, parents=True)

for test in args.tests:
    cmd = test["cmd"]
    ngpus = torch.cuda.device_count() if test["ngpus"] == "local" else get_num_gpus()
    out_path = f"{str(out_path)}_{n_gpus}_gpus_{test}.txt"
    try:
        subprocess.run(f"{cmd} > {out_path}", shell=True)
    except Exception:
        # save traceback to outpath if something goes wrong
        exception = traceback.format_exc()
        with open(out_path, "w") as f:
            f.write(exception) 
