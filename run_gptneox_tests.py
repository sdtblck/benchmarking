import argparse
import socket
import torch
from pathlib import Path
import random
import subprocess
import yaml
import re 
import json 

def get_unused_port():
    # sometimes we encounter the error "TCP address in use", and the killall script doesn't work that well
    # to avoid this, we get a new open port each time we run a test
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        addr = s.getsockname()
    return int(addr[1])


def get_num_gpus():
    if Path("/job/hostfile").exists():
        return sum(
            [
                int(line.strip().split("slots=")[-1])
                for line in open("/job/hostfile").readlines()
            ]
        )
    else:
        return torch.cuda.device_count()

def get_gpu_memory_mb():
    return torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)

def data_dir_is_shared(data_dir, verbose=False):
    if not Path("/job/hosts").exists():
        return True  # single node

    # try writing a random string to the file
    n = str(random.randint(1, 1000))
    pth = str((Path(data_dir) / ".tmp_testfile.txt").resolve())
    with open(pth, "w") as f:
        f.write(n)

    out = subprocess.stderr if verbose else subprocess.DEVNULL
    # read across all hosts
    output = [
        i
        for i in subprocess.check_output(
            f"pdsh -R ssh -w ^/job/hosts 'cat {pth}'", shell=True, stderr=out
        )
        .decode()
        .split("\n")
        if i
    ]
    output = [i.split(":")[-1].strip() for i in output]

    if (
        len(output) == len(open("/job/hosts").readlines())
        and len(list(set(output))) == 1
    ):
        ret = True
    else:
        ret = False

    try:
        Path(pth).unlink()
    except:
        pass
    
    return ret

MEMORY_REQ_MB = {"1B": 40536}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        "--outpath",
        type=str,
        help="A prefix path specifying where results of the tests will be saved as a .json file. \
                                                        The general save format will be: {prefix}_{n_gpus}_gpus_{test_name}.json",
        default="",
    )
    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        help="A shared directory accessible by all nodes to tokenize test pretraining data to. An error will be raised if this directory isn't shared.",
        default="/mnt/ssd-1/data",
    )

    parser.add_argument(
        "-t",
        "--tests",
        type=str,
        nargs="+",
        help="Which tests to run. Default is to run allreduce on a single node, and across all nodes",
        default=["1B"],
    )
    args = parser.parse_args()

    # validate args:
    assert data_dir_is_shared(args.data_dir), f"{args.data_dir} should be shared between nodes"
    for test in args.tests:
        path_to_config = Path("benchmarking_configs") / f"{test}.yml"
        assert path_to_config.exists(), f"{test}.yml does not exist"

    # clone gpt-neox if it doesn't exist
    if not (Path.home() / "gpt-neox").exists():
        subprocess.run(
            f"cd ~; git clone https://github.com/EleutherAI/gpt-neox", shell=True
        )
        subprocess.run("cd ~/gpt-neox/; bash tools/syncdir.sh '~/gpt-neox'", shell=True)

    # prepare data
    subprocess.run(f"bash ~/gpt-neox/tools/sync_cmd.sh 'cd ~/gpt-neox/; python3 prepare_data.py -d {args.data_dir}'", shell=True)

    for test in args.tests:
        
        # check we have enough gpu memory
        memory_req = MEMORY_REQ_MB.get(test, None)
        if memory_req is not None:
            mem = get_gpu_memory_mb()
            assert mem > memory_req, f"The test {test} requires {memory_req}MB of GPU RAM, but you only have {mem}MB available."

        test_config = Path("benchmarking_configs") / f"{test}.yml"
        tmp_path = Path.home() / "gpt-neox" / "configs" / "tmp.yml"
        log_path = (Path.home() / ".tmp_log.txt").resolve()
        save_path = f"{args.outpath}_{get_num_gpus()}_gpus_{test}.json".strip("_")

        # make runtime modifications to config
        data_dir = Path(args.data_dir)
        vocab_file = str(data_dir / "gpt2-vocab.json")
        merge_file = str(data_dir / "gpt2-merges.txt")
        assert Path(vocab_file).exists(), f"vocab file {vocab_file} does not exist"
        assert Path(merge_file).exists(), f"merge file {merge_file} does not exist"
        data_path = str(data_dir / "enron" / "enron_text_document")
        assert Path(f"{data_path}.idx").exists(), f".idx file {data_path}.idx does not exist"
        assert Path(f"{data_path}.bin").exists(), f".bin file {data_path}.bin does not exist"

        with open(test_config, 'r') as f:
            config = yaml.safe_load(f)
        
        config["data-path"] = data_path
        config["vocab-file"] = vocab_file
        config["merge-file"] = merge_file
        config["master_port"] = get_unused_port()

        with open(tmp_path, 'w') as f:
            yaml.dump(config, f)
        
        # run training
        subprocess.run(f"cd ~/gpt-neox/; ./deepy.py train.py configs/tmp.yml | tee {str(log_path)}", shell=True)

        # parse average TFLOPS from logs, save to json
        with open(log_path) as f:
            logs = f.read()
        
        flops = re.findall(r'approx flops per GPU: (.*?)TFLOPS', logs)
        flops = [float(f) for f in flops]
        avg_flops = sum(flops) / len(flops)
        print('\n\n')
        print(f'AVERAGE FLOPS: {avg_flops} TFLOPS')

        with open(save_path, 'w') as f:
            json.dump({"average_tflops": avg_flops, "all_tflops": flops}, f)

        print(f'saved to {save_path}')
