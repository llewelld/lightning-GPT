import os

os.environ["WORLD_SIZE"] = "4"
os.environ["PMI_SIZE"] = "4"

# MPI_LOCALRANKID
# Local sequential index of the process on the node
# See nowhere
local_rank = int(os.environ["MPI_LOCALRANKID"])

# PMI_RANK
# The relative process ID of the current process with mpirun
# See https://doku.lrz.de/job-farming-with-slurm-11481293.html#JobfarmingwithSLURM-Taskidentifier
# See https://github.com/intel/torch-ccl?tab=readme-ov-file#usage
global_rank = int(os.environ["PMI_RANK"])

# PMI_SIZE
world_size = int(os.environ["PMI_SIZE"])

# LOCAL_WORLD_SIZE
local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])

#os.environ["RANK"] = int(local_rank)
#os.environ["LOCAL_RANK"] = int(local_rank)
#os.environ["MPI_LOCALRANKID"] = int(local_rank)
#os.environ["PMI_RANK"] = int(local_rank)
#os.environ["SLURM_PROCID"] = int(local_rank)
#os.environ["ZE_AFFINITY_MASK"] = int(local_rank)
#os.environ["GLOBAL_RANK"] = int(global_rank)
#os.environ["CCL_LOCAL_RANK"] = int(local_rank)
#os.environ["LOCAL_WORLD_SIZE"] = int(local_world_size)
#os.environ["CCL_LOCAL_SIZE"] = int(local_world_size)
#os.environ["MPI_LOCALNRANKS"] = int(local_world_size)

os.environ["SLURM_PROCID"] = os.environ["PMI_RANK"]
os.environ["ZE_AFFINITY_MASK"] = str(local_rank)

# ZE_AFFINITY_MASK
# List of devices we want the process to see
# See https://www.intel.com/content/www/us/en/developer/articles/technical/flattening-gpu-tile-hierarchy.html
os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "FLAT"
#os.environ["SLURM_LOCALID"] = str(local_rank)
print("MPI: local_rank: {}".format(local_rank))
print("MPI: procid: {}".format(os.environ["SLURM_PROCID"]))

from argparse import ArgumentParser
from urllib.request import urlopen
from urllib.error import URLError

import lightning as L
import torch
from torch.utils.data import DataLoader

# Import intel_extension_for_pytorch
import intel_extension_for_pytorch as ipex

from lightning_gpt import callbacks, data, models
from lightning.pytorch.utilities import rank_zero_info

from xpuaccelerator import XPUAccelerator
from torch.distributed import init_process_group, destroy_process_group
import oneccl_bindings_for_pytorch
from lightning.pytorch.strategies import DDPStrategy

FILENAME = "shakespeare_input.txt"
URL = f"https://cs.stanford.edu/people/karpathy/char-rnn/{FILENAME}"

def main(args):

    try:
        if os.path.exists(FILENAME):
            with open(FILENAME, "r") as f:
                text = f.read()
        else:
            with urlopen(URL) as f:
                text = f.read()
    except URLError as e:
        print(f"Unable to retrieve file from the URL: {URL}. Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    init_process_group(backend='ccl')
    print("MPI: world size: {}".format(torch.distributed.get_world_size()))
    print("MPI: rank: {}".format(torch.distributed.get_rank()))


    train_dataset = data.CharDataset(text, args.block_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    GPT_class = None
    extra_kwargs = {}

    if args.model_type == "None":
        args.model_type = None

    if args.implementation == "mingpt":
        GPT_class = models.MinGPT
        extra_kwargs.update(
            dict(
                embd_pdrop=0.1,
                resid_pdrop=0.1,
                attn_pdrop=0.1,
            )
        )

    elif args.implementation == "nanogpt":
        GPT_class = models.NanoGPT
        extra_kwargs["dropout"] = 0.1

    else:
        raise ValueError(f"Unsupported implementation {args.implementation}")

    if args.strategy.startswith("deepspeed"):
        if GPT_class == models.MinGPT:
            GPT_class = models.DeepSpeedMinGPT
        elif GPT_class == models.NanoGPT:
            GPT_class = models.DeepSpeedNanoGPT
        else:
            raise ValueError(f"Implementation {args.implementation} not supported with DeepSpeed")
        extra_kwargs["offload"] = False

    elif args.strategy == "fsdp_native":
        if GPT_class == models.MinGPT:
            GPT_class = models.FSDPMinGPT
        elif GPT_class == models.NanoGPT:
            GPT_class = models.FSDPNanoGPT
        else:
            raise ValueError(f"Implementation {args.implementation} not supported with FSDP")

    callback_list = []

    if torch.xpu.is_available():
        ipex.set_fp32_math_mode(mode=ipex.FP32MathMode.FP32, device='xpu')
        torch.set_float32_matmul_precision("high")
        callback_list.append(callbacks.XPUMetricsCallback())

    model = GPT_class(
        vocab_size=train_dataset.vocab_size,
        block_size=train_dataset.block_size,
        model_type=args.model_type,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        weight_decay=0.1,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        **extra_kwargs,
    )

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError(
                f"The current torch version ({torch.__version__}) does not have support for compile."
                "Please install torch >= 1.14 or disable compile."
            )
        model = torch.compile(model)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        callback_list.append(callbacks.CUDAMetricsCallback())

    accelerator = XPUAccelerator()

    ddp = DDPStrategy(accelerator=accelerator, process_group_backend="ccl")
    print("MPI: device count: {}".format(torch.xpu.device_count()))
    print("MPI: device available: {}".format(torch.xpu.is_available()))
    print("MPI: current device: {}".format(torch.xpu.current_device()))

    trainer = L.Trainer.from_argparse_args(
        args,
        gradient_clip_val=1.0,
        callbacks=callback_list,
        #accelerator="xpu",
        strategy=ddp,
        enable_checkpointing=False,
    )

    trainer.fit(model, train_loader)

    if False:
        context = "Friends of my soul"  # Prime with something
        x = train_dataset.to_tokens(context, model.device)
        y = model.generate(x, max_new_tokens=1000, temperature=1.0, top_k=10)[0]
        rank_zero_info(train_dataset.from_tokens(y))

    #destroy_process_group()

if __name__ == "__main__":
    L.seed_everything(42)

    parser = ArgumentParser()
    parser = L.Trainer.add_argparse_args(parser)

    parser.add_argument("--model_type", default="gpt2", type=str)
    parser.add_argument("--n_layer", type=int)
    parser.add_argument("--n_head", type=int)
    parser.add_argument("--n_embd", type=int)
    parser.add_argument("--learning_rate", default=3e-4, type=float)
    parser.add_argument("--block_size", default=128, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--compile", default=None, choices=[None, "dynamo"])
    parser.add_argument("--implementation", default="mingpt", choices=["mingpt", "nanogpt"])
    args = parser.parse_args()

    main(args)
