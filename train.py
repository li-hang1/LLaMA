from LLaMA import LLaMA
from dataset import TitleGenDataset, collate_fn
from utils import load_yaml, load_ckpt, run_one_epoch

from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch import nn
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import os


def main(config):

    # DDP init
    rank = int(os.environ.get('RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.cuda.set_device(local_rank)

    is_main = (rank == 0)

    train_dataset = TitleGenDataset(vocab_path=config["data"]["vocab_path"], data_path=config["data"]["train_data_path"])
    val_dataset = TitleGenDataset(vocab_path=config["data"]["vocab_path"], data_path=config["data"]["val_data_path"])
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], sampler=train_sampler, num_workers=20,
                                  pin_memory=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], sampler=val_sampler, num_workers=20,
                                pin_memory=True, collate_fn=collate_fn)
    model_config = config["model"]
    model = LLaMA(**model_config).cuda()
    model = DDP(model, device_ids=[local_rank])
    optimizer = AdamW(model.parameters(), lr=config["lr"], betas=(0.9,0.999))
    criterion = nn.CrossEntropyLoss(reduction='none')

    best_val = float("inf")
    start_epoch = 0

    ckpt_path = os.path.join(config["pretrained"], "best.pth")
    if config["resume"] and os.path.exists(ckpt_path):
        start_epoch, best_val = load_ckpt(ckpt_path, model, optimizer)
        if is_main:
            print(f"Resumed from {ckpt_path}, epoch={start_epoch}, best_val={best_val:.4f}")


    for epoch in range(start_epoch, config["epochs"]):
        train_sampler.set_epoch(epoch)

        train_loss = run_one_epoch(model, train_dataloader, criterion, optimizer, epoch, train=True, log_interval=1000,
                                   is_main=is_main)
        if is_main:
            print(f"Epoch: {epoch} | Train loss: {train_loss:.4f}")
            if (epoch+1) % config["eval_epoch_interval"] == 0:
                val_loss = run_one_epoch(model, val_dataloader, criterion, optimizer=None, train=False)
                print(f"Epoch: {epoch} | Val loss: {val_loss:.4f}")
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"model": model.module.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "epoch": epoch,
                                "best_val": best_val}, os.path.join(config["pretrained"], f"best.pth"))
                    print(f"Best validation model saved. Epoch: {epoch}, Validation loss: {best_val:.4f}")

    dist.destroy_process_group()

if __name__ == '__main__':
    config = load_yaml("configs/config.yaml")
    main(config)



