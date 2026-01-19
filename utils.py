import json
import yaml
import torch

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_ckpt(path, model, optimizer):
    ckpt = torch.load(path, map_location="cuda")
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("epoch", 0), ckpt.get("best_val", float("inf"))

def compute_masked_loss(model, batch, criterion):
    input_ids = batch["input_ids"].cuda()
    target_ids = batch["target_ids"].cuda()
    pad_mask = batch["pad_mask"].cuda()
    content_lens = batch["content_lens"]

    logits = model(input_ids, content_lens)  # [B, max_len, vocab_size]
    B, max_len, vocab_size = logits.shape

    loss_all = criterion(logits.view(B*max_len, vocab_size), target_ids.view(B*max_len))
    loss_all = loss_all.reshape(B, max_len)

    title_mask = torch.zeros((B, max_len), dtype=torch.float32, device=logits.device)
    for i in range(B):
        c = content_lens[i]
        title_mask[i, c:] = 1.0

    final_mask = title_mask * pad_mask

    loss = (loss_all * final_mask).sum() / final_mask.sum()

    return loss

def run_one_epoch(model, dataloader, criterion, optimizer, epoch=0, train=True, log_interval=100, is_main=False):
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    step_count = 0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(dataloader):
            loss = compute_masked_loss(model=model, batch=batch, criterion=criterion)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            step_count += 1

            if train and is_main and (batch_idx+1) % log_interval == 0:
                print(f"Epoch: {epoch} batch_idx: {batch_idx+1} | Loss {loss.item():.4f}")

    return total_loss / step_count

