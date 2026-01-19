from utils import load_yaml
import torch
import os

from LLaMA import LLaMA
from dataset import Tokenizer


config_path = "configs/config.yaml"
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(config):
    model_config = config["model"]
    model = LLaMA(**model_config).to(device)

    ckpt_path = config["pretrained"]
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Loaded pretrained weights from {ckpt_path}")
    print(f"epoch: {ckpt['epoch']}")
    print(f"val_loss: {ckpt['best_val']}")
    return model

@torch.no_grad()
def generate_title(model, tokenizer, content_text, max_new_tokens=500, temperature=1.0, top_k=0, top_p=1.0):

    bos_id, sep_id, eos_id = 2, 4, 3

    content_ids = tokenizer.text_to_ids(content_text)

    input_ids = [bos_id] + content_ids + [sep_id]
    content_len = len(input_ids)

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    content_lens = torch.tensor([content_len], device=device)

    for _ in range(max_new_tokens):
        logits = model(input_ids, content_lens)  # [1, seq_len, vocab_size]
        next_logits = logits[0, -1] / temperature  # temperature决定模型有多敢乱选，大于1拉平概率分布，小于1拉尖概率分布

        if top_k > 0:
            values, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < values[-1]] = -float("inf")

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)  # 从大到小
            probs = torch.softmax(sorted_logits, dim=-1)
            cum_probs = torch.cumsum(probs, dim=-1)  # 最右边一列肯定是1

            cutoff = cum_probs > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()  # 第一列不变，后面整体右移一列，最后一列没有了
            cutoff[..., 0] = False

            sorted_logits[cutoff] = -float("inf")
            next_logits = torch.zeros_like(next_logits).scatter(-1, sorted_indices, sorted_logits)

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # 按概率采样

        if next_token.item() == eos_id:
            break

        input_ids = torch.cat([input_ids, next_token.view(1, 1)], dim=1)

    title_ids = input_ids[0, content_len:].tolist()
    return tokenizer.ids_to_text(title_ids)


def main():
    config = load_yaml(config_path)

    tokenizer = Tokenizer(config["data"]["vocab_path"])
    model = load_model(config)

    print("输入正文，回车生成标题（输入 q 退出）\n")

    while True:
        text = input("正文：").strip()
        if text.lower() in {"q", "quit", "exit"}:
            break

        title = generate_title(model=model, tokenizer=tokenizer, content_text=text, max_new_tokens=500, temperature=1.0,
                               top_k=0, top_p=1.0)
        print(f"标题：{title}\n")

if __name__ == "__main__":
    main()
