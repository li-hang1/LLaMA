import torch
from torch.utils.data import Dataset, DataLoader
from tokenizer import Tokenizer
from utils import load_json

class TitleGenDataset(Dataset):
    def __init__(self, vocab_path, data_path, bos_id=2, sep_id=4, eos_id=3):
        """
        vocab_path: path to vocabulary file (vocab.txt)
        data_path: path to data.json, contain [ {"content": [...], "title": [...]}, ... ]
        """
        self.tokenizer = Tokenizer(vocab_path)
        self.data = load_json(data_path)
        self.bos_id = bos_id
        self.sep_id = sep_id
        self.eos_id = eos_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        content_title_dict = self.data[idx]
        content = self.tokenizer.text_to_ids(content_title_dict["content"])
        title = self.tokenizer.text_to_ids(content_title_dict["title"])

        input_ids = [self.bos_id] + content + [self.sep_id] + title
        target_idx = content + [self.sep_id] + title + [self.eos_id]

        content_lens = len(content) + 1

        return torch.tensor(input_ids), torch.tensor(target_idx), content_lens


def collate_fn(batch, pad_id=0):
    """
    batch: list of (input_ids, target_ids, content_lens)
    return: input_batch:[B, max_len], target_batch:[B, max_len], pad_mask:[B, max_len], content_lens:tuple [B]
    """
    input_ids, target_ids, content_lens = zip(*batch)

    # 每个样本的真实长度
    lengths = [len(x) for x in input_ids]
    max_len = max(lengths)

    batch_size = len(batch)

    input_batch = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    target_batch = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    pad_mask = torch.zeros((batch_size, max_len), dtype=torch.long)

    for i in range(batch_size):
        l = lengths[i]
        # 填充 input、target
        input_batch[i, :l] = input_ids[i]
        target_batch[i, :l] = target_ids[i]

        # attention_mask：真实token=1，<PAD>=0
        pad_mask[i, :l] = 1

    return {
        "input_ids": input_batch,        # [B, max_len]
        "target_ids": target_batch,      # [B, max_len]
        "pad_mask": pad_mask,            # [B, max_len], 用于 pad mask
        "content_lens": content_lens     # [B]
    }


if __name__ == "__main__":
    data = TitleGenDataset(vocab_path="sohu_data/vocab.txt", data_path="sohu_data/data.json")
    dataloader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)
    for pair in dataloader:
        input_ids, target_ids, pad_mask, content_lens = pair
        print(input_ids.shape, target_ids.shape, input_ids.dtype, target_ids.dtype, pad_mask.shape, content_lens)
        break

