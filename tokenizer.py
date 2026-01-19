class Tokenizer:
    def __init__(self, vocab_file, unk_token="<UNK>"):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_list = [line.rstrip('\n') for line in f]

        self.token2id = {token: idx for idx, token in enumerate(vocab_list)}
        self.id2token = {idx: token for idx, token in enumerate(vocab_list)}
        self.unk_id = self.token2id.get(unk_token, 0)

    def text_to_ids(self, text):
        ids = [self.token2id.get(c, self.unk_id) for c in text]
        return ids

    def ids_to_text(self, ids):
        return "".join([self.id2token.get(i, "[UNK]") for i in ids])

if __name__ == "__main__":
    tokenizer = Tokenizer("sohu_data/vocab.txt")
    text = ""
    ids = tokenizer.text_to_ids(text)
    print(ids)


