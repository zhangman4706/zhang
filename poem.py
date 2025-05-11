import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from gensim.models import Word2Vec
from torch.utils.data import Dataset, DataLoader

class PoetryDataset(Dataset):
    def __init__(self, w1, word_2_index, all_data):
        self.w1 = w1
        self.word_2_index = word_2_index
        self.all_data = all_data

    def __getitem__(self, index):
        poetry = self.all_data[index]
        indices = [self.word_2_index[i] for i in poetry]
        xs, ys = indices[:-1], indices[1:]
        return self.w1[xs], np.array(ys).astype(np.int64)

    def __len__(self):
        return len(self.all_data)


class PoetryModel(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.all_data, (self.w1, self.word_2_index, self.index_2_word) = self.prepare_data()
        self.word_size, self.embedding_dim = self.w1.shape

        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=params["hidden_num"],
                            batch_first=True, num_layers=2)
        self.linear = nn.Linear(params["hidden_num"], self.word_size)
        self.dropout = nn.Dropout(0.3)
        self.loss_fn = nn.CrossEntropyLoss()

        self.to(self.device)

    def prepare_data(self):
        file = "poetry_7.txt"
        param_file = "word_vec.pkl"
        #train_num = self.params["train_num"]

        if os.path.exists(param_file):
            all_data = open(file, "r", encoding="utf-8").read().split("\n")
            return all_data, pickle.load(open(param_file, "rb"))

        text = open(file, "r", encoding="utf-8").read()
        with open("split_7.txt", "w", encoding="utf-8") as f:
            f.write(" ".join(text))
        sentences = [list(line.strip()) for line in text.split("\n") if line.strip()]
        model = Word2Vec(sentences, vector_size=self.params["embedding_num"], min_count=1)
        pickle.dump([model.syn1neg, model.wv.key_to_index, model.wv.index_to_key], open(param_file, "wb"))
        return sentences, (model.syn1neg, model.wv.key_to_index, model.wv.index_to_key)

    def forward(self, x, h_0=None, c_0=None):
        if h_0 is None or c_0 is None:
            h_0 = torch.zeros(2, x.size(0), self.params["hidden_num"]).to(self.device)
            c_0 = torch.zeros(2, x.size(0), self.params["hidden_num"]).to(self.device)
        output, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        output = self.linear(self.dropout(output.contiguous().view(-1, self.params["hidden_num"])))
        return output, (h_n, c_n)

    def train_model(self, save_path="poetry_model.pth"):
        dataloader = DataLoader(PoetryDataset(self.w1, self.word_2_index, self.all_data),
                                batch_size=self.params["batch_size"], shuffle=True)
        optimizer = self.params["optimizer"](self.parameters(), lr=self.params["lr"])

        for epoch in range(self.params["epochs"]):
            for i, (x, y) in enumerate(dataloader):
                self.train()
                x, y = x.to(self.device), y.to(self.device)
                pred, _ = self(x)
                loss = self.loss_fn(pred, y.view(-1))
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                if i % self.params["batch_num_test"] == 0:
                    print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
                    print("生成示例：", self.generate_auto())

        self.save(save_path)

    def generate_auto(self, length=32):
        self.eval()
        idx = np.random.randint(0, self.word_size)
        word = self.index_2_word[idx]
        result = word
        h, c = None, None
        for _ in range(length - 1):
            x = torch.tensor(self.w1[idx][None][None]).to(self.device)
            output, (h, c) = self(x, h, c)
            idx = int(torch.argmax(output[-1]))
            result += self.index_2_word[idx]
        return result

    def generate_acrostic(self):
        self.eval()
        while True:
            heads = input("请输入四个汉字：")[:4]
            if not heads:
                print(self.generate_auto())
                continue
            result, puncts = "", ["，", "。", "，", "。"]
            for i, ch in enumerate(heads):
                try:
                    idx = self.word_2_index[ch]
                except KeyError:
                    idx = np.random.randint(0, self.word_size)
                result += self.index_2_word[idx]
                h, c = None, None
                for _ in range(6):
                    x = torch.tensor(self.w1[idx][None][None]).to(self.device)
                    output, (h, c) = self(x, h, c)
                    idx = int(torch.argmax(output[-1]))
                    result += self.index_2_word[idx]
                result += puncts[i]
            print(result)

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"模型已保存到 {path}")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.to(self.device)
        print(f"模型已从 {path} 加载")
        
    


# --------------------- main 入口 ---------------------
if __name__ == "__main__":
    params = {
        "batch_size": 32,
        "epochs": 300,
        "lr": 0.003,
        "hidden_num": 64,
        "embedding_num": 128,
        "train_num": 1000,
        "optimizer": torch.optim.AdamW,
        "batch_num_test": 100,
    }

    model = PoetryModel(params)
    model.train_model("poetry_model.pth")
    model.generate_acrostic()
