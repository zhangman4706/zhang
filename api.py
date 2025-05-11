from fastapi import FastAPI, Query
from pydantic import BaseModel
import torch
import numpy as np
from poem import PoetryModel

# 初始化模型参数
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

# 初始化模型并加载已训练的权重
model = PoetryModel(params)
model.load("poetry_model.pth")
model.eval()

# 创建 FastAPI 实例
app = FastAPI(title="古诗生成API", description="用于生成古诗和藏头诗的接口", version="1.0")


# 随机生成古诗
@app.get("/generate")
def generate_poetry(length: int = Query(32, description="生成古诗的长度")):
    try:
        result = model.generate_auto(length)
        return {"poem": result}
    except Exception as e:
        return {"error": str(e)}


# 藏头诗生成
@app.get("/acrostic")
def generate_acrostic(head: str = Query(..., min_length=1, max_length=4, description="四个字的藏头")):
    try:
        heads = head[:4]
        result, puncts = "", ["，", "。", "，", "。"]
        for i, ch in enumerate(heads):
            try:
                idx = model.word_2_index[ch]
            except KeyError:
                idx = np.random.randint(0, model.word_size)
            result += model.index_2_word[idx]
            h, c = None, None
            for _ in range(6):
                x = torch.tensor(model.w1[idx][None][None]).to(model.device)
                output, (h, c) = model(x, h, c)
                idx = int(torch.argmax(output[-1]))
                result += model.index_2_word[idx]
            result += puncts[i]
        return {"acrostic_poem": result}
    except Exception as e:
        return {"error": str(e)}
#http://localhost:8000/docs
#uvicorn api:app --host 0.0.0.0 --port 8000