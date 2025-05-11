import torch
import numpy as np
from poem import PoetryModel  # 假设你的模型代码文件名是 poetry_model.py

def test_model_interactive(model_path="poetry_model.pth"):
    # 模型参数应和训练时一致
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

    # 初始化并加载模型
    model = PoetryModel(params)
    model.load(model_path)

    # 开始交互循环
    while True:
        print("\n================== 古诗生成 ==================")
        print("1. 自动生成古诗")
        print("2. 生成藏头诗")
        print("3. 退出程序")
        choice = input("请输入选项（1/2/3）：").strip()

        if choice == "1":
            print("\n【自动生成古诗】：")
            print(model.generate_auto())

        elif choice == "2":
            heads = input("\n请输入四个汉字作为藏头（例：春风化雨）：").strip()[:4]
            if not heads or len(heads) < 1:
                print("输入有误，请重新输入。")
                continue

            result = ""
            puncts = ["，", "。", "，", "。"]
            for i, ch in enumerate(heads):
                try:
                    idx = model.word_2_index[ch]
                except KeyError:
                    print(f"字 '{ch}' 不在词表中，使用随机字代替。")
                    idx = np.random.randint(0, model.word_size)
                result += model.index_2_word[idx]
                h, c = None, None
                for _ in range(6):
                    x = torch.tensor(model.w1[idx][None][None]).to(model.device)
                    output, (h, c) = model(x, h, c)
                    idx = int(torch.argmax(output[-1]))
                    result += model.index_2_word[idx]
                result += puncts[i]
            print("\n【生成结果】：")
            print(result)

        elif choice == "3":
            print("退出程序。再见！")
            break

        else:
            print("无效选项，请输入 1、2 或 3。")


if __name__ == "__main__":
    test_model_interactive("poetry_model.pth")
