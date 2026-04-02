import numpy as np
import os
import pickle
from tokenizer import Tokenizer  # 确保你的类名是对的

# 1. 初始化 Tokenizer
with open("../data/bpe_vocab.pkl", "rb") as f:
    vocab = pickle.load(f)
with open("../data/bpe_merges.pkl", "rb") as f:
    merges = pickle.load(f)
tokenizer = Tokenizer(vocab, merges)

# 2. 读取并分词
print("正在读取文本并分词...")
with open("../data/TinyStoriesV2-GPT4-train.txt", "r", encoding="utf-8") as f:
    text = f.read()

ids = tokenizer.encode(text)
ids_array = np.array(ids, dtype=np.uint16)

# 3. 保存到指定的报错路径
target_path = "../data/TinyStoriesV2-GPT4-train/token_ids.npy"
os.makedirs(os.path.dirname(target_path), exist_ok=True)
np.save(target_path, ids_array)

print(f"成功！文件已生成在: {target_path}")