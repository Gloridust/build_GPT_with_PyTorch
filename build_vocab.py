import json
import os
import glob
import numpy as np

def extract_text(data):
    """
    从不同格式的数据中提取文本
    支持以下格式：
    1. {"question": "...", "answer": "..."}
    2. {"id": "...", "text": "...", "score": ...}
    """
    if isinstance(data, str):
        data = json.loads(data)
    
    if "question" in data and "answer" in data:
        # QA格式
        return [data["question"], data["answer"]]
    elif "text" in data:
        # 纯文本格式
        return [data["text"]]
    else:
        print(f"警告：未知的数据格式: {data.keys()}")
        return []

def build_vocab(jsonl_dir, output_path, min_vocab_size=5000, max_vocab_size=50000):
    """
    从多个jsonl文件构建词表
    
    Args:
        jsonl_dir: jsonl文件所在目录
        output_path: 词表输出路径
        min_vocab_size: 最小词表大小，默认5000
        max_vocab_size: 最大词表大小，默认50000
    """
    # 读取所有文本
    texts = []
    word_freq = {}  # 用于统计词频
    
    # 获取所有jsonl文件
    jsonl_files = glob.glob(os.path.join(jsonl_dir, "*.jsonl"))
    
    if not jsonl_files:
        raise Exception(f"在 {jsonl_dir} 目录下没有找到jsonl文件！")
    
    print(f"找到以下jsonl文件:")
    for file in jsonl_files:
        print(f"- {os.path.basename(file)}")
    
    # 读取所有文件的数据
    for file in jsonl_files:
        print(f"\n处理文件: {os.path.basename(file)}")
        with open(file, 'r', encoding='utf-8') as r:
            for line in r:
                if not line:
                    continue
                try:
                    # 提取文本
                    file_texts = extract_text(line)
                    texts.extend(file_texts)
                except json.JSONDecodeError:
                    print(f"警告：无法解析JSON行: {line[:100]}...")
                except Exception as e:
                    print(f"警告：处理数据时出错: {str(e)}")
                    continue
    
    # 统计词频
    for t in texts:
        if not t:
            continue
        for word in t.strip():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # 按频率排序
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # 自动优化词表大小
    suggested_size = optimize_vocab_size(
        word_freq, 
        min_size=min_vocab_size,
        max_size=max_vocab_size
    )
    
    print(f"建议的词表大小: {suggested_size}")
    proceed = input(f"是否使用建议的词表大小? (y/n): ")
    
    if proceed.lower() == 'y':
        max_vocab_size = suggested_size
    
    # 选择最常见的词构建词表（保留一定数量的位置给特殊token）
    max_words = max_vocab_size - 3  # 减去特殊token的数量
    words = [word for word, _ in sorted_words[:max_words]]
    words.sort()  # 按字典序排序
    
    # 特殊Token
    word2id = {
        "<pad>": 0,  # 填充
        "<unk>": 1,  # 未知
        "<sep>": 2   # 分隔/结束
    }
    
    # 构建词表
    word2id.update({word: i + len(word2id) for i, word in enumerate(words)})
    id2word = list(word2id.keys())
    
    vocab = {
        "word2id": word2id,
        "id2word": id2word
    }
    
    # 保存词表
    vocab_json = json.dumps(vocab, ensure_ascii=False, indent=2)
    with open(output_path, 'w', encoding='utf-8') as w:
        w.write(vocab_json)
    
    print(f"\n词表统计:")
    print(f"总词数: {len(word_freq)}")
    print(f"选取词数: {len(id2word)}")
    print(f"覆盖率: {sum(freq for word, freq in sorted_words[:max_words]) / sum(word_freq.values()):.2%}")

def optimize_vocab_size(word_freq, min_size=5000, max_size=50000, target_coverage=0.95):
    """
    优化词表大小的选择
    """
    total_words = len(word_freq)
    total_freq = sum(word_freq.values())
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # 计算不同大小的覆盖率
    sizes = []
    coverages = []
    for size in range(min_size, max_size + 1, 1000):
        cumsum = sum(freq for _, freq in sorted_words[:size])
        coverage = cumsum / total_freq
        sizes.append(size)
        coverages.append(coverage)
        
        # 如果达到目标覆盖率，可以停止
        if coverage >= target_coverage:
            break
    
    # 找到覆盖率收益开始变缓的拐点
    gains = np.diff(coverages) / np.diff(sizes)
    elbow_idx = np.argmin(gains) + 1
    suggested_size = sizes[elbow_idx]
    
    return suggested_size

def main():
    jsonl_dir = "data/jsonl"
    vocab_file = "data/vocab.json"
    
    # 确保jsonl目录存在
    if not os.path.exists(jsonl_dir):
        os.makedirs(jsonl_dir)
        print(f"已创建目录: {jsonl_dir}")
        print("请将jsonl文件放入该目录后重新运行")
        return
        
    build_vocab(jsonl_dir, vocab_file, min_vocab_size=5000, max_vocab_size=50000)

if __name__ == '__main__':
    main() 