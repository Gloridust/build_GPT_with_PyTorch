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
    
    print(f"\n建议的词表大小: {suggested_size}")
    proceed = input(f"是否使用建议的词表大小? (y/n): ")
    
    if proceed.lower() == 'y':
        max_vocab_size = suggested_size
    else:
        # 如果不使用建议值，让用户输入自定义大小
        custom_size = input(f"请输入自定义词表大小 ({min_vocab_size}-{max_vocab_size}): ")
        try:
            custom_size = int(custom_size)
            if min_vocab_size <= custom_size <= max_vocab_size:
                max_vocab_size = custom_size
            else:
                print(f"输入值超出范围，使用默认值: {max_vocab_size}")
        except ValueError:
            print(f"输入无效，使用默认值: {max_vocab_size}")
    
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
    if not word_freq:
        print("警告：词频字典为空")
        return min_size
    
    total_words = len(word_freq)
    total_freq = sum(word_freq.values())
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    # 如果总词数小于最小词表大小，直接返回总词数
    if total_words < min_size:
        print(f"警告：总词数({total_words})小于最小词表大小({min_size})")
        return total_words
    
    # 计算不同大小的覆盖率
    sizes = []
    coverages = []
    
    # 确保至少有两个点以计算增益
    step_size = min(1000, (max_size - min_size) // 10)
    if step_size < 1:
        step_size = 1
    
    for size in range(min_size, min(max_size + 1, total_words + 1), step_size):
        cumsum = sum(freq for _, freq in sorted_words[:size])
        coverage = cumsum / total_freq
        sizes.append(size)
        coverages.append(coverage)
        
        # 如果达到目标覆盖率，可以停止
        if coverage >= target_coverage:
            break
    
    # 如果没有足够的点来计算拐点，返回最小值
    if len(sizes) < 2:
        print("警告：数据点不足以计算拐点")
        return min_size
    
    # 计算覆盖率增益
    coverages = np.array(coverages)
    sizes = np.array(sizes)
    gains = np.diff(coverages) / np.diff(sizes)
    
    # 如果增益数组为空，返回最小值
    if len(gains) == 0:
        print("警告：无法计算覆盖率增益")
        return min_size
    
    # 找到覆盖率收益开始变缓的拐点
    elbow_idx = np.argmin(gains)
    suggested_size = sizes[elbow_idx]
    
    # 打印详细信息
    print("\n词表大小优化分析:")
    print(f"总词数: {total_words}")
    print(f"最小词表大小: {min_size}")
    print(f"最大词表大小: {max_size}")
    print(f"目标覆盖率: {target_coverage:.2%}")
    print(f"建议词表大小: {suggested_size}")
    print(f"对应覆盖率: {coverages[elbow_idx]:.2%}")
    
    # 绘制覆盖率曲线
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(sizes, coverages, 'b-', label='Coverage')
        plt.axvline(suggested_size, color='r', linestyle='--', label='Suggested Size')
        plt.xlabel('Vocabulary Size')
        plt.ylabel('Coverage')
        plt.title('Vocabulary Size vs Coverage')
        plt.grid(True)
        plt.legend()
        plt.savefig('data/vocab_coverage.png')
        plt.close()
        print("覆盖率曲线已保存到 data/vocab_coverage.png")
    except Exception as e:
        print(f"警告：无法绘制覆盖率曲线: {str(e)}")
    
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