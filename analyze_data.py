import json
from tokenizer import Tokenizer
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import glob

def analyze_sequence_lengths(jsonl_dir, tokenizer):
    """分析数据集中序列的长度分布"""
    lengths = []
    
    # 获取所有jsonl文件
    jsonl_files = glob.glob(os.path.join(jsonl_dir, "*.jsonl"))
    
    if not jsonl_files:
        raise Exception(f"在 {jsonl_dir} 目录下没有找到jsonl文件！")
    
    print(f"找到以下jsonl文件:")
    for file in jsonl_files:
        print(f"- {os.path.basename(file)}")
        
    # 分析所有文件
    for file in jsonl_files:
        print(f"\n分析文件: {os.path.basename(file)}")
        with open(file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Analyzing data"):
                data = json.loads(line)
                question = data['question']
                answer = data['answer']
                
                # 计算问答对的总长度
                tokens, _ = tokenizer.encode(question, answer)
                lengths.append(len(tokens))
    
    lengths = np.array(lengths)
    
    # 计算统计信息
    percentiles = np.percentile(lengths, [50, 75, 90, 95, 99])
    
    print(f"\n序列长度统计:")
    print(f"最小长度: {lengths.min()}")
    print(f"最大长度: {lengths.max()}")
    print(f"平均长度: {lengths.mean():.2f}")
    print(f"中位数长度: {np.median(lengths):.2f}")
    print(f"标准差: {lengths.std():.2f}")
    print(f"\n分位数统计:")
    print(f"50%的序列长度 <= {percentiles[0]:.0f}")
    print(f"75%的序列长度 <= {percentiles[1]:.0f}")
    print(f"90%的序列长度 <= {percentiles[2]:.0f}")
    print(f"95%的序列长度 <= {percentiles[3]:.0f}")
    print(f"99%的序列长度 <= {percentiles[4]:.0f}")
    
    # 绘制长度分布直方图
    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, density=True)
    plt.axvline(percentiles[2], color='r', linestyle='--', label='90th percentile')
    plt.axvline(percentiles[3], color='g', linestyle='--', label='95th percentile')
    plt.xlabel('Sequence Length')
    plt.ylabel('Density')
    plt.title('Distribution of Sequence Lengths')
    plt.legend()
    plt.grid(True)
    plt.savefig('data/length_distribution.png')
    plt.close()
    
    return percentiles

def main():
    jsonl_dir = "data/jsonl"
    vocab_path = "data/vocab.json"
    
    # 确保jsonl目录存在
    if not os.path.exists(jsonl_dir):
        os.makedirs(jsonl_dir)
        print(f"已创建目录: {jsonl_dir}")
        print("请将jsonl文件放入该目录后重新运行")
        return
    
    # 初始化tokenizer
    tokenizer = Tokenizer(vocab_path)
    
    # 分析序列长度
    percentiles = analyze_sequence_lengths(jsonl_dir, tokenizer)
    
    # 建议的max_length设置
    suggested_max_length = int(percentiles[2])  # 使用90分位数
    print(f"\n建议的max_length设置: {suggested_max_length}")
    
    # 更新train.py中的max_length
    update_max_length = input("\n是否要更新train.py中的max_length? (y/n): ")
    if update_max_length.lower() == 'y':
        with open('train.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 替换max_length的值
        content = content.replace(
            'max_length = 128',  # 注意这里使用当前的默认值
            f'max_length = {suggested_max_length}'
        )
        
        with open('train.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"已更新train.py中的max_length为{suggested_max_length}")

if __name__ == "__main__":
    main() 