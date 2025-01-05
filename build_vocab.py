import json
import os
import glob

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

def build_vocab(jsonl_dir, output_path):
    """
    从多个jsonl文件构建词表
    
    Args:
        jsonl_dir: jsonl文件所在目录
        output_path: 词表输出路径
    """
    # 读取所有文本
    texts = []
    
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
    
    # 拆分Token
    words = set()
    for t in texts:
        if not t:
            continue
        for word in t.strip():
            words.add(word)
    words = list(words)
    words.sort()
    
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
        
    print(f"\n词表构建完成,共 {len(id2word)} 个token")

def main():
    jsonl_dir = "data/jsonl"
    vocab_file = "data/vocab.json"
    
    # 确保jsonl目录存在
    if not os.path.exists(jsonl_dir):
        os.makedirs(jsonl_dir)
        print(f"已创建目录: {jsonl_dir}")
        print("请将jsonl文件放入该目录后重新运行")
        return
        
    build_vocab(jsonl_dir, vocab_file)

if __name__ == '__main__':
    main() 