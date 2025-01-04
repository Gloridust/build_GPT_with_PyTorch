import json

def build_vocab(file_path, output_path):
    # 读取所有文本
    texts = []
    with open(file_path, 'r', encoding='utf-8') as r:
        for line in r:
            if not line:
                continue
            line = json.loads(line)
            question = line["question"]
            answer = line["answer"]
            texts.append(question)
            texts.append(answer)
            
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
        
    print(f"词表构建完成,共 {len(id2word)} 个token")

def main():
    train_file = "data/train.jsonl"
    vocab_file = "data/vocab.json"
    build_vocab(train_file, vocab_file)

if __name__ == '__main__':
    main() 