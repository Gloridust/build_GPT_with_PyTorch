import json
import os

def split_dataset(input_file, output_dir, train_size=10000, val_size=1000):
    """
    将原始数据集分割成训练集和验证集
    
    Args:
        input_file: 输入的jsonl文件路径
        output_dir: 输出目录
        train_size: 训练集大小
        val_size: 验证集大小
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # 读取数据
    data = []
    with open(input_file, "r", encoding='utf-8') as f:
        for line in f:
            if not line or line == "":
                continue
            data.append(line)
            
    # 添加身份数据
    identity_data = [
        {"question": "你是谁", "answer": "我是EthanGpt,一个简易的小助手"},
        {"question": "你叫什么", "answer": "我是EthanGpt,一个简易的小助手"},
        {"question": "你的名字是什么", "answer": "我是EthanGpt,一个简易的小助手"},
        {"question": "你叫啥", "answer": "我是EthanGpt,一个简易的小助手"},
        {"question": "你名字是啥", "answer": "我是EthanGpt,一个简易的小助手"},
        {"question": "你是什么身份", "answer": "我是EthanGpt,一个简易的小助手"},
        {"question": "你的全名是什么", "answer": "我是EthanGpt,一个简易的小助手"},
        {"question": "你自称什么", "answer": "我是EthanGpt,一个简易的小助手"},
        {"question": "你的称号是什么", "answer": "我是EthanGpt,一个简易的小助手"},
        {"question": "你的昵称是什么", "answer": "我是EthanGpt,一个简易的小助手"}
    ]
    
    for item in identity_data:
        data.append(json.dumps(item, ensure_ascii=False) + "\n")
    
    # 分割数据
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    
    # 保存训练集
    train_file = os.path.join(output_dir, "train.json")
    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(train_data)
    print(f"训练集已保存: {train_file}, 共 {len(train_data)} 条数据")
    
    # 保存验证集
    val_file = os.path.join(output_dir, "val.json")
    with open(val_file, "w", encoding="utf-8") as f:
        f.writelines(val_data)
    print(f"验证集已保存: {val_file}, 共 {len(val_data)} 条数据")

def main():
    input_file = "data/train.jsonl"
    output_dir = "data"
    split_dataset(input_file, output_dir)

if __name__ == "__main__":
    main() 