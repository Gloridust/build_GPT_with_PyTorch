import json
import os
import glob

def load_jsonl_files(jsonl_dir):
    """
    加载指定目录下的所有jsonl文件
    
    Args:
        jsonl_dir: jsonl文件所在目录
    Returns:
        data: 所有数据的列表
    """
    data = []
    # 获取目录下所有的jsonl文件
    jsonl_files = glob.glob(os.path.join(jsonl_dir, "*.jsonl"))
    
    if not jsonl_files:
        raise Exception(f"在 {jsonl_dir} 目录下没有找到jsonl文件！")
    
    print(f"找到以下jsonl文件:")
    for file in jsonl_files:
        print(f"- {os.path.basename(file)}")
    
    # 读取所有文件的数据
    for file in jsonl_files:
        print(f"\n处理文件: {os.path.basename(file)}")
        with open(file, "r", encoding='utf-8') as f:
            for line in f:
                if not line or line == "":
                    continue
                data.append(line)
        print(f"已读取 {len(data)} 条数据")
    
    return data

def split_dataset(jsonl_dir, output_dir, val_ratio=0.1):
    """
    将原始数据集分割成训练集和验证集
    
    Args:
        jsonl_dir: jsonl文件所在目录
        output_dir: 输出目录
        val_ratio: 验证集占总数据的比例，默认0.1
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 读取所有jsonl文件的数据
    data = load_jsonl_files(jsonl_dir)
    
    # 读取身份数据
    identity_file = os.path.join(output_dir, "identity_data.json")
    with open(identity_file, "r", encoding='utf-8') as f:
        identity_data = json.load(f)["identity_data"]
    
    # 将身份数据添加到训练集的开头，增加权重
    identity_lines = [json.dumps(item, ensure_ascii=False) + "\n" for item in identity_data]
    # 重复添加多次以增加权重
    for _ in range(10):  # 重复10次
        data = identity_lines + data
    
    total_size = len(data)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    print(f"\n数据集统计:")
    print(f"总数据量: {total_size}")
    print(f"训练集大小: {train_size}")
    print(f"验证集大小: {val_size}")
    
    # 分割数据
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # 保存训练集
    train_file = os.path.join(output_dir, "train.json")
    with open(train_file, "w", encoding="utf-8") as f:
        f.writelines(train_data)
    print(f"\n训练集已保存: {train_file}, 共 {len(train_data)} 条数据")
    
    # 保存验证集
    val_file = os.path.join(output_dir, "val.json")
    with open(val_file, "w", encoding="utf-8") as f:
        f.writelines(val_data)
    print(f"验证集已保存: {val_file}, 共 {len(val_data)} 条数据")

def convert_to_qa_format(data):
    """将不同格式的数据转换为QA格式"""
    if isinstance(data, str):
        data = json.loads(data)
    
    if "question" in data and "answer" in data:
        # 已经是QA格式
        return data
    elif "text" in data:
        # 将长文本分割成多个段落作为问答对
        text = data["text"]
        # 简单的分段策略：按句号分割
        sentences = text.split("。")
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if len(sentences) >= 2:
            # 使用第一句作为问题，其余作为答案
            return {
                "question": sentences[0] + "。",
                "answer": "。".join(sentences[1:]) + "。"
            }
        else:
            # 如果文本太短，使用默认模式
            return {
                "question": "请简要概括这段文字的内容。",
                "answer": text
            }
    else:
        print(f"警告：未知的数据格式: {data.keys()}")
        return None

def main():
    jsonl_dir = "data/jsonl"
    output_dir = "data"
    
    # 确保jsonl目录存在
    if not os.path.exists(jsonl_dir):
        os.makedirs(jsonl_dir)
        print(f"已创建目录: {jsonl_dir}")
        print("请将jsonl文件放入该目录后重新运行")
        return
    
    split_dataset(jsonl_dir, output_dir)

if __name__ == "__main__":
    main() 