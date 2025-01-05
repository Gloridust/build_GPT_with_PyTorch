import torch
from torch.utils.data import Dataset
import json
import numpy as np

class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        if data_path:
            print(f"\n加载数据文件: {data_path}")
            with open(data_path, "r", encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    try:
                        if not line or line.isspace():
                            continue
                        json_line = json.loads(line)
                        
                        # 验证数据格式
                        if not isinstance(json_line, dict):
                            print(f"警告：第{i}行不是有效的JSON对象")
                            continue
                            
                        if "question" not in json_line or "answer" not in json_line:
                            print(f"警告：第{i}行缺少必要的字段")
                            print(f"数据内容: {json_line}")
                            continue
                            
                        question = json_line["question"]
                        answer = json_line["answer"]
                        
                        # 验证字段类型
                        if not isinstance(question, str) or not isinstance(answer, str):
                            print(f"警告：第{i}行的question或answer不是字符串类型")
                            continue
                            
                        self.data.append({
                            "question": question,
                            "answer": answer
                        })
                        
                    except json.JSONDecodeError as e:
                        print(f"警告：第{i}行JSON解析失败")
                        print(f"行内容: {line[:100]}...")
                        print(f"错误信息: {str(e)}")
                    except Exception as e:
                        print(f"警告：处理第{i}行时出错")
                        print(f"错误信息: {str(e)}")
                        
            print(f"数据加载完成,共 {len(self.data)} 条")
            
            # 如果没有成功加载任何数据，抛出异常
            if not self.data:
                raise Exception("没有加载到任何有效数据！请检查数据文件格式。")

    def preprocess(self, question, answer):
        encode, att_mask = self.tokenizer.encode(question, answer, 
                                               max_length=self.max_length,
                                               pad_to_max_length=True)
        input_ids = encode[:-1]  # 去掉最后一个token作为输入
        att_mask = att_mask[:-1]
        labels = encode[1:]  # 从第二个token开始作为标签
        return input_ids, att_mask, labels

    def __getitem__(self, index):
        item_data = self.data[index]
        input_ids, att_mask, labels = self.preprocess(**item_data)
        return {
            "input_ids": torch.LongTensor(np.array(input_ids)),
            "attention_mask": torch.LongTensor(np.array(att_mask)), 
            "labels": torch.LongTensor(np.array(labels))
        }

    def __len__(self):
        return len(self.data) 