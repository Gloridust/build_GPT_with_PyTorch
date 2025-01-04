import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from tqdm import tqdm
import time

from tokenizer import Tokenizer
from model import GPTModel
from qa_dataset import QADataset

def train_model(model, train_loader, val_loader, optimizer, criterion,
                device, num_epochs, model_output_dir, writer):
    batch_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        time1 = time.time()
        model.train()
        
        for index, data in enumerate(tqdm(train_loader, file=sys.stdout, 
                                        desc=f"Train Epoch: {epoch}")):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(input_ids, attention_mask)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            
            writer.add_scalar('Loss/train', loss, batch_step)
            batch_step += 1
            
            # 每100步打印一次loss
            if index % 100 == 0 or index == len(train_loader) - 1:
                time2 = time.time()
                time_per_step = (time2 - time1) / (index + 0.0001)
                tqdm.write(
                    f"Step {index}, Epoch {epoch} - "
                    f"Loss: {loss:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
                    f"Time/step: {time_per_step:.4f}s"
                )
                
        # 验证
        model.eval()
        val_loss = validate_model(model, criterion, device, val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f"Validation Loss: {val_loss:.4f}, Epoch: {epoch}")
        
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(model_output_dir, "best.pt")
            print(f"Saving best model to {best_model_path}, epoch: {epoch}")
            torch.save(model.state_dict(), best_model_path)
            
        # 保存最新模型
        last_model_path = os.path.join(model_output_dir, "last.pt")
        print(f"Saving last model to {last_model_path}, epoch: {epoch}")
        torch.save(model.state_dict(), last_model_path)

def validate_model(model, criterion, device, val_loader):
    running_loss = 0.0
    with torch.no_grad():
        for data in tqdm(val_loader, file=sys.stdout, desc="Validating"):
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device) 
            labels = data['labels'].to(device)
            
            outputs, _ = model(input_ids, attention_mask)
            loss = criterion(outputs, labels.view(-1))
            running_loss += loss.item()
            
    return running_loss / len(val_loader)

def main():
    # 配置参数
    train_json_path = "data/train.json"
    val_json_path = "data/val.json"
    vocab_path = "data/vocab.json"
    max_length = 120
    epochs = 15
    batch_size = 128
    lr = 1e-4
    model_output_dir = "output"
    logs_dir = "logs"
    
    # 创建必要的目录
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 初始化tokenizer
    tokenizer = Tokenizer(vocab_path)
    
    # 模型参数
    model_param = {
        "d_model": 768,
        "d_ff": 2048,
        "d_k": 64,
        "d_v": 64,
        "n_layers": 6,
        "n_heads": 8,
        "max_pos": 1800,
        "device": device,
        "vocab_size": tokenizer.get_vocab_size()
    }
    
    # 初始化模型
    model = GPTModel(**model_param)
    model = model.to(device)
    
    # 数据加载
    print("Loading training data...")
    train_dataset = QADataset(train_json_path, tokenizer, max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    
    print("Loading validation data...")
    val_dataset = QADataset(val_json_path, tokenizer, max_length)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # 初始化优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    
    # 初始化tensorboard
    writer = SummaryWriter(logs_dir)
    
    # 开始训练
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=epochs,
        model_output_dir=model_output_dir,
        writer=writer
    )
    
    writer.close()

if __name__ == '__main__':
    main() 