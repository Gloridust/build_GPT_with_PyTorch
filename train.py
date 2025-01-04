import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from tqdm import tqdm
import time
import json

from tokenizer import Tokenizer
from model import GPTModel
from qa_dataset import QADataset
from utils import TrainingLogger

def get_device():
    """获取可用的设备,优先级: cuda > mps > cpu"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def train_model(model, train_loader, val_loader, optimizer, criterion,
                device, num_epochs, model_output_dir, writer, logger, start_epoch=0):
    """
    训练模型
    Args:
        start_epoch: 开始训练的epoch,用于恢复训练
    """
    batch_step = 0
    best_val_loss = logger.best_val_loss  # 从logger获取最佳loss
    
    for epoch in range(start_epoch, num_epochs):
        time1 = time.time()
        model.train()
        
        epoch_train_loss = 0.0
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
            epoch_train_loss += loss.item()
            
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
                
        # 计算平均训练loss
        avg_train_loss = epoch_train_loss / len(train_loader)
        
        # 验证
        model.eval()
        val_loss = validate_model(model, criterion, device, val_loader)
        writer.add_scalar('Loss/val', val_loss, epoch)
        print(f"Validation Loss: {val_loss:.4f}, Epoch: {epoch}")
        
        # 记录训练信息
        logger.log_epoch(
            epoch=epoch,
            train_loss=avg_train_loss,
            val_loss=val_loss,
            lr=optimizer.param_groups[0]['lr']
        )
        
        # 保存checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss
        }
        
        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = os.path.join(model_output_dir, "best.pt")
            print(f"Saving best model to {best_model_path}, epoch: {epoch}")
            torch.save(checkpoint, best_model_path)
            
        # 保存最新模型
        last_model_path = os.path.join(model_output_dir, "last.pt")
        print(f"Saving last model to {last_model_path}, epoch: {epoch}")
        torch.save(checkpoint, last_model_path)

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

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """
    加载checkpoint
    Args:
        device: 设备对象，用于确保加载的状态在正确的设备上
    """
    if not os.path.exists(checkpoint_path):
        return None, 0, float('inf')
        
    print(f"Loading checkpoint from {checkpoint_path}")
    # 根据设备加载checkpoint
    if device.type == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 确保优化器状态在正确的设备上
    optimizer_state = checkpoint['optimizer_state_dict']
    for state in optimizer_state['state'].values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    optimizer.load_state_dict(optimizer_state)
    
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    
    return model, start_epoch, best_val_loss

def pretrain_identity(model, tokenizer, device, model_output_dir, max_length):
    """预训练身份数据"""
    # 读取身份数据
    identity_file = "data/identity_data.json"
    with open(identity_file, "r", encoding='utf-8') as f:
        identity_data = json.load(f)["identity_data"]
    
    # 创建身份数据的数据集
    identity_dataset = QADataset(None, tokenizer, max_length)
    identity_dataset.data = identity_data
    
    identity_loader = DataLoader(
        identity_dataset,
        batch_size=len(identity_data),  # 一次性处理所有身份数据
        shuffle=True
    )
    
    # 初始化优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    
    # 预训练100轮
    print("Pretraining identity data...")
    model.train()
    for epoch in tqdm(range(100), desc="Identity pretraining"):
        for data in identity_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(input_ids, attention_mask)
            loss = criterion(outputs, labels.view(-1))
            loss.backward()
            optimizer.step()
            
        if (epoch + 1) % 10 == 0:
            tqdm.write(f"Identity pretraining epoch {epoch+1}, loss: {loss.item():.4f}")
    
    # 保存预训练的模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(model_output_dir, "identity_pretrained.pt"))

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
    
    # 添加恢复训练参数
    resume_training = True  # 是否恢复训练
    checkpoint_path = os.path.join(model_output_dir, "best.pt")  # 默认从best模型恢复
    
    # 创建必要的目录
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # 设备配置
    device = get_device()
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
    
    # 预训练身份数据
    if not os.path.exists(os.path.join(model_output_dir, "identity_pretrained.pt")):
        pretrain_identity(model, tokenizer, device, model_output_dir, max_length)
    else:
        # 加载预训练的身份数据
        identity_checkpoint = torch.load(os.path.join(model_output_dir, "identity_pretrained.pt"))
        model.load_state_dict(identity_checkpoint['model_state_dict'])
        print("Loaded pretrained identity model")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    
    # 尝试加载checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    
    if resume_training and os.path.exists(checkpoint_path):
        model, start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, checkpoint_path, device)
        print(f"Resuming training from epoch {start_epoch}")
    
    # 确保所有模型参数都在正确的设备上
    model = model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    
    # 数据加载
    print("Loading training data...")
    train_dataset = QADataset(train_json_path, tokenizer, max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
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
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).to(device)
    
    # 初始化tensorboard
    writer = SummaryWriter(logs_dir)
    
    # 初始化logger(使用已有的best_val_loss)
    logger = TrainingLogger(logs_dir, best_val_loss)
    
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
        writer=writer,
        logger=logger,
        start_epoch=start_epoch  # 添加start_epoch参数
    )
    
    writer.close()

if __name__ == '__main__':
    main() 