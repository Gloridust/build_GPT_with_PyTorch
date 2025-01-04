import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

class TrainingLogger:
    def __init__(self, log_dir, best_val_loss=float('inf')):
        self.log_dir = log_dir
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.best_val_loss = best_val_loss
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建训练记录文件
        self.log_file = os.path.join(log_dir, f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        # 如果是恢复训练，记录初始状态
        if best_val_loss != float('inf'):
            log_data = {
                'resume_training': True,
                'initial_best_val_loss': best_val_loss
            }
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data) + '\n')
        
    def log_epoch(self, epoch, train_loss, val_loss, lr):
        """记录每个epoch的信息"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # 更新最佳验证loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
        
        # 保存训练记录
        log_data = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': lr,
            'best_val_loss': self.best_val_loss
        }
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_data) + '\n')
        
        # 绘制loss趋势图
        self.plot_losses()
        
    def plot_losses(self):
        """绘制loss趋势图"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label='Train Loss')
        plt.plot(self.epochs, self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # 保存图片
        plt.savefig(os.path.join(self.log_dir, 'loss_plot.png'))
        plt.close() 