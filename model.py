import torch, os, swanlab
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
import numpy as np
from typing import Optional, Dict, Any
from datetime import datetime
from utils import *


class ResidualBlock(nn.Module):
    """
    残差块，包含两个全连接层和跳跃连接
    """
    def __init__(self, input_dim, hidden_dim, output_dim,dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 如果输入输出维度不同，添加线性映射
        self.shortcut = nn.Identity()
        if input_dim != output_dim:
            self.shortcut = nn.Linear(input_dim, output_dim)
            
    def forward(self, x):
        """
        前向传播
        """
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        
        # 跳跃连接
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResidualNetwork(nn.Module):
    """
    残差网络模型
    输入：irt信息 + 文本嵌入
    输出：irt数据与目标值的残差
    """
    def __init__(self, 
                 irt_data_dim: int,
                 text_embedding_dim: int,
                 output_dim: int,
                 proj_dim: int = 256,
                 hidden_dim: int = 64,
                 num_blocks: int = 16,
                 dropout: float = 0.1):
        """
        初始化残差网络
        
        Args:
            irt_data_dim: irt相关数据的维度
            text_embedding_dim: 文本嵌入的维度
            hidden_dim: 隐藏层维度
            num_blocks: 残差块数量
            dropout: dropout 概率
        """
        super(ResidualNetwork, self).__init__()
        
        self.irt_data_dim = irt_data_dim
        self.text_embedding_dim = text_embedding_dim
        self.proj_dim = proj_dim
        self.output_dim = output_dim
        
        # 投影层
        self.irt_layer = nn.Linear(self.irt_data_dim, self.proj_dim)
        self.embedding_layer = nn.Linear(self.text_embedding_dim, self.proj_dim)

        # 输入层
        self.input_layer = nn.Linear(2*self.proj_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        
        # 残差块堆叠
        self.residual_blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.residual_blocks.append(
                ResidualBlock(hidden_dim, hidden_dim * 2, hidden_dim, dropout)
            )
        
        # 输出层，预测残差
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, irt_data: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            irt_data: irt数据 [batch_size, irt_data_dim]
            text_embedding: 文本嵌入 [batch_size, text_embedding_dim]
            
        Returns:
            residual: 预测的残差 [batch_size, irt_data_dim]
        """
        # 投影irt数据和文本嵌入后拼接
        x = torch.cat([self.irt_layer(irt_data), self.embedding_layer(text_embedding)], dim=1)
        
        # 输入层
        x = F.relu(self.input_bn(self.input_layer(x)))
        
        # 通过残差块
        for block in self.residual_blocks:
            x = block(x)
        
        # 输出残差
        residual = self.output_layer(x)
        
        return residual
    
    def predict(self, irt_data: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        """
        预测校正后的值 = irt数据 + 残差
        
        Args:
            irt_data: irt数据 [batch_size, irt_data_dim]
            text_embedding: 文本嵌入 [batch_size, text_embedding_dim]
            
        Returns:
            corrected: 校正后的值 [batch_size, irt_data_dim]
        """
        residual = self.forward(irt_data, text_embedding)
        corrected = irt_data[:,10:] + residual
        return corrected


class ResidualLoss(nn.Module):
    """
    残差网络损失函数
    包含：
    1. 残差预测的 MSE 损失
    2. 校正后值的 MSE 损失
    3. L2 正则化项
    """
    def __init__(self, 
                 alpha: float = 0.5,
                 beta: float = 0.0001,
                 reduction: str = 'mean'):
        """
        初始化损失函数
        
        Args:
            alpha: 残差损失和校正损失的权重平衡参数
            beta: L2 正则化系数
            reduction: 损失缩减方式
        """
        super(ResidualLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse_loss = nn.MSELoss(reduction=reduction)
        self.reduction = reduction
        
    def forward(self, 
                predicted_residual: torch.Tensor,
                predicted_corrected: Optional[torch.Tensor],
                target: torch.Tensor,
                irt_data: torch.Tensor,
                model: nn.Module) -> torch.Tensor:
        """
        计算损失
        
        Args:
            predicted_residual: 预测的残差 [batch_size, irt_data_dim]
            predicted_corrected: 预测的校正后值 [batch_size, irt_data_dim]
            target: 目标值 [batch_size, irt_data_dim]
            irt_data: irt数据 [batch_size, irt_data_dim]
            model: 模型实例，用于计算 L2 正则化
            
        Returns:
            loss: 总损失
        """
        # 真实残差
        true_residual = target - irt_data[:, 10:]
        
        # 残差预测损失
        # print(predicted_residual.shape)
        # print(true_residual.shape)
        # exit()
        residual_loss = self.mse_loss(predicted_residual, true_residual)
        
        # 校正后值的损失
        if predicted_corrected is not None:
            correction_loss = self.mse_loss(predicted_corrected, target)
        else:
            correction_loss = 0.0
        
        # 加权组合
        base_loss = (1 - self.alpha) * residual_loss + self.alpha * correction_loss
        
        # L2 正则化
        l2_reg = torch.tensor(0.0, device=predicted_residual.device)
        for param in model.parameters():
            l2_reg += torch.norm(param) ** 2
        l2_reg = self.beta * l2_reg
        
        total_loss = base_loss + l2_reg
        
        return total_loss


class SwanLabTrainer:
    """
    集成 SwanLab 监控的残差网络训练器
    """
    def __init__(self,
                 model: ResidualNetwork,
                 loss_fn: ResidualLoss,
                 optimizer: torch.optim.Optimizer,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                 swanlab_config: Optional[Dict[str, Any]] = None,
                 checkpoint_dir: str = './checkpoints',
                 early_stopping_patience: int = 10,
                 save_best_only: bool = True,
                 eval_interval: int = 0):
        """
        初始化训练器
        
        Args:
            model: 残差网络模型
            loss_fn: 损失函数
            optimizer: 优化器
            device: 训练设备
            scheduler: 学习率调度器
            swanlab_config: SwanLab 配置字典，包含：
                - project: 项目名称
                - experiment_name: 实验名称
                - config: 实验配置参数
                - enabled: 是否启用 SwanLab（默认 True）
            checkpoint_dir: 模型检查点保存目录
            early_stopping_patience: 早停耐心值
            save_best_only: 是否只保存最佳模型
            eval_interval: 每隔多少 step（batch）进行一次评估。
                0 表示仅在每个 epoch 结束时评估（默认）；
                正整数表示按 step 间隔评估，此时 epoch 结束仍会追加一次评估。
        """
        self.model = model.to(device)
        self.eval_interval = eval_interval
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        # 训练监控
        self.swanlab_enabled = False
        if swanlab_config is not None:
            self._init_swanlab(swanlab_config)
        
        # 检查点保存
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # 早停机制
        self.early_stopping_patience = early_stopping_patience
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_mse': [],
            'val_mae': [],
            'lr': []
        }
        
    def _init_swanlab(self, config: Dict[str, Any]):
        """
        初始化 SwanLab
        
        Args:
            config: SwanLab 配置
        """
        try:
            import swanlab
            
            project = config.get('project', 'irt')
            experiment_name = config.get('experiment_name', None)
            exp_config = config.get('config', {})
            
            if experiment_name is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                experiment_name = f'residual_net_{timestamp}'
            
            swanlab.init(
                project=project,
                experiment_name=experiment_name,
                config={
                    **exp_config,
                    'model_type': 'ResidualNetwork',
                    'irt_data_dim': self.model.irt_data_dim,
                    'text_embedding_dim': self.model.text_embedding_dim,
                }
            )
            
            self.swanlab_enabled = True
            print(f"✓ SwanLab 初始化成功：{project}/{experiment_name}")
            
        except ImportError:
            print("⚠ 警告：未安装 swanlab，请运行 'pip install swanlab' 启用监控功能")
            self.swanlab_enabled = False
        except Exception as e:
            print(f"⚠ 警告：SwanLab 初始化失败：{e}")
            self.swanlab_enabled = False
    
    def _log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ''):
        """
        记录指标到 SwanLab
        
        Args:
            metrics: 指标字典
            step: 训练步数（epoch）
            prefix: 指标前缀
        """
        if self.swanlab_enabled:
            import swanlab
            log_dict = {f"{prefix}{k}": v for k, v in metrics.items()}
            swanlab.log(log_dict, step=step)
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        保存模型检查点
        
        Args:
            epoch: 当前 epoch
            metrics: 评估指标
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'history': self.history,
            'model_config': {
                'irt_data_dim': self.model.irt_data_dim,
                'text_embedding_dim': self.model.text_embedding_dim,
            }
        }
        
        # 保存最新模型
        if not self.save_best_only:
            checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"  ✓ 保存最佳模型：{best_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载模型检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
        print(f"✓ 加载检查点：{checkpoint_path} (epoch {checkpoint['epoch']})")
        return checkpoint['epoch']
    
    def train_epoch(self,
                    dataloader: DataLoader,
                    epoch: int,
                    global_step: int = 0,
                    val_dataloader: Optional[DataLoader] = None,
                    force_eval_at_end: bool = False) -> Dict[str, Any]:
        """
        训练一个 epoch
        
        Args:
            dataloader: 训练数据加载器
            epoch: 当前 epoch
            global_step: 进入本 epoch 前的全局 step 计数
            val_dataloader: 验证数据加载器（用于按 step 间隔评估）
            force_eval_at_end: epoch 结束时是否强制进行一次评估
            
        Returns:
            dict: 包含 avg_loss、最终 global_step、以及（可能的）val_metrics
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        last_val_metrics = None
        
        for batch_idx, batch in enumerate(dataloader):
            irt_data = batch[0].to(self.device)
            text_embedding = batch[1].to(self.device)
            target = batch[2].to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            predicted_residual = self.model(irt_data, text_embedding)
            # predicted_corrected = self.model.predict(irt_data, text_embedding)
            
            # 计算损失
            loss = self.loss_fn(
                predicted_residual,
                None,
                target,
                irt_data,
                self.model
            )
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            global_step += 1
            
            total_loss += loss.item()
            batch_count += 1
            
            # 每 10 个 batch 记录一次训练 loss
            if self.swanlab_enabled and (batch_idx + 1) % 10 == 0:
                self._log_metrics(
                    {'batch_loss': loss.item()},
                    step=global_step,
                    prefix='train/'
                )
            
            # 按 eval_interval 进行周期性评估
            if (self.eval_interval > 0
                    and val_dataloader is not None
                    and global_step % self.eval_interval == 0):
                val_metrics = self.evaluate(val_dataloader, global_step)
                last_val_metrics = val_metrics
                self._after_eval(val_metrics, global_step, epoch)
                self.model.train()
        
        # epoch 结束时的强制评估（若启用）
        if force_eval_at_end and val_dataloader is not None:
            # 避免与最后一次周期性评估重复
            if (self.eval_interval <= 0
                    or (global_step % self.eval_interval != 0)):
                val_metrics = self.evaluate(val_dataloader, global_step)
                last_val_metrics = val_metrics
                self._after_eval(val_metrics, global_step, epoch)
                self.model.train()
        
        avg_loss = total_loss / max(batch_count, 1)
        
        # 记录学习率
        current_lr = self.optimizer.param_groups[0]['lr']
        self.history['lr'].append(current_lr)
        
        return {
            'avg_loss': avg_loss,
            'global_step': global_step,
            'val_metrics': last_val_metrics,
        }
    
    def evaluate(self, dataloader: DataLoader, step: int) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            dataloader: 评估数据加载器
            step: 当前全局训练步数（用于日志记录）
            
        Returns:
            metrics: 评估指标字典
        """
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_mae = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                irt_data = batch[0].to(self.device)
                text_embedding = batch[1].to(self.device)
                target = batch[2].to(self.device)
                
                # 前向传播
                predicted_residual = self.model(irt_data, text_embedding)
                predicted_corrected = self.model.predict(irt_data, text_embedding)
                
                # 计算损失
                loss = self.loss_fn(
                    predicted_residual,
                    predicted_corrected,
                    target,
                    irt_data,
                    self.model
                )
                
                # 计算 MSE 和 MAE
                mse = F.mse_loss(predicted_corrected, target)
                mae = F.l1_loss(predicted_corrected, target)
                
                total_loss += loss.item()
                total_mse += mse.item()
                total_mae += mae.item()
                batch_count += 1
        
        metrics = {
            'loss': total_loss / batch_count,
            'mse': total_mse / batch_count,
            'mae': total_mae / batch_count
        }
        
        # 记录到 SwanLab
        self._log_metrics(metrics, step=step, prefix='val/')
        
        return metrics
    
    def _after_eval(self,
                    val_metrics: Dict[str, float],
                    step: int,
                    epoch: int) -> None:
        """
        评估完成后的处理：更新 history、保存检查点、调度 scheduler、早停等。
        该方法会在每个触发的评估点（无论是 epoch 末尾还是 step 间隔）被调用。
        """
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_mse'].append(val_metrics['mse'])
        self.history['val_mae'].append(val_metrics['mae'])
        
        # 检查是否为最佳模型
        is_best = val_metrics['loss'] < self.best_loss
        if is_best:
            self.best_loss = val_metrics['loss']
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # 保存检查点
        if self.save_best_only:
            self.save_checkpoint(epoch, val_metrics, is_best=is_best)
        
        # 调度学习率
        if self.scheduler is not None:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['loss'])
            else:
                self.scheduler.step()
        
        # 打印日志
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"Eval | Epoch {epoch} | Step {step} | "
              f"Val Loss: {val_metrics['loss']:.6f} | "
              f"Val MSE: {val_metrics['mse']:.6f} | "
              f"Val MAE: {val_metrics['mae']:.6f} | "
              f"LR: {current_lr:.6f}")
    
    def train(self,
              train_dataloader: DataLoader,
              val_dataloader: Optional[DataLoader] = None,
              epochs: int = 100,
              log_interval: int = 1):
        """
        完整训练流程
        
        Args:
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器（可选）
            epochs: 训练轮数
            log_interval: 日志打印间隔（以 epoch 为单位，用于无验证或无 step 级评估时的回退日志）
        """
        print(f"\n开始训练，共 {epochs} 轮")
        print(f"设备：{self.device}")
        print(f"SwanLab 监控：{'已启用' if self.swanlab_enabled else '未启用'}")
        if self.eval_interval > 0 and val_dataloader is not None:
            print(f"评估间隔：每 {self.eval_interval} 步评估一次")
        else:
            print("评估间隔：每个 epoch 结束时评估")
        print("-" * 60)
        
        global_step = 0
        
        for epoch in range(1, epochs + 1):
            # 训练一个 epoch（可能在内部按 step 间隔进行评估）
            epoch_result = self.train_epoch(
                dataloader=train_dataloader,
                epoch=epoch,
                global_step=global_step,
                val_dataloader=val_dataloader,
                force_eval_at_end=True,
            )
            global_step = epoch_result['global_step']
            train_loss = epoch_result['avg_loss']
            self.history['train_loss'].append(train_loss)
            
            # 无验证集时的基础日志
            if val_dataloader is None:
                if epoch % log_interval == 0:
                    print(f"Epoch {epoch:3d}/{epochs} | "
                          f"Train Loss: {train_loss:.6f} | "
                          f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
                self.save_checkpoint(epoch, {'loss': train_loss})
            else:
                # 使用 epoch 内最后一次 val_metrics 做 epoch 级别的汇总打印
                last_val = epoch_result.get('val_metrics')
                if last_val is not None and epoch % log_interval == 0:
                    print(f"Epoch {epoch:3d}/{epochs} | "
                          f"Train Loss: {train_loss:.6f} | "
                          f"Val Loss: {last_val['loss']:.6f} | "
                          f"Val MSE: {last_val['mse']:.6f} | "
                          f"Val MAE: {last_val['mae']:.6f} | "
                          f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 早停检查
            if self.patience_counter >= self.early_stopping_patience:
                print(f"\n早停触发：验证损失在 {self.early_stopping_patience} 次评估内未改善")
                break
        
        print("-" * 60)
        print("训练完成！")
        
        # 完成训练时关闭 SwanLab
        if self.swanlab_enabled:
            import swanlab
            swanlab.finish()
            print("✓ SwanLab 实验已关闭")
        
        return self.history


# 使用示例
if __name__ == '__main__':

    # 示例：创建模型和数据
    irt_data_dim = 12
    text_embedding_dim = 4096
    output_dim = 2
    
    # 创建模型
    model = ResidualNetwork(
        irt_data_dim=irt_data_dim,
        text_embedding_dim=text_embedding_dim,
        output_dim=output_dim,
        hidden_dim=64,
        num_blocks=3,
        dropout=0.1
    )
    
    # 创建损失函数
    loss_fn = ResidualLoss(alpha=0.5, beta=0.0001)
    
    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 创建学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # 创建示例数据
    my_dataset = MyDataset(train_irt_params_path='/mnt/fanzhaoji/DataCompress/IRT/20260729/unprocessed_training_irt_data.jsonl',
                                    item_embeddings_path='/mnt/fanzhaoji/DataCompress/IRT/Meta-Llama-3-8B_global_item_embeddings.npy',
                                    archor_count=5,
                                    
                                    )
    # dataloader = DataLoader(training_dataset,batch_size=5)
    
    train_size = int(0.9 * len(my_dataset))
    val_size = 10000
    test_size = len(my_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(my_dataset, [train_size, val_size, test_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False)


    # 创建训练器（带 SwanLab 监控）
    trainer = SwanLabTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        eval_interval=20,
        swanlab_config={
            'project': 'irt-residual-network',
            'experiment_name': 'residual_net_demo',
            'config': {
                'hidden_dim': 64,
                'num_blocks': 3,
                'dropout': 0.1,
                'batch_size': 32,
                'learning_rate': 0.001,
            }
        },
        checkpoint_dir='./checkpoints',
        early_stopping_patience=10,
        save_best_only=True
    )
    
    # 训练
    print("开始训练...")
    history = trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=100,
        log_interval=1
    )
    
    print("训练完成！")
    print(f"最佳验证损失：{trainer.best_loss:.6f}")