import torch
import torch.nn as nn

class ProteinRegressor(nn.Module):
    def __init__(self, 
                 input_dim=480,    # 与config.py中的INPUT_DIM保持一致
                 num_heads=4,      # 从16降至8，减少显存占用
                 hidden_dim=256,   # 从1024降至512，减少显存占用
                 num_layers=3,     # 从6降至4，减少显存占用
                 output_dim=3):    # 预测3维Cα坐标（x,y,z）
        super().__init__()
        self.input_dim = input_dim
        
        # Transformer编码器层（适配GPU训练，优化显存）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,  # 关键：输入格式为(batch, seq_len, dim)，与训练数据匹配
            dropout=0.1,       # 添加dropout，防止过拟合同时减少部分显存占用
            activation="relu"  # 选用relu激活，计算效率更高
        )
        
        # Transformer编码器
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False  # 关闭嵌套张量，适配旧版本PyTorch
        )
        
        # 回归头（预测蛋白质Cα坐标，轻量化设计）
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout，减少过拟合
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, mask):
        """
        前向传播函数
        :param x: 输入嵌入张量，形状为(batch, seq_len, input_dim)
        :param mask: 有效位置掩码，形状为(batch, seq_len)（True表示有效位置）
        :return: 预测坐标张量，形状为(batch, seq_len, output_dim)
        """
        # 调整掩码格式：Transformer的src_key_padding_mask要求True表示需要屏蔽，且与输入同设备
        key_padding_mask = (~mask).to(x.device).bool()
        
        # Transformer编码（提取序列特征）
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        
        # 回归头预测3维Cα坐标
        x = self.regressor(x)
        
        return x


class FullModel(nn.Module):
    """包装器模型：包含训练时的回归器并暴露 `infer_pdb` 方法，
    便于序列化后在其它环境直接通过 `torch.load` 使用。
    """
    def __init__(self, input_dim=480):
        super().__init__()
        self.model = ProteinRegressor(input_dim=input_dim)

    def forward(self, x, mask):
        return self.model(x, mask)

    def infer_pdb(self, embeddings, mask, return_coords_only=True):
        """最小实现：返回预测坐标数组。可以根据需要扩展为返回 PDB 文本。"""
        self.eval()
        with torch.no_grad():
            if not isinstance(embeddings, torch.Tensor):
                embeddings = torch.tensor(embeddings, dtype=torch.float32)
            if not isinstance(mask, torch.Tensor):
                mask = torch.tensor(mask, dtype=torch.bool)

            device = next(self.parameters()).device
            embeddings = embeddings.to(device)
            mask = mask.to(device)

            coords = self.forward(embeddings, mask)

        if return_coords_only:
            return coords.cpu().numpy()
        return coords