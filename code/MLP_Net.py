import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class MLP(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features, activation=nn.ReLU(), dropout=0.0):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        layers = []
        
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(in_features, hidden_sizes[0]))
        layers.append(activation)
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # 添加后续隐藏层
        for i in range(len(hidden_sizes)-1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # 最后一层到输出层
        layers.append(nn.Linear(hidden_sizes[-1], out_features))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def visualize_mlp_process(image_path, mlp_model, verbose=False, device='cuda'):
    # 设置设备
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    mlp_model = mlp_model.to(device)
    
    # 读取图像
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img) / 255.0
    
    h, w = img_array.shape[:2]
    
    # 转换为张量并移动到设备
    input_tensor = torch.FloatTensor(img_array).view(-1, 3).to(device)
    
    # 前向传播
    with torch.no_grad():
        output = mlp_model(input_tensor)
    
    # 数据查看部分需要将数据移回CPU
    if verbose:
        print("\n=== 输出数据详情 ===")
        print(f"输出张量形状: {output.shape}")
        print(f"设备: {output.device}")
        
        # 将数据复制到CPU
        output_cpu = output.cpu()
        print("\n前5个像素的输出值:")
        print(output_cpu[:5].numpy())
        
        print("\n统计信息:")
        print(f"最大值: {output_cpu.max().item():.4f}")
        print(f"最小值: {output_cpu.min().item():.4f}")
        print(f"平均值: {output_cpu.mean().item():.4f}")
        print(f"标准差: {output_cpu.std().item():.4f}")

    # 转换回图像格式
    output_img = output.cpu().view(h, w, 3).numpy()
    output_img = np.clip(output_img, 0, 1)

    # 可视化
    plt.figure(figsize=(12, 6))
    
    # 原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(img_array)
    plt.title(f'Original Image ({h}x{w})')
    
    # 处理结果
    plt.subplot(1, 2, 2)
    plt.imshow(output_img)
    plt.title(f'MLP Processed\nHidden Layers: {len(mlp_model.hidden_sizes)}')
    
    # 网络结构信息
    activation_layers = sum(1 for layer in mlp_model.net if isinstance(layer, nn.modules.activation.ReLU))
    print(f"网络结构信息：")
    print(f"- 隐藏层数量: {len(mlp_model.hidden_sizes)}")
    print(f"- 激活函数层: {activation_layers} (ReLU)")
    print(f"- Dropout层: {sum(1 for layer in mlp_model.net if isinstance(layer, nn.Dropout))}")
    
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 初始化MLP并移动到设备
    mlp = MLP(
        in_features=3,
        hidden_sizes=[64, 32],
        out_features=3,
        activation=nn.ReLU(),
        dropout=0.1
    ).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    visualize_mlp_process("IMG/lena.jpg", mlp, verbose=True) 