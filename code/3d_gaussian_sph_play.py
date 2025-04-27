#!/usr/bin/env python3
import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors

class GaussianSphereRepresentation:
    def __init__(self, num_gaussians=1000):
        """
        初始化3D高斯球表示模型
        
        参数:
            num_gaussians: 用于表示图像的高斯球数量
        """
        self.num_gaussians = num_gaussians
        self.gaussians = None
        self.original_resolution = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gaussian_params = None  # 新增参数容器
        
    def load_model(self, filepath, device=None):
        """
        从PyTorch文件加载高斯球表示模型参数
        
        参数:
            filepath: 模型文件路径
            device: 指定加载到的设备，默认使用初始化时的设备
        """
        if device is None:
            device = self.device
        
        try:
            # 加载状态字典
            state_dict = torch.load(filepath, map_location=device)
            
            # 恢复基本配置
            self.num_gaussians = state_dict['num_gaussians']
            self.original_resolution = state_dict['original_resolution']
            
            # 创建参数字典
            self.gaussian_params = {
                'positions': torch.nn.Parameter(state_dict['positions'].to(device)),
                'colors': torch.nn.Parameter(state_dict['colors'].to(device)),
                'opacities': torch.nn.Parameter(state_dict['opacities'].to(device)),
                'rotations': torch.nn.Parameter(state_dict['rotations'].to(device)),
                'scales': torch.nn.Parameter(state_dict['scales'].to(device))
            }
            
            # 恢复gaussians字典
            self.gaussians = {
                'radii': state_dict['radii'].to(device),
                **self.gaussian_params
            }
            
            print(f"模型成功从 {filepath} 加载")
            print(f"加载的高斯球数量: {len(self.gaussian_params['positions'])}")
            return True
        except Exception as e:
            print(f"加载模型时出错: {str(e)}")
            return False
            
    def render_tensor(self, resolution=None, chunks=4):
        """改进的内存优化渲染版本，支持分块渲染"""
        # 在开始渲染前强制释放所有缓存
        torch.cuda.empty_cache()
        
        if self.gaussians is None:
            return torch.zeros((10,10,3), dtype=torch.float32)
            
        # 使用原始分辨率
        h, w = resolution if resolution else self.original_resolution
        
        # 修改为使用GPU加速
        positions = self.gaussian_params['positions'].to(self.device)
        colors = self.gaussian_params['colors'].to(self.device)
        scales = self.gaussian_params['scales'].to(self.device)
        rotations = self.gaussian_params['rotations'].to(self.device)
        opacities = self.gaussian_params['opacities'].to(self.device)

        # 相机参数 (简单正交投影)
        focal_length = 1.0
        view_matrix = torch.tensor([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ], dtype=torch.float32, device=self.device)

        # 将高斯转换到相机坐标系
        cam_positions = torch.matmul(
            torch.cat([
                positions, 
                torch.ones(len(positions), 1, device=self.device)
            ], dim=1), 
            view_matrix.T
        )[:, :3]

        # 只渲染正向的高斯
        valid_mask = cam_positions[:, 2] > 0
        cam_positions = cam_positions[valid_mask]
        colors = colors[valid_mask]
        scales = scales[valid_mask]
        rotations = rotations[valid_mask]
        opacities = opacities[valid_mask]

        # 计算2D投影坐标 (透视投影)
        x_coords = (cam_positions[:, 0] / cam_positions[:, 2] * focal_length + 1) * w / 2
        y_coords = (-cam_positions[:, 1] / cam_positions[:, 2] * focal_length + 1) * h / 2
        z_inv = 1.0 / cam_positions[:, 2]

        # 使用向量化操作代替循环
        x_coords = (x_coords.clamp(0, w-1)).long()
        y_coords = (y_coords.clamp(0, h-1)).long()
        
        # 使用分块渲染减少内存占用
        chunk_size = len(cam_positions) // chunks + 1
        
        # 创建输出图像和累积权重图
        final_image = torch.zeros((h, w, 3), dtype=torch.float32, device=self.device)
        accumulated_weights = torch.zeros((h, w, 1), dtype=torch.float32, device=self.device)
        
        # 分块处理高斯点
        for chunk_idx in range(chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(cam_positions))
            
            if start_idx >= end_idx:
                break
            
            # 获取当前块的数据
            chunk_x_coords = x_coords[start_idx:end_idx]
            chunk_y_coords = y_coords[start_idx:end_idx]
            chunk_colors = colors[start_idx:end_idx]
            chunk_scales = scales[start_idx:end_idx]
            chunk_opacities = opacities[start_idx:end_idx]
            chunk_z_inv = z_inv[start_idx:end_idx]
            
            # 创建空间索引网格 - 使用稀疏计算
            grid_x, grid_y = torch.meshgrid(
                torch.arange(w, device=self.device),
                torch.arange(h, device=self.device),
                indexing='xy'
            )
            
            # 计算当前块高斯的权重矩阵
            dx = grid_x[None, :, :] - chunk_x_coords[:, None, None]
            dy = grid_y[None, :, :] - chunk_y_coords[:, None, None]
            
            # 优化内存使用的权重计算
            dist = (dx**2 + dy**2) / (chunk_scales[:, 0, None, None]*100 + 1e-6)**2
            weights = torch.exp(-torch.clamp(dist, max=10)) * chunk_opacities[:, None, None]
            
            # 计算颜色贡献
            color_contrib = weights[..., None] * chunk_colors[:, None, None, :] * chunk_z_inv[:, None, None, None]
            
            # 过滤掉无效像素
            valid_mask = (chunk_x_coords >= 0) & (chunk_x_coords < w) & (chunk_y_coords >= 0) & (chunk_y_coords < h)
            color_contrib = color_contrib * valid_mask[:, None, None, None]
            
            # 累加到最终图像
            final_image += color_contrib.sum(dim=0)
            accumulated_weights += weights.sum(dim=0)[..., None]
            
            # 每块处理完后释放内存
            del dx, dy, dist, weights, color_contrib
            torch.cuda.empty_cache()
        
        # 归一化
        final_image = torch.where(
            accumulated_weights > 1e-6, 
            final_image / accumulated_weights, 
            torch.tensor(0.0, device=self.device)
        )
        
        return final_image
    
    def render(self, resolution=None):
        """保持原有接口，但内部调用张量版本"""
        return self.render_tensor(resolution).detach().cpu().numpy()

def main():
    print("开始载入高斯球模型并渲染图像...")
    
    # 创建一个新的模型实例
    gs_model = GaussianSphereRepresentation()
    
    # 指定模型路径
    model_path = "/home/lab/3D_Gaussian_Splatting/code/saved_models/gaussian_model_lena.pt"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在!")
        return
    
    # 加载保存的模型
    if not gs_model.load_model(model_path):
        print("模型加载失败，程序终止")
        return
    
    print("开始渲染图像...")
    
    # 渲染图像
    rendered_image = gs_model.render()
    
    # 确保渲染成功
    if rendered_image is None or rendered_image.size == 0:
        print("渲染失败，无法获取有效图像")
        return
    
    # 保存渲染的图像
    output_path = "rendered_gaussian_lena.png"
    cv2.imwrite(output_path, (rendered_image * 255).astype(np.uint8))
    print(f"图像已保存到: {output_path}")
    
    # 显示渲染结果
    plt.figure(figsize=(10, 8))
    plt.imshow(rendered_image)
    plt.axis('off')
    plt.title('Rendered from Loaded Gaussian Model')
    plt.tight_layout()
    plt.savefig("rendered_gaussian_display.png")
    plt.show()
    
    print("渲染完成!")

if __name__ == "__main__":
    main()