import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
import os

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
        
    def _generate_point_cloud(self, image, num_points=2000):
        """
        生成简化的SfM点云（模拟真实SfM流程）
        """
        # 使用ORB特征检测器模拟SfM特征点
        orb = cv2.ORB_create(nfeatures=num_points)
        keypoints = orb.detect(image, None)
        
        # 生成伪深度（根据亮度模拟深度）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape)==3 else image
        h, w = gray.shape
        
        points = []
        for kp in keypoints:
            x, y = kp.pt
            intensity = gray[int(y), int(x)]
            # 模拟深度：亮度越高距离越近
            z = 0.5 + (intensity / 255.0) * 2.0  # 深度范围[0.5, 2.5]
            points.append([x/w*2-1, -(y/h*2-1), z])  # 归一化坐标
        
        return np.array(points)
    
    def _knn_radius(self, points, k=3):
        """计算每个点的K近邻平均距离"""
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
        distances, _ = nbrs.kneighbors(points)
        return distances[:, 1:].mean(axis=1)  # 排除自身

    def initialize_gaussians(self, image):
        """
        基于SfM点云初始化高斯球
        """
        # 生成模拟SfM点云
        point_cloud = self._generate_point_cloud(image, self.num_gaussians)
        
        # 检查点云数据有效性
        if not np.isfinite(point_cloud).all():
            raise ValueError("生成的初始点云包含非法值")
        
        # 计算K近邻半径
        radii = self._knn_radius(point_cloud, k=3)
        
        # 修改为使用Parameter存储可训练参数
        self.gaussian_params = {
            'positions': torch.nn.Parameter(torch.tensor(point_cloud, dtype=torch.float32).to(self.device)),
            'colors': torch.nn.Parameter(torch.tensor(self._sample_colors(image, point_cloud), 
                                  dtype=torch.float32).to(self.device)),
            'opacities': torch.nn.Parameter(torch.tensor(np.random.uniform(0.7, 0.95, len(point_cloud)), 
                                   dtype=torch.float32).to(self.device)),
            'rotations': torch.nn.Parameter(torch.rand(len(point_cloud), 4).to(self.device)),
            'scales': torch.nn.Parameter(torch.ones((len(point_cloud), 3)).to(self.device) * 0.1)
        }
        # 非训练参数保持原样
        self.gaussians = {
            'radii': torch.tensor(radii, dtype=torch.float32).to(self.device),
            **self.gaussian_params
        }
        self.original_resolution = image.shape[:2]  # 保存原始分辨率
        return self.gaussians
    
    def _sample_colors(self, image, points):
        """从图像中采样颜色"""
        h, w = image.shape[:2]
        colors = []
        for pt in points:
            x = int((pt[0] + 1)/2 * w)
            y = int((-pt[1] + 1)/2 * h)
            x = np.clip(x, 0, w-1)
            y = np.clip(y, 0, h-1)
            colors.append(image[y, x]/255.0)
        return np.array(colors)
    
    def _ssim_loss(self, img1, img2, window_size=11):
        # 修改输入维度处理
        def rgb2gray(img):
            return 0.299 * img[...,0] + 0.587 * img[...,1] + 0.114 * img[...,2]
        
        # 转换到灰度并增加维度 [B, C, H, W]
        img1_gray = rgb2gray(img1).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        img2_gray = rgb2gray(img2).unsqueeze(0).unsqueeze(0)
        
        # 创建高斯窗口（保持单通道）
        weights = torch.FloatTensor(np.exp(-0.5*np.square(np.linspace(-1.5,1.5,window_size)))).to(img1.device)
        weights = (weights.outer(weights)).view(1,1,window_size,window_size)
        weights /= weights.sum()
        
        # 计算统计量
        mu1 = torch.nn.functional.conv2d(img1_gray, weights, padding=window_size//2)
        mu2 = torch.nn.functional.conv2d(img2_gray, weights, padding=window_size//2)
        
        sigma1_sq = torch.nn.functional.conv2d(img1_gray**2, weights, padding=window_size//2) - mu1**2
        sigma2_sq = torch.nn.functional.conv2d(img2_gray**2, weights, padding=window_size//2) - mu2**2
        sigma12 = torch.nn.functional.conv2d(img1_gray*img2_gray, weights, padding=window_size//2) - mu1*mu2
        
        # SSIM参数
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1*mu2 + C1)*(2*sigma12 + C2)) / \
                  ((mu1**2 + mu2**2 + C1)*(sigma1_sq + sigma2_sq + C2))
        return 1 - ssim_map.mean()

    def optimize_gaussians(self, target_image, iterations=100):
        """
        改进的优化流程，包含学习率调度和梯度累积
        """
        # 准备目标图像
        h, w = self.original_resolution
        target_tensor = torch.tensor(cv2.resize(target_image, (w, h)) / 255.0, 
                                   dtype=torch.float32).to(self.device)
        
        # 使用更复杂的学习率策略
        optimizer = torch.optim.Adam([
            {'params': self.gaussian_params['positions'], 'lr': 0.0005},  # 略微提高
            {'params': self.gaussian_params['colors'], 'lr': 0.01},
            {'params': self.gaussian_params['opacities'], 'lr': 0.02},
            {'params': self.gaussian_params['scales'], 'lr': 0.002},
            {'params': self.gaussian_params['rotations'], 'lr': 0.0005}
        ])
        
        # 添加学习率调度器 - 余弦退火重启
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
        
        # 动态权重混合损失
        loss_weights = {'l1': 0.8, 'ssim': 0.2, 'perceptual': 0.1}  # 添加感知损失权重
        
        # 梯度累积步数 - 可提高内存效率
        accumulation_steps = 2
        
        # 随机扰动步长 - 帮助跳出局部最小值
        perturbation_step = 30
        perturbation_scale = 0.01
        
        best_loss = float('inf')
        patience_counter = 0
        patience_threshold = 20  # 早停耐心值
        
        # 启用混合精度训练
        scaler = torch.cuda.amp.GradScaler()
        
        # 释放并预留GPU内存
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)  # 增加到90%
        
        for i in range(iterations):
            optimizer.zero_grad()
            
            # 随机扰动参数以跳出局部最小值
            if i > 0 and i % perturbation_step == 0:
                with torch.no_grad():
                    # 对位置和缩放参数添加随机扰动
                    self.gaussian_params['positions'].data += torch.randn_like(
                        self.gaussian_params['positions']) * perturbation_scale * (1.0 - i/iterations)
                    self.gaussian_params['scales'].data *= (1.0 + torch.randn_like(
                        self.gaussian_params['scales']) * perturbation_scale * (1.0 - i/iterations))
            
            # 混合精度训练
            with torch.cuda.amp.autocast():
                rendered_tensor = self.render_tensor()
                
                # 计算混合损失
                l1_loss = torch.nn.L1Loss()(rendered_tensor, target_tensor)
                ssim_loss = self._ssim_loss(rendered_tensor, target_tensor)
                
                # 添加额外的感知损失项 - 使用梯度图作为简单的感知特征
                sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(self.device).view(1, 1, 3, 3)
                sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(self.device).view(1, 1, 3, 3)
                
                # 转为灰度并计算梯度
                gray_rendered = 0.299 * rendered_tensor[...,0] + 0.587 * rendered_tensor[...,1] + 0.114 * rendered_tensor[...,2]
                gray_target = 0.299 * target_tensor[...,0] + 0.587 * target_tensor[...,1] + 0.114 * target_tensor[...,2]
                
                gray_rendered = gray_rendered.unsqueeze(0).unsqueeze(0)
                gray_target = gray_target.unsqueeze(0).unsqueeze(0)
                
                grad_x_rendered = torch.nn.functional.conv2d(gray_rendered, sobel_x, padding=1)
                grad_y_rendered = torch.nn.functional.conv2d(gray_rendered, sobel_y, padding=1)
                grad_x_target = torch.nn.functional.conv2d(gray_target, sobel_x, padding=1)
                grad_y_target = torch.nn.functional.conv2d(gray_target, sobel_y, padding=1)
                
                perceptual_loss = torch.nn.functional.mse_loss(grad_x_rendered, grad_x_target) + \
                                 torch.nn.functional.mse_loss(grad_y_rendered, grad_y_target)
                
                # 权重衰减 - 随着训练的进行，提高SSIM和感知损失的权重
                progress = i / iterations
                l1_weight = loss_weights['l1'] * (1.0 - progress*0.3)  # 逐渐降低L1权重
                ssim_weight = loss_weights['ssim'] * (1.0 + progress)  # 逐渐提高SSIM权重
                perceptual_weight = loss_weights['perceptual'] * (1.0 + progress*2.0)  # 逐渐提高感知损失权重
                
                total_loss = l1_weight*l1_loss + ssim_weight*ssim_loss + perceptual_weight*perceptual_loss
                
                # 添加可变正则化项 - 随训练进行减小
                reg_coef = 0.01 * (1.0 - progress)
                reg_loss = torch.mean(torch.abs(self.gaussian_params['positions'])) * reg_coef
                total_loss += reg_loss
                
                # 梯度累积
                total_loss = total_loss / accumulation_steps
            
            # 使用混合精度训练的反向传播
            scaler.scale(total_loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                # 梯度裁剪 - 动态裁剪阈值
                clip_value = 1.0 * (1.0 - progress*0.5)  # 随着训练进行，适当降低裁剪阈值
                torch.nn.utils.clip_grad_norm_(self.gaussian_params['positions'], clip_value)
                torch.nn.utils.clip_grad_norm_(self.gaussian_params['scales'], clip_value*0.1)
                
                # 更新参数
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            
            # 参数约束（添加更严格的限制）
            with torch.no_grad():
                self.gaussian_params['opacities'].data = self.gaussian_params['opacities'].clamp(0.1, 0.95)
                self.gaussian_params['colors'].data = self.gaussian_params['colors'].clamp(0.05, 0.95)
                self.gaussian_params['scales'].data = self.gaussian_params['scales'].clamp(0.03, 0.4)
                # 防止位置参数爆炸
                self.gaussian_params['positions'].data = self.gaussian_params['positions'].clamp(-3.5, 3.5)
            
            # 检查NaN并提前终止
            if torch.isnan(total_loss):
                print("检测到NaN损失值，恢复最佳参数并终止优化")
                break
            
            # 保存最佳参数
            if total_loss * accumulation_steps < best_loss:
                best_loss = total_loss * accumulation_steps
                best_params = {k: v.clone() for k,v in self.gaussian_params.items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            # 早停策略
            if patience_counter > patience_threshold:
                print(f"损失值停滞不前达{patience_threshold}次，应用早停策略")
                
                # 学习率重启并减小扰动
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                
                # 重置耐心计数器
                patience_counter = 0
                
                # 如果学习率已经很低，则终止优化
                if optimizer.param_groups[0]['lr'] < 1e-6:
                    print("学习率过低，终止优化")
                    break
            
            # 可视化训练进度
            if (i+1) % 2 == 0:
                print(f"Iter {i+1}/{iterations} Loss: {(total_loss * accumulation_steps).item():.6f} "
                     f"(L1: {l1_loss.item():.4f}, SSIM: {ssim_loss.item():.4f}, "
                     f"Percept: {perceptual_loss.item():.4f}, LR: {optimizer.param_groups[0]['lr']:.6f})")

        # 恢复最佳参数
        for k, v in best_params.items():
            self.gaussian_params[k].data = v
        return self.gaussians
    
    def render_tensor(self, resolution=None, chunks=4):
        """改进的内存优化渲染版本，支持分块渲染"""
        # 在开始渲染前强制释放所有缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
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
    
    def visualize_3d(self):
        """可视化3D高斯球表示"""
        if self.gaussians is None:
            print("请先初始化高斯球")
            return
            
        # 添加数据有效性检查
        positions = self.gaussian_params['positions'].detach().cpu().numpy()
        
        # 检查NaN/Inf
        if not np.isfinite(positions).all():
            print("警告：检测到非法坐标值（NaN/Inf），无法可视化")
            return
        
        # 创建3D坐标系
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 计算坐标范围（增加容错机制）
        coord_ranges = []
        mid_points = []
        for dim in range(3):
            dim_min = positions[:, dim].min()
            dim_max = positions[:, dim].max()
            
            # 处理所有点在同一位置的情况
            if dim_max - dim_min < 1e-6:
                dim_min -= 0.5
                dim_max += 0.5
            
            coord_ranges.append(dim_max - dim_min)
            mid_points.append((dim_min + dim_max) / 2)
        
        max_range = max(coord_ranges) / 2
        
        # 设置坐标轴范围
        ax.set_xlim(mid_points[0] - max_range, mid_points[0] + max_range)
        ax.set_ylim(mid_points[1] - max_range, mid_points[1] + max_range)
        ax.set_zlim(mid_points[2] - max_range, mid_points[2] + max_range)
        
        # 添加detach()并转换为numpy
        positions = self.gaussian_params['positions'].detach().cpu().numpy()
        colors = self.gaussian_params['colors'].detach().cpu().numpy()
        radii = self.gaussians['radii'].detach().cpu().numpy()
        
        # 绘制散点图，点大小由半径决定
        scatter = ax.scatter(
            positions[:, 0], 
            positions[:, 1], 
            positions[:, 2],
            c=colors,
            s=radii*1000,
            alpha=0.6,
            depthshade=True  # 启用深度阴影
        )
        
        # 添加颜色条
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.5)
        cbar.set_label('Color Value')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Gaussian Sphere Representation')
        
        plt.tight_layout()

    def visualize_comparison(self, original_image, rendered_image):
        """
        并排显示原始图像和渲染结果对比
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 显示原始图像（转换BGR到RGB）
        ax1.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Original Image')
        ax1.axis('off')
        
        # 显示渲染结果
        ax2.imshow(rendered_image)
        ax2.set_title('Gaussian Splatting Rendering')
        ax2.axis('off')
        
        plt.tight_layout()
        plt.show()

    def save_model(self, filepath):
        """
        保存高斯球表示模型参数到PyTorch文件
        
        参数:
            filepath: 保存模型的文件路径 (建议使用.pt或.pth后缀)
        """
        if self.gaussian_params is None:
            print("错误: 模型尚未初始化，无法保存")
            return False
        
        # 创建模型状态字典
        state_dict = {
            # 基本配置
            'num_gaussians': self.num_gaussians,
            'original_resolution': self.original_resolution,
            
            # 保存高斯参数（所有参数的实际数据）
            'positions': self.gaussian_params['positions'].data.cpu(),  # 转移到CPU
            'colors': self.gaussian_params['colors'].data.cpu(),
            'opacities': self.gaussian_params['opacities'].data.cpu(),
            'rotations': self.gaussian_params['rotations'].data.cpu(),
            'scales': self.gaussian_params['scales'].data.cpu(),
            'radii': self.gaussians['radii'].cpu()  # 非训练参数
        }
        
        try:
            # 保存到文件
            torch.save(state_dict, filepath)
            print(f"模型成功保存到: {filepath}")
            print(f"保存的高斯球数量: {len(state_dict['positions'])}")
            return True
        except Exception as e:
            print(f"保存模型时出错: {str(e)}")
            return False

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


def main():
    # 加载图片（保持原始分辨率）
    image_path = "./IMG/lena.jpg"
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"无法加载图片: {image_path}")
        return
    
    # GPU内存配置
    torch.cuda.empty_cache()
    # 设置PyTorch为确定性模式以提高稳定性
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # 启用基准优化
    
    # 创建3D高斯球表示 - 增加高斯数量
    gs_model = GaussianSphereRepresentation(num_gaussians=10000)  # 增加高斯数量
    
    # 将图像调整为更合理的尺寸（如果需要）
    scale_factor = 1.0  # 可根据GPU内存调整
    if scale_factor != 1.0:
        h, w = original_image.shape[:2]
        original_image = cv2.resize(original_image, (int(w*scale_factor), int(h*scale_factor)))
    
    gs_model.original_resolution = original_image.shape[:2]
    
    # 初始化高斯球
    gaussians = gs_model.initialize_gaussians(original_image)
    
    # 使用多阶段优化策略
    print("第1阶段优化: 基础细节恢复")
    gs_model.optimize_gaussians(original_image, iterations=300)  # 更多迭代次数
    
    # 保存中间结果（可选）
    intermediate_render = gs_model.render()
    cv2.imwrite("intermediate_result.png", (intermediate_render*255).astype(np.uint8))
    
    print("\n第2阶段优化: 细节增强")
    # 使用更高学习率和更多迭代进行第二阶段优化
    gs_model.optimize_gaussians(original_image, iterations=300)
    
    # 渲染最终结果
    rendered = gs_model.render()
    
    # 显示3D高斯球和渲染对比
    gs_model.visualize_3d()
    gs_model.visualize_comparison(original_image, rendered)
    
    # 保存结果（可选）
    cv2.imwrite("final_result.png", (rendered*255).astype(np.uint8))
    
    # 打印GPU内存使用情况
    print(f"峰值GPU内存使用: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"当前GPU内存使用: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

    # 保存模型
    model_path = "./saved_models/gaussian_model_lena.pt"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)  # 确保目录存在
    gs_model.save_model(model_path)


def load_and_render():
    # 创建一个新的模型实例
    gs_model = GaussianSphereRepresentation()
    
    # 加载保存的模型
    model_path = "./saved_models/gaussian_model_lena.pt"
    gs_model.load_model(model_path)
    
    # 渲染图像
    rendered = gs_model.render()
    
    # 显示渲染结果
    plt.imshow(rendered)
    plt.axis('off')
    plt.title('Rendered from Loaded Model')
    plt.show()
    
    # 如果需要，可以继续训练
    # gs_model.optimize_gaussians(some_image, iterations=100)


if __name__ == "__main__":
    main() 