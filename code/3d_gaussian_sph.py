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
        完整的优化流程（修复NaN问题版本）
        """
        # 准备目标图像
        h, w = self.original_resolution
        target_tensor = torch.tensor(cv2.resize(target_image, (w, h)) / 255.0, 
                                   dtype=torch.float32).to(self.device)
        
        # 调整优化器参数（降低学习率，添加梯度裁剪）
        optimizer = torch.optim.Adam([
            {'params': self.gaussian_params['positions'], 'lr': 0.0001},  # 降低10倍
            {'params': self.gaussian_params['colors'], 'lr': 0.005},
            {'params': self.gaussian_params['opacities'], 'lr': 0.01},
            {'params': self.gaussian_params['scales'], 'lr': 0.001},
            {'params': self.gaussian_params['rotations'], 'lr': 0.0001}
        ])
        
        # 混合损失权重
        loss_weights = {'l1': 0.8, 'ssim': 0.2}
        
        best_loss = float('inf')
        for i in range(iterations):
            optimizer.zero_grad()
            
            # 添加渲染保护机制
            with torch.autograd.detect_anomaly():
                rendered_tensor = self.render_tensor()
                
                # 计算混合损失
                l1_loss = torch.nn.L1Loss()(rendered_tensor, target_tensor)
                ssim_loss = self._ssim_loss(rendered_tensor, target_tensor)
                total_loss = loss_weights['l1']*l1_loss + loss_weights['ssim']*ssim_loss
                
                # 添加正则化项防止数值爆炸
                reg_loss = torch.mean(torch.abs(self.gaussian_params['positions'])) * 0.01
                total_loss += reg_loss

                # 反向传播和优化（添加梯度裁剪）
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gaussian_params['positions'], 1.0)
                torch.nn.utils.clip_grad_norm_(self.gaussian_params['scales'], 0.1)
                optimizer.step()
            
            # 参数约束（添加更严格的限制）
            with torch.no_grad():
                self.gaussian_params['opacities'].data = self.gaussian_params['opacities'].clamp(0.1, 0.9)
                self.gaussian_params['colors'].data = self.gaussian_params['colors'].clamp(0.1, 0.9)
                self.gaussian_params['scales'].data = self.gaussian_params['scales'].clamp(0.05, 0.3)
                # 防止位置参数爆炸
                self.gaussian_params['positions'].data = self.gaussian_params['positions'].clamp(-3, 3)
            
            # 检查NaN并提前终止
            if torch.isnan(total_loss):
                print("检测到NaN损失值，恢复最佳参数并终止优化")
                break
            
            # 保存最佳参数
            if total_loss < best_loss:
                best_loss = total_loss
                best_params = {k: v.clone() for k,v in self.gaussian_params.items()}
            
            print(f"Iter {i+1}/{iterations} Loss: {total_loss.item():.4f} "
                 f"(L1: {l1_loss.item():.4f}, SSIM: {ssim_loss.item():.4f})")

        # 恢复最佳参数
        for k, v in best_params.items():
            self.gaussian_params[k].data = v
        return self.gaussians
    
    def render_tensor(self, resolution=None):
        """强制内存优化的渲染版本"""
        # 在开始渲染前强制释放所有缓存
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # 重置内存统计
        
        # 设置临时内存限制（根据实际可用内存调整）
        max_mem = torch.cuda.get_device_properties(0).total_memory * 0.8  # 使用80%显存
        torch.cuda.set_per_process_memory_fraction(0.8)  # 强制限制内存使用
        
        if self.gaussians is None:
            return torch.zeros((10,10,3), dtype=torch.float32)  # 返回空图像
            
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
        
        # 创建空间索引网格
        grid_x, grid_y = torch.meshgrid(
            torch.arange(w, device=self.device),
            torch.arange(h, device=self.device),
            indexing='xy'
        )
        
        # 计算所有高斯的权重矩阵 [N, H, W]
        dx = grid_x[None, :, :] - x_coords[:, None, None]
        dy = grid_y[None, :, :] - y_coords[:, None, None]
        dist = (dx**2 + dy**2) / (scales[:, 0, None, None]*100 + 1e-6)**2  # 防止除以零
        weights = torch.exp(-torch.clamp(dist, max=10)) * opacities[:, None, None]  # 限制指数输入范围
        
        # 计算颜色贡献 [N, H, W, 3]
        color_contrib = weights[..., None] * colors[:, None, None, :] * z_inv[:, None, None, None]
        
        # 使用索引掩码进行累加
        valid_mask = (x_coords >= 0) & (x_coords < w) & (y_coords >= 0) & (y_coords < h)
        color_contrib = color_contrib * valid_mask[:, None, None, None]
        
        # 使用张量操作代替循环累加
        image = color_contrib.sum(dim=0)  # [H, W, 3]
        alpha_sum = weights.sum(dim=0)[..., None]  # [H, W, 1]
        
        # 归一化
        image = torch.where(alpha_sum > 1e-6, image / alpha_sum, torch.tensor(0.0, device=self.device))
        
        # 在渲染结束后立即释放中间变量
        del color_contrib, weights, dx, dy
        torch.cuda.empty_cache()
        
        return image
    
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


def main():
    # 加载图片（保持原始分辨率）
    image_path = "./IMG/lena.jpg"
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"无法加载图片: {image_path}")
        return
    
    # 创建3D高斯球表示
    gs_model = GaussianSphereRepresentation(num_gaussians=2000)
    gs_model.original_resolution = original_image.shape[:2]  # 保存原始分辨率
    
    # 初始化高斯球
    gaussians = gs_model.initialize_gaussians(original_image)
    
    # 初始化后直接调用优化方法
    gs_model.optimize_gaussians(original_image, iterations=500)
    
    # 渲染并显示对比
    rendered = gs_model.render()
    
    # 显示3D高斯球和渲染对比
    gs_model.visualize_3d()  # 原有的3D可视化
    gs_model.visualize_comparison(original_image, rendered)  # 新增的对比可视化


if __name__ == "__main__":
    main() 