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
        
        # 计算K近邻半径
        radii = self._knn_radius(point_cloud, k=3)
        
        # 初始化高斯参数
        self.gaussians = {
            'positions': point_cloud,
            'radii': radii,
            'colors': self._sample_colors(image, point_cloud),
            'opacities': np.random.uniform(0.7, 0.95, len(point_cloud)),
            'rotations': np.random.rand(len(point_cloud), 4),  # 四元数
            'scales': np.ones((len(point_cloud), 3)) * 0.1  # 初始缩放
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
    
    def optimize_gaussians(self, target_image, iterations=100):
        """
        优化高斯球参数以更好地拟合目标图像
        
        参数:
            target_image: 目标图像
            iterations: 优化迭代次数
        """
        # 这里应该实现优化算法，例如梯度下降
        # 为简化示例，这里只展示一个框架
        for i in range(iterations):
            # 1. 渲染当前高斯球
            rendered = self.render()
            
            # 2. 计算与目标图像的差异
            # loss = compute_loss(rendered, target_image)
            
            # 3. 更新高斯球参数
            # self.update_parameters(loss)
            
            print(f"Iteration {i+1}/{iterations}")
        
        return self.gaussians
    
    def render(self, resolution=None):
        """
        基于高斯泼溅的渲染实现
        """
        if self.gaussians is None:
            return np.zeros((10,10,3))  # 返回空图像
            
        # 使用原始分辨率
        h, w = resolution if resolution else self.original_resolution
        
        # 转换为PyTorch张量加速计算
        positions = torch.tensor(self.gaussians['positions'], dtype=torch.float32)
        colors = torch.tensor(self.gaussians['colors'], dtype=torch.float32)
        scales = torch.tensor(self.gaussians['scales'], dtype=torch.float32)
        rotations = torch.tensor(self.gaussians['rotations'], dtype=torch.float32)
        opacities = torch.tensor(self.gaussians['opacities'], dtype=torch.float32)

        # 创建渲染画布 [H, W, 3]
        image = torch.zeros((h, w, 3), dtype=torch.float32)
        alpha_sum = torch.zeros((h, w, 1), dtype=torch.float32)
        
        # 相机参数 (简单正交投影)
        focal_length = 1.0
        view_matrix = torch.tensor([
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ], dtype=torch.float32)

        # 将高斯转换到相机坐标系
        cam_positions = torch.matmul(
            torch.cat([positions, torch.ones(len(positions), 1)], dim=1), 
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
        x_proj = (cam_positions[:, 0] / cam_positions[:, 2] * focal_length + 1) * w / 2
        y_proj = (-cam_positions[:, 1] / cam_positions[:, 2] * focal_length + 1) * h / 2
        z_inv = 1.0 / cam_positions[:, 2]

        # 为每个高斯生成2D协方差矩阵
        # (这里简化实现，实际需要根据旋转和缩放计算)
        cov2d = torch.stack([
            scales[:,0]*100, 
            torch.zeros_like(scales[:,0]),
            torch.zeros_like(scales[:,0]),
            scales[:,1]*100
        ], dim=1).reshape(-1,2,2)

        # 遍历所有高斯（实际应使用空间加速结构）
        for i in range(len(cam_positions)):
            x = int(x_proj[i])
            y = int(y_proj[i])
            if 0 <= x < w and 0 <= y < h:
                # 计算高斯权重
                dx = torch.arange(w) - x
                dy = torch.arange(h) - y
                grid_x, grid_y = torch.meshgrid(dx, dy, indexing='xy')
                dist = (grid_x**2 + grid_y**2) / (scales[i,0]*100)**2
                weight = torch.exp(-dist) * opacities[i]
                
                # 累加颜色
                image += weight[..., None] * colors[i] * z_inv[i]
                alpha_sum += weight[..., None]

        # 归一化并转换为numpy数组
        image = torch.where(alpha_sum > 0, image / alpha_sum, torch.tensor(0.0))
        return image.numpy()
    
    def visualize_3d(self):
        """可视化3D高斯球表示"""
        if self.gaussians is None:
            print("请先初始化高斯球")
            return
            
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = self.gaussians['positions']
        colors = self.gaussians['colors']
        radii = self.gaussians['radii']
        
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
        
        # 设置等比例轴
        max_range = np.array([
            positions[:,0].max()-positions[:,0].min(),
            positions[:,1].max()-positions[:,1].min(),
            positions[:,2].max()-positions[:,2].min()
        ]).max() / 2.0
        
        mid_x = (positions[:,0].max()+positions[:,0].min()) * 0.5
        mid_y = (positions[:,1].max()+positions[:,1].min()) * 0.5
        mid_z = (positions[:,2].max()+positions[:,2].min()) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
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
    
    # 渲染并显示对比
    rendered = gs_model.render()
    
    # 显示3D高斯球和渲染对比
    gs_model.visualize_3d()  # 原有的3D可视化
    gs_model.visualize_comparison(original_image, rendered)  # 新增的对比可视化


if __name__ == "__main__":
    main() 