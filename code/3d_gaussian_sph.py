import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GaussianSphereRepresentation:
    def __init__(self, num_gaussians=1000):
        """
        初始化3D高斯球表示模型
        
        参数:
            num_gaussians: 用于表示图像的高斯球数量
        """
        self.num_gaussians = num_gaussians
        self.gaussians = None
        
    def initialize_gaussians(self, image):
        """
        根据输入图像初始化高斯球
        
        参数:
            image: 输入图像, numpy数组, shape为(H, W, 3)
        """
        # 将图像转换为灰度图以简化处理
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        h, w = gray.shape
        
        # 创建高斯球参数: 位置(x, y, z), 半径r, 颜色(r, g, b), 透明度alpha
        positions = []
        radii = []
        colors = []
        alphas = []
        
        # 根据图像亮度采样点
        points = []
        for i in range(h):
            for j in range(w):
                if gray[i, j] > 30:  # 忽略过暗的像素
                    points.append((j, i, gray[i, j]))
                    
        # 如果有足够的点，进行采样
        if len(points) > self.num_gaussians:
            indices = np.random.choice(len(points), self.num_gaussians, replace=False)
            points = [points[i] for i in indices]
        
        # 为每个采样点创建高斯球
        for j, i, intensity in points:
            # 归一化坐标到[-1, 1]范围
            x = j / w * 2 - 1
            y = -(i / h * 2 - 1)  # 翻转y轴使其向上为正
            z = 0  # 所有点初始在同一平面
            
            # 根据亮度设置球半径
            r = 0.01 + (intensity / 255.0) * 0.03
            
            # 获取颜色
            if len(image.shape) == 3:
                color = image[i, j] / 255.0
            else:
                color = np.array([intensity, intensity, intensity]) / 255.0
                
            positions.append([x, y, z])
            radii.append(r)
            colors.append(color)
            alphas.append(0.7)  # 设置透明度
            
        self.gaussians = {
            'positions': np.array(positions),
            'radii': np.array(radii),
            'colors': np.array(colors),
            'alphas': np.array(alphas)
        }
        
        return self.gaussians
    
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
    
    def render(self, resolution=(512, 512)):
        """
        渲染高斯球表示
        
        参数:
            resolution: 输出图像分辨率
        
        返回:
            渲染后的图像
        """
        # 这里应该实现基于高斯球的渲染
        # 简化示例中，我们只返回一个可视化结果
        
        h, w = resolution
        image = np.zeros((h, w, 3), dtype=np.float32)
        
        # 实际实现中，这里应该渲染每个高斯球到图像上
        
        return image
    
    def visualize_3d(self):
        """
        可视化3D高斯球表示
        """
        if self.gaussians is None:
            print("请先初始化高斯球")
            return
            
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = self.gaussians['positions']
        colors = self.gaussians['colors']
        radii = self.gaussians['radii']
        
        # 绘制散点图，点大小由半径决定
        ax.scatter(
            positions[:, 0], positions[:, 1], positions[:, 2],
            c=colors, s=radii*1000, alpha=0.6
        )
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Gaussian Sphere Representation')
        
        plt.tight_layout()
        plt.show()


def main():
    # 加载图片
    image_path = "./IMG/lena.jpg"
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法加载图片: {image_path}")
        return
        
    # 调整图像大小以加快处理
    image = cv2.resize(image, (256, 256))
    
    # 创建3D高斯球表示
    gs_model = GaussianSphereRepresentation(num_gaussians=2000)
    
    # 初始化高斯球
    gaussians = gs_model.initialize_gaussians(image)
    
    # 可视化3D表示
    gs_model.visualize_3d()
    
    # 优化高斯球参数（这一步可选）
    # gs_model.optimize_gaussians(image, iterations=50)
    
    # 再次可视化优化后的结果
    # gs_model.visualize_3d()


if __name__ == "__main__":
    main() 