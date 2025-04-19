import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import os
import sys
import torch
import torchvision.transforms as transforms
from concurrent.futures import ThreadPoolExecutor
from skimage import exposure, restoration

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import load_config, get_image_paths

def check_gpu_availability():
    """检查GPU是否可用"""
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("GPU不可用，将使用CPU处理")
        return False

def process_single_image(img_path, config, preprocessed_dir, device):
    """处理单张图像"""
    try:
        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"警告：无法读取图像 {img_path}")
            return False
        
        # 图像预处理参数
        target_size = config['preprocessing']['image_size']
        quality = config['preprocessing']['quality']
        
        # 调整图像大小
        img = cv2.resize(img, target_size)
        
        # 使用GPU进行图像处理（如果可用）
        if device.type == 'cuda' and config['preprocessing']['use_gpu']:
            # 转换为PyTorch张量并移至GPU
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().to(device) / 255.0
            
            # 应用图像增强
            if config['preprocessing']['noise_reduction']:
                # 使用高斯滤波进行降噪
                gaussian_blur = transforms.GaussianBlur(kernel_size=3, sigma=0.5)
                img_tensor = gaussian_blur(img_tensor)
            
            if config['preprocessing']['contrast_enhancement']:
                # 对比度增强
                img_tensor = transforms.functional.adjust_contrast(img_tensor, 1.5)
            
            if config['preprocessing']['color_balance']:
                # 色彩平衡
                img_tensor = transforms.functional.adjust_saturation(img_tensor, 1.2)
                img_tensor = transforms.functional.adjust_hue(img_tensor, 0.05)
            
            # 转回CPU并转为NumPy数组
            img = (img_tensor.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            # CPU处理
            # 根据质量设置进行图像增强
            if quality == 'high':
                # 直方图均衡化
                img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
                img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
                img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
                
                # 锐化
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                img = cv2.filter2D(img, -1, kernel)
            
            # 额外的图像增强（基于配置）
            if config['preprocessing']['noise_reduction']:
                # 使用非局部均值去噪
                img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            
            if config['preprocessing']['contrast_enhancement']:
                # CLAHE对比度增强
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            if config['preprocessing']['color_balance']:
                # 自动白平衡
                img = exposure.equalize_adapthist(img, clip_limit=0.03)
                img = (img * 255).astype(np.uint8)
        
        # 保存预处理后的图像
        output_path = preprocessed_dir / Path(img_path).name
        cv2.imwrite(str(output_path), img)
        return True
    except Exception as e:
        print(f"处理图像 {img_path} 时出错: {str(e)}")
        return False

def preprocess_images(config):
    """预处理图像数据"""
    image_paths = get_image_paths(config)
    output_dir = Path(config['paths']['output_dir'])
    preprocessed_dir = output_dir / 'preprocessed'
    preprocessed_dir.mkdir(exist_ok=True)
    
    # 检查GPU是否可用
    use_gpu = config['preprocessing']['use_gpu']
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    if use_gpu and device.type == 'cuda':
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU进行图像处理")
    
    # 批处理大小
    batch_size = config['preprocessing']['batch_size']
    
    print(f"开始预处理 {len(image_paths)} 张图像...")
    
    # 使用多线程处理图像
    successful = 0
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for img_path in image_paths:
            futures.append(executor.submit(
                process_single_image, img_path, config, preprocessed_dir, device
            ))
        
        # 显示进度条并收集结果
        for future in tqdm(futures, total=len(futures)):
            if future.result():
                successful += 1
    
    print(f"图像预处理完成！成功处理 {successful}/{len(image_paths)} 张图像")
    return str(preprocessed_dir)

if __name__ == "__main__":
    config = load_config()
    preprocess_images(config)