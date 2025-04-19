import yaml
import os

def load_config(config_path="configs/config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 确保输出目录存在
    os.makedirs(config['paths']['output_dir'], exist_ok=True)
    os.makedirs(config['paths']['colmap_workspace'], exist_ok=True)
    
    return config

def get_image_paths(config):
    """获取原始图像路径列表"""
    raw_data_dir = config['paths']['raw_data']
    image_extensions = ['.jpg', '.jpeg', '.png']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend([
            os.path.join(raw_data_dir, f) 
            for f in os.listdir(raw_data_dir) 
            if f.lower().endswith(ext)
        ])
    
    return sorted(image_paths) 