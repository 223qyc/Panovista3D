import os
import subprocess
from pathlib import Path
import pycolmap
from tqdm import tqdm
import sys
import time
import logging
import torch
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.config import load_config

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_gpu_availability():
    """检查GPU是否可用"""
    if torch.cuda.is_available():
        logger.info(f"GPU可用: {torch.cuda.get_device_name(0)}")
        return True
    else:
        logger.info("GPU不可用，将使用CPU处理")
        return False

def run_colmap_sparse_reconstruction(config, image_dir):
    """运行COLMAP稀疏重建"""
    start_time = time.time()
    workspace_path = Path(config['paths']['colmap_workspace'])
    database_path = workspace_path / "database.db"
    
    # 检查GPU是否可用
    use_gpu = config['reconstruction']['use_gpu']
    gpu_available = check_gpu_availability()
    use_gpu = use_gpu and gpu_available
    
    # 设置线程数
    num_threads = config['reconstruction']['num_threads']
    if num_threads <= 0:
        num_threads = multiprocessing.cpu_count()
    
    try:
        # 特征提取
        logger.info("开始特征提取...")
        extract_start = time.time()
        pycolmap.extract_features(
            str(database_path),  # 转换为字符串
            str(image_dir),      # 转换为字符串
            camera_model="SIMPLE_RADIAL",  # 使用默认相机模型
            sift_options=pycolmap.SiftExtractionOptions(
                max_num_features=config['preprocessing']['max_features'],
                gpu_index="0" if use_gpu else "-1",  # 修改为使用 gpu_index 参数
                num_threads=num_threads,
                darkness_adaptivity=use_gpu  # 仅在GPU模式下启用暗光适应
            )
        )
        logger.info(f"特征提取完成，耗时: {time.time() - extract_start:.2f}秒")
        
        # 特征匹配
        logger.info("开始特征匹配...")
        match_start = time.time()
        pycolmap.match_exhaustive(
            str(database_path),
            sift_options=pycolmap.SiftMatchingOptions(
                gpu_index="0" if use_gpu else "-1",  # 修改为使用 gpu_index 参数
                num_threads=num_threads
            )
        )
        logger.info(f"特征匹配完成，耗时: {time.time() - match_start:.2f}秒")
        
        # 稀疏重建
        logger.info("开始稀疏重建...")
        mapping_start = time.time()
        maps = pycolmap.incremental_mapping(
            str(database_path),
            str(image_dir),
            str(workspace_path / "sparse"),
            min_num_matches=config['reconstruction']['sparse']['min_num_matches'],
            min_tri_angle=config['reconstruction']['sparse']['min_tri_angle'],
            max_reproj_error=config['reconstruction']['sparse']['max_reproj_error'],
            num_threads=num_threads,
            refine_focal_length=True,
            refine_principal_point=True,
            refine_extra_params=True,
            refine_extrinsics=True,
            max_refinement_iterations=config['reconstruction']['sparse']['refine_iterations']
        )
        logger.info(f"稀疏重建完成，耗时: {time.time() - mapping_start:.2f}秒")
        
        if not maps:
            logger.error("稀疏重建失败，没有生成有效的重建结果")
            return None
            
        logger.info(f"稀疏重建成功，共生成 {len(maps)} 个模型")
        logger.info(f"稀疏重建总耗时: {time.time() - start_time:.2f}秒")
        return maps
        
    except Exception as e:
        logger.error(f"稀疏重建过程中出错: {str(e)}")
        return None

def run_colmap_dense_reconstruction(config):
    """运行COLMAP稠密重建"""
    start_time = time.time()
    workspace_path = Path(config['paths']['colmap_workspace'])
    sparse_path = workspace_path / "sparse"
    dense_path = workspace_path / "dense"
    
    # 检查GPU是否可用
    use_gpu = config['reconstruction']['use_gpu']
    gpu_available = check_gpu_availability()
    use_gpu = use_gpu and gpu_available
    
    # 设置线程数
    num_threads = config['reconstruction']['num_threads']
    if num_threads <= 0:
        num_threads = multiprocessing.cpu_count()
    
    try:
        # 图像去畸变
        logger.info("开始图像去畸变...")
        undistort_start = time.time()
        pycolmap.undistort_images(
            str(sparse_path),
            str(dense_path),
            config['paths']['raw_data'],
            max_image_size=2000  # 限制最大图像尺寸以提高性能
        )
        logger.info(f"图像去畸变完成，耗时: {time.time() - undistort_start:.2f}秒")
        
        # 深度图估计
        logger.info("开始深度图估计...")
        stereo_start = time.time()
        pycolmap.patch_match_stereo(
            str(dense_path),
            min_depth=config['reconstruction']['dense']['min_depth'],
            max_depth=config['reconstruction']['dense']['max_depth'],
            num_samples=config['reconstruction']['dense']['num_samples'],
            patch_size=config['reconstruction']['dense']['patch_size'],
            window_radius=config['reconstruction']['dense']['window_radius'],
            window_step=config['reconstruction']['dense']['window_step'],
            filter_min_ncc=config['reconstruction']['dense']['filter_min_ncc'],
            filter_min_triangulation_angle=config['reconstruction']['dense']['filter_min_triangulation_angle'],
            cache_size=32,  # 缓存大小，单位GB
            gpu_index=0 if use_gpu else -1,  # 使用GPU时设为0，否则为-1
            num_threads=num_threads
        )
        logger.info(f"深度图估计完成，耗时: {time.time() - stereo_start:.2f}秒")
        
        # 深度图融合
        logger.info("开始深度图融合...")
        fusion_start = time.time()
        try:
            pycolmap.stereo_fusion(
                str(dense_path / "fused.ply"),
                str(dense_path),
                min_num_pixels=2,
                max_reproj_error=2.0,  # 最大重投影误差
                max_depth_error=0.1,   # 最大深度误差
                max_normal_error=20.0, # 最大法向量误差
                num_threads=num_threads
            )
            logger.info(f"深度图融合完成，耗时: {time.time() - fusion_start:.2f}秒")
        except Exception as e:
            logger.error(f"深度图融合失败: {str(e)}")
            # 尝试使用备用参数重新融合
            logger.info("尝试使用备用参数重新融合...")
            try:
                pycolmap.stereo_fusion(
                    str(dense_path / "fused.ply"),
                    str(dense_path),
                    min_num_pixels=1,  # 降低最小像素数要求
                    max_reproj_error=3.0,  # 增大最大重投影误差
                    num_threads=num_threads
                )
                logger.info("使用备用参数深度图融合成功")
            except Exception as e2:
                logger.error(f"使用备用参数深度图融合仍然失败: {str(e2)}")
                raise
        
        # 检查生成的点云文件
        fused_ply_path = dense_path / "fused.ply"
        if not fused_ply_path.exists():
            raise FileNotFoundError(f"融合点云文件不存在: {fused_ply_path}")
            
        logger.info(f"稠密重建总耗时: {time.time() - start_time:.2f}秒")
        return fused_ply_path
        
    except Exception as e:
        logger.error(f"稠密重建过程中出错: {str(e)}")
        return None

def reconstruct_scene(config):
    """重建场景主函数"""
    start_time = time.time()
    logger.info("开始3D场景重建...")
    
    # 创建输出目录
    output_path = Path(config['paths']['output_dir']) / "reconstruction"
    output_path.mkdir(exist_ok=True)
    
    # 运行稀疏重建
    image_dir = Path(config['paths']['output_dir']) / "preprocessed"
    if not image_dir.exists() or len(list(image_dir.glob("*.*"))) == 0:
        logger.error(f"预处理图像目录不存在或为空: {image_dir}")
        raise FileNotFoundError(f"预处理图像目录不存在或为空: {image_dir}")
    
    logger.info(f"使用预处理图像: {image_dir}")
    maps = run_colmap_sparse_reconstruction(config, image_dir)
    
    if not maps:
        logger.error("稀疏重建失败！尝试使用备用参数重新运行...")
        # 尝试使用备用参数
        backup_config = config.copy()
        backup_config['reconstruction']['sparse']['min_num_matches'] = 10  # 降低最小匹配点数要求
        backup_config['reconstruction']['sparse']['max_reproj_error'] = 5.0  # 增大最大重投影误差
        maps = run_colmap_sparse_reconstruction(backup_config, image_dir)
        
        if not maps:
            logger.error("使用备用参数稀疏重建仍然失败！")
            raise RuntimeError("稀疏重建失败！")
        else:
            logger.info("使用备用参数稀疏重建成功")
    
    # 保存稀疏重建结果
    logger.info("保存稀疏重建结果...")
    for i, m in enumerate(maps):
        sparse_output_path = output_path / f"sparse_{i}"
        sparse_output_path.mkdir(exist_ok=True)
        m.write(str(sparse_output_path))
        logger.info(f"稀疏重建模型 {i} 已保存至: {sparse_output_path}")
    
    # 运行稠密重建
    logger.info("开始稠密重建...")
    dense_model_path = run_colmap_dense_reconstruction(config)
    
    if not dense_model_path or not dense_model_path.exists():
        logger.error("稠密重建失败！")
        # 尝试使用稀疏点云作为备用
        logger.info("将使用稀疏点云作为最终结果")
        try:
            # 从稀疏重建中提取点云
            sparse_model = maps[0]  # 使用第一个模型
            points3D = sparse_model.points3D
            
            # 创建点云数据
            import numpy as np
            points = np.zeros((len(points3D), 3))
            colors = np.zeros((len(points3D), 3))
            
            for i, (_, p) in enumerate(points3D.items()):
                points[i] = p.xyz
                colors[i] = p.rgb
            
            # 创建Open3D点云
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
            
            # 保存为PLY文件
            sparse_ply_path = output_path / "dense.ply"  # 仍然命名为dense.ply以保持一致性
            o3d.io.write_point_cloud(str(sparse_ply_path), pcd)
            logger.info(f"已将稀疏点云保存为: {sparse_ply_path}")
            dense_model_path = sparse_ply_path
        except Exception as e:
            logger.error(f"提取稀疏点云时出错: {str(e)}")
            raise RuntimeError("重建失败！无法生成有效的点云模型")
    else:
        # 复制稠密重建结果
        import shutil
        dense_output_path = output_path / "dense.ply"
        shutil.copy2(str(dense_model_path), str(dense_output_path))
        logger.info(f"稠密重建结果已保存至: {dense_output_path}")
        dense_model_path = dense_output_path
    
    # 记录重建信息
    with open(output_path / "reconstruction_info.txt", "w") as f:
        f.write(f"重建时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"稀疏重建模型数量: {len(maps)}\n")
        f.write(f"点云文件: {dense_model_path}\n")
        f.write(f"总耗时: {time.time() - start_time:.2f}秒\n")
    
    logger.info(f"重建完成！结果保存在: {output_path}")
    logger.info(f"总耗时: {time.time() - start_time:.2f}秒")
    return output_path

if __name__ == "__main__":
    config = load_config()
    reconstruct_scene(config)