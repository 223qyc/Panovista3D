import open3d as o3d
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os
import time
import logging
import torch
import threading

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

def process_point_cloud(pcd, config):
    """处理点云数据"""
    start_time = time.time()
    logger.info("开始处理点云...")
    
    # 原始点云信息
    original_points = len(pcd.points)
    logger.info(f"原始点云包含 {original_points} 个点")
    
    # 移除离群点
    logger.info("移除离群点...")
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    logger.info(f"移除离群点后剩余 {len(pcd.points)} 个点 (移除了 {original_points - len(pcd.points)} 个点)")
    
    # 根据点云密度设置进行下采样
    density = config['visualization']['point_cloud_density']
    if density == "high":
        voxel_size = 0.005
    elif density == "medium":
        voxel_size = 0.01
    else:  # low
        voxel_size = 0.02
    
    logger.info(f"使用体素大小 {voxel_size} 进行下采样...")
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    logger.info(f"下采样后剩余 {len(pcd.points)} 个点")
    
    # 估计法向量
    logger.info("估计点云法向量...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(k=20)
    
    logger.info(f"点云处理完成，耗时: {time.time() - start_time:.2f}秒")
    return pcd

def setup_camera_and_lighting(vis, config):
    """设置相机和光照"""
    # 获取视图控制
    view_control = vis.get_view_control()
    
    # 设置相机位置
    view_control.set_zoom(0.8)
    view_control.set_front([0, 0, -1])
    view_control.set_lookat([0, 0, 0])
    view_control.set_up([0, 1, 0])
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = config['visualization']['point_size']
    render_option.background_color = np.asarray(config['visualization']['background_color'])
    
    # 设置光照
    if config['visualization']['lighting']:
        render_option.light_on = True
        render_option.point_show_normal = True
    else:
        render_option.light_on = False
        render_option.point_show_normal = False
    
    # 设置材质
    material_type = config['visualization']['material_type']
    if material_type == "normals":
        render_option.point_color_option = o3d.visualization.PointColorOption.Normal
    elif material_type == "depth":
        render_option.point_color_option = o3d.visualization.PointColorOption.ZCoordinate
    else:  # defaultLit
        render_option.point_color_option = o3d.visualization.PointColorOption.Default
    
    # 显示坐标系
    if config['visualization']['show_coordinate_frame']:
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(coordinate_frame)

def save_screenshot(vis, config):
    """保存可视化截图"""
    screenshot_path = Path(config['visualization']['screenshot_path'])
    screenshot_path.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = screenshot_path / f"screenshot_{timestamp}.png"
    vis.capture_screen_image(str(filename))
    logger.info(f"截图已保存至: {filename}")
    return filename

def visualize_point_cloud(config):
    """可视化点云"""
    # 加载点云
    point_cloud_path = Path(config['paths']['output_dir']) / "reconstruction" / "dense.ply"
    if not point_cloud_path.exists():
        raise FileNotFoundError(f"点云文件不存在: {point_cloud_path}")
    
    logger.info(f"加载点云: {point_cloud_path}")
    pcd = o3d.io.read_point_cloud(str(point_cloud_path))
    
    # 处理点云
    pcd = process_point_cloud(pcd, config)
    
    # 创建可视化窗口
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="3D重建点云可视化", width=1280, height=720)
    
    # 添加点云
    vis.add_geometry(pcd)
    
    # 设置相机和光照
    setup_camera_and_lighting(vis, config)
    
    # 添加键盘回调函数
    if config['visualization']['interactive']:
        # 截图功能 - 按S键
        def screenshot_callback(vis):
            save_screenshot(vis, config)
            return False
        vis.register_key_callback(ord('S'), screenshot_callback)
        
        # 切换点云显示模式 - 按M键
        def toggle_point_mode(vis):
            opt = vis.get_render_option()
            current_mode = opt.point_color_option
            if current_mode == o3d.visualization.PointColorOption.Default:
                opt.point_color_option = o3d.visualization.PointColorOption.Normal
                logger.info("切换到法向量显示模式")
            elif current_mode == o3d.visualization.PointColorOption.Normal:
                opt.point_color_option = o3d.visualization.PointColorOption.ZCoordinate
                logger.info("切换到深度显示模式")
            else:
                opt.point_color_option = o3d.visualization.PointColorOption.Default
                logger.info("切换到默认显示模式")
            return False
        vis.register_key_callback(ord('M'), toggle_point_mode)
        
        # 切换光照 - 按L键
        def toggle_lighting(vis):
            opt = vis.get_render_option()
            opt.light_on = not opt.light_on
            logger.info(f"光照: {'开启' if opt.light_on else '关闭'}")
            return False
        vis.register_key_callback(ord('L'), toggle_lighting)
        
        # 显示帮助信息
        logger.info("交互式控制:")
        logger.info("  鼠标左键 - 旋转")
        logger.info("  鼠标右键 - 平移")
        logger.info("  鼠标滚轮 - 缩放")
        logger.info("  S键 - 保存截图")
        logger.info("  M键 - 切换显示模式")
        logger.info("  L键 - 切换光照")
        logger.info("  Q/ESC - 退出")
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

def create_mesh(config):
    """创建网格模型"""
    # 加载点云
    point_cloud_path = Path(config['paths']['output_dir']) / "reconstruction" / "dense.ply"
    pcd = o3d.io.read_point_cloud(str(point_cloud_path))
    
    # 点云预处理
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pcd = pcd.select_by_index(ind)
    
    # 估计法向量
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # 泊松重建
    logger.info("开始泊松重建...")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, 
            depth=9,
            linear_fit=True
        )
    
    # 移除低密度顶点
    logger.info("移除低密度顶点...")
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    # 网格简化
    logger.info("网格简化...")
    target_triangles = int(len(mesh.triangles) * config['visualization']['mesh_simplification'])
    mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    
    # 平滑处理
    logger.info("网格平滑处理...")
    mesh = mesh.filter_smooth_taubin(number_of_iterations=5)
    
    # 计算顶点法向量
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    
    # 保存网格
    output_path = Path(config['paths']['output_dir']) / "reconstruction" / "mesh.ply"
    o3d.io.write_triangle_mesh(str(output_path), mesh)
    logger.info(f"网格已保存至: {output_path}")
    
    return mesh

def visualize_mesh(config):
    """可视化网格模型"""
    try:
        # 检查是否已有网格文件
        mesh_path = Path(config['paths']['output_dir']) / "reconstruction" / "mesh.ply"
        if mesh_path.exists():
            logger.info(f"加载现有网格模型: {mesh_path}")
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            # 确保法向量已计算
            if not mesh.has_vertex_normals():
                logger.info("计算顶点法向量...")
                mesh.compute_vertex_normals()
        else:
            logger.info("创建新的网格模型...")
            mesh = create_mesh(config)
    
        # 创建可视化窗口
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="3D重建网格可视化", width=1280, height=720)
        
        # 添加网格
        vis.add_geometry(mesh)
        
        # 设置相机和光照
        setup_camera_and_lighting(vis, config)
        
        # 添加键盘回调函数
        if config['visualization']['interactive']:
            # 截图功能 - 按S键
            def screenshot_callback(vis):
                save_screenshot(vis, config)
                return False
            vis.register_key_callback(ord('S'), screenshot_callback)
            
            # 切换渲染模式 - 按M键
            def toggle_render_mode(vis):
                opt = vis.get_render_option()
                if opt.mesh_show_wireframe:
                    opt.mesh_show_wireframe = False
                    opt.mesh_show_back_face = True
                    logger.info("切换到实体模式")
                elif opt.mesh_show_back_face:
                    opt.mesh_show_back_face = False
                    opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Flat
                    logger.info("切换到平面着色模式")
                elif opt.mesh_shade_option == o3d.visualization.MeshShadeOption.Flat:
                    opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Smooth
                    logger.info("切换到平滑着色模式")
                else:
                    opt.mesh_show_wireframe = True
                    opt.mesh_shade_option = o3d.visualization.MeshShadeOption.Default
                    logger.info("切换到线框模式")
                return False
            vis.register_key_callback(ord('M'), toggle_render_mode)
            
            # 切换光照 - 按L键
            def toggle_lighting(vis):
                opt = vis.get_render_option()
                opt.light_on = not opt.light_on
                logger.info(f"光照: {'开启' if opt.light_on else '关闭'}")
                return False
            vis.register_key_callback(ord('L'), toggle_lighting)
            
            # 显示帮助信息
            logger.info("交互式控制:")
            logger.info("  鼠标左键 - 旋转")
            logger.info("  鼠标右键 - 平移")
            logger.info("  鼠标滚轮 - 缩放")
            logger.info("  S键 - 保存截图")
            logger.info("  M键 - 切换渲染模式")
            logger.info("  L键 - 切换光照")
            logger.info("  Q/ESC - 退出")
        
        # 运行可视化
        vis.run()
        vis.destroy_window()
    except Exception as e:
        logger.error(f"可视化网格模型时出错: {str(e)}")

def export_high_quality_render(config):
    """导出高质量渲染图像"""
    try:
        # 检查网格文件是否存在
        mesh_path = Path(config['paths']['output_dir']) / "reconstruction" / "mesh.ply"
        if not mesh_path.exists():
            logger.info("网格文件不存在，先创建网格...")
            mesh = create_mesh(config)
        else:
            mesh = o3d.io.read_triangle_mesh(str(mesh_path))
            if not mesh.has_vertex_normals():
                mesh.compute_vertex_normals()
        
        # 创建可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080, visible=False)  # 不显示窗口
        
        # 添加网格
        vis.add_geometry(mesh)
        
        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.background_color = np.asarray(config['visualization']['background_color'])
        render_option.light_on = True
        render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Smooth
        render_option.mesh_show_back_face = True
        
        # 设置相机位置
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)
        view_control.set_front([0, 0, -1])
        view_control.set_lookat([0, 0, 0])
        view_control.set_up([0, 1, 0])
        
        # 渲染并保存高质量图像
        vis.poll_events()
        vis.update_renderer()
        
        # 保存图像
        screenshot_path = Path(config['visualization']['screenshot_path'])
        screenshot_path.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = screenshot_path / f"high_quality_render_{timestamp}.png"
        vis.capture_screen_image(str(filename), do_render=True)
        logger.info(f"高质量渲染已保存至: {filename}")
        
        vis.destroy_window()
        return filename
    except Exception as e:
        logger.error(f"导出高质量渲染时出错: {str(e)}")
        return None

if __name__ == "__main__":
    try:
        config = load_config()
        
        while True:
            print("\n===== 3D重建可视化工具 =====\n")
            print("1. 可视化点云")
            print("2. 可视化网格")
            print("3. 可视化点云和网格")
            print("4. 导出高质量渲染图像")
            print("0. 退出")
            
            choice = input("\n请选择可视化选项 (0-4): ")
            
            if choice == "1":
                visualize_point_cloud(config)
            elif choice == "2":
                visualize_mesh(config)
            elif choice == "3":
                # 先显示点云，然后显示网格
                logger.info("先显示点云，关闭窗口后将显示网格...")
                visualize_point_cloud(config)
                visualize_mesh(config)
            elif choice == "4":
                export_high_quality_render(config)
            elif choice == "0":
                logger.info("退出程序")
                break
            else:
                logger.error("无效的选择！请重新输入。")
    except Exception as e:
        logger.error(f"运行时出错: {str(e)}")