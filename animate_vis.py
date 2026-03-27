import streamlit as st
import folium
from folium.plugins import TimestampedGeoJson
import pandas as pd
import json
import os
import re
from datetime import datetime, timedelta
import networkx as nx
import osmnx as ox

# 页面配置
st.set_page_config(page_title="新能源物流车队协同调度可视化", layout="wide")
st.title("🚛 新能源物流车队协同调度动画系统")

# 使用 session_state 来持久化数据
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = None

class SimulationVisualizer:
    """仿真结果可视化器"""
    
    def __init__(self):
        self.depot_location = None
        self.charging_stations = []
        self.task_points = []
        self.vehicles = []
        self.simulation_log = []
        self.graph = None
        self.distance_matrix = None
        self.snapped_points = None
        self.data_loaded = False
        self.log_parsed = False
        
    def load_data(self):
        """加载仿真数据"""
        try:
            # 加载点位数据
            if os.path.exists("data/snapped_points.csv"):
                self.snapped_points = pd.read_csv("data/snapped_points.csv", index_col=0)
                st.success(f"✅ 成功加载 {len(self.snapped_points)} 个点位数据")
                
                # 重置点位列表
                self.depot_location = None
                self.charging_stations = []
                self.task_points = []
                
                # 解析点位类型
                for idx, row in self.snapped_points.iterrows():
                    point_data = {
                        'id': idx,
                        'lat': row['latitude'],
                        'lon': row['longitude'],
                        'node_id': row['node_id'],
                        'type': row.get('type', 'Unknown')
                    }
                    
                    type_str = str(row.get('type', ''))
                    if 'Depot' in type_str or '仓库' in type_str:
                        self.depot_location = point_data
                    elif 'Charging' in type_str or '充电站' in type_str:
                        self.charging_stations.append(point_data)
                    elif 'Task' in type_str or '任务' in type_str:
                        self.task_points.append(point_data)
                        
            # 加载距离矩阵
            if os.path.exists("data/distance_matrix.csv"):
                self.distance_matrix = pd.read_csv("data/distance_matrix.csv", index_col=0)
                st.success(f"✅ 成功加载 {self.distance_matrix.shape[0]}×{self.distance_matrix.shape[1]} 距离矩阵")
                
            # 加载路网图
            graph_files = ["shanghai_china.graphml", "guangzhou_china.graphml", "beijing_china.graphml"]
            for graph_file in graph_files:
                if os.path.exists(graph_file):
                    self.graph = ox.load_graphml(graph_file)
                    # 确保节点ID为整数
                    self.graph = nx.relabel_nodes(self.graph, int)
                    st.success(f"✅ 成功加载路网图 {graph_file}，包含 {len(self.graph.nodes)} 个节点")
                    break
            
            self.data_loaded = True
            return True
            
        except Exception as e:
            st.error(f"❌ 数据加载失败: {e}")
            self.data_loaded = False
            return False
    
    def parse_simulation_log(self, log_text):
        """解析仿真日志"""
        try:
            lines = log_text.strip().split('\n')
            simulation_data = []
            current_time = 0
            
            for line_idx, line in enumerate(lines):
                # 提取时间信息
                time_match = re.search(r'--- 时间: (\d+) 分钟 ---', line)
                if time_match:
                    current_time = int(time_match.group(1))
                    continue
                
                # 匹配状态行：[状态] Vehicle(...)
                if '[状态]' in line and 'Vehicle(' in line:
                    try:
                        # 提取车辆信息 - 改进的正则表达式
                        vehicle_match = re.search(r'Vehicle\(id=(\d+), loc=(\d+), bat=([\d.]+)kWh, status=(\w+), plan=(\d+)\)', line)
                        if vehicle_match:
                            vehicle_id = int(vehicle_match.group(1))
                            location = int(vehicle_match.group(2))
                            battery = float(vehicle_match.group(3))
                            status = vehicle_match.group(4)
                            plan = int(vehicle_match.group(5))
                            
                            simulation_data.append({
                                'time': current_time,
                                'vehicle_id': vehicle_id,
                                'location': location,
                                'battery': battery,
                                'status': status,
                                'plan': plan
                            })
                            
                    except Exception as e:
                        continue
                        
            self.simulation_log = simulation_data
            self.log_parsed = True
            return len(simulation_data) > 0
            
        except Exception as e:
            st.error(f"❌ 日志解析失败: {e}")
            self.log_parsed = False
            return False
    
    def get_location_coordinates(self, location_node):
        """获取节点的经纬度坐标"""
        if not self.graph or not location_node:
            return None
            
        try:
            location_node = int(location_node)
            if location_node in self.graph.nodes:
                node_data = self.graph.nodes[location_node]
                return (node_data['y'], node_data['x'])  # (lat, lon)
        except:
            pass
            
        # 如果节点不存在，尝试从snapped_points查找
        if self.snapped_points is not None:
            for idx, row in self.snapped_points.iterrows():
                if str(row['node_id']) == str(location_node):
                    return (row['latitude'], row['longitude'])
                    
        return None
    
    def create_static_map_base(self):
        """创建静态地图基础"""
        # 确定地图中心点
        if self.depot_location:
            center_lat, center_lon = self.depot_location['lat'], self.depot_location['lon']
        elif self.simulation_log:
            # 使用第一个车辆的初始位置
            first_location = self.simulation_log[0]['location']
            coords = self.get_location_coordinates(first_location)
            if coords:
                center_lat, center_lon = coords
                # 临时设置仓库位置
                self.depot_location = {'lat': coords[0], 'lon': coords[1], 'node_id': first_location}
            else:
                center_lat, center_lon = 31.2304, 121.4737  # 默认上海
        else:
            center_lat, center_lon = 31.2304, 121.4737  # 默认上海
            
        # 创建地图
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # 添加仓库标记
        if self.depot_location:
            folium.Marker(
                location=[self.depot_location['lat'], self.depot_location['lon']],
                popup=f"🏭 中央仓库 (节点: {self.depot_location.get('node_id', 'Unknown')})",
                tooltip="中央仓库",
                icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
            ).add_to(m)
        
        # 添加充电站标记
        for i, station in enumerate(self.charging_stations):
            folium.Marker(
                location=[station['lat'], station['lon']],
                popup=f"⚡ 充电站 S{i+1} (节点: {station['node_id']})",
                tooltip=f"充电站 S{i+1}",
                icon=folium.Icon(color='green', icon='bolt', prefix='fa')
            ).add_to(m)
        
        # 添加任务点标记
        for i, task in enumerate(self.task_points):
            folium.Marker(
                location=[task['lat'], task['lon']],
                popup=f"📦 任务点 {i+1} (节点: {task['node_id']})",
                tooltip=f"任务点 {i+1}",
                icon=folium.Icon(color='blue', icon='cube', prefix='fa')
            ).add_to(m)
            
        return m
    
    def create_vehicle_path_polylines(self, map_obj):
        """在地图上绘制车辆完整路径的静态轨迹"""
        if not self.simulation_log or not self.graph:
            return
            
        # 按车辆分组并提取路径
        vehicle_paths = {}
        
        for entry in self.simulation_log:
            vehicle_id = entry['vehicle_id']
            location = entry['location']
            
            if vehicle_id not in vehicle_paths:
                vehicle_paths[vehicle_id] = []
                
            coords = self.get_location_coordinates(location)
            if coords and coords not in vehicle_paths[vehicle_id]:
                vehicle_paths[vehicle_id].append(coords)
        
        # 为每辆车绘制路径
        colors = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightblue']
        
        for vehicle_id, path_coords in vehicle_paths.items():
            if len(path_coords) >= 2:
                color = colors[vehicle_id % len(colors)]
                
                folium.PolyLine(
                    locations=path_coords,
                    color=color,
                    weight=3,
                    opacity=0.4,
                    popup=f"🚛 车辆 {vehicle_id} 行驶轨迹"
                ).add_to(map_obj)
    
    def create_timestamped_geojson(self):
        """创建基于时间的GeoJSON数据用于动画"""
        if not self.simulation_log:
            return None
            
        features = []
        
        # 按时间和车辆组织数据
        time_vehicle_data = {}
        for entry in self.simulation_log:
            time_key = entry['time']
            vehicle_id = entry['vehicle_id']
            
            if time_key not in time_vehicle_data:
                time_vehicle_data[time_key] = {}
            time_vehicle_data[time_key][vehicle_id] = entry
        
        # 为每个时间点创建GeoJSON features
        for time_minute, vehicles_data in time_vehicle_data.items():
            for vehicle_id, vehicle_data in vehicles_data.items():
                coords = self.get_location_coordinates(vehicle_data['location'])
                if not coords:
                    continue
                
                # 根据状态确定颜色和图标
                status = vehicle_data['status']
                status_info = self.get_status_display_info(status)
                
                # 计算电量百分比
                battery_percent = vehicle_data['battery']
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [coords[1], coords[0]]  # [lon, lat]
                    },
                    "properties": {
                        "time": f"2024-01-01T{time_minute//60:02d}:{time_minute%60:02d}:00",
                        "popup": f"""
                        <div style='font-family: Arial; font-size: 12px; width: 200px;'>
                            <b style='color: {status_info['color']};'>{status_info['icon']} 车辆 {vehicle_id}</b><br>
                            <b>状态:</b> {status_info['display']}<br>
                            <b>电量:</b> {battery_percent:.1f} kWh<br>
                            <b>计划:</b> {vehicle_data['plan']} 个任务<br>
                            <b>位置:</b> {vehicle_data['location']}<br>
                            <b>时间:</b> {time_minute} 分钟
                        </div>
                        """,
                        "tooltip": f"车辆{vehicle_id} | {status_info['display']} | {battery_percent:.0f}kWh",
                        "icon": "circle",
                        "iconstyle": {
                            "fillColor": status_info['color'],
                            "color": "black",
                            "weight": 2,
                            "fillOpacity": 0.8,
                            "radius": 10 + vehicle_id * 2  # 不同车辆稍微不同大小
                        }
                    }
                }
                features.append(feature)
        
        return {
            "type": "FeatureCollection",
            "features": features
        }
    
    def get_status_display_info(self, status):
        """根据状态返回显示信息"""
        status_map = {
            'IDLE': {'color': 'gray', 'icon': '⏸️', 'display': '待命中'},
            'MOVING_TO_TASK': {'color': 'blue', 'icon': '🚛', 'display': '前往任务'},
            'MOVING_TO_STATION': {'color': 'cyan', 'icon': '🔋', 'display': '前往充电站'},
            'MOVING_TO_DEPOT': {'color': 'purple', 'icon': '🏠', 'display': '返回仓库'},
            'CHARGING': {'color': 'green', 'icon': '⚡', 'display': '充电中'},
            'SERVICING': {'color': 'orange', 'icon': '📦', 'display': '服务中'},
        }
        return status_map.get(status, {'color': 'black', 'icon': '❓', 'display': status})
    
    def create_animated_map(self):
        """创建完整的动画地图"""
        # 创建基础地图
        m = self.create_static_map_base()
        if not m:
            return None
        
        # 添加车辆路径
        self.create_vehicle_path_polylines(m)
        
        # 创建时间动画数据
        timestamped_geojson = self.create_timestamped_geojson()
        if timestamped_geojson:
            # 添加时间动画图层
            TimestampedGeoJson(
                timestamped_geojson,
                period="PT1M",  # 每分钟一帧
                add_last_point=True,
                auto_play=False,  # 默认不自动播放，让用户手动控制
                loop=True,
                max_speed=3,  # 播放速度
                loop_button=True,
                date_options="HH:mm",
                time_slider_drag_update=True,
                duration="PT2S"  # 每帧持续2秒
            ).add_to(m)
        
        # 添加图层控制
        folium.LayerControl().add_to(m)
        
        return m
    
    def get_data_status(self):
        """获取数据状态信息"""
        return {
            'data_loaded': self.data_loaded,
            'log_parsed': self.log_parsed,
            'has_depot': self.depot_location is not None,
            'num_stations': len(self.charging_stations),
            'num_tasks': len(self.task_points),
            'num_log_entries': len(self.simulation_log)
        }

def main():
    """主函数"""
    # 初始化或获取可视化器实例
    if st.session_state.visualizer is None:
        st.session_state.visualizer = SimulationVisualizer()
    
    visualizer = st.session_state.visualizer
    
    st.sidebar.header("📊 控制面板")
    
    # 显示当前状态
    status = visualizer.get_data_status()
    st.sidebar.subheader("🔍 当前状态")
    
    status_text = f"""
    📁 **数据加载**: {'✅ 完成' if status['data_loaded'] else '❌ 未完成'}
    📋 **日志解析**: {'✅ 完成' if status['log_parsed'] else '❌ 未完成'}
    🏭 **仓库点**: {'✅ 已识别' if status['has_depot'] else '❌ 未识别'}
    ⚡ **充电站**: {status['num_stations']} 个
    📦 **任务点**: {status['num_tasks']} 个
    📊 **日志记录**: {status['num_log_entries']} 条
    """
    st.sidebar.markdown(status_text)
    
    # 数据加载状态
    st.sidebar.subheader("1️⃣ 数据加载")
    if st.sidebar.button("🔄 加载数据文件"):
        with st.spinner("正在加载数据..."):
            success = visualizer.load_data()
            if success:
                st.sidebar.success("数据加载完成!")
                st.rerun()  # 刷新页面以显示新状态
    
    # 仿真日志输入
    st.sidebar.subheader("2️⃣ 仿真日志")
    
    # 文件上传选项
    uploaded_file = st.sidebar.file_uploader("上传日志文件", type=['txt', 'log'], key="log_upload")
    
    log_input = ""
    if uploaded_file is not None:
        try:
            log_input = uploaded_file.read().decode('utf-8')
            st.sidebar.success(f"✅ 成功加载日志文件 ({len(log_input)} 字符)")
        except Exception as e:
            st.sidebar.error(f"❌ 文件读取失败: {e}")
    else:
        # 文本框输入
        log_input = st.sidebar.text_area(
            "或直接粘贴仿真日志:",
            height=200,
            help="粘贴simulation_gurobi.py的输出日志",
            placeholder="将您的仿真日志粘贴在这里...",
            key="log_textarea"
        )
    
    # 解析日志按钮
    if st.sidebar.button("📋 解析日志") and log_input.strip():
        with st.spinner("正在解析仿真日志..."):
            success = visualizer.parse_simulation_log(log_input)
            if success:
                st.sidebar.success(f"✅ 成功解析 {len(visualizer.simulation_log)} 条记录")
                st.rerun()  # 刷新页面以显示新状态
            else:
                st.sidebar.error("❌ 日志解析失败，请检查日志格式")
    
    # 清除数据按钮
    st.sidebar.subheader("🗑️ 数据管理")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("清除数据", help="清除已加载的点位数据"):
            visualizer.data_loaded = False
            visualizer.depot_location = None
            visualizer.charging_stations = []
            visualizer.task_points = []
            visualizer.graph = None
            visualizer.snapped_points = None
            visualizer.distance_matrix = None
            st.sidebar.success("数据已清除")
            st.rerun()
    
    with col2:
        if st.button("清除日志", help="清除已解析的仿真日志"):
            visualizer.log_parsed = False
            visualizer.simulation_log = []
            st.sidebar.success("日志已清除")
            st.rerun()
    
    # 主要内容区域
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🗺️ 车队调度动画")
        
        # 检查准备状态
        ready_for_animation = status['data_loaded'] and status['log_parsed']
        
        if not ready_for_animation:
            st.warning("⚠️ 请先完成以下步骤：")
            if not status['data_loaded']:
                st.write("- 🔄 加载数据文件")
            if not status['log_parsed']:
                st.write("- 📋 解析仿真日志")
        
        # 生成地图按钮
        if st.button("🎬 生成动画地图", type="primary", disabled=not ready_for_animation):
            if ready_for_animation:
                with st.spinner("正在生成动画地图..."):
                    try:
                        animated_map = visualizer.create_animated_map()
                        
                        if animated_map:
                            # 使用streamlit-folium显示地图
                            try:
                                from streamlit_folium import st_folium
                                map_data = st_folium(
                                    animated_map, 
                                    width=1000, 
                                    height=600, 
                                    returned_objects=["last_clicked"],
                                    key="animation_map"
                                )
                                
                                # 显示使用说明
                                st.info("""
                                🎮 **动画控制说明：**
                                - 点击播放按钮开始动画
                                - 拖拽时间滑块查看特定时间点
                                - 点击车辆标记查看详细状态信息
                                - 使用图层控制开关显示不同元素
                                """)
                                
                                # 显示地图交互信息
                                if map_data and map_data.get('last_clicked'):
                                    st.write("🎯 **最后点击位置:**", map_data['last_clicked'])
                                
                            except ImportError:
                                st.error("❌ 请安装streamlit-folium: `pip install streamlit-folium`")
                                
                                # 备用方案：保存HTML文件
                                map_html = "temp_animation_map.html"
                                animated_map.save(map_html)
                                st.success(f"📁 地图已保存到 {map_html}")
                                
                                # 提供下载链接
                                with open(map_html, 'r', encoding='utf-8') as f:
                                    html_content = f.read()
                                st.download_button(
                                    label="📥 下载HTML地图文件",
                                    data=html_content,
                                    file_name="vehicle_animation.html",
                                    mime="text/html"
                                )
                        else:
                            st.error("❌ 地图生成失败，请检查数据完整性")
                            
                    except Exception as e:
                        st.error(f"❌ 地图生成过程中出现错误: {e}")
            else:
                st.error("⚠️ 请先完成数据加载和日志解析")
    
    with col2:
        st.subheader("📈 统计信息")
        
        if visualizer.simulation_log:
            # 基本统计
            total_records = len(visualizer.simulation_log)
            vehicle_ids = set(entry['vehicle_id'] for entry in visualizer.simulation_log)
            max_time = max(entry['time'] for entry in visualizer.simulation_log) if visualizer.simulation_log else 0
            
            st.metric("总记录数", total_records)
            st.metric("车辆数量", len(vehicle_ids))
            st.metric("仿真时长", f"{max_time} 分钟")
            
            # 状态分布统计
            status_counts = {}
            for entry in visualizer.simulation_log:
                status = entry['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            st.subheader("🔄 状态分布")
            for status, count in status_counts.items():
                status_info = visualizer.get_status_display_info(status)
                st.write(f"{status_info['icon']} **{status_info['display']}**: {count}")
                
            # 电量统计
            battery_levels = [entry['battery'] for entry in visualizer.simulation_log]
            if battery_levels:
                st.subheader("🔋 电量统计")
                st.write(f"最高电量: {max(battery_levels):.1f} kWh")
                st.write(f"最低电量: {min(battery_levels):.1f} kWh")
                avg_battery = sum(battery_levels) / len(battery_levels)
                st.write(f"平均电量: {avg_battery:.1f} kWh")
        
        # 数据概览
        if status['data_loaded']:
            st.subheader("📍 点位概览")
            if visualizer.depot_location:
                st.write(f"🏭 仓库: 1个")
            st.write(f"⚡ 充电站: {len(visualizer.charging_stations)}个")
            st.write(f"📦 任务点: {len(visualizer.task_points)}个")
            
        # 帮助信息
        st.subheader("❓ 使用帮助")
        st.write("""
        **数据要求:**
        - `data/snapped_points.csv`: 点位坐标
        - `shanghai_china.graphml`: 路网图
        - 仿真日志: 包含车辆状态的文本
        
        **操作步骤:**
        1. 🔄 点击"加载数据文件"
        2. 📋 粘贴或上传仿真日志
        3. 📋 点击"解析日志"
        4. 🎬 点击"生成动画地图"
        
        **功能说明:**
        - 🔄 自动解析车辆轨迹
        - 🎬 时间轴动画播放
        - 📊 实时状态显示
        - 🗺️ 交互式地图控制
        """)
        
        # 调试信息
        if st.checkbox("显示调试信息"):
            st.subheader("🔍 调试信息")
            st.write("**Visualizer对象ID:**", id(visualizer))
            st.write("**Session State Keys:**", list(st.session_state.keys()))
            if visualizer.simulation_log:
                st.write("**日志前3条:**")
                for i, entry in enumerate(visualizer.simulation_log[:3]):
                    st.write(f"  {i+1}: {entry}")

if __name__ == "__main__":
    main()