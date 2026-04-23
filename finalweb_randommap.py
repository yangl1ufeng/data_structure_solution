import streamlit as st
import folium
from folium.plugins import TimestampedGeoJson
import pandas as pd
from streamlit_folium import st_folium
import path_processor
import json
import random
import os
import re
from datetime import datetime, timedelta
import networkx as nx
import osmnx as ox
import subprocess
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
import threading
import time

# 页面配置
st.set_page_config(
    page_title="新能源物流车队协同调度系统 - 集成版",
    page_icon="🚚",
    layout="wide"
)

# 状态管理初始化
if "center" not in st.session_state:
    st.session_state["center"] = [31.2304, 121.4737]
if "zoom" not in st.session_state:
    st.session_state["zoom"] = 10
if "points" not in st.session_state:
    st.session_state["points"] = []
if "last_path_geometry" not in st.session_state:
    st.session_state["last_path_geometry"] = None
if "visualizer" not in st.session_state:
    st.session_state.visualizer = None
if "simulation_log" not in st.session_state:
    st.session_state.simulation_log = ""
if "current_stage" not in st.session_state:
    st.session_state.current_stage = "setup"  # setup, processing, visualization
if "auto_simulation_running" not in st.session_state:
    st.session_state.auto_simulation_running = False

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
            os.makedirs("data", exist_ok=True) 
            # 加载点位数据
            if os.path.exists("data/snapped_points.csv"):
                self.snapped_points = pd.read_csv("data/snapped_points.csv", index_col=0)
                
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
                
            # 加载路网图
            graph_files = ["shanghai_china.graphml", "guangzhou_china.graphml", "beijing_china.graphml"]
            for graph_file in graph_files:
                if os.path.exists(graph_file):
                    self.graph = ox.load_graphml(graph_file)
                   #self.graph = nx.relabel_nodes(self.graph, int)
                    break
            
            self.data_loaded = True
            return True
            
        except Exception as e:
            st.error(f"❌ 数据加载失败: {e}")
            self.data_loaded = False
            return False

   
    def generate_random_scenario(self, num_tasks=10, num_stations=5, vehicle_count=5):
        """
        在已加载的路网中随机生成场景，并同步到 session_state["points"]
        """
        if not hasattr(self, 'graph') or self.graph is None:
            st.error("❌ 尚未加载路网图，无法生成场景！")
            return None

        # 获取所有节点的ID和属性
        all_nodes = list(self.graph.nodes(data=True))
        
        # 1. 随机选取点位
        depot_node = random.choice(all_nodes)
        station_samples = random.sample(all_nodes, num_stations)
        task_samples = random.sample(all_nodes, num_tasks)
        
        # 2. 构造 path_processor 需要的标准点位列表格式 (关键！)
        new_points = []
        
        # 仓库
        new_points.append({
            "type": "🏭 中央仓库 (Depot)",
            "latitude": float(depot_node[1]['y']),
            "longitude": float(depot_node[1]['x'])
        })
        
        # 充电站
        for n in station_samples:
            new_points.append({
                "type": "⚡ 充电站 (Charging Station)",
                "latitude": float(n[1]['y']),
                "longitude": float(n[1]['x'])
            })
            
        # 任务点
        for n in task_samples:
            new_points.append({
                "type": "📦 任务目标点 (Task Point)",
                "latitude": float(n[1]['y']),
                "longitude": float(n[1]['x'])
            })
        
        # 3. 更新全局状态，这样 Tab 1 的列表和地图也会同步更新
        st.session_state["points"] = new_points
        
        # 4. 原有的 JSON 保存逻辑可以保留（如果你的仿真脚本需要读取它）
        config = {
            "vehicle_count": vehicle_count,
            "depot": {"id": int(depot_node[0]), "lat": float(depot_node[1]['y']), "lon": float(depot_node[1]['x'])},
            "stations": [{"id": int(n[0]), "lat": float(n[1]['y']), "lon": float(n[1]['x'])} for n in station_samples],
            "tasks": [{"id": f"T{i}", "node_id": int(n[0]), "lat": float(n[1]['y']), "lon": float(n[1]['x'])} for i, n in enumerate(task_samples)]
        }
        with open("custom_scenario.json", "w", encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
            
        return new_points

    def parse_simulation_log(self, log_text):
        """解析仿真日志"""
        try:
            if log_text is None:
                return False

            lines = log_text.strip().split('\n')
            simulation_data = []
            current_time = 0
            
            for line_idx, line in enumerate(lines):
                # 提取时间信息
                time_match = re.search(r'--- 时间: (\d+) 分钟 ---', line)
                if time_match:
                    current_time = int(time_match.group(1))
                    continue
                
                # 匹配状态行
                if '[状态]' in line and 'Vehicle(' in line:
                    try:
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
                return (node_data['y'], node_data['x'])
        except:
            pass
            
        if self.snapped_points is not None:
            for idx, row in self.snapped_points.iterrows():
                if str(row['node_id']) == str(location_node):
                    return (row['latitude'], row['longitude'])
                    
        return None
    
    def create_static_map_base(self):
        """创建静态地图基础"""
        if self.depot_location:
            center_lat, center_lon = self.depot_location['lat'], self.depot_location['lon']
        elif self.simulation_log:
            first_location = self.simulation_log[0]['location']
            coords = self.get_location_coordinates(first_location)
            if coords:
                center_lat, center_lon = coords
                self.depot_location = {'lat': coords[0], 'lon': coords[1], 'node_id': first_location}
            else:
                center_lat, center_lon = 31.2304, 121.4737
        else:
            center_lat, center_lon = 31.2304, 121.4737
            
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        if self.depot_location:
            folium.Marker(
                location=[self.depot_location['lat'], self.depot_location['lon']],
                popup=f"🏭 中央仓库 (节点: {self.depot_location.get('node_id', 'Unknown')})",
                tooltip="中央仓库",
                icon=folium.Icon(color='red', icon='warehouse', prefix='fa')
            ).add_to(m)
        
        for i, station in enumerate(self.charging_stations):
            folium.Marker(
                location=[station['lat'], station['lon']],
                popup=f"⚡ 充电站 S{i+1} (节点: {station['node_id']})",
                tooltip=f"充电站 S{i+1}",
                icon=folium.Icon(color='green', icon='bolt', prefix='fa')
            ).add_to(m)
        
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
            
        vehicle_paths = {}
        
        for entry in self.simulation_log:
            vehicle_id = entry['vehicle_id']
            location = entry['location']
            
            if vehicle_id not in vehicle_paths:
                vehicle_paths[vehicle_id] = []
                
            coords = self.get_location_coordinates(location)
            if coords and coords not in vehicle_paths[vehicle_id]:
                vehicle_paths[vehicle_id].append(coords)
        
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
        
        time_vehicle_data = {}
        for entry in self.simulation_log:
            time_key = entry['time']
            vehicle_id = entry['vehicle_id']
            
            if time_key not in time_vehicle_data:
                time_vehicle_data[time_key] = {}
            time_vehicle_data[time_key][vehicle_id] = entry
        
        for time_minute, vehicles_data in time_vehicle_data.items():
            for vehicle_id, vehicle_data in vehicles_data.items():
                coords = self.get_location_coordinates(vehicle_data['location'])
                if not coords:
                    continue
                
                status = vehicle_data['status']
                status_info = self.get_status_display_info(status)
                
                battery_percent = vehicle_data['battery']
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [coords[1], coords[0]]
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
                            "radius": 10 + vehicle_id * 2
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
        m = self.create_static_map_base()
        if not m:
            return None
        
        self.create_vehicle_path_polylines(m)
        
        timestamped_geojson = self.create_timestamped_geojson()
        if timestamped_geojson:
            TimestampedGeoJson(
                timestamped_geojson,
                period="PT1M",
                add_last_point=True,
                auto_play=False,
                loop=True,
                max_speed=3,
                loop_button=True,
                date_options="HH:mm",
                time_slider_drag_update=True,
                duration="PT2S"
            ).add_to(m)
        
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

def run_simulation_background(progress_container):
    """在后台运行仿真"""
    try:
        progress_bar = progress_container.progress(0)
        status_text = progress_container.text("正在运行仿真...")
        
        # 设置Python路径并运行仿真脚本
        result=subprocess.run(
            [sys.executable, "simulation_gurobi.py"],
            capture_output=True,
            text=True,
            encoding='utf-8',    # <-- 强制使用 UTF-8 编码读中文
            errors='ignore',     # <-- 遇到怪字符直接跳过，不要报错
            timeout=180
        )
        
        progress_bar.progress(80)
        status_text.text("仿真完成，处理输出...")
        
        if result.returncode == 0:
            st.session_state.simulation_log = result.stdout
            progress_bar.progress(100)
            status_text.text("✅ 仿真执行成功！")
            return True, result.stdout
        else:
            error_msg = f"仿真执行失败：\n{result.stderr}"
            st.error(error_msg)
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        error_msg = "⏰ 仿真执行超时（超过3分钟）"
        st.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"❌ 运行仿真时出现错误: {e}"
        st.error(error_msg)
        return False, str(e)
    finally:
        st.session_state.auto_simulation_running = False

def main():
    """主函数"""
    st.title("🚚 新能源物流车队协同调度系统 - 集成版")
    
    # 初始化可视化器
    if st.session_state.visualizer is None:
        st.session_state.visualizer = SimulationVisualizer()
        # === 修改：初始化后立刻尝试加载地图文件 ===
        st.session_state.visualizer.load_data() 
    
    visualizer = st.session_state.visualizer
    
    # 创建标签页
    tab1, tab2, tab3 = st.tabs(["📍 节点配置", "🚀 自动化流程", "📊 动画可视化"])
    
    # === 标签页1: 节点配置 ===
    with tab1:
        st.header("📍 节点配置与路径计算")
        
        # 页面布局
        sidebar_config, main_map_config = st.columns([1, 3])
        
        with sidebar_config:
            st.subheader("控制面板")
            
            # 节点类型选择
            point_type = st.radio(
                "选择要添加的节点类型:",
                ("🏭 中央仓库 (Depot)", "⚡ 充电站 (Charging Station)", "📦 任务目标点 (Task Point)"),
                key="point_type_selector"
            )
            
            st.info("""
            **操作说明:**
            1. 选择节点类型
            2. 在右侧地图上点击添加节点
            3. 中央仓库只能设置一个
            4. 建议至少添加: 1个仓库 + 2个充电站 + 3个任务点
            """)
            
            # 已选点位列表
            st.subheader("已选点位列表")
            if st.session_state["points"]:
                for i in reversed(range(len(st.session_state["points"]))):
                    point = st.session_state["points"][i]
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.text(f"ID: {i+1} - {point['type']}")
                        st.caption(f"Lat: {point['latitude']:.4f}, Lon: {point['longitude']:.4f}")
                    with col2:
                        if st.button("删除", key=f"delete_{i}", use_container_width=True):
                            st.session_state["points"].pop(i)
                            st.rerun()
                    st.divider()
                
                # 导出配置
                points_df = pd.DataFrame(st.session_state["points"])
                json_data = points_df.to_json(orient="records", indent=4)
                
                st.download_button(
                    label="📥 导出配置",
                    data=json_data,
                    file_name="fleet_config.json",
                    mime="application/json",
                )
            else:
                st.warning("当前未添加任何点位。")
            
            # 路径计算
            st.subheader("🌍 路径规划")
            selected_city = st.selectbox(
                "计算区域:",
                ("Shanghai, China", "Guangzhou, China"),
                key="city_selector"
            )
            
            if st.button("🚀 计算最短路径", use_container_width=True, type="primary"):
                if not st.session_state["points"]:
                    st.error("点位列表为空，无法计算。")
                else:
                    with st.spinner(f"正在处理 '{selected_city}' 的路网并计算路径..."):
                        try:
                            snapped_df, dist_matrix_df, example_path_geom = path_processor.process_all(
                                st.session_state["points"],
                                place_name=selected_city
                            )
                            
                            if snapped_df is not None:
                                st.success("✅ 路径计算完成！数据已保存到 'data' 文件夹。")
                                st.session_state["last_path_geometry"] = example_path_geom
                                st.session_state.current_stage = "processing"
                                st.rerun()
                            else:
                                st.error("路径计算失败。")
                        except Exception as e:
                            st.error(f"❌ 路径计算出错: {e}")
        
        with main_map_config:
            st.subheader("🗺️ 节点配置地图")
            
            # 创建地图
            m = folium.Map(
                location=st.session_state["center"],
                zoom_start=st.session_state["zoom"],
                tiles="OpenStreetMap"
            )
            
            # 点位样式
            POINT_STYLE = {
                "🏭 中央仓库 (Depot)": {"icon": "home", "color": "red"},
                "⚡ 充电站 (Charging Station)": {"icon": "bolt", "color": "green"},
                "📦 任务目标点 (Task Point)": {"icon": "cube", "color": "blue"},
            }
            
            # 添加已有点位
            for point in st.session_state["points"]:
                style = POINT_STYLE[point["type"]]
                folium.Marker(
                    location=[point["latitude"], point["longitude"]],
                    tooltip=f"{point['type']}<br>Lat: {point['latitude']:.4f}<br>Lon: {point['longitude']:.4f}",
                    icon=folium.Icon(icon=style["icon"], color=style["color"], prefix='fa')
                ).add_to(m)
            
            # 绘制路径
            if st.session_state["last_path_geometry"]:
                folium.PolyLine(
                    locations=st.session_state["last_path_geometry"],
                    color="purple",
                    weight=5,
                    opacity=0.8,
                    tooltip="计算出的最短路径"
                ).add_to(m)
            
            # 显示地图并处理交互
            map_data = st_folium(m, width='100%', height=600, key="config_map")
            
            # 处理点击事件
            if map_data and map_data["last_clicked"]:
                lat = map_data["last_clicked"]["lat"]
                lon = map_data["last_clicked"]["lng"]
                
                if point_type == "🏭 中央仓库 (Depot)":
                    depot_exists = any(p["type"] == "🏭 中央仓库 (Depot)" for p in st.session_state["points"])
                    if depot_exists:
                        st.error("❌ 中央仓库只能设置一个！请先删除已有仓库。")
                    else:
                        st.session_state["points"].append({"type": point_type, "latitude": lat, "longitude": lon})
                        st.rerun()
                else:
                    st.session_state["points"].append({"type": point_type, "latitude": lat, "longitude": lon})
                    st.rerun()
    
    # === 标签页2: 自动化流程 ===
    with tab2:
        st.header("🚀 自动化仿真流程")
        
        # 流程状态显示
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if os.path.exists("data/snapped_points.csv") and os.path.exists("data/distance_matrix.csv"):
                st.success("✅ **步骤1: 数据准备**\n\n数据文件已就绪")
                data_ready = True
            else:
                st.warning("⏳ **步骤1: 数据准备**\n\n请先在节点配置页面计算路径")
                data_ready = False
        
        with col2:
            if st.session_state.simulation_log:
                st.success("✅ **步骤2: 仿真执行**\n\n仿真日志已生成")
                sim_ready = True
            else:
                if data_ready:
                    st.info("🔄 **步骤2: 仿真执行**\n\n准备运行仿真")
                else:
                    st.warning("⏳ **步骤2: 仿真执行**\n\n等待数据准备")
                sim_ready = False
        
        with col3:
            if sim_ready:
                st.info("🎬 **步骤3: 动画生成**\n\n可以生成可视化动画")
            else:
                st.warning("⏳ **步骤3: 动画生成**\n\n等待仿真完成")
        
        st.divider()
                # --- 修改 1: 一键自动化按钮 (去掉了 disabled 判断) ---
        if st.button("🚀 一键自动化执行", type="primary", use_container_width=True):
            # 去掉了 if data_ready 的二次检查，让程序直接尝试运行
            if not st.session_state.auto_simulation_running:
                st.session_state.auto_simulation_running = True
                
                # 创建进度显示区域
                progress_container = st.empty()
                
                with st.spinner("正在执行自动化流程..."):
                    # 步骤1: 加载数据
                    st.info("📁 步骤1/3: 加载数据文件...")
                    # 即使文件不存在，我们也让它尝试加载，看看报错信息
                    visualizer.load_data()
                    
                    # 步骤2: 运行仿真
                    st.info("⚙️ 步骤2/3: 运行仿真引擎...")
                    success, log_output = run_simulation_background(progress_container)
                    
                    if success:
                        # 步骤3: 解析日志
                        st.info("📋 步骤3/3: 解析仿真日志...")
                        visualizer.parse_simulation_log(log_output)
                        
                        st.success("🎉 自动化流程完成！请切换到动画可视化页面查看结果。")
                        st.session_state.current_stage = "visualization"
                    
                progress_container.empty()
        
        # 手动执行选项
        st.subheader("📋 手动控制选项")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # --- 修改 2: 重新加载数据按钮 (改为永远可用) ---
            if st.button("🔄 重新加载数据"):
                with st.spinner("加载数据中..."):
                    success = visualizer.load_data()
                    if success:
                        st.success("✅ 数据加载完成")
                    else:
                        st.error("❌ 数据加载失败：请检查 data 文件夹下是否有 csv 文件")
        
        with col2:
            # --- 修改 3: 运行仿真按钮 (去掉了 data_ready 限制) ---
            if st.button("⚙️ 运行仿真", disabled=st.session_state.auto_simulation_running):
                st.session_state.auto_simulation_running = True
                progress_container = st.empty()
                
                with st.spinner("运行仿真中..."):
                    success, log_output = run_simulation_background(progress_container)
                    if success:
                        visualizer.parse_simulation_log(log_output)
                        st.success("✅ 仿真完成")
                
                progress_container.empty()


        
        # 仿真日志显示
        if st.session_state.simulation_log:
            with st.expander("📄 查看仿真日志"):
                st.text_area("仿真输出:", value=st.session_state.simulation_log, height=300, key="sim_log_display")
        
        # 文件上传选项
        st.subheader("📤 或上传现有日志")
        uploaded_file = st.file_uploader("上传仿真日志文件", type=['txt', 'log'], key="manual_log_upload")
        
        if uploaded_file is not None:
            try:
                log_content = uploaded_file.read().decode('utf-8')
                st.session_state.simulation_log = log_content
                
                if visualizer.parse_simulation_log(log_content):
                    st.success(f"✅ 成功解析日志文件 ({len(visualizer.simulation_log)} 条记录)")
                else:
                    st.error("❌ 日志解析失败")
            except Exception as e:
                st.error(f"❌ 文件读取失败: {e}")
    
    # === 标签页3: 动画可视化 ===
        # === 标签页3: 动画可视化 ===
    with tab3:
        st.header("📊 仿真结果动画可视化")
        
        # 【修改点 1】：在 Tab 3 开始处初始化一个显示状态
        if "show_map_active" not in st.session_state:
            st.session_state.show_map_active = False

        # --- 以下内容完全保持你原来的代码逻辑 ---
        try:
            status = visualizer.get_data_status()
            if not isinstance(status, dict):
                status = {
                    'data_loaded': False, 'log_parsed': False, 'has_depot': False,
                    'num_stations': 0, 'num_tasks': 0, 'num_log_entries': 0
                }
        except Exception as e:
            st.error(f"❌ 获取状态信息失败: {e}")
            status = {'data_loaded': False, 'log_parsed': False}
        
        ready_for_animation = status.get('data_loaded', False) and status.get('log_parsed', False)
        
        # 侧边栏部分 (完全保留你的原始代码)
        with st.sidebar:
            st.header("📊 系统状态")
            try:
                status_text = f"""
                📁 **数据加载**: {'✅ 完成' if status.get('data_loaded', False) else '❌ 未完成'}
                📋 **日志解析**: {'✅ 完成' if status.get('log_parsed', False) else '❌ 未完成'}
                🏭 **仓库点**: {'✅ 已识别' if status.get('has_depot', False) else '❌ 未识别'}
                ⚡ **充电站**: {status.get('num_stations', 0)} 个
                📦 **任务点**: {status.get('num_tasks', 0)} 个
                📊 **日志记录**: {status.get('num_log_entries', 0)} 条
                """
                st.markdown(status_text)
            except Exception as e:
                st.error(f"❌ 状态显示失败: {e}")

            st.sidebar.markdown("---")
            st.sidebar.subheader("🎯 随机场景生成器")
            s_tasks = st.sidebar.slider("任务数量", 5, 100, 20)
            s_stations = st.sidebar.slider("充电站数量", 2, 15, 5)
            s_vehicles = st.sidebar.slider("车队规模", 2, 20, 8)
            
            if st.sidebar.button("🎲 生成并应用新场景", use_container_width=True):
                # ... (这里是你之前的随机生成逻辑，保持不变) ...
                if visualizer.graph is None:
                    visualizer.load_data()

                if visualizer.graph is not None:
                    with st.spinner("正在生成随机点并计算路网路径..."):
                        random_points = visualizer.generate_random_scenario(s_tasks, s_stations, s_vehicles)
                        if random_points:
                            snapped_df, dist_matrix_df, example_path_geom = path_processor.process_all(
                                random_points, place_name="Shanghai, China"
                            )
                            if snapped_df is not None:
                                visualizer.load_data()
                                st.session_state["last_path_geometry"] = example_path_geom
                                # 生成新场景时，把之前的地图关闭
                                st.session_state.show_map_active = False 
                                st.sidebar.success(f"✅ 成功应用新场景：{s_tasks}个任务点")
                                st.rerun() 
                else:
                    st.sidebar.error("❌ 请先确保地图已加载！")

            if visualizer.simulation_log and len(visualizer.simulation_log) > 0:
                # ... (统计信息部分，保持不变) ...
                st.header("📈 统计信息")
                try:
                    total_records = len(visualizer.simulation_log)
                    vehicle_ids = set(entry['vehicle_id'] for entry in visualizer.simulation_log)
                    max_time = max(entry['time'] for entry in visualizer.simulation_log) if visualizer.simulation_log else 0
                    st.metric("总记录数", total_records)
                    st.metric("车辆数量", len(vehicle_ids))
                    st.metric("仿真时长", f"{max_time} 分钟")
                except: pass
            
            # 数据管理与导出 (保持不变)
            st.header("⚙️ 数据与导出")
            try:
                if st.session_state.simulation_log:
                    st.download_button(label="📥 下载仿真日志", data=st.session_state.simulation_log, 
                                     file_name=f"simulation_log.txt", mime="text/plain", use_container_width=True)
                if status.get('data_loaded', False) and visualizer.snapped_points is not None:
                    csv_data = visualizer.snapped_points.to_csv()
                    st.download_button(label="📥 下载点位数据", data=csv_data, 
                                     file_name="snapped_points_export.csv", mime="text/csv", use_container_width=True)
            except Exception as e: st.error(f"❌ 导出错误: {e}")

            if st.button("🗑️ 清空所有数据", use_container_width=True):
                visualizer.data_loaded = False
                visualizer.simulation_log = []
                st.session_state.show_map_active = False # 清空时关闭地图
                st.rerun()

        # --- 主要可视化区域 ---
        if not ready_for_animation:
            st.warning("⚠️ 请先完成自动化仿真流程，或使用左侧“随机场景生成器”生成新场景。")
            st.info("💡 只有当后台存在仿真日志时，才能显示车辆移动动画。")
        else:
            col1, col2 = st.columns([3, 1])
            with col1:
                # 【修改点 2】：点击按钮只改变状态，地图显示放在按钮外面
                if st.button("🎬 生成动画地图", type="primary", use_container_width=True):
                    st.session_state.show_map_active = True
                
                # 只要状态为 True，地图就一直渲染，不会因为刷新而消失
                if st.session_state.show_map_active:
                    with st.spinner("正在生成动画地图..."):
                        try:
                            animated_map = visualizer.create_animated_map()
                            if animated_map:
                                st_folium(animated_map, width=1000, height=600, key="animation_map_v1")
                        except Exception as e:
                            st.error(f"❌ 地图生成错误: {e}")
            
            with col2:
                st.subheader("📋 快捷说明")
                st.write("点击左侧按钮生成车辆轨迹动画。")
                st.write("蓝色：前往任务")
                st.write("绿色：充电中")
                st.write("紫色：回库中")

    
if __name__ == "__main__":
    main()
