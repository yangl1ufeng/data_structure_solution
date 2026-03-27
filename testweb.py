import streamlit as st
import folium
import pandas as pd
from streamlit_folium import st_folium
import path_processor  # <--- 1. 导入新创建的模块
# --- 页面配置 ---
st.set_page_config(
    page_title="新能源车队协同调度系统",
    page_icon="🚚",
    layout="wide",  # 设置为宽屏布局
)

# --- 状态管理初始化 ---
# 使用 st.session_state 来持久化存储应用运行过程中的数据
if "center" not in st.session_state:
    # 初始化地图中心点（以上海为例）
    st.session_state["center"] = [31.2304, 121.4737]
if "zoom" not in st.session_state:
    # 初始化地图缩放级别
    st.session_state["zoom"] = 10
if "points" not in st.session_state:
    # 初始化点位列表，用于存储所有添加的节点
    st.session_state["points"] = []
if "last_path_geometry" not in st.session_state:
    st.session_state["last_path_geometry"] = None # 用于存储最后计算的路径
    
# --- 页面标题 ---
st.title("🚚 新能源车队协同调度系统")

# --- 页面布局：左侧边栏和右侧主区域 ---
# 定义两列，左侧用于控制，右侧用于显示地图
sidebar, main_map = st.columns([1, 3])

# --- 左侧边栏：控制面板 ---
with sidebar:
    st.header("📍 控制面板")

    # 1. 节点类型选择
    point_type = st.radio(
        "请选择要添加的节点类型:",
        ("🏭 中央仓库 (Depot)", "⚡ 充电站 (Charging Station)", "📦 任务目标点 (Task Point)"),
        key="point_type_selector"
    )

    st.info("""
    **操作说明:**
    1.  在上方选择要添加的节点类型。
    2.  在右侧地图上点击鼠标左键，即可添加对应节点。
    3.  **中央仓库** 全局只能设置一个。
    """)

    # 4. 数据展示与删除
    st.header("已选点位列表")
    if st.session_state["points"]:
        # 为了能够实现删除，我们不再使用 st.dataframe，而是手动渲染列表
        # 遍历点位列表，为每个点位添加删除按钮
        # 使用 reversed() 来避免删除时索引错乱的问题
        for i in reversed(range(len(st.session_state["points"]))):
            point = st.session_state["points"][i]
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"ID: {i+1} - {point['type']}")
                st.caption(f"Lat: {point['latitude']:.4f}, Lon: {point['longitude']:.4f}")
            with col2:
                # 为每个按钮创建一个唯一的 key
                if st.button("删除", key=f"delete_{i}", use_container_width=True):
                    # 从 session_state 中移除这个点
                    st.session_state["points"].pop(i)
                    # 立即重新运行脚本以更新UI
                    st.rerun()
            st.divider()

        # 将点位数据转换为 Pandas DataFrame 以便导出
        points_df = pd.DataFrame(st.session_state["points"])
        json_data = points_df.to_json(orient="records", indent=4)

        # 5. 提供下载按钮
        st.download_button(
            label="📥 保存配置 / 导出 JSON",
            data=json_data,
            file_name="fleet_config.json",
            mime="application/json",
        )
    else:
        st.warning("当前未添加任何点位。")
        # 6. 添加路径计算功能
    st.header("🌍 路径规划")
    # 添加城市选择器
    selected_city = st.selectbox(
        "请选择计算区域:",
        ("Shanghai, China", "Guangzhou, China"),
        key="city_selector"
    )

    if st.button("🚀 计算所有点间最短路径", use_container_width=True):
        if not st.session_state["points"]:
            st.error("点位列表为空，无法计算。")
        else:
            with st.spinner(f"正在处理 '{selected_city}' 的路网并计算路径，请稍候..."):
                # 调用核心处理函数，并传入选择的城市
                snapped_df, dist_matrix_df, example_path_geom = path_processor.process_all(
                    st.session_state["points"],
                    place_name=selected_city
                )
                
                if snapped_df is not None:
                    st.success("计算完成！结果已保存到 'data' 文件夹。")
                    # 将示例路径保存到 session state 以便在地图上绘制
                    st.session_state["last_path_geometry"] = example_path_geom
                    # 强制刷新以在地图上显示新路径
                    st.rerun()
                else:
                    st.error("点位列表为空，无法计算。")


# --- 右侧主区域：地图显示 ---
with main_map:
    # 2. 地图初始化
    m = folium.Map(
        location=st.session_state["center"],
        zoom_start=st.session_state["zoom"],
        tiles="OpenStreetMap"  # 使用默认的 OSM 底图
    )

    # 3. 视觉区分：定义不同类型点的图标和颜色
    POINT_STYLE = {
        "🏭 中央仓库 (Depot)": {"icon": "home", "color": "red"},
        "⚡ 充电站 (Charging Station)": {"icon": "bolt", "color": "green"},
        "📦 任务目标点 (Task Point)": {"icon": "cube", "color": "blue"},
    }

    # 将已有的点位添加到地图上
    for point in st.session_state["points"]:
        style = POINT_STYLE[point["type"]]
        folium.Marker(
            location=[point["latitude"], point["longitude"]],
            tooltip=f"{point['type']}<br>Lat: {point['latitude']:.4f}<br>Lon: {point['longitude']:.4f}",
            icon=folium.Icon(icon=style["icon"], color=style["color"], prefix='fa')
        ).add_to(m)

      # 在地图上绘制最后计算的路径
    if st.session_state["last_path_geometry"]:
        folium.PolyLine(
            locations=st.session_state["last_path_geometry"],
            color="purple",
            weight=5,
            opacity=0.8,
            tooltip="计算出的最短路径"
        ).add_to(m)
        
    # 渲染 Folium 地图并捕获交互事件
    map_data = st_folium(m, width='100%', height=600)

    # 4. 地图点击交互处理
    if map_data and map_data["last_clicked"]:
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]

        # 检查是否是添加中央仓库
        if point_type == "🏭 中央仓库 (Depot)":
            # 容错处理：检查是否已经存在中央仓库
            depot_exists = any(p["type"] == "🏭 中央仓库 (Depot)" for p in st.session_state["points"])
            if depot_exists:
                st.sidebar.error("错误：中央仓库只能设置一个！请先在列表中移除已有的仓库。")
            else:
                st.session_state["points"].append({"type": point_type, "latitude": lat, "longitude": lon})
                st.rerun() # 添加成功后立即刷新页面以更新地图和列表
        else:
            # 添加充电站或任务点
            st.session_state["points"].append({"type": point_type, "latitude": lat, "longitude": lon})
            st.rerun() # 添加成功后立即刷新页面
