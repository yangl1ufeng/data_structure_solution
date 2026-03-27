import os
import osmnx as ox
import networkx as nx
import pandas as pd
import json

# --- 全局配置与缓存 ---
# 缓存已计算的路径，避免重复计算。格式: {(source_node, target_node): {"length": 123, "path": [...]}}
PATH_CACHE = {}

def get_road_network(place_name, network_type="drive"):
    """
    加载路网图。如果本地存在对应的 .graphml 文件，则直接加载；
    否则，从网络下载并以城市名保存。

    Args:
        place_name (str): 查询的地点名称，如 "Guangzhou, China"。
        network_type (str): 'drive', 'walk', 'bike', 'all' 等。

    Returns:
        networkx.MultiDiGraph: 路网图。
    """
    # 根据城市名生成文件名，例如 "guangzhou_china.graphml"
    filename = f"{place_name.replace(' ', '_').replace(',', '').lower()}.graphml"
    
    if os.path.exists(filename):
        print(f"从本地文件 '{filename}' 加载路网图...")
        graph = ox.load_graphml(filename)
    else:
        print(f"本地未找到路网图，正在从网络下载 '{place_name}' 的路网...")
        print("这个过程可能需要几分钟，请耐心等待...")
        graph = ox.graph_from_place(place_name, network_type=network_type)
        ox.save_graphml(graph, filename)
        print(f"路网图已保存到 '{filename}' 以便下次使用。")
    
    print(f"'{place_name}' 的路网图加载完成。")
    return graph

def snap_points_to_network(graph, points_list):
    """
    将地理坐标点列表匹配到路网最近的节点上。

    Args:
        graph (networkx.MultiDiGraph): 路网图。
        points_list (list): 包含点位信息的字典列表，每个字典需有 'latitude' 和 'longitude'。

    Returns:
        pandas.DataFrame: 包含原始信息和匹配到的节点ID (node_id) 的 DataFrame。
    """
    print("正在将点位匹配到路网节点...")
    # 提取经纬度
    lats = [p['latitude'] for p in points_list]
    lons = [p['longitude'] for p in points_list]

    # 使用 osmnx 批量匹配最近节点
    nearest_nodes = ox.distance.nearest_nodes(graph, lons, lats)

    # 创建 DataFrame 存储结果
    snapped_points_df = pd.DataFrame(points_list)
    snapped_points_df['node_id'] = nearest_nodes
    
    print("点位匹配完成。")
    return snapped_points_df

def get_shortest_path(graph, source_node, target_node, weight='length'):
    """
    计算两个节点之间的最短路径，并使用缓存。

    Args:
        graph (networkx.MultiDiGraph): 路网图。
        source_node (int): 起点节点ID。
        target_node (int): 终点节点ID。
        weight (str): 用于计算路径长度的边的属性（如 'length' 表示距离）。

    Returns:
        dict: 包含路径长度 ('length') 和节点列表 ('path') 的字典，如果不可达则返回 None。
    """
    # 检查缓存
    if (source_node, target_node) in PATH_CACHE:
        return PATH_CACHE[(source_node, target_node)]

    try:
        # 使用 Dijkstra 算法计算最短路径长度和路径
        path_length = nx.shortest_path_length(graph, source=source_node, target=target_node, weight=weight)
        node_path = nx.shortest_path(graph, source=source_node, target=target_node, weight=weight)
        
        result = {"length": path_length, "path": node_path}
        # 存入缓存
        PATH_CACHE[(source_node, target_node)] = result
        return result
    except nx.NetworkXNoPath:
        # 处理不可达情况
        print(f"警告: 从节点 {source_node} 到 {target_node} 不存在路径。")
        return None

def create_distance_matrix(graph, snapped_points_df):
    """
    生成所有点位之间的距离矩阵。

    Args:
        graph (networkx.MultiDiGraph): 路网图。
        snapped_points_df (pd.DataFrame): 已匹配到节点的点位 DataFrame。

    Returns:
        pd.DataFrame: 距离矩阵，索引和列都是点位的原始索引。
    """
    print("正在生成距离矩阵...")
    num_points = len(snapped_points_df)
    # 初始化一个空的 DataFrame
    dist_matrix = pd.DataFrame(index=snapped_points_df.index, columns=snapped_points_df.index, dtype=float)

    node_ids = snapped_points_df['node_id']

    for i in range(num_points):
        for j in range(num_points):
            if i == j:
                dist_matrix.iloc[i, j] = 0.0
                continue

            source_node = node_ids.iloc[i]
            target_node = node_ids.iloc[j]
            
            path_info = get_shortest_path(graph, source_node, target_node)
            
            if path_info:
                dist_matrix.iloc[i, j] = path_info['length']
            else:
                # 如果不可达，可以设置为无穷大或一个标记值
                dist_matrix.iloc[i, j] = float('inf')
    
    print("距离矩阵生成完毕。")
    return dist_matrix

def get_path_geometry(graph, node_path):
    """
    从节点路径获取地理坐标序列。

    Args:
        graph (networkx.MultiDiGraph): 路网图。
        node_path (list): 节点ID列表。

    Returns:
        list: [(lat, lon), ...] 格式的坐标点列表。
    """
    # 使用 .loc 批量获取节点属性，速度更快
    path_nodes = graph.nodes(data=True)
    coords = [(path_nodes[node]['y'], path_nodes[node]['x']) for node in node_path]
    return coords

def save_results(snapped_points_df, distance_matrix_df, folder="data"):
    """
    将匹配点和距离矩阵保存到本地文件。

    Args:
        snapped_points_df (pd.DataFrame): 匹配后的点位信息。
        distance_matrix_df (pd.DataFrame): 距离矩阵。
        folder (str): 保存文件的文件夹名称。
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    snapped_points_path = os.path.join(folder, "snapped_points.csv")
    distance_matrix_path = os.path.join(folder, "distance_matrix.csv")

    snapped_points_df.to_csv(snapped_points_path, index_label="point_index")
    distance_matrix_df.to_csv(distance_matrix_path)
    
    print(f"结果已保存到 '{folder}' 文件夹下。")

# --- 主执行逻辑示例 ---
def process_all(points_from_session_state, place_name):
    """
    一个集成的函数，执行从加载数据到保存结果的全过程。

    Args:
        points_from_session_state (list): 来自 Streamlit session state 的点位列表。
        place_name (str): 要加载路网的城市名称。
    """
    if not points_from_session_state:
        print("点位列表为空，无法处理。")
        return None, None, None

    # 1. 根据传入的 place_name 加载路网
    G = get_road_network(place_name=place_name)

    # 2. 点位匹配
    snapped_points = snap_points_to_network(G, points_from_session_state)

    # 3. 计算距离矩阵
    distance_matrix = create_distance_matrix(G, snapped_points)

    # 4. 保存结果
    save_results(snapped_points, distance_matrix)

    # 5. (示例) 获取点0到点1的路径几何信息
    path_geometry = None
    if len(snapped_points) >= 2:
        node_ids = snapped_points['node_id']
        path_info = get_shortest_path(G, node_ids.iloc[0], node_ids.iloc[1])
        if path_info:
            path_geometry = get_path_geometry(G, path_info['path'])
            print(f"\n示例：从点0到点1的路径长度为 {path_info['length']:.2f} 米。")
    
    return snapped_points, distance_matrix, path_geometry

if __name__ == '__main__':
    # --- 这是一个用于独立测试的示例 ---
    # 模拟从 st.session_state 获取的数据
    mock_session_state_points = [
        {'type': '🏭 中央仓库 (Depot)', 'latitude': 31.2304, 'longitude': 121.4737},
        {'type': '📦 任务目标点 (Task Point)', 'latitude': 31.235, 'longitude': 121.48},
        {'type': '⚡ 充电站 (Charging Station)', 'latitude': 31.22, 'longitude': 121.46}
    ]

    # 测试时也需要传入城市名
    process_all(mock_session_state_points, place_name="Shanghai, China")
