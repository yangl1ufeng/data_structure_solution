import osmnx as ox
import os

# 定义文件名
file_name = "shanghai_old_china.graphml"

print(f"正在从互联网下载上海路网数据并生成 {file_name}...")
print("这可能需要几分钟，请保持网络畅通...")

try:
    # 根据上海市中心的坐标下载路网（这个范围涵盖了你 CSV 里的点位）
    # 这里的 31.23, 121.47 是上海人民广场附近
    graph = ox.graph_from_point((31.23, 121.47), dist=10000, network_type='drive')
    
    # 保存为 graphml 文件
    ox.save_graphml(graph, file_name)
    print(f"成功！文件已生成：{os.path.abspath(file_name)}")
    print("现在你可以重新运行 python simulation_gurobi.py 了。")

except Exception as e:
    print(f"生成失败: {e}")
