import osmnx as ox
import networkx as nx
import sys
import io

# 修复输出编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 配置 OSMnx
ox.settings.use_cache = True
ox.settings.log_console = True

print("正在下载上海路网数据...")

try:
    # 下载上海的路网（驾车网络）
    G = ox.graph_from_place("Shanghai, China", network_type="drive")

    # 验证数据
    print(f"节点数：{G.number_of_nodes()}")
    print(f"边数：{G.number_of_edges()}")

    if G.number_of_nodes() == 0:
        print("错误：下载的路网数据为空！")
        sys.exit(1)

    # 保存为 graphml 格式
    output_path = "shanghai_china.graphml"
    ox.save_graphml(G, filepath=output_path)

    # 验证文件
    import os

    file_size = os.path.getsize(output_path)
    print(f"文件已保存：{output_path}")
    print(f"文件大小：{file_size / 1024 / 1024:.2f} MB")

    if file_size < 1000:
        print("警告：文件太小，可能下载失败！")

    print("下载完成！")

except Exception as e:
    print(f"下载失败：{e}")
    import traceback

    traceback.print_exc()
