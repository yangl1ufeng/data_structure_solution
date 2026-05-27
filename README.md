# data_structure_solution
# 🚛 新能源物流车队协同调度系统

本项目是一个集成 Web 可视化、混合整数线性规划（MILP）与启发式调度算法的新能源物流车队路径规划与派单仿真系统。

## 🌟 核心功能

* **多算法调度引擎**：
  * **全局最优 (Gurobi)**：基于 MILP 模型，带有严格的时间窗、电量流平衡与载重约束，适用于中小型规模的精确求解。
  * **启发式贪心算法**：支持”距离最近优先”与”最大载重优先”，内置智能电量中转机制（前往任务点 + 寻找最近充电站），可毫秒级处理上千个任务的大规模调度。
  * **遗传算法 (GA)**：元启发式方法，任务排列染色体 + 贪心分裂解码器，锦标赛选择 + OX交叉 + 多策略变异，适合大规模优化。
  * **Q-Learning 强化学习**：离散化状态空间（电量/距离/重量/紧急度），ε-greedy 探索策略，Q表持久化与启发式初始化，自主学习调度策略。
  * **拍卖竞标多智能体**：每辆车作为独立智能体，基于成本（距离+电池风险+时间窗+载重利用率）出价竞标，多轮迭代 + 冲突解决。
  * **模拟退火 (SA)**：元启发式方法，5种邻域操作（交换/重排/重分配/增删充电站），Metropolis 接受准则，从贪心初始解出发全局搜索优化。
* **大规模仿真支持**：基于字典缓存机制 (Path Cache) 优化路网测距查询，极大地提升了建图与调度计算速度。
* **交互式 Web UI (Streamlit + Folium)**：
  * 支持在所选城市区域（如上海、广州）一键随机生成最高 5000 个任务点。
  * 支持自定义车队规模（最高 100 辆车）与基础参数调节。
  * 动态地图仿真回放 (TimestampedGeoJson)，以专属颜色、醒目边界及动态提示框直观展示每辆车的运行轨迹、载荷与电量状态。
* **智能电量管理**：实时预估耗电量，避免车辆抛锚；在电量不足时自动规划前往最近的空闲充电站进行补电，并无缝衔接后续任务。

## ⚙️ 环境依赖

* Python 3.8+
* [Streamlit](https://streamlit.io/)
* [Folium](https://python-visualization.github.io/folium/) / streamlit-folium
* [NetworkX](https://networkx.org/)
* [Gurobi Optimizer](https://www.gurobi.com/) (可能需配置有效许可证 `@gurobi.lic` 以解锁大规模变量限制)
* Pandas / Numpy

## 🚀 快速启动

1. **安装依赖**：
   在终端中运行以下命令安装必备库：
   ```bash
   pip install streamlit folium streamlit-folium networkx gurobipy pandas numpy
   ```

2. **配置 Gurobi 许可证**：
   如果你需要使用 Gurobi 求解大规模任务，请将申请到的 `gurobi.lic` 文件放置在项目根目录 `d:\code\python_code\solution_web\local\` 下。

3. **运行系统**：
   在终端中启动 Streamlit 服务：
   ```bash
   cd d:\code\python_code\solution_web\local
   streamlit run finalweb.py
   ```

4. **系统使用流程**：
   * 在左侧边栏配置**仿真参数**（车辆数、调度策略等）。
   * 在页面中选择目标城市，点击**🎲 一键随机生成**或手动添加任务节点。
   * 点击**🚀 计算最短路径**进行路网贴合与距离初始化。
   * 点击**一键自动化执行**或**运行仿真**，等待后台引擎计算完毕并在右侧加载可视化回放动画。

## 📁 主要文件结构

* `finalweb.py`: Streamlit 前端主程序，负责 UI 渲染、地图可视化与后台进程调度。
* `simulation.py`: 仿真核心引擎，管理时间轴、车辆状态流转以及日志生成。
* `scheduler_gurobi.py`: 基于 Gurobi 的 MILP 数学规划求解器。
* `scheduler_greedy.py`: 基于启发式规则的快速调度器。
* `scheduler_genetic.py`: 基于遗传算法 (GA) 的元启发式调度器。
* `scheduler_rl.py`: 基于 Q-Learning 的强化学习调度器。
* `scheduler_auction.py`: 基于拍卖竞标的多智能体调度器。
* `scheduler_sa.py`: 基于模拟退火 (SA) 的元启发式调度器。
* `gurobi.lic`: (需用户提供) Gurobi 求解器许可证

## 📋 项目概览

本系统包含以下核心功能：
- **智能调度算法**：基于 Gurobi 求解器的 EVRP（Electric Vehicle Routing Problem）优化
- **协同运输**：自动拆分大型任务，支持多车协同配送
- **实时仿真**：车辆移动、充电、任务执行的动态仿真
- **可视化展示**：基于 Streamlit + Folium 的交互式地图界面
- **路网计算**：集成 OSMnx 进行真实道路网络分析

## 🛠️ 技术栈

- **后端**：Python 3.8+
- **优化器**：Gurobi Optimizer 10.0+
- **地图处理**：OSMnx、NetworkX
- **前端可视化**：Streamlit、Folium
- **数据处理**：Pandas、GeoPandas
- **科学计算**：NumPy、Matplotlib

## ⚙️ 环境配置

### 1. 克隆项目

```bash
git clone https://github.com/yangl1ufeng/data_structure_solution
cd data_structure_solution
```

### 2. 创建虚拟环境

#### 方式一：使用 venv（推荐）

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### 方式二：使用 conda

```bash
conda create -n evrp python=3.9
conda activate evrp
```

### 3. 安装依赖包

#### 选项 A：一键安装（推荐）

```bash
# 确保虚拟环境已激活 你会看到(.venv)在路径前
pip install --upgrade pip
pip install -r requirements.txt
```

#### 选项 B：分步安装（解决兼容性问题）

```bash
# 确保虚拟环境已激活 你会看到(.venv)在路径前

# 1. 安装地理空间基础库
pip install shapely fiona pyproj rtree

# 2. 安装地理数据处理
pip install geopandas

# 3. 安装路网分析工具
pip install osmnx

# 4. 安装Web应用框架
pip install streamlit streamlit-folium folium

# 5. 安装数据处理和可视化
pip install pandas networkx matplotlib plotly requests

# 6. 安装优化器（需要许可证）
pip install gurobipy
```

### 4. Gurobi 许可证配置

Gurobi 是商业优化器，需要有效许可证：

#### 学术许可证（免费）
```bash
# 1. 注册学术许可证：https://www.gurobi.com/academia/
# 2. 获取许可证密钥后运行：
grbgetkey YOUR_LICENSE_KEY
```

#### 试用许可证
```bash
# 下载并安装 Gurobi，自动包含试用许可证
# 访问：https://www.gurobi.com/downloads/
```

### 5. 验证安装

```bash
# 测试核心依赖
python -c "import streamlit, osmnx, folium, pandas, networkx; print('✅ 基础包安装成功')"

# 测试 Gurobi
python -c "import gurobipy; print('✅ Gurobi 安装成功')"
```

## 📁 项目结构

```
local/
├── finalweb.py                    # 集成系统主程序 ⭐
├── data/                          # 数据文件夹（自动生成）
│   ├── snapped_points.csv         # 地点坐标数据
│   └── distance_matrix.csv        # 距离矩阵
├── cache/                         # 缓存文件夹
│   └── q_table.json               # Q-Learning Q表持久化
├── simulation.py           # 仿真引擎
├── scheduler_gurobi.py            # Gurobi MILP调度器
├── scheduler_greedy.py            # 贪心启发式调度器
├── scheduler_genetic.py           # 遗传算法调度器（元启发式）
├── scheduler_rl.py                # Q-Learning调度器（强化学习）
├── scheduler_auction.py           # 拍卖竞标调度器（多智能体）
├── scheduler_sa.py                # 模拟退火调度器（元启发式）
├── path_processor.py              # 路径处理工具
├── requirements.txt               # 依赖列表
├── README.md                      # 项目说明
└── *.graphml                      # 路网数据（自动下载）
```

## 🚀 快速开始

### 1. 数据准备

确保 `data/` 文件夹包含必要数据文件：
- 如果没有数据文件，首次运行会自动引导您创建

### 2. 地图节点配置（首次使用）

```bash
# 激活虚拟环境
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 运行集成系统（含节点配置功能）
streamlit run finalweb.py
```

在打开的网页中，切换到"节点配置"标签页：
1. 选择节点类型（仓库、充电站、任务点）
2. 在地图上点击添加节点
3. 点击"处理路径数据"生成所需文件

### 3. 运行仿真系统

```bash
# 运行核心仿真引擎
python simulation.py
```

### 4. 启动可视化界面

```bash
# 运行可视化系统
streamlit run finalweb.py
```

然后在浏览器中访问 `http://localhost:8501`

## 📊 功能特性

### 核心算法
- **EVRP 优化**：电动车辆路径问题求解
- **协同运输**：大型任务自动拆分为多车协同
- **智能充电**：基于电量阈值的充电决策
- **动态调度**：实时任务分配和路径优化
- **元启发式方法**：遗传算法 (GA) + 模拟退火 (SA) 全局搜索
- **强化学习**：Q-Learning 自主学习调度策略
- **多智能体**：拍卖竞标去中心化决策

### 可视化功能
- **动态地图**：实时显示车辆位置和移动轨迹
- **状态监控**：车辆电量、载重、任务状态
- **性能分析**：完成率、效率指标、得分统计
- **交互操作**：任务点选择、参数调整

## ⚠️ 常见问题

### OSMnx 安装失败
```bash
# 使用 conda 安装（推荐）
conda install -c conda-forge osmnx

# 或者先安装依赖再安装 osmnx
pip install shapely fiona pyproj rtree
pip install osmnx
```

### Gurobi 许可证问题
```bash
# 检查许可证状态
python -c "import gurobipy; m = gurobipy.Model(); print('许可证有效')"

# 重新激活许可证
grbgetkey YOUR_LICENSE_KEY
```

### Streamlit 模块未找到
```bash
# 确保在正确的虚拟环境中
which python  # 检查 Python 路径
pip list | grep streamlit  # 检查是否安装

# 重新安装
pip install --force-reinstall streamlit
```

### 数据文件缺失
1. 运行 `streamlit run finalweb.py`
2. 切换到"节点配置"标签页，在地图上选择节点
3. 点击"处理路径数据"按钮
4. 等待生成完成

## 📈 系统要求

- **操作系统**：Windows 10+, macOS 10.14+, Ubuntu 18.04+
- **Python 版本**：3.8 - 3.11
- **内存**：建议 8GB+
- **磁盘空间**：至少 2GB
- **网络**：初次运行需要下载地图数据

## 🤝 贡献指南

1. Fork 本项目
2. 创建特性分支：`git checkout -b feature/your-feature`
3. 提交更改：`git commit -am 'Add some feature'`
4. 推送到分支：`git push origin feature/your-feature`
5. 创建 Pull Request

## 📄 许可证

本项目基于 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- **作者**：DBD
- **邮箱**：1352861763@qq.com
- **项目链接**：https://github.com/yangl1ufeng/data_structure_solution

## 🙏 致谢

- [Gurobi](https://www.gurobi.com/) - 优化求解器
- [OSMnx](https://github.com/gboeing/osmnx) - 路网数据处理
- [Streamlit](https://streamlit.io/) - Web应用框架
- [Folium](https://github.com/python-visualization/folium) - 地图可视化

---

⭐ 如果这个项目对您有帮助，请给个 Star！
