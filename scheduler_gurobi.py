# 必须先导 os，先设置环境变量！
import os

# 强制配置环境变量，使 Gurobi 读取当前目录下的许可证文件
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["GRB_LICENSE_FILE"] = os.path.join(current_dir, "gurobi.lic")

# 设置完环境变量后，再导入 gurobipy
import gurobipy as gp
from gurobipy import GRB
import networkx as nx

class GurobiEVRPScheduler:
    """
    基于严格 MILP (混合整数线性规划) 的 EVRPTW 调度器
    实现了带时间窗、载重和连续电量流的多车路线排班系统。
    """

    def __init__(self, time_limit=10, gap_tolerance=0.05):
        self.time_limit = time_limit
        self.gap_tolerance = gap_tolerance
        self.big_M = 100000.0  # 大M法常数

    def solve_assignment(self, vehicles, pending_tasks, stations, dist_helper, graph):
        # ---> 新增: 距离查询本地缓存字典，极大幅度提升重复查询速度
        dist_cache = {}
        def fast_get_dist(loc1, loc2):
            if (loc1, loc2) in dist_cache:
                return dist_cache[(loc1, loc2)]
            dist = dist_helper.get_distance(loc1, loc2)
            dist_cache[(loc1, loc2)] = dist
            return dist

        # --- 预处理: 提前剔除无法连通的孤岛任务，根本不将其纳入数学模型 ---
        reachable_tasks = []
        for t in pending_tasks:
            is_reachable = False
            for k in vehicles:
                # 使用缓存函数
                dist_to_task = fast_get_dist(k.current_location, t.location_node)
                if dist_to_task is not None:
                    is_reachable = True
                    break
            if is_reachable:
                reachable_tasks.append(t)
        
        pending_tasks = reachable_tasks
        if not pending_tasks:
            return {str(k.id): [] for k in vehicles}

        model = gp.Model("EVRPTW_Routing")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', self.time_limit)
        model.setParam('MIPGap', self.gap_tolerance)
        # 推荐为大规模问题增加的参数
        model.setParam('MIPFocus', 1)

        # -----------------------------
        # 1. 图节点构建与预处理剪枝
        # -----------------------------
        task_nodes = [f"TASK_{t.id}" for t in pending_tasks]
        station_nodes = [f"STATION_{s.id}" for s in stations]
        
        V_tasks = {f"TASK_{t.id}": t for t in pending_tasks}
        V_stations = {f"STATION_{s.id}": s for s in stations}
        
        location_map = {}
        for t in pending_tasks: location_map[f"TASK_{t.id}"] = t.location_node
        for s in stations: location_map[f"STATION_{s.id}"] = s.location_node
        for k in vehicles:
            location_map[f"START_{k.id}"] = k.current_location
            location_map[f"END_{k.id}"] = k.depot_node
        
        # --- 步骤 1.5: 构建真实的合法边 (剔除无路径的死点) ---
        valid_edges = []
        for k in vehicles:
            start_k = f"START_{k.id}"
            end_k = f"END_{k.id}"
            
            V_k = [start_k, end_k] + task_nodes + station_nodes
            
            for i in V_k:
                for j in V_k:
                    if i == j or i == end_k or j == start_k:
                        continue
                        
                    loc_i = location_map[i]
                    loc_j = location_map[j]
                    
                    dist_val = fast_get_dist(loc_i, loc_j)
                    if dist_val is None:
                        if i == start_k and j == end_k:
                            valid_edges.append((i, j, k.id))
                        continue
                        
                    valid_edges.append((i, j, k.id))
                    
        valid_edges = list(set(valid_edges))

        # -----------------------------
        # 2. 声明字典并初始化决策变量
        # -----------------------------
        x = {}
        y = {}
        t = {}  
        b = {}
        u = {}
        charge_amt = {}

        # 声明全局弃单变量 y
        for task_node in task_nodes:
            y[task_node] = model.addVar(vtype=GRB.BINARY, name=f"y_{task_node}")
            
        # 声明边访问变量 x
        for i, j, k_id in valid_edges:
            x[i, j, k_id] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}_{k_id}")

        for k in vehicles:
            start_k = f"START_{k.id}"
            end_k = f"END_{k.id}"
            
            V_k = [start_k, end_k] + task_nodes + station_nodes
            
            for i in V_k:
                # 计算“逃生”到最近充电站的最短距离
                min_escape_dist = 0.0
                if i in task_nodes:
                    min_escape = float('inf')
                    for s in station_nodes:
                        d_s = fast_get_dist(location_map[i], location_map[s])
                        if d_s is not None and d_s < min_escape:
                            min_escape = d_s
                    # 如果周围没充电站，则要求至少能开回仓库
                    if min_escape == float('inf'):
                        d_d = fast_get_dist(location_map[i], location_map[end_k])
                        min_escape = d_d if d_d is not None else 0.0
                    min_escape_dist = min_escape
                
                # 当前节点到最近充电站所需的电量 (安全底线)
                escape_energy = min_escape_dist * k.consumption_rate_dist
                
                t[i, k.id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"t_{i}_{k.id}")
                b[i, k.id] = model.addVar(vtype=GRB.CONTINUOUS, lb=escape_energy, ub=k.max_battery, name=f"b_{i}_{k.id}")
                u[i, k.id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=k.max_payload, name=f"u_{i}_{k.id}")
                
                if i in station_nodes:
                    charge_amt[i, k.id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=k.max_battery, name=f"charge_{i}_{k.id}")

        # -----------------------------
        # 3. 核心约束逻辑
        # -----------------------------
        for k in vehicles:
            start_k = f"START_{k.id}"
            end_k = f"END_{k.id}"
            V_sub = task_nodes + station_nodes

            # a. 流平衡约束 (Flow Balance)
            # 起点出度为 1
            model.addConstr(gp.quicksum(x[start_k, j, k.id] for j in V_sub + [end_k] if (start_k, j, k.id) in valid_edges) == 1)
            # 终点入度为 1
            model.addConstr(gp.quicksum(x[i, end_k, k.id] for i in V_sub + [start_k] if (i, end_k, k.id) in valid_edges) == 1)
            
            # 中间节点流平衡 (入度 = 出度)
            for j in V_sub:
                model.addConstr(
                    gp.quicksum(x[i, j, k.id] for i in [start_k] + V_sub if (i, j, k.id) in valid_edges) == 
                    gp.quicksum(x[j, p, k.id] for p in V_sub + [end_k] if (j, p, k.id) in valid_edges)
                )

            # b. 起点状态初始化
            model.addConstr(b[start_k, k.id] == k.current_battery)
            model.addConstr(u[start_k, k.id] == k.max_payload) # 默认满载出发
            # 假设当前仿真时间为0（相对时间）
            model.addConstr(t[start_k, k.id] == 0)

            # c. MTZ 连续流传递与大M法约束
            for i, j, k_id in valid_edges:
                if k_id != k.id: continue
                
                # 使用映射表获取物理坐标
                loc_i = location_map[i]
                loc_j = location_map[j]
                
                dist_val = fast_get_dist(loc_i, loc_j)
                dist = 10000.0 if dist_val is None else dist_val
                
                travel_time = (dist / 1000.0) / k.speed_kmh * 60.0 # 分钟
                
                # 🔥 核心修复：补上最大载重造成的耗电率倍数，与仿真底层的安全校验严格对齐
                energy_cost = dist * (k.consumption_rate_dist + k.max_payload * k.consumption_rate_payload)
                
                # (1) 时间流传递
                # t_j >= t_i + travel_time + service_time - M(1 - x)
                service_time_i = 0
                if i in task_nodes: service_time_i = 10 # 卸货10分钟
                elif i in station_nodes: service_time_i = 30 # 默认最短充电锁定时间
                
                model.addConstr(t[j, k.id] >= t[i, k.id] + travel_time + service_time_i - self.big_M * (1 - x[i, j, k.id]))

                # (2) 电量流传递
                if i in station_nodes:
                    model.addConstr(b[j, k.id] <= b[i, k.id] + charge_amt[i, k.id] - energy_cost + self.big_M * (1 - x[i, j, k.id]))
                else:
                    model.addConstr(b[j, k.id] <= b[i, k.id] - energy_cost + self.big_M * (1 - x[i, j, k.id]))
                    
                # (3) 载重流传递
                demands_j = V_tasks[j].weight if j in task_nodes else 0
                model.addConstr(u[j, k.id] <= u[i, k.id] - demands_j + self.big_M * (1 - x[i, j, k.id]))

        # d. 任务约束：每个任务必须被一辆车访问（由 y 控制是否放弃）
        for task_node in task_nodes:
            model.addConstr(
                # 🔥 修复：将 k.id 替换为生成器里的 k_id，并将 task_node 替换为 j
                gp.quicksum(x[i, j, k_id] for i, j, k_id in valid_edges if j == task_node) == y[task_node],
                name=f"visit_{task_node}"
            )
            
            # e. 软时间窗约束
            task = V_tasks[task_node]
            for k in vehicles:
                # 记录超时变量 (非负)
                late_var = model.addVar(lb=0.0, name=f"late_{task_node}_{k.id}")
                model.addConstr(late_var >= t[task_node, k.id] - task.deadline)

        # -----------------------------
        # 4. 目标函数构造
        # -----------------------------
        REWARD_BASE = 10000
        PENALTY_DROP = 50000
        
        # 最大化总分 = 任务奖励 - 距离消耗 - 迟到惩罚 - 充电次数惩罚
        obj_expr = gp.quicksum(y[n] * REWARD_BASE for n in task_nodes)
        obj_expr -= gp.quicksum((1 - y[n]) * (PENALTY_DROP + V_tasks[n].weight) for n in task_nodes) # 弃单惩罚
        
        for i, j, k_id in valid_edges:
            # 🔥 修复：目标函数中也只需通过映射表换取物理坐标
            dist_val = fast_get_dist(location_map[i], location_map[j])
            dist = 0 if dist_val is None else dist_val
            obj_expr -= x[i, j, k_id] * dist * 0.01 # 减去行驶距离惩罚
            
        # 充电惩罚
        for k in vehicles:
            for s in station_nodes:
                station_visits = gp.quicksum(x[i, s, k.id] for i, jj, kk in valid_edges if jj == s and kk == k.id)
                obj_expr -= station_visits * 500

        model.setObjective(obj_expr, GRB.MAXIMIZE)
        model.optimize()

        # -----------------------------
        # 5. 结果解析为 Command List
        # -----------------------------
        assignments = {str(k.id): [] for k in vehicles}
        
        if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
            for k in vehicles:
                start_k = f"START_{k.id}"
                end_k = f"END_{k.id}"
                
                curr_node = start_k
                route_commands = []
                
                # 追踪有向图链路
                while curr_node != end_k:
                    next_node = None
                    for j in [end_k] + task_nodes + station_nodes:
                        if (curr_node, j, k.id) in valid_edges and x[curr_node, j, k.id].X > 0.5:
                            next_node = j
                            break
                            
                    if not next_node or next_node == end_k:
                        break
                        
                    if next_node in task_nodes:
                        t_obj = V_tasks[next_node]
                        route_commands.append(f"TASK:{t_obj.id}")
                    elif next_node in station_nodes:
                        s_obj = V_stations[next_node]
                        route_commands.append(f"CHARGE:{s_obj.id}")
                        
                    curr_node = next_node
                    
                assignments[str(k.id)] = route_commands
                print(f"  [EVRPTW] 车辆 {k.id} 生成连续路径排序: {route_commands}")
        
        return assignments