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

    def _solve_two_stage(self, vehicles, pending_tasks, stations, dist_helper, graph):
        """两阶段策略（适用于大规模静态调度）：
        Stage 1: 贪心分配任务到车辆
        Stage 2: Gurobi 优化每辆车的路线（每车 5-15 个任务，快速求解）
        """
        from scheduler_greedy import GreedyEVRPScheduler
        greedy = GreedyEVRPScheduler(strategy_type="nearest")
        greedy_assignments = greedy.solve_assignment(vehicles, pending_tasks, stations, dist_helper, graph)

        final_assignments = {str(v.id): [] for v in vehicles}
        vehicle_map = {str(v.id): v for v in vehicles}

        for v_id_str, greedy_route in greedy_assignments.items():
            v = vehicle_map.get(v_id_str)
            if not v or not greedy_route:
                continue

            # 提取贪心路线中的任务列表
            assigned_task_ids = []
            charge_stops = []
            for cmd in greedy_route:
                if cmd.startswith("TASK:"):
                    assigned_task_ids.append(cmd.split(":", 1)[1])
                elif cmd.startswith("CHARGE:"):
                    charge_stops.append(cmd.split(":", 1)[1])

            if not assigned_task_ids:
                continue

            # 只选取该车辆被分配的任务，构建小规模 MILP
            my_tasks = [t for t in pending_tasks if t.id in assigned_task_ids]
            if len(my_tasks) <= 3:
                # 任务太少，直接用贪心结果
                final_assignments[v_id_str] = greedy_route
                continue

            # Stage 2: Gurobi 优化单车路线
            opt_route = self._solve_single_vehicle_route(
                v, my_tasks, stations, dist_helper, graph
            )
            if opt_route:
                final_assignments[v_id_str] = opt_route
            else:
                final_assignments[v_id_str] = greedy_route  # 回退到贪心

        return final_assignments

    def _solve_single_vehicle_route(self, vehicle, my_tasks, stations, dist_helper, graph):
        """对单车+少量任务构建小规模 MILP，秒级求解"""
        if not my_tasks:
            return []

        task_nodes = [f"T_{t.id}" for t in my_tasks]
        station_nodes = [f"S_{s.id}" for s in stations]
        start_n = "START"
        end_n = "END"

        V_t = {f"T_{t.id}": t for t in my_tasks}
        V_s = {f"S_{s.id}": s for s in stations}

        loc_map = {start_n: vehicle.current_location, end_n: vehicle.depot_node}
        for t in my_tasks:
            loc_map[f"T_{t.id}"] = t.location_node
        for s in stations:
            loc_map[f"S_{s.id}"] = s.location_node

        # 预计算距离
        all_n = [start_n, end_n] + task_nodes + station_nodes
        d_cache = {}
        for i in all_n:
            for j in all_n:
                if i == j:
                    continue
                d = dist_helper.get_distance(loc_map[i], loc_map[j])
                d_cache[(i, j)] = d if (d is not None and d != float('inf')) else None

        # 弧段剪枝
        max_range = vehicle.max_battery / (vehicle.consumption_rate_dist + vehicle.max_payload * vehicle.consumption_rate_payload)
        valid_e = set()
        for i in all_n:
            for j in all_n:
                if i == j or i == end_n or j == start_n:
                    continue
                d = d_cache.get((i, j))
                if d is None or d > max_range * 1.2:
                    continue
                valid_e.add((i, j))

        valid_e_list = list(valid_e)

        if len(valid_e_list) < 3:
            return None

        model = gp.Model("SingleVehicle")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', max(3, self.time_limit // 3))
        model.setParam('MIPGap', self.gap_tolerance)

        x = model.addVars(valid_e_list, vtype=GRB.BINARY, name="x")
        y = model.addVars(task_nodes, vtype=GRB.BINARY, name="y")

        t_vars = {}
        b_vars = {}
        u_vars = {}
        for i in all_n:
            t_vars[i] = model.addVar(lb=0.0, name=f"t_{i}")
            b_vars[i] = model.addVar(lb=0.0, ub=vehicle.max_battery, name=f"b_{i}")
            u_vars[i] = model.addVar(lb=0.0, ub=vehicle.max_payload, name=f"u_{i}")

        charge_vars = {}
        for s_node in station_nodes:
            charge_vars[s_node] = model.addVar(lb=0.0, ub=vehicle.max_battery, name=f"chg_{s_node}")

        # 流平衡
        mid_nodes = task_nodes + station_nodes
        model.addConstr(gp.quicksum(x[start_n, j] for j in mid_nodes + [end_n] if (start_n, j) in valid_e) == 1)
        model.addConstr(gp.quicksum(x[i, end_n] for i in mid_nodes + [start_n] if (i, end_n) in valid_e) == 1)
        for j in mid_nodes:
            in_flow = gp.quicksum(x[i, j] for i in [start_n] + mid_nodes if (i, j) in valid_e)
            out_flow = gp.quicksum(x[j, p] for p in mid_nodes + [end_n] if (j, p) in valid_e)
            model.addConstr(in_flow == out_flow)

        # 起点初始化
        model.addConstr(b_vars[start_n] == vehicle.current_battery)
        model.addConstr(u_vars[start_n] == vehicle.max_payload)
        model.addConstr(t_vars[start_n] == 0)

        cons_rate = vehicle.consumption_rate_dist + vehicle.max_payload * vehicle.consumption_rate_payload
        big_M = 100000.0

        for i, j in valid_e_list:
            dist = d_cache.get((i, j), 10000) or 10000
            travel_time = (dist / 1000.0) / vehicle.speed_kmh * 60.0
            energy_cost = dist * cons_rate

            svc = 0
            if i in task_nodes:
                svc = 10
            elif i in station_nodes:
                svc = 30

            model.addConstr(t_vars[j] >= t_vars[i] + travel_time + svc - big_M * (1 - x[i, j]))
            if i in station_nodes:
                model.addConstr(b_vars[j] <= b_vars[i] + charge_vars[i] - energy_cost + big_M * (1 - x[i, j]))
            else:
                model.addConstr(b_vars[j] <= b_vars[i] - energy_cost + big_M * (1 - x[i, j]))

            demand_j = V_t[j].weight if j in task_nodes else 0
            model.addConstr(u_vars[j] <= u_vars[i] - demand_j + big_M * (1 - x[i, j]))

        # 任务访问约束
        for tn in task_nodes:
            in_e = [(i, j) for i, j in valid_e_list if j == tn]
            model.addConstr(gp.quicksum(x[i, j] for i, j in in_e) == y[tn])

        # 迟到惩罚
        late_vars = {}
        for tn in task_nodes:
            task = V_t[tn]
            lv = model.addVar(lb=0.0, name=f"late_{tn}")
            model.addConstr(lv >= t_vars[tn] - task.deadline)
            late_vars[tn] = lv

        # 目标函数
        REWARD = 10000
        DROP_PENALTY = 50000
        LATE_RATE = 200
        obj = gp.quicksum(y[tn] * REWARD for tn in task_nodes)
        obj -= gp.quicksum((1 - y[tn]) * (DROP_PENALTY + V_t[tn].weight) for tn in task_nodes)
        for i, j in valid_e_list:
            obj -= x[i, j] * (d_cache.get((i, j), 0) or 0) * 0.01
        for s_node in station_nodes:
            in_e = [(i, j) for i, j in valid_e_list if j == s_node]
            obj -= gp.quicksum(x[i, j] for i, j in in_e) * 500
        for tn, lv in late_vars.items():
            obj -= lv * LATE_RATE
        model.setObjective(obj, GRB.MAXIMIZE)

        try:
            model.optimize()
        except gp.GurobiError:
            return None

        if model.SolCount == 0:
            return None

        # 解析路线
        route = []
        curr = start_n
        while curr != end_n:
            next_n = None
            for j in mid_nodes + [end_n]:
                if (curr, j) in valid_e and x[curr, j].X > 0.5:
                    next_n = j
                    break
            if not next_n or next_n == end_n:
                break
            if next_n in task_nodes:
                route.append(f"TASK:{V_t[next_n].id}")
            elif next_n in station_nodes:
                route.append(f"CHARGE:{V_s[next_n].id}")
            curr = next_n

        return route

    def solve_assignment(self, vehicles, pending_tasks, stations, dist_helper, graph):
        # 大规模调度 → 两阶段策略（待分配任务 > 20 或待分配任务数 * 车辆数 > 50）
        if len(pending_tasks) > 20 or len(pending_tasks) * len(vehicles) > 50:
            print(f"  [Gurobi] 大规模调度({len(pending_tasks)}任务×{len(vehicles)}车) → 两阶段: 贪心分配 + Gurobi优化单车路线")
            return self._solve_two_stage(vehicles, pending_tasks, stations, dist_helper, graph)
        # --- 新增预处理: 提前剔除无法连通的孤岛任务，根本不将其纳入数学模型 ---
        reachable_tasks = []
        for t in pending_tasks:
            is_reachable = False
            for k in vehicles:
                dist_to_task = dist_helper.get_distance(k.current_location, t.location_node)
                # 只要网路中至少有一辆车能开到这个点，它就不是死点
                if dist_to_task is not None:
                    is_reachable = True
                    break
            if is_reachable:
                reachable_tasks.append(t)
            # 如果不可达，连 print 占位都不需要提供，直接在系统层面“蒸发”这个任务
        
        # 覆盖为清洗后的安全任务集
        pending_tasks = reachable_tasks
        
        # 如果所有任务都是死点（或本来就没任务），直接返回空分配，省去求解
        if not pending_tasks:
            return {str(k.id): [] for k in vehicles}

        model = gp.Model("EVRPTW_Routing")
        model.setParam('OutputFlag', 0)
        model.setParam('TimeLimit', self.time_limit)
        model.setParam('MIPGap', self.gap_tolerance)

        # -----------------------------
        # 1. 图节点构建与弧段剪枝
        # -----------------------------
        task_nodes = [f"TASK_{t.id}" for t in pending_tasks]
        station_nodes = [f"STATION_{s.id}" for s in stations]

        V_tasks = {f"TASK_{t.id}": t for t in pending_tasks}
        V_stations = {f"STATION_{s.id}": s for s in stations}

        # 建立 虚拟ID -> 物理坐标 的映射字典
        location_map = {}
        for t in pending_tasks: location_map[f"TASK_{t.id}"] = t.location_node
        for s in stations: location_map[f"STATION_{s.id}"] = s.location_node
        for k in vehicles:
            location_map[f"START_{k.id}"] = k.current_location
            location_map[f"END_{k.id}"] = k.depot_node

        # 预计算所有节点对之间的距离
        all_nodes = []
        for k in vehicles:
            all_nodes.extend([f"START_{k.id}", f"END_{k.id}"])
        all_nodes.extend(task_nodes)
        all_nodes.extend(station_nodes)
        all_nodes = list(set(all_nodes))  # 去重

        dist_cache = {}
        for i in all_nodes:
            for j in all_nodes:
                if i == j:
                    continue
                key = (i, j)
                if key not in dist_cache:
                    loc_i = location_map[i]
                    loc_j = location_map[j]
                    d = dist_helper.get_distance(loc_i, loc_j)
                    dist_cache[key] = d if (d is not None and d != float('inf')) else None

        # 弧段剪枝：只保留距离在电池续航范围内的边
        valid_edges = set()
        edges_by_target = {}  # 按目标节点预分组，加速约束构建
        for k in vehicles:
            start_k = f"START_{k.id}"
            end_k = f"END_{k.id}"
            max_range = k.max_battery / (k.consumption_rate_dist + k.max_payload * k.consumption_rate_payload)

            V_k = [start_k, end_k] + task_nodes + station_nodes

            for i in V_k:
                for j in V_k:
                    if i == j or i == end_k or j == start_k:
                        continue
                    d = dist_cache.get((i, j))
                    if d is None:
                        continue  # 跳过不可达的边
                    if d > max_range * 1.5:
                        # 距离超过续航 1.5 倍，不可能直接行驶（即使用中途充电站也极不可能选这条边）
                        continue
                    edge = (i, j, k.id)
                    valid_edges.add(edge)
                    if j not in edges_by_target:
                        edges_by_target[j] = []
                    edges_by_target[j].append(edge)

        valid_edges_list = list(valid_edges)
        print(f"  [Gurobi] 弧段剪枝: {len(valid_edges_list)} 条有效边 (剪枝前约 {len(all_nodes)**2 * len(vehicles)} 条)")

        # -----------------------------
        # 2. 决策变量定义
        # -----------------------------
        # x[i, j, k]: 车辆 k 是否驶过边 (i, j)
        x = model.addVars(valid_edges_list, vtype=GRB.BINARY, name="x")
        
        # y[i]: 任务 i 是否被完成 (允许弃单惩罚)
        y = model.addVars(task_nodes, vtype=GRB.BINARY, name="y")

        # 连续流变量 (针对每个节点和每辆车)
        t = {} # 到达时间 time
        b = {} # 离开电量 battery
        u = {} # 剩余载重 payload
        charge_amt = {} # 充电量
        
        for k in vehicles:
            start_k = f"START_{k.id}"
            end_k = f"END_{k.id}"
            V_k = [start_k, end_k] + task_nodes + station_nodes
            
            for i in V_k:
                t[i, k.id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name=f"t_{i}_{k.id}")
                b[i, k.id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=k.max_battery, name=f"b_{i}_{k.id}")
                u[i, k.id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=k.max_payload, name=f"u_{i}_{k.id}")
                if i in station_nodes:
                    charge_amt[i, k.id] = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, ub=k.max_battery, name=f"charge_{i}_{k.id}")

        # -----------------------------
        # 3. 核心约束逻辑
        # -----------------------------
        # 距离缓存已在上方预计算，此处将 None 转为大数用于约束中的 M 项
        _dist_for_constr = {}
        for key, d in dist_cache.items():
            _dist_for_constr[key] = d if d is not None else 100000.0

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
            for i, j, k_id in valid_edges_list:
                if k_id != k.id: continue

                dist = _dist_for_constr.get((i, j), 0)
                
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

        # d. 任务约束：每个任务必须被一辆车访问（使用预分组边，O(1) 查找）
        late_vars = {}
        for task_node in task_nodes:
            incoming = edges_by_target.get(task_node, [])
            model.addConstr(
                gp.quicksum(x[i, j, k_id] for i, j, k_id in incoming) == y[task_node],
                name=f"visit_{task_node}"
            )

            # e. 软时间窗约束
            task = V_tasks[task_node]
            for k in vehicles:
                late_var = model.addVar(lb=0.0, name=f"late_{task_node}_{k.id}")
                model.addConstr(late_var >= t[task_node, k.id] - task.deadline)
                late_vars[(task_node, k.id)] = late_var

        # -----------------------------
        # 4. 目标函数构造
        # -----------------------------
        REWARD_BASE = 10000
        PENALTY_DROP = 50000
        LATE_PENALTY_PER_MIN = 200

        obj_expr = gp.quicksum(y[n] * REWARD_BASE for n in task_nodes)
        obj_expr -= gp.quicksum((1 - y[n]) * (PENALTY_DROP + V_tasks[n].weight) for n in task_nodes)

        for i, j, k_id in valid_edges_list:
            dist = _dist_for_constr.get((i, j), 0)
            obj_expr -= x[i, j, k_id] * dist * 0.01

        # 充电惩罚 — 使用预分组边
        for s in station_nodes:
            incoming = edges_by_target.get(s, [])
            station_visits = gp.quicksum(x[i, j, k_id] for i, j, k_id in incoming)
            obj_expr -= station_visits * 500

        # 迟到惩罚 — 对每个任务-车辆组合的迟到时间进行惩罚
        for (task_node, k_id), late_var in late_vars.items():
            obj_expr -= late_var * LATE_PENALTY_PER_MIN

        model.setObjective(obj_expr, GRB.MAXIMIZE)

        # --- 热启动：用贪心解初始化 Gurobi ---
        try:
            from scheduler_greedy import GreedyEVRPScheduler
            greedy = GreedyEVRPScheduler(strategy_type="nearest")
            greedy_sol = greedy.solve_assignment(vehicles, pending_tasks, stations, dist_helper, graph)

            # 将贪心路径转换为 Gurobi 变量赋值
            for v in vehicles:
                route = greedy_sol.get(str(v.id), [])
                if not route:
                    continue
                start_k = f"START_{v.id}"
                end_k = f"END_{v.id}"
                prev_node = start_k
                for cmd in route:
                    if cmd.startswith("TASK:"):
                        tid = cmd.split(":", 1)[1]
                        curr_node = f"TASK_{tid}"
                    elif cmd.startswith("CHARGE:"):
                        sid = cmd.split(":", 1)[1]
                        curr_node = f"STATION_{sid}"
                    else:
                        continue
                    edge = (prev_node, curr_node, v.id)
                    if edge in valid_edges:
                        x[edge].Start = 1.0
                    prev_node = curr_node
                # 最后一段: 回仓库
                edge = (prev_node, end_k, v.id)
                if edge in valid_edges:
                    x[edge].Start = 1.0
                # 标记任务被访问
                for cmd in route:
                    if cmd.startswith("TASK:"):
                        tid = cmd.split(":", 1)[1]
                        tn = f"TASK_{tid}"
                        if tn in y:
                            y[tn].Start = 1.0
        except Exception as e:
            pass  # 热启动失败不影响求解

        # --- 求解参数：优先寻找可行解 ---
        model.setParam('MIPFocus', 1)        # 侧重寻找可行解
        model.setParam('NoRelHeurTime', 5)   # 5 秒启发式搜索
        model.setParam('Heuristics', 0.5)    # 更多启发式时间
        if self.time_limit >= 10:
            model.setParam('NoRelHeurTime', min(10, self.time_limit // 3))

        try:
            model.optimize()
            if model.SolCount > 0:
                print(f"  [Gurobi] 求解完成: 状态={model.status}, 目标值={model.ObjVal:.0f}, "
                      f"Gap={model.MIPGap*100:.1f}%, 耗时={model.Runtime:.1f}s")
            else:
                print(f"  [Gurobi] 未找到可行解 (状态={model.status}, 耗时={model.Runtime:.1f}s)")
        except gp.GurobiError as e:
            if "size-limited" in str(e) or "too large" in str(e):
                print("  [Gurobi] 警告: 当前许可证有规模限制，自动降级为贪心策略。")
                print("  [Gurobi] 提示: 在 https://gurobi.com/academia/ 注册免费学术许可证即可解除限制。")
            else:
                print("  [Gurobi] 求解错误: %s" % str(e))
            # 降级为贪心策略
            from scheduler_greedy import GreedyEVRPScheduler
            fallback = GreedyEVRPScheduler(strategy_type="nearest")
            return fallback.solve_assignment(vehicles, pending_tasks, stations, dist_helper, graph)

        # -----------------------------
        # 5. 结果解析为 Command List
        # -----------------------------
        assignments = {str(k.id): [] for k in vehicles}

        if model.SolCount == 0:
            # 无可行解 → 回退到贪心
            print("  [Gurobi] 未找到可行解，降级为贪心策略。")
            from scheduler_greedy import GreedyEVRPScheduler
            fallback = GreedyEVRPScheduler(strategy_type="nearest")
            return fallback.solve_assignment(vehicles, pending_tasks, stations, dist_helper, graph)

        if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.INTERRUPTED] and model.SolCount > 0:
            for k in vehicles:
                start_k = f"START_{k.id}"
                end_k = f"END_{k.id}"
                
                curr_node = start_k
                route_commands = []
                
                # 追踪有向图链路
                while curr_node != end_k:
                    next_node = None
                    for j in task_nodes + station_nodes + [end_k]:
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