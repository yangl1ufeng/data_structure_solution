import gurobipy as gp
from gurobipy import GRB
import networkx as nx

class GurobiEVRPScheduler:
    """
    高级 Gurobi EVRP 调度器 - 深度优化版本
    目标: 最大化总评分 (Total Score Maximization)
    特性: 智能充电阈值、任务冲突过滤、动态起点同步
    """

    def __init__(self, time_limit=5, gap_tolerance=0.05):
        self.time_limit = time_limit
        self.gap_tolerance = gap_tolerance
        
        # 优化参数
        self.distance_penalty_gamma = 0.001  # 距离惩罚系数
        self.charging_penalty_lambda = 10.0  # 充电次数惩罚系数
        self.charging_startup_cost = 5.0     # 中等电量充电启动成本
        
        # 智能充电阈值
        self.high_battery_threshold = 0.8    # 80% 以上禁充
        self.medium_battery_threshold = 0.5  # 50-80% 惩罚充电
        
    def preprocess_data(self, vehicles, pending_tasks, stations, dist_helper, graph):
        """
        数据预处理：过滤冲突任务、构建智能充电决策空间
        """
        # === 1. 任务准入与冲突过滤 ===
        print("  [预处理] 开始任务冲突过滤...")
        
        # 收集所有车辆计划中的任务ID
        locked_task_ids = set()
        for vehicle in vehicles:
            for plan_item in vehicle.plan_queue:
                if plan_item.startswith("TASK:"):
                    task_id = plan_item.split(":", 1)[1]
                    locked_task_ids.add(task_id)
        
        # 严格过滤：只保留 PENDING 且未锁定的任务
        filtered_tasks = []
        for task in pending_tasks:
            if task.status == "PENDING" and task.id not in locked_task_ids:
                filtered_tasks.append(task)
            else:
                print(f"  [过滤] 任务 {task.id} 被过滤 (状态:{task.status}, 锁定:{task.id in locked_task_ids})")
        
        print(f"  [预处理] 任务过滤完成: {len(pending_tasks)} -> {len(filtered_tasks)}")
        
        # === 2. 智能充电站决策空间构建 ===
        print("  [预处理] 构建智能充电决策空间...")
        
        # 为每辆车构建个性化的可访问充电站列表
        vehicle_charging_options = {}
        for vehicle in vehicles:
            battery_percentage = vehicle.current_battery / vehicle.max_battery
            available_stations = []
            
            if battery_percentage <= self.high_battery_threshold:
                # 80% 以下才允许充电
                for station in stations:
                    # 检查从当前位置是否能到达充电站
                    distance = dist_helper.get_distance(vehicle.current_location, station.location_node)
                    if distance is not None and distance < float('inf'):
                        charge_option = {
                            'station': station,
                            'distance': distance,
                            'battery_level': battery_percentage,
                            'penalty_cost': 0.0 if battery_percentage <= self.medium_battery_threshold 
                                          else self.charging_startup_cost
                        }
                        available_stations.append(charge_option)
                
                print(f"  [充电策略] 车辆 {vehicle.id} (电量:{battery_percentage:.1%}) -> 可充电站: {len(available_stations)}")
            else:
                print(f"  [充电策略] 车辆 {vehicle.id} (电量:{battery_percentage:.1%}) -> 禁止充电 (>80%)")
            
            vehicle_charging_options[vehicle.id] = available_stations
        
        # === 3. 动态起点同步 ===
        print("  [预处理] 同步车辆动态起点...")
        vehicle_origins = {}
        for vehicle in vehicles:
            vehicle_origins[vehicle.id] = vehicle.current_location
            print(f"  [起点] 车辆 {vehicle.id} 当前位置: {vehicle.current_location}")
        
        return {
            'filtered_tasks': filtered_tasks,
            'vehicle_charging_options': vehicle_charging_options,
            'vehicle_origins': vehicle_origins
        }

    def solve_assignment(self, vehicles, pending_tasks, stations, dist_helper, graph):
        """
        构建并求解优化后的 MILP 模型
        """
        if not vehicles or not pending_tasks:
            return {str(v.id): [] for v in vehicles}

        # === 数据预处理 ===
        preprocessed = self.preprocess_data(vehicles, pending_tasks, stations, dist_helper, graph)
        filtered_tasks = preprocessed['filtered_tasks']
        vehicle_charging_options = preprocessed['vehicle_charging_options']
        vehicle_origins = preprocessed['vehicle_origins']
        
        if not filtered_tasks:
            print("  [Gurobi] 无有效任务可分配")
            return {str(v.id): [] for v in vehicles}

        try:
            # === 创建模型 ===
            model = gp.Model("EVRP_Optimized")
            model.setParam('OutputFlag', 0)
            model.setParam('TimeLimit', self.time_limit)
            model.setParam('MIPGap', self.gap_tolerance)

            # === 决策变量 ===
            # 任务分配变量: y[task_id, vehicle_id]
            y = {}
            for task in filtered_tasks:
                for vehicle in vehicles:
                    y[task.id, vehicle.id] = model.addVar(vtype=GRB.BINARY, 
                                                          name=f"y_{task.id}_{vehicle.id}")
            
            # 充电决策变量: z[vehicle_id, station_id]
            z = {}
            charging_distance = {}
            charging_penalty = {}
            
            for vehicle in vehicles:
                for charge_option in vehicle_charging_options[vehicle.id]:
                    station = charge_option['station']
                    var_key = (vehicle.id, station.id)
                    z[var_key] = model.addVar(vtype=GRB.BINARY, 
                                            name=f"z_{vehicle.id}_{station.id}")
                    charging_distance[var_key] = charge_option['distance']
                    charging_penalty[var_key] = charge_option['penalty_cost']

            # === 约束条件 ===
            # 1. 每个任务最多分配给一辆车
            for task in filtered_tasks:
                model.addConstr(
                    gp.quicksum(y[task.id, v.id] for v in vehicles) <= 1,
                    name=f"task_assignment_{task.id}"
                )

            # 2. 每辆车最多分配一个任务（简化版本）
            for vehicle in vehicles:
                model.addConstr(
                    gp.quicksum(y[t.id, vehicle.id] for t in filtered_tasks) <= 1,
                    name=f"vehicle_capacity_{vehicle.id}"
                )

            # 3. 每辆车最多选择一个充电站
            for vehicle in vehicles:
                available_stations = [opt['station'].id for opt in vehicle_charging_options[vehicle.id]]
                if available_stations:
                    model.addConstr(
                        gp.quicksum(z[vehicle.id, s_id] for s_id in available_stations) <= 1,
                        name=f"charging_choice_{vehicle.id}"
                    )

            # 4. 载重约束
            for vehicle in vehicles:
                total_weight = gp.quicksum(y[t.id, vehicle.id] * t.weight for t in filtered_tasks)
                model.addConstr(total_weight <= vehicle.max_payload, 
                               name=f"payload_{vehicle.id}")

            # === 优化后的目标函数 ===
            # Maximize (TotalScore - γ × TotalDistance - λ × NumChargingEvents)
            
            # 总得分项
            total_score = 0
            for task in filtered_tasks:
                for vehicle in vehicles:
                    # 简化得分计算：基础分 + 时间效益
                    base_score = 100
                    time_bonus = max(0, task.deadline - task.creation_time - 20)  # 假设20分钟完成
                    task_score = base_score + min(50, time_bonus)
                    total_score += y[task.id, vehicle.id] * task_score

            # 总距离惩罚项
            total_distance_penalty = 0
            for task in filtered_tasks:
                for vehicle in vehicles:
                    origin = vehicle_origins[vehicle.id]
                    task_distance = dist_helper.get_distance(origin, task.location_node)
                    if task_distance is not None:
                        total_distance_penalty += y[task.id, vehicle.id] * task_distance * self.distance_penalty_gamma

            # 充电距离成本
            charging_distance_cost = 0
            for (v_id, s_id), var in z.items():
                charging_distance_cost += var * charging_distance[(v_id, s_id)] * self.distance_penalty_gamma

            # 智能充电惩罚项 (λ × NumChargingEvents + 启动成本)
            charging_events_penalty = 0
            for (v_id, s_id), var in z.items():
                # 充电次数惩罚
                charging_events_penalty += var * self.charging_penalty_lambda
                # 启动成本惩罚
                charging_events_penalty += var * charging_penalty[(v_id, s_id)]

            # 组合目标函数
            objective = (total_score - total_distance_penalty - 
                        charging_distance_cost - charging_events_penalty)
            
            model.setObjective(objective, GRB.MAXIMIZE)

            # === 求解 ===
            model.optimize()

            # === 解析结果 ===
            assignments = {str(v.id): [] for v in vehicles}
            
            if model.status == GRB.OPTIMAL or model.status == GRB.TIME_LIMIT:
                # 解析任务分配
                for task in filtered_tasks:
                    for vehicle in vehicles:
                        if y[task.id, vehicle.id].x > 0.5:
                            assignments[str(vehicle.id)].append(f"TASK:{task.id}")
                            print(f"  [Gurobi] 任务 {task.id} -> 车辆 {vehicle.id}")

                # 解析充电决策
                for (v_id, s_id), var in z.items():
                    if var.x > 0.5:
                        assignments[str(v_id)].insert(0, f"CHARGE:{s_id}")  # 充电放在任务前
                        print(f"  [Gurobi] 车辆 {v_id} -> 充电站 {s_id}")

                print(f"  [Gurobi] 优化完成，目标值: {model.objVal:.2f}")
            else:
                print(f"  [Gurobi] 求解失败，状态: {model.status}")

            return assignments

        except Exception as e:
            print(f"  [Gurobi] 异常: {e}")
            return {str(v.id): [] for v in vehicles}