# simulation_engine.py

import networkx as nx
import heapq
import random
import math
import pandas as pd
import os

# --- 引入 Gurobi 调度器 ---
# 确保 scheduler_gurobi.py 在同一目录下
from scheduler_gurobi import GurobiEVRPScheduler 

# --- 0. 距离辅助类 ---
class DistanceHelper:
    """封装距离矩阵以便快速查询，支持实时降级计算"""
    # 变更: 增加 graph 参数
    def __init__(self, distance_matrix_df, snapped_points_df, G=None):
        self._matrix = distance_matrix_df
        self._index_to_node = snapped_points_df['node_id'].to_dict()
        self._node_to_index = {v: k for k, v in self._index_to_node.items()}
        self.G = G # 保存图引用

    def get_distance(self, from_node, to_node):
        """根据节点ID查询距离，矩阵缺失时实时计算"""
        # 1. 尝试查表
        try:
            from_idx = self._node_to_index[from_node]
            to_idx = self._node_to_index[to_node]
            # 确保索引存在于矩阵行列中
            if str(from_idx) in self._matrix.index and str(to_idx) in self._matrix.columns:
                distance = self._matrix.loc[str(from_idx), str(to_idx)]
                if distance != float('inf'):
                    return distance
        except KeyError:
            pass # 节点不在 CSV 索引表中
        
        # 2. 查表失败，尝试 NetworkX 实时计算 (保底方案)
        if self.G:
            try:
                # 使用 weight='weight' 或 'length'
                dist = nx.shortest_path_length(self.G, source=from_node, target=to_node, weight='weight')
                return dist
            except nx.NetworkXNoPath:
                return float('inf') # 真的不可达
            except nx.NodeNotFound:
                return float('inf')
        
        return None

# --- 1. 核心实体类定义 ---

class Task:
    """任务类"""
    def __init__(self, task_id, location_node, weight, creation_time, deadline):
        self.id = str(task_id)          # <--- 关键修复：强制转换为字符串
        self.location_node = location_node  # 任务地点的路网节点ID
        self.weight = weight                # 任务重量 (kg)
        self.creation_time = creation_time  # 任务生成时间 (分钟)
        self.deadline = deadline            # 任务最晚完成时间 (分钟)
        self.status = "PENDING"             # 状态: PENDING, ASSIGNED, COMPLETED, FAILED

    def __repr__(self):
        return f"Task(id={self.id}, loc={self.location_node}, w={self.weight}, dl={self.deadline})"
    
    # --- 新增：允许 Task 对象之间进行比较 ---
    def __lt__(self, other):
        # 当创建时间相同时，按 ID 字符串进行比较，确保排序确定性
        return str(self.id) < str(other.id)

class ChargingStation:
    """充电站类"""
    def __init__(self, station_id, location_node, num_chargers):
        self.id = station_id
        self.location_node = location_node  # 充电站的路网节点ID
        self.num_chargers = num_chargers    # 充电桩数量
        self.charging_vehicles = set()      # 正在充电的车辆ID集合
        self.queue = []                     # 等待充电的车辆ID队列

    def is_available(self):
        """检查是否有空闲充电桩"""
        return len(self.charging_vehicles) < self.num_chargers

    def add_to_queue(self, vehicle_id):
        """车辆加入等待队列"""
        if vehicle_id not in self.queue:
            self.queue.append(vehicle_id)

    def start_charging(self, vehicle_id):
        """开始为车辆充电"""
        if self.is_available() and vehicle_id not in self.charging_vehicles:
            if vehicle_id in self.queue:
                self.queue.remove(vehicle_id)
            self.charging_vehicles.add(vehicle_id)
            return True
        return False

    def stop_charging(self, vehicle_id):
        """车辆停止充电"""
        if vehicle_id in self.charging_vehicles:
            self.charging_vehicles.remove(vehicle_id)

class Vehicle:
    """新能源物流车类 (智能体)"""
    def __init__(self, vehicle_id, depot_node, max_battery, max_payload,
                 charge_rate, consumption_rate_dist, consumption_rate_payload, speed_kmh=40):
        self.id = vehicle_id
        self.depot_node = depot_node
        self.max_battery = max_battery            # 单位: kWh
        self.max_payload = max_payload            # 单位: kg
        self.charge_rate = charge_rate            # 单位: kWh/min
        
        # --- 单位换算 ---
        self.speed_kmh = speed_kmh
        self.meters_per_min = (speed_kmh * 1000) / 60.0 
        
        self.consumption_rate_dist = consumption_rate_dist      
        self.consumption_rate_payload = consumption_rate_payload 

        # 动态状态
        self.status = "IDLE"
        self.current_location = depot_node
        self.current_battery = max_battery
        self.current_payload = 0
        self.current_task = None
        self.path = []
        self.destination_station = None
        self.current_edge_progress = 0.0
        
        # --- 全新逻辑：待执行计划队列 ---
        # 存储格式举例: ['TASK:1', 'CHARGE:S1', 'TASK:2_part1']
        self.plan_queue = [] 

    def __repr__(self):
        return f"Vehicle(id={self.id}, loc={self.current_location}, bat={self.current_battery:.2f}kWh, status={self.status}, plan={len(self.plan_queue)})"

    def _calculate_consumption(self, distance, payload):
        """计算耗电量"""
        return (distance * self.consumption_rate_dist) + (payload * distance * self.consumption_rate_payload)

    def execute_plan(self, G, tasks_map, stations_map):
        """执行计划队列中的指令"""
        if self.status != "IDLE":
            return

        if not self.plan_queue:
            if self.current_location != self.depot_node and self.status == "IDLE":
                self.go_to_depot(G)
            return

        command = self.plan_queue.pop(0)
        cmd_type, target_id = command.split(":")
        
        print(f"  [执行] 车辆 {self.id} 开始执行: {command}")

        if cmd_type == "TASK":
            task = tasks_map.get(target_id)
            if task and task.status == "PENDING":
                self.assign_task(task, G)
            else:
                print(f"    -> 任务 {target_id} 状态异常 ({task.status if task else 'None'}), 跳过。")
                self.execute_plan(G, tasks_map, stations_map) 

        elif cmd_type == "CHARGE":
            station = stations_map.get(target_id)
            if station:
                self.go_to_station(station, G)

    def assign_task(self, task, G):
        """前往执行任务"""
        try:
            path_to_task = nx.shortest_path(G, source=self.current_location, target=task.location_node, weight='weight')
            self.current_task = task
            self.path = path_to_task[1:]
            self.status = "MOVING_TO_TASK"
            self.current_edge_progress = 0.0
            task.status = "ASSIGNED"
            print(f"    -> 前往任务 {task.id}")
        except nx.NetworkXNoPath:
            print(f"    -> 无法到达任务 {task.id}")

    def move(self, G):
        """车辆移动一个时间步"""
        if self.status in ["MOVING_TO_TASK", "MOVING_TO_STATION", "MOVING_TO_DEPOT"]:
            if not self.path:
                if self.status == "MOVING_TO_TASK":
                    self.status = "SERVICING"
                    self.current_payload = self.current_task.weight
                    print(f"  [到达] 车辆 {self.id} 到达任务点 {self.current_task.id}。")
                elif self.status == "MOVING_TO_STATION":
                    self.status = "CHARGING" 
                    print(f"  [到达] 车辆 {self.id} 到达充电站。")
                elif self.status == "MOVING_TO_DEPOT":
                    self.status = "IDLE"
                    print(f"  [到达] 车辆 {self.id} 返回仓库。")
                return

            meters_remaining_budget = self.meters_per_min
            
            while meters_remaining_budget > 0 and self.path:
                next_node = self.path[0]
                try:
                    edge_data = G.edges[self.current_location, next_node, 0]
                    total_edge_length = edge_data.get('length', edge_data.get('weight', 50.0))
                except KeyError:
                    total_edge_length = 50.0
                
                dist_needed = total_edge_length - self.current_edge_progress
                
                if dist_needed <= meters_remaining_budget:
                    consumption = self._calculate_consumption(dist_needed, self.current_payload)
                    self.current_battery -= consumption
                    meters_remaining_budget -= dist_needed
                    self.path.pop(0)
                    self.current_location = next_node
                    self.current_edge_progress = 0.0
                else:
                    consumption = self._calculate_consumption(meters_remaining_budget, self.current_payload)
                    self.current_battery -= consumption
                    self.current_edge_progress += meters_remaining_budget
                    meters_remaining_budget = 0
            
            if self.current_battery <= 0:
                print(f"  [警告] 车辆 {self.id} 电量耗尽！")
                self.current_battery = 0
                self.status = "IDLE"

    def charge(self, station):
        """车辆充电一个时间步"""
        if self.status == "CHARGING" and self.id in station.charging_vehicles:
            self.current_battery += self.charge_rate
            if self.current_battery >= self.max_battery:
                self.current_battery = self.max_battery
                self.status = "IDLE" 
                station.stop_charging(self.id)
                self.current_edge_progress = 0.0 
                print(f"  [充电完成] 车辆 {self.id} 在充电站 {station.id} 充满电。")

    def service_task(self, current_time, G):
        """处理任务"""
        if self.status == "SERVICING":
            task_to_return = self.current_task
            
            if task_to_return is None:
                print(f"  [错误] 车辆 {self.id} 处于 SERVICING 状态但 current_task 为 None!")
                return None

            task_to_return.status = "COMPLETED"
            self.go_to_depot(G)
            self.current_task = None 
            return task_to_return
        return None

    def go_to_station(self, station, G):
        try:
            path_to_station = nx.shortest_path(G, source=self.current_location, target=station.location_node, weight='weight')
            self.path = path_to_station[1:]
            self.status = "MOVING_TO_STATION"
            self.destination_station = station
            self.current_edge_progress = 0.0 
            print(f"  [移动] 车辆 {self.id} 前往充电站 {station.id}。")
        except nx.NetworkXNoPath:
            pass

    def go_to_depot(self, G):
        if self.current_location == self.depot_node:
            self.status = "IDLE"
            return
        try:
            path_to_depot = nx.shortest_path(G, source=self.current_location, target=self.depot_node, weight='weight')
            self.path = path_to_depot[1:]
            self.status = "MOVING_TO_DEPOT"
            self.current_payload = 0 
            self.current_edge_progress = 0.0 
            print(f"  [移动] 车辆 {self.id} 完成任务，返回仓库。")
        except nx.NetworkXNoPath:
            pass


# --- 2. 仿真器主类 ---

class Simulator:
    """物流仿真引擎"""
    def __init__(self, graph, vehicle_configs, station_configs, initial_tasks, dist_helper):
        self.G = graph
        self.dist_helper = dist_helper 
        self.current_time = 0
        self.score = 0

        self.depot_node = vehicle_configs[0]['depot_node'] 
        self.vehicles = [Vehicle(**vc) for vc in vehicle_configs]
        self.stations = {sc['station_id']: ChargingStation(**sc) for sc in station_configs}
        
        self.tasks = {t.id: t for t in initial_tasks}
        self.task_event_queue = []
        self.scheduler = GurobiEVRPScheduler(time_limit=5)

    def is_simulation_finished(self):
        """检查是否所有任务都已处理完毕，且所有车辆均已空闲且无待办计划"""
        if len(self.task_event_queue) > 0:
            return False
            
        for task in self.tasks.values():
            if task.status in ["PENDING", "ASSIGNED"]:
                return False
                
        for v in self.vehicles:
            if v.status != "IDLE" or len(v.plan_queue) > 0:
                return False
                
        return True

    def add_dynamic_task(self, task):
        max_fleet_payload = 0
        if self.vehicles:
            max_fleet_payload = max(v.max_payload for v in self.vehicles)
        else:
            max_fleet_payload = 1000 

        if task.weight > max_fleet_payload:
            num_splits = math.ceil(task.weight / max_fleet_payload)
            sub_weight = task.weight / num_splits
            
            print(f"  [系统] 任务 {task.id} (重 {task.weight}kg) 超过单车运力上限 ({max_fleet_payload}kg)，拆分为 {num_splits} 个子任务以进行协同运输。")

            for i in range(num_splits):
                sub_task_id = f"{task.id}_part{i+1}"
                sub_task = Task(
                    task_id=sub_task_id, 
                    location_node=task.location_node, 
                    weight=sub_weight, 
                    creation_time=task.creation_time, 
                    deadline=task.deadline
                )
                heapq.heappush(self.task_event_queue, (sub_task.creation_time, sub_task))
        else:
            heapq.heappush(self.task_event_queue, (task.creation_time, task))

    def _process_station_queues(self):
        for station in self.stations.values():
            while station.is_available() and station.queue:
                vehicle_id_to_charge = station.queue.pop(0)
                if station.start_charging(vehicle_id_to_charge):
                    for v in self.vehicles:
                        if v.id == vehicle_id_to_charge:
                            print(f"  [充电] 车辆 {v.id} 在充电站 {station.id} 开始充电。")
                            break
    
    def _update_score(self, task):
        base_reward = 100
        alpha = 1.0  
        beta = 2.0   
        max_bonus = 50.0

        completion_time = self.current_time
        time_diff = task.deadline - completion_time
        
        early_bonus = 0
        late_penalty = 0

        if time_diff >= 0:
            early_bonus = min(max_bonus, time_diff * alpha)
        else:
            late_penalty = abs(time_diff) * beta
            
        task_score = base_reward + early_bonus - late_penalty
        self.score += task_score
        
        print(f"  [计分] 任务 {task.id} 完成！(DL:{task.deadline}, T:{completion_time}) 基础:{base_reward} 提前:{early_bonus:.1f} 逾期:-{late_penalty:.1f} -> 得分:{task_score:.1f}, 总分:{self.score:.1f}")

    def _check_failed_tasks(self):
        for task in self.tasks.values():
            if task.status in ["PENDING", "ASSIGNED"] and self.current_time > task.deadline:
                task.status = "FAILED"
                self.score -= 200 
                print(f"  [失败] 任务 {task.id} 超时未完成！扣分: 200, 总分: {self.score}")

    def step(self):
        print(f"\n--- 时间: {self.current_time} 分钟 ---")

        while self.task_event_queue and self.task_event_queue[0][0] <= self.current_time:
            _, new_task = heapq.heappop(self.task_event_queue)
            self.tasks[new_task.id] = new_task
            print(f"  [事件] 新任务生成: {new_task}")

        pending_tasks = [t for t in self.tasks.values() if t.status == "PENDING"]
        idle_vehicles = [v for v in self.vehicles if v.status == "IDLE" and not v.plan_queue]
        
        if pending_tasks and idle_vehicles:
            print(f"  [调度] 触发全局优化... (待分配: {len(pending_tasks)}, 空闲车: {len(idle_vehicles)})")
            
            assignments = self.scheduler.solve_assignment(
                idle_vehicles, 
                pending_tasks, 
                self.stations.values(), 
                self.dist_helper, 
                self.G
            )
            
            if not assignments:
                print("  [警告] 调度器返回了空字典 (可能是不可行)")

            for v_id, route in assignments.items():
                if route:
                    target_vehicle = next((v for v in self.vehicles if str(v.id) == str(v_id)), None)
                    if target_vehicle:
                        target_vehicle.plan_queue = route
                        print(f"  [调度] 车辆 {target_vehicle.id} 更新计划: {route}")
                    else:
                        print(f"  [错误] 调度器返回未知车辆ID: {v_id}")
                else:
                    print(f"  [调度] 车辆 {v_id} 分配到空路径 (原地待命)")

        self._check_failed_tasks()
        self._process_station_queues()

        for v in self.vehicles:
            v.execute_plan(self.G, self.tasks, self.stations)
            
            if "MOVING" in v.status:
                v.move(self.G)
            elif v.status == "CHARGING":
                if v.destination_station and v.id not in v.destination_station.charging_vehicles:
                    v.destination_station.add_to_queue(v.id)
                    print(f"  [排队] 车辆 {v.id} 在充电站 {v.destination_station.id} 等待充电。")
                elif v.destination_station:
                    v.charge(v.destination_station)
            elif v.status == "SERVICING":
                completed_task = v.service_task(self.current_time, self.G) 
                if completed_task:
                    print(f"  [调试] 收到完成任务: {completed_task.id}，准备计分...") 
                    self._update_score(completed_task)
                else:
                    print(f"  [调试] 车辆 {v.id} 完成服务但未返回任务对象。")
            
            print(f"  [状态] {v}")

        self.current_time += 1

    def run(self, max_simulation_duration=1440):
        """运行整个仿真直到全部任务完成或达到安全超时限制"""
        print("=== 仿真开始 ===")
        
        print(f"--- 初始状态 (时间: 0 分钟前) ---")
        for v in self.vehicles:
            print(f"  [初始] {v}")
            if v.current_location != self.depot_node:
                 print(f"  [警告] 车辆 {v.id} 初始化位置异常！")

        while not self.is_simulation_finished() and self.current_time < max_simulation_duration:
            self.step()

        print(f"\n=== 仿真结束 (时间: {self.current_time} 分钟) ===")
        if self.current_time >= max_simulation_duration:
            print("⚠️ 警告: 达到最大保障仿真时长，强制退出（可能发生了死锁或某些任务无法到达）")
        print(f"最终得分: {self.score}")


# --- 3. 仿真设置与运行示例 ---

if __name__ == '__main__':
    DATA_FOLDER = "data"
    SNAPPED_POINTS_FILE = os.path.join(DATA_FOLDER, "snapped_points.csv")
    DISTANCE_MATRIX_FILE = os.path.join(DATA_FOLDER, "distance_matrix.csv")
    ROAD_NETWORK_FILE = "shanghai_china.graphml" 

    print("正在加载数据文件...")
    try:
        snapped_points_df = pd.read_csv(SNAPPED_POINTS_FILE, index_col="point_index")
        distance_matrix_df = pd.read_csv(DISTANCE_MATRIX_FILE, index_col=0)
        
        G = nx.read_graphml(ROAD_NETWORK_FILE)
        G = nx.relabel_nodes(G, int) 
        
        print("正在清理路网边权重数据类型...")
        for u, v, k, data in G.edges(keys=True, data=True):
            raw_len = data.get('length', data.get('weight', 1.0))
            try:
                if isinstance(raw_len, list):
                    val = float(raw_len[0])
                else:
                    val = float(raw_len)
            except (ValueError, TypeError):
                val = 1.0 
            
            data['weight'] = val
            data['length'] = val

        print("所有数据加载成功！")

    except Exception as e:
        print(f"❌ 严重错误：数据加载或处理失败。\n详细信息: {e}")
        exit()
    
    dist_helper = DistanceHelper(distance_matrix_df, snapped_points_df, G)

    depot_info = snapped_points_df[snapped_points_df['type'].str.contains("Depot")]
    if depot_info.empty:
        raise ValueError("错误: 在 snapped_points.csv 中未找到仓库 (Depot) 点。")
    DEPOT_NODE = depot_info.iloc[0]['node_id']
    
    print(f"正在检查节点 {DEPOT_NODE} 是否在图 G 中...")
    if DEPOT_NODE not in G:
        print(f"❌ 严重错误: 仓库节点 {DEPOT_NODE} 不在加载的路网图 G 中！")
        print("可能原因：")
        print("1. 生成CSV时使用的路网与当前加载的 graphml 文件版本不一致。")
        print("2. 节点ID类型不匹配（虽然已尝试int转换）。")
        print(f"图中前5个节点ID示例: {list(G.nodes())[:5]}")
        exit()
    else:
        print(f"✅ 仓库节点检查通过。")

    station_info = snapped_points_df[snapped_points_df['type'].str.contains("Charging Station")]
    for _, row in station_info.iterrows():
        sid = row['node_id']
        if sid not in G:
             print(f"❌ 警告: 充电站节点 {sid} 不在图中，将被跳过。")
    
    station_configs = [
        {"station_id": f"S{i+1}", "location_node": row['node_id'], "num_chargers": 2} 
        for i, row in station_info.iterrows()
        if row['node_id'] in G 
    ]

    task_point_info = snapped_points_df[snapped_points_df['type'].str.contains("Task Point")]
    
    vehicle_configs = [
        {
            "vehicle_id": 1, 
            "depot_node": DEPOT_NODE, 
            "max_battery": 100,           
            "max_payload": 1000, 
            "charge_rate": 2,             
            "speed_kmh": 40,              
            "consumption_rate_dist": 0.00025,    # <--- 在这里修改基础耗电率
            "consumption_rate_payload": 0.0000001 # <--- 在这里修改与载重相关的额外耗电率
        },
        {
            "vehicle_id": 2, 
            "depot_node": DEPOT_NODE, 
            "max_battery": 100, 
            "max_payload": 1000, 
            "charge_rate": 2, 
            "speed_kmh": 40,
            "consumption_rate_dist": 0.00025,    # <--- 同上
            "consumption_rate_payload": 0.0000001 # <--- 同上
        }
    ]

    simulator = Simulator(G, vehicle_configs, station_configs, [], dist_helper)

    if not task_point_info.empty:
        valid_tasks = []
        for i in range(len(task_point_info)):
             tid = task_point_info.iloc[i]['node_id']
             if tid in G:
                 valid_tasks.append(tid)
             else:
                 print(f"⚠️ 任务点节点 {tid} 不在图中，已跳过。")
        
        print(f"找到 {len(valid_tasks)} 个有效任务点，正在生成任务...")
        
        for i, node_id in enumerate(valid_tasks):
            new_task_id = i + 1
            
            if random.random() < 0.8:
                weight = random.randint(100, 800)
            else:
                weight = random.randint(801, 1500)
            
            creation_time = i * 10 
            
            time_window = random.randint(60, 120) if weight < 800 else random.randint(90, 180)
            deadline = creation_time + time_window

            simulator.add_dynamic_task(Task(
                task_id=new_task_id, 
                location_node=node_id, 
                weight=weight, 
                creation_time=creation_time, 
                deadline=deadline
            ))
            
    else:
        print("警告: 未在 snapped_points.csv 中找到任务点，将不会生成任何任务。")

    # --- E. 运行仿真: 指定最大断路运行时长，而非绝对运行时间 ---
    simulator.run(max_simulation_duration=1440) 