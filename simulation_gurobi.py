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
        """
        计算耗电量
        Args:
            distance: 距离 (米)
            payload: 载重 (kg)
        Returns:
            kWh
        """
        return (distance * self.consumption_rate_dist) + (payload * distance * self.consumption_rate_payload)

    def execute_plan(self, G, tasks_map, stations_map):
        """
        执行计划队列中的指令
        替代原来的 _decide_next_task
        """
        # 如果正在干活（移动、充电、服务中），不需要从计划里取新指令
        if self.status != "IDLE":
            return

        # 如果计划空了，尝试回仓库（如果不在一起）
        if not self.plan_queue:
            if self.current_location != self.depot_node and self.status == "IDLE":
                self.go_to_depot(G)
            return

        # 获取下一条指令
        command = self.plan_queue.pop(0)
        cmd_type, target_id = command.split(":")
        
        print(f"  [执行] 车辆 {self.id} 开始执行: {command}")

        if cmd_type == "TASK":
            task = tasks_map.get(target_id)
            if task and task.status == "PENDING":
                self.assign_task(task, G)
            else:
                print(f"    -> 任务 {target_id} 状态异常 ({task.status if task else 'None'}), 跳过。")
                self.execute_plan(G, tasks_map, stations_map) # 递归尝试下一个

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

    # --- 这里只保留物理动作 (Move, Charge, Service) ---
    def move(self, G):
        """车辆移动一个时间步"""
        if self.status in ["MOVING_TO_TASK", "MOVING_TO_STATION", "MOVING_TO_DEPOT"]:
            if not self.path:
                # 到达目的地处理
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
                self.current_edge_progress = 0.0 # 确保重置
                print(f"  [充电完成] 车辆 {self.id} 在充电站 {station.id} 充满电。")

    def service_task(self, current_time, G):
        """处理任务"""
        if self.status == "SERVICING":
            # 1. 先保存任务对象的引用
            task_to_return = self.current_task
            
            if task_to_return is None:
                print(f"  [错误] 车辆 {self.id} 处于 SERVICING 状态但 current_task 为 None!")
                return None

            task_to_return.status = "COMPLETED"
            
            # 2. 执行回仓逻辑 (此时 go_to_depot 可能会修改 self.current_task，所以上面必须先保存)
            self.go_to_depot(G)
            
            # 3. 安全地清理引用 (如果 go_to_depot 里没清，这里清)
            self.current_task = None 
            
            return task_to_return
        return None

    def go_to_station(self, station, G):
        try:
            path_to_station = nx.shortest_path(G, source=self.current_location, target=station.location_node, weight='weight')
            self.path = path_to_station[1:]
            self.status = "MOVING_TO_STATION"
            self.destination_station = station
            self.current_edge_progress = 0.0 # 重置进度
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
            # self.current_task = None  <--- 注释掉这一行！
            # 原因: service_task 调用此方法后，需要返回 self.current_task 给主循环计分。
            # 如果在这里设为 None，service_task 返回的就是 None，由于是引用传递。
            # 这行应该在 service_task 返回后，或者下次分配任务时清理。
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

        # 初始化实体
        self.depot_node = vehicle_configs[0]['depot_node'] 
        self.vehicles = [Vehicle(**vc) for vc in vehicle_configs]
        self.stations = {sc['station_id']: ChargingStation(**sc) for sc in station_configs}
        
        # 修复：确保字典键与 Task.id 一致 (Task.id 现在强制为 str)
        self.tasks = {t.id: t for t in initial_tasks}
        
        # 用于动态生成任务的堆，按出现时间排序
        self.task_event_queue = []

        # --- 新增：初始化调度器 ---
        self.scheduler = GurobiEVRPScheduler(time_limit=5)

    # --- 修改开始：实现协同运输（任务拆分）逻辑 ---
    def add_dynamic_task(self, task):
        """
        将一个未来的动态任务加入事件队列
        如果任务过重，自动拆分为多个子任务以便多车协同
        """
        # 1. 获取车队最大单车运力
        max_fleet_payload = 0
        if self.vehicles:
            max_fleet_payload = max(v.max_payload for v in self.vehicles)
        else:
            max_fleet_payload = 1000 # 保底默认值

        # 2. 判断是否需要协同（拆分）
        if task.weight > max_fleet_payload:
            # 计算需要拆成几份 (向上取整)
            # 例如: 1200kg / 1000kg = 1.2 -> 2份
            num_splits = math.ceil(task.weight / max_fleet_payload)
            
            # 平均分配重量 (也可以按最大运力切分，这里用平均分配比较均衡)
            # 1200 / 2 = 600kg
            sub_weight = task.weight / num_splits
            
            print(f"  [系统] 任务 {task.id} (重 {task.weight}kg) 超过单车运力上限 ({max_fleet_payload}kg)，拆分为 {num_splits} 个子任务以进行协同运输。")

            for i in range(num_splits):
                # 创建子任务，ID例如 "3_part1", "3_part2"
                sub_task_id = f"{task.id}_part{i+1}"
                sub_task = Task(
                    task_id=sub_task_id, 
                    location_node=task.location_node, 
                    weight=sub_weight, 
                    creation_time=task.creation_time, 
                    deadline=task.deadline
                )
                
                # 将子任务加入队列
                heapq.heappush(self.task_event_queue, (sub_task.creation_time, sub_task))
        else:
            # 不需要拆分，直接加入
            heapq.heappush(self.task_event_queue, (task.creation_time, task))
    # --- 修改结束 ---

    def _process_station_queues(self):
        """处理所有充电站的排队和充电逻辑"""
        for station in self.stations.values():
            # 检查是否有车辆充电完成，释放充电桩
            # (在 Vehicle.charge() 中处理)

            # 如果有空位且队列里有车，则让队首车辆开始充电
            while station.is_available() and station.queue:
                vehicle_id_to_charge = station.queue.pop(0)
                if station.start_charging(vehicle_id_to_charge):
                    # 找到对应的车辆对象，更新其状态
                    for v in self.vehicles:
                        if v.id == vehicle_id_to_charge:
                            print(f"  [充电] 车辆 {v.id} 在充电站 {station.id} 开始充电。")
                            break
    
    def _update_score(self, task):
        """
        根据完成的任务更新分数 (对齐 Gurobi 目标函数)
        逻辑:
        1. 基础分: +100
        2. 效率奖: 提前 1分钟 +1分 (上限50分)
        3. 逾期罚: 迟到 1分钟 -2分 (无下限)
        """
        base_reward = 100
        alpha = 1.0  # 提前奖励系数
        beta = 2.0   # 逾期惩罚系数
        max_bonus = 50.0

        # 实际完成时间 (假设在这里 current_time 即为完成时间)
        # 注意: 这里的 self.current_time 是整型分钟
        completion_time = self.current_time
        
        time_diff = task.deadline - completion_time
        
        # 计算具体奖惩
        early_bonus = 0
        late_penalty = 0

        if time_diff >= 0:
            # 提前完成 (Diff 为正)
            early_bonus = min(max_bonus, time_diff * alpha)
        else:
            # 逾期完成 (Diff 为负)
            late_penalty = abs(time_diff) * beta
            
        # 计算该任务最终得分
        task_score = base_reward + early_bonus - late_penalty
        self.score += task_score
        
        print(f"  [计分] 任务 {task.id} 完成！(DL:{task.deadline}, T:{completion_time}) 基础:{base_reward} 提前:{early_bonus:.1f} 逾期:-{late_penalty:.1f} -> 得分:{task_score:.1f}, 总分:{self.score:.1f}")

    def _check_failed_tasks(self):
        """检查并处理超时的任务"""
        for task in self.tasks.values():
            if task.status in ["PENDING", "ASSIGNED"] and self.current_time > task.deadline:
                task.status = "FAILED"
                self.score -= 200 # 惩罚
                print(f"  [失败] 任务 {task.id} 超时未完成！扣分: 200, 总分: {self.score}")


    def step(self):
        """仿真向前推进一个时间步 (1分钟)"""
        print(f"\n--- 时间: {self.current_time} 分钟 ---")

        # 1. 生成当前时间点的新任务
        while self.task_event_queue and self.task_event_queue[0][0] <= self.current_time:
            _, new_task = heapq.heappop(self.task_event_queue)
            # 修复：确保键与 ID 一致
            self.tasks[new_task.id] = new_task
            print(f"  [事件] 新任务生成: {new_task}")

        # --- 新增：核心调度逻辑 ---
        # 如果有待分配任务，且有车空闲
        pending_tasks = [t for t in self.tasks.values() if t.status == "PENDING"]
        # 只选取完全空闲且没有计划的车参与调度
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
                    # --- 新增: 显式打印空路径分配 ---
                    print(f"  [调度] 车辆 {v_id} 分配到空路径 (原地待命)")
        # ------------------------

        # 2. 检查失败任务
        self._check_failed_tasks()

        # 3. 处理充电站队列
        self._process_station_queues()

        # 4. 遍历所有车辆，执行决策和行动
        for v in self.vehicles:
            # --- 变更点：使用 execute_plan 替代 decide ---
            v.execute_plan(self.G, self.tasks, self.stations)
            
            # 行动
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
                    print(f"  [调试] 收到完成任务: {completed_task.id}，准备计分...") # 新增调试
                    self._update_score(completed_task)
                else:
                    # 如果状态是 SERVICING 但没返回任务，这很不正常
                    print(f"  [调试] 车辆 {v.id} 完成服务但未返回任务对象。")
            
            print(f"  [状态] {v}")

        # 5. 时间步进
        self.current_time += 1

    def run(self, simulation_duration):
        """运行整个仿真"""
        print("=== 仿真开始 ===")
        
        # --- 新增：在开始前打印初始位置，确认车辆都在仓库 ---
        print(f"--- 初始状态 (时间: 0 分钟前) ---")
        for v in self.vehicles:
            print(f"  [初始] {v}")
            if v.current_location != self.depot_node:
                 print(f"  [警告] 车辆 {v.id} 初始化位置异常！")
        # -----------------------------------------------

        while self.current_time < simulation_duration:
            self.step()
        print(f"\n=== 仿真结束 (时间: {self.current_time} 分钟) ===")
        print(f"最终得分: {self.score}")


# --- 3. 仿真设置与运行示例 ---

if __name__ == '__main__':
    # --- A. 从文件加载路网、点位和距离信息 ---
    DATA_FOLDER = "data"
    SNAPPED_POINTS_FILE = os.path.join(DATA_FOLDER, "snapped_points.csv")
    DISTANCE_MATRIX_FILE = os.path.join(DATA_FOLDER, "distance_matrix.csv")
    
    # 确保这里是正确的文件名
    ROAD_NETWORK_FILE = "shanghai_china.graphml" 

    print("正在加载数据文件...")
    try:
        snapped_points_df = pd.read_csv(SNAPPED_POINTS_FILE, index_col="point_index")
        distance_matrix_df = pd.read_csv(DISTANCE_MATRIX_FILE, index_col=0)
        
        # ！！！关键修改：添加 simplify=False 参数，或者确保加载方式与生成时一致！！！
        G = nx.read_graphml(ROAD_NETWORK_FILE)
        
        # 将节点 ID 转换为整数 (GraphML有时会把ID存为字符串)
        # 这一步非常关键！CSV读出来是int，GraphML读出来可能是str
        G = nx.relabel_nodes(G, int) # 强制将图的节点ID转换为整数类型
        
        # --- ！！！新增：数据类型清洗！！！ ---
        print("正在清理路网边权重数据类型...")
        for u, v, k, data in G.edges(keys=True, data=True):
            # 1. 优先获取 'length'，如果缺失则尝试 'weight'，最后默认为 1.0
            raw_len = data.get('length', data.get('weight', 1.0))
            
            # 2. 强制类型转换: str -> float
            try:
                # 有些 dirty data 可能是列表，比如 ['5.5', '6.6']，取第一个
                if isinstance(raw_len, list):
                    val = float(raw_len[0])
                else:
                    val = float(raw_len)
            except (ValueError, TypeError):
                val = 1.0 # 转换失败时的保底值
            
            # 3. 统一赋值给 'weight' 和 'length'
            data['weight'] = val
            data['length'] = val
        # ------------------------------------

        print("所有数据加载成功！")

    except Exception as e:
        print(f"❌ 严重错误：数据加载或处理失败。\n详细信息: {e}")
        exit()
    
    # 变更: 传入 G 给 DistanceHelper
    dist_helper = DistanceHelper(distance_matrix_df, snapped_points_df, G)

    # 解析点位信息
    depot_info = snapped_points_df[snapped_points_df['type'].str.contains("Depot")]
    if depot_info.empty:
        raise ValueError("错误: 在 snapped_points.csv 中未找到仓库 (Depot) 点。")
    DEPOT_NODE = depot_info.iloc[0]['node_id']
    
    # --- ！！！新增：关键节点 ID 检查！！！ ---
    print(f"正在检查节点 {DEPOT_NODE} 是否在图 G 中...")
    if DEPOT_NODE not in G:
        print(f"❌ 严重错误: 仓库节点 {DEPOT_NODE} 不在加载的路网图 G 中！")
        print("可能原因：")
        print("1. 生成CSV时使用的路网与当前加载的 graphml 文件版本不一致。")
        print("2. 节点ID类型不匹配（虽然已尝试int转换）。")
        # 尝试打印图中前几个节点ID以供调试
        print(f"图中前5个节点ID示例: {list(G.nodes())[:5]}")
        exit()
    else:
        print(f"✅ 仓库节点检查通过。")

    station_info = snapped_points_df[snapped_points_df['type'].str.contains("Charging Station")]
    # 同样检查充电站节点
    for _, row in station_info.iterrows():
        sid = row['node_id']
        if sid not in G:
             print(f"❌ 警告: 充电站节点 {sid} 不在图中，将被跳过。")
    
    station_configs = [
        {"station_id": f"S{i+1}", "location_node": row['node_id'], "num_chargers": 2} 
        for i, row in station_info.iterrows()
        if row['node_id'] in G # 只添加存在的站点
    ]

    task_point_info = snapped_points_df[snapped_points_df['type'].str.contains("Task Point")]
    
    # 车辆配置 (需要在主程序底部修改)
    vehicle_configs = [
        {
            "vehicle_id": 1, 
            "depot_node": DEPOT_NODE, 
            "max_battery": 100,           # 100 kWh
            "max_payload": 1000, 
            "charge_rate": 2,             # 2 kWh/min (120kW快充)
            
            # --- 关键修改 ---
            "speed_kmh": 40,              # 速度: 40 km/h -> 约 667 米/分
            "consumption_rate_dist": 0.00025, 
            "consumption_rate_payload": 0.0000001
        },
        {
            "vehicle_id": 2, 
            "depot_node": DEPOT_NODE, 
            "max_battery": 100, 
            "max_payload": 1000, 
            "charge_rate": 2, 
            "speed_kmh": 40,
            "consumption_rate_dist": 0.00025,
            "consumption_rate_payload": 0.0000001
        }
    ]

    # --- C. 创建仿真器实例 ---
    simulator = Simulator(G, vehicle_configs, station_configs, [], dist_helper)

    # --- D. 基于任务点动态生成任务 ---
    # 随机从已选的任务点中生成任务
    if not task_point_info.empty:
        valid_tasks = []
        # 修改 1: 去掉 min(3, ...) 的限制，读取所有任务点
        for i in range(len(task_point_info)):
             tid = task_point_info.iloc[i]['node_id']
             if tid in G:
                 valid_tasks.append(tid)
             else:
                 print(f"⚠️ 任务点节点 {tid} 不在图中，已跳过。")
        
        # 修改 2: 使用循环动态生成任务，而不是写死 if
        print(f"找到 {len(valid_tasks)} 个有效任务点，正在生成任务...")
        
        for i, node_id in enumerate(valid_tasks):
            # 简单的动态参数生成逻辑示例
            # task_id 从 1 开始
            new_task_id = i + 1
            
            # --- 修改开始: 随机生成任务重量 ---
            # 80% 概率生成普通任务 (100kg - 800kg)
            # 20% 概率生成重型任务 (801kg - 1500kg)，触发协同运输测试
            if random.random() < 0.8:
                weight = random.randint(100, 800)
            else:
                weight = random.randint(801, 1500)
            # --- 修改结束 ---
            
            # 为了避免所有任务一开始全部堆积，让任务按顺序每 10 分钟出一个
            # 您也可以用 random.randint(0, 50) 这种让它们在一段时间内随机冒出来
            creation_time = i * 10 
            
            # --- 修改开始: 截止时间也应该随机一点 ---
            # 基础截止时间 = 创建时间 + 随机预留窗口 (60分钟 ~ 120分钟)
            # 重型任务给予更多时间
            time_window = random.randint(60, 120) if weight < 800 else random.randint(90, 180)
            deadline = creation_time + time_window
            # --- 修改结束 ---

            simulator.add_dynamic_task(Task(
                task_id=new_task_id, 
                location_node=node_id, 
                weight=weight, 
                creation_time=creation_time, 
                deadline=deadline
            ))
            
    else:
        print("警告: 未在 snapped_points.csv 中找到任务点，将不会生成任何任务。")


    # --- E. 运行仿真 ---
    simulator.run(simulation_duration=150)