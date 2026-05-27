"""
Q-Learning 强化学习调度器 —— 强化学习方法
使用 Q-Learning 训练车辆调度策略，自适应不同场景。

状态空间: 车辆电量/距离/载重 + 任务重量/紧急度 → 离散化
动作空间: 为车辆选择下一个任务
奖励设计: 任务完成正奖励，超时/失败负惩罚
"""

import random
import math
import time
import os
import pickle
import json


class QLearningEVRPScheduler:
    """基于Q-Learning的EVRP调度器"""

    def __init__(self, learning_rate=0.1, discount_factor=0.9,
                 epsilon=0.2, epsilon_decay=0.995, epsilon_min=0.05,
                 time_limit=10):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.time_limit = time_limit

        # Q表: {(state_key, action_key): q_value}
        self.q_table = {}
        self._init_heuristic_q()

        # 持久化路径
        self._qtable_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cache", "q_table.json"
        )

        # 统计
        self.episode_count = 0
        self._dist_cache = {}
        self._cons_cache = {}

    def _init_heuristic_q(self):
        """用启发式知识初始化Q表，加速收敛"""
        # 预置一些合理的初始Q值偏向
        self._heuristic_defaults = {
            # (电量充足, 距离近, 重量轻, 不紧急) → 高分
            ("high", "near", "light", "normal"): 8.0,
            # (电量充足, 距离近, 重量轻, 紧急) → 更高分
            ("high", "near", "light", "urgent"): 10.0,
            # (电量充足, 距离远, 任何重量) → 较低分
            ("high", "far", "heavy", "normal"): 3.0,
            # (电量低, 距离近) → 一般
            ("low", "near", "light", "urgent"): 5.0,
            # (电量低, 距离远) → 低分（倾向于不选）
            ("low", "far", "heavy", "urgent"): 1.0,
            # 默认偏向
            ("medium", "medium", "medium", "normal"): 5.0,
        }
        # 不直接写入Q表，在查询时作为默认偏置使用

    def _get_default_q(self, state_key):
        """根据启发式规则返回默认Q值偏置"""
        batt, dist, weight, urgency = state_key
        # 电量越充足越好
        batt_score = {"high": 3, "medium": 2, "low": 0}
        # 距离越近越好
        dist_score = {"near": 4, "medium": 2, "far": 0}
        # 紧急的任务优先
        urg_score = {"urgent": 3, "normal": 1, "relaxed": 0}

        base = 3.0
        base += batt_score.get(batt, 1)
        base += dist_score.get(dist, 1)
        base += urg_score.get(urgency, 1)
        return base

    def _discretize_battery(self, ratio):
        if ratio > 0.6:
            return "high"
        elif ratio > 0.25:
            return "medium"
        return "low"

    def _discretize_distance(self, dist, max_range):
        if dist is None or dist == float('inf'):
            return "far"
        if dist < max_range * 0.2:
            return "near"
        elif dist < max_range * 0.5:
            return "medium"
        return "far"

    def _discretize_weight(self, weight, max_payload):
        if weight < max_payload * 0.3:
            return "light"
        elif weight < max_payload * 0.7:
            return "medium"
        return "heavy"

    def _discretize_urgency(self, deadline, current_time):
        remaining = deadline - current_time
        if remaining < 30:
            return "urgent"
        elif remaining < 90:
            return "normal"
        return "relaxed"

    def _make_state_key(self, vehicle, task, current_time=0):
        """构建离散化状态键"""
        batt_ratio = vehicle.current_battery / max(vehicle.max_battery, 1)
        batt = self._discretize_battery(batt_ratio)

        max_range = vehicle.max_battery / max(
            vehicle.consumption_rate_dist + vehicle.max_payload * vehicle.consumption_rate_payload,
            0.0001
        )

        dist = self._dist_cache.get((vehicle.current_location, task.location_node))
        if dist is None:
            dist = float('inf')
        d = self._discretize_distance(dist, max_range)

        w = self._discretize_weight(task.weight, vehicle.max_payload)
        u = self._discretize_urgency(task.deadline, current_time)

        return (batt, d, w, u)

    def _get_q(self, state_key, action):
        key = (state_key, action)
        if key not in self.q_table:
            self.q_table[key] = self._get_default_q(state_key)
        return self.q_table[key]

    def _set_q(self, state_key, action, value):
        self.q_table[(state_key, action)] = value

    def _update_q(self, state_key, action, reward, next_state_key, next_actions):
        """Q-Learning更新规则"""
        old_q = self._get_q(state_key, action)
        if next_actions:
            max_next_q = max(self._get_q(next_state_key, a) for a in next_actions)
        else:
            max_next_q = 0
        new_q = old_q + self.lr * (reward + self.gamma * max_next_q - old_q)
        self._set_q(state_key, action, new_q)
        return new_q

    def save_q_table(self):
        """持久化Q表"""
        try:
            os.makedirs(os.path.dirname(self._qtable_path), exist_ok=True)
            # 将tuple key转为字符串
            serializable = {}
            for k, v in self.q_table.items():
                sk = str(k)
                serializable[sk] = v
            with open(self._qtable_path, 'w') as f:
                json.dump(serializable, f)
        except Exception:
            pass

    def load_q_table(self):
        """加载Q表"""
        try:
            if os.path.exists(self._qtable_path):
                with open(self._qtable_path, 'r') as f:
                    serialized = json.load(f)
                for k, v in serialized.items():
                    # 解析字符串key
                    key = eval(k)
                    self.q_table[key] = v
                return True
        except Exception:
            pass
        return False

    def solve_assignment(self, vehicles, pending_tasks, stations, dist_helper, graph):
        if not pending_tasks:
            return {str(v.id): [] for v in vehicles}
        if not vehicles:
            return {}

        start_time = time.time()
        self.episode_count += 1

        # 衰减epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # 构建距离缓存
        self._dist_cache = {}
        self._cons_cache = {}

        def fast_dist(loc1, loc2):
            key = (loc1, loc2)
            if key not in self._dist_cache:
                self._dist_cache[key] = dist_helper.get_distance(loc1, loc2)
            return self._dist_cache[key]

        def get_cons_rate(v):
            if v.id not in self._cons_cache:
                self._cons_cache[v.id] = (
                    v.consumption_rate_dist + v.max_payload * v.consumption_rate_payload
                )
            return self._cons_cache[v.id]

        stations_list = list(stations)
        vehicle_list = list(vehicles)
        task_list = list(pending_tasks)

        # 预计算任务到最近充电站距离
        task_min_station_dist = {}
        for t in task_list:
            min_d = float('inf')
            for s in stations_list:
                d = fast_dist(t.location_node, s.location_node)
                if d is not None and d < min_d:
                    min_d = d
            if min_d == float('inf'):
                d = fast_dist(t.location_node, vehicle_list[0].depot_node)
                min_d = d if d is not None else 50000
            task_min_station_dist[t.id] = min_d

        assignments = {str(v.id): [] for v in vehicle_list}
        assigned_tasks = set()

        # 追踪车辆虚拟状态
        v_battery = {v.id: v.current_battery for v in vehicle_list}
        v_location = {v.id: v.current_location for v in vehicle_list}
        v_at_depot = {v.id: (v.current_location == v.depot_node) for v in vehicle_list}

        task_map = {t.id: t for t in task_list}

        # 迭代构建调度方案
        MAX_ROUNDS = min(len(task_list) * 2, 200)
        current_time = 0

        for _round in range(MAX_ROUNDS):
            if time.time() - start_time > self.time_limit:
                break

            # 收集可用的任务和车辆
            available_tasks = [t for t in task_list if t.id not in assigned_tasks]
            if not available_tasks:
                break

            idle_vehicles = [v for v in vehicle_list
                           if v_battery[v.id] > v.max_battery * 0.05]

            if not idle_vehicles:
                break

            # 为每个(车辆, 任务)对计算Q值
            best_assignment = None
            best_q = float('-inf')

            for v in idle_vehicles:
                for t in available_tasks:
                    if t.weight > v.max_payload:
                        continue

                    dist_to_task = fast_dist(v_location[v.id], t.location_node)
                    if dist_to_task is None or dist_to_task == float('inf'):
                        continue

                    cons_rate = get_cons_rate(v)
                    min_to_station = task_min_station_dist.get(t.id, 50000)

                    # 可行性检查
                    req_battery = (dist_to_task + min_to_station) * cons_rate
                    can_do = v_battery[v.id] >= req_battery

                    if not can_do:
                        # 检查是否能通过充电完成
                        batt_to_task = dist_to_task * cons_rate
                        if v_battery[v.id] < batt_to_task:
                            can_charge = False
                            for s in stations_list:
                                d_to_s = fast_dist(v_location[v.id], s.location_node)
                                if d_to_s is None:
                                    continue
                                if v_battery[v.id] >= d_to_s * cons_rate:
                                    d_s_to_task = fast_dist(s.location_node, t.location_node)
                                    if d_s_to_task is None:
                                        continue
                                    if v.max_battery >= (d_s_to_task + min_to_station) * cons_rate:
                                        can_charge = True
                                        break
                            if not can_charge:
                                continue
                        else:
                            continue

                    # 构建状态键
                    state_key = self._make_state_key(v, t, current_time)

                    # Q值查询（epsilon-greedy的利用部分）
                    q_val = self._get_q(state_key, "assign")

                    # epsilon-greedy 探索
                    if random.random() < self.epsilon:
                        q_val += random.uniform(-2, 2)

                    if q_val > best_q:
                        best_q = q_val
                        best_assignment = (v, t)

            if best_assignment is None:
                break

            # 执行最佳分配
            v, t = best_assignment
            dist_to_task = fast_dist(v_location[v.id], t.location_node)
            cons_rate = get_cons_rate(v)
            min_to_station = task_min_station_dist.get(t.id, 50000)

            # 确定是否需要充电
            req_battery = (dist_to_task + min_to_station) * cons_rate
            plan_items = []

            if v_battery[v.id] < req_battery:
                # 找最佳充电站
                best_sid = None
                best_s_cost = float('inf')
                for s in stations_list:
                    d_to_s = fast_dist(v_location[v.id], s.location_node)
                    d_s_to_task = fast_dist(s.location_node, t.location_node)
                    if d_to_s is None or d_s_to_task is None:
                        continue
                    if v_battery[v.id] >= d_to_s * cons_rate:
                        cost = d_to_s + d_s_to_task
                        if cost < best_s_cost:
                            best_s_cost = cost
                            best_sid = s.id
                if best_sid:
                    plan_items.append(f"CHARGE:{best_sid}")

            plan_items.append(f"TASK:{t.id}")
            assignments[str(v.id)].extend(plan_items)
            assigned_tasks.add(t.id)

            # 更新虚拟状态
            if dist_to_task is not None:
                v_battery[v.id] -= dist_to_task * cons_rate
            v_location[v.id] = t.location_node
            v_at_depot[v.id] = False

            # Q-Learning 更新: 用即时奖励更新Q值
            state_key = self._make_state_key(v, t, current_time)
            reward = 10.0  # 成功分配的正奖励

            # 找下一个可用的任务作为next_state
            remaining = [rt for rt in task_list if rt.id not in assigned_tasks]
            if remaining:
                next_t = remaining[0]
                next_state_key = self._make_state_key(v, next_t, current_time)
                next_actions = ["assign", "skip"]
                self._update_q(state_key, "assign", reward, next_state_key, next_actions)
            else:
                self._update_q(state_key, "assign", reward, None, [])

        # 标记失败任务
        for t in task_list:
            if t.id not in assigned_tasks:
                t.status = "FAILED"

        print(f"  [Q-Learning] Episode {self.episode_count}: "
              f"{len(assigned_tasks)}/{len(task_list)} 任务已分配, "
              f"epsilon={self.epsilon:.3f}, Q表大小={len(self.q_table)}")

        # 定期保存Q表
        if self.episode_count % 10 == 0:
            self.save_q_table()

        return assignments
