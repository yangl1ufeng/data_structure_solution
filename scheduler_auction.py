"""
拍卖竞标多智能体调度器 —— 多智能体方法
每辆车作为独立智能体，通过拍卖机制竞标任务。
基于成本计算（距离+电量+时间窗+载重利用率）出价，实现去中心化调度决策。

机制: 组合拍卖 + 冲突解决 + 迭代多轮竞标
"""

import math
import time


class AuctionEVRPScheduler:
    """基于拍卖的多智能体EVRP调度器"""

    def __init__(self, num_rounds=5, time_limit=10,
                 dist_weight=0.5, battery_weight=2.0,
                 urgency_weight=3.0, payload_weight=1.5):
        self.num_rounds = num_rounds
        self.time_limit = time_limit
        self.dist_weight = dist_weight
        self.battery_weight = battery_weight
        self.urgency_weight = urgency_weight
        self.payload_weight = payload_weight

        self._dist_cache = {}
        self._cons_cache = {}

    def solve_assignment(self, vehicles, pending_tasks, stations, dist_helper, graph):
        if not pending_tasks:
            return {str(v.id): [] for v in vehicles}
        if not vehicles:
            return {}

        start_time = time.time()

        # 构建缓存
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
        task_map = {t.id: t for t in task_list}

        # 预计算任务到最近充电站的距离
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

        # 车辆状态追踪
        v_battery = {v.id: v.current_battery for v in vehicle_list}
        v_location = {v.id: v.current_location for v in vehicle_list}
        v_payload_used = {v.id: 0.0 for v in vehicle_list}
        v_at_depot = {v.id: (v.current_location == v.depot_node) for v in vehicle_list}

        assignments = {str(v.id): [] for v in vehicle_list}
        assigned_tasks = set()

        def vehicle_can_reach(v, t):
            """检查车辆是否能到达任务并安全返回"""
            if t.weight > v.max_payload:
                return False, None

            dist_to_task = fast_dist(v_location[v.id], t.location_node)
            if dist_to_task is None or dist_to_task == float('inf'):
                return False, None

            cons_rate = get_cons_rate(v)
            min_to_station = task_min_station_dist.get(t.id, 50000)
            req_direct = (dist_to_task + min_to_station) * cons_rate

            if v_battery[v.id] >= req_direct:
                return True, None  # 无需充电

            # 需要充电的路径
            batt_to_task = dist_to_task * cons_rate
            if v_battery[v.id] < batt_to_task:
                # 找最佳充电中转站
                best_station = None
                best_extra = float('inf')
                for s in stations_list:
                    d_to_s = fast_dist(v_location[v.id], s.location_node)
                    if d_to_s is None:
                        continue
                    if v_battery[v.id] >= d_to_s * cons_rate:
                        d_s_to_task = fast_dist(s.location_node, t.location_node)
                        if d_s_to_task is None:
                            continue
                        batt_from_s = (d_s_to_task + min_to_station) * cons_rate
                        if v.max_battery >= batt_from_s:
                            extra = d_to_s + d_s_to_task
                            if extra < best_extra:
                                best_extra = extra
                                best_station = s.id
                if best_station:
                    return True, best_station
                return False, None
            return False, None

        def compute_bid(v, t, needs_charge_station):
            """计算车辆对任务的出价（成本越低越好）"""
            dist_to_task = fast_dist(v_location[v.id], t.location_node)
            if dist_to_task is None:
                return float('inf')

            cons_rate = get_cons_rate(v)
            max_range = v.max_battery / max(cons_rate, 0.0001)

            # 1. 距离成本
            dist_cost = (dist_to_task / max(max_range, 1)) * self.dist_weight

            # 2. 电池风险成本
            batt_ratio = v_battery[v.id] / max(v.max_battery, 1)
            battery_risk = (1.0 - batt_ratio) * self.battery_weight

            # 3. 时间紧迫度奖励（负成本 = 更愿意竞标）
            remaining_time = t.deadline  # deadline 是相对截止时间
            if remaining_time < 30:
                urgency_bonus = -self.urgency_weight * 2.0
            elif remaining_time < 60:
                urgency_bonus = -self.urgency_weight
            else:
                urgency_bonus = 0.0

            # 4. 载重利用率奖励
            payload_ratio = t.weight / max(v.max_payload, 1)
            payload_bonus = -payload_ratio * self.payload_weight

            # 5. 充电额外成本
            charge_penalty = 0.0
            if needs_charge_station:
                charge_penalty = self.battery_weight * 1.5

            # 6. 如果在仓库，有优势（满电出发）
            depot_bonus = -0.5 if v_at_depot[v.id] else 0.0

            bid = (dist_cost + battery_risk + charge_penalty +
                   urgency_bonus + payload_bonus + depot_bonus)
            return bid

        # ============ 多轮拍卖 ============
        for round_idx in range(self.num_rounds):
            if time.time() - start_time > self.time_limit:
                break

            unassigned = [t for t in task_list if t.id not in assigned_tasks]
            if not unassigned:
                break

            # 每轮拍卖：每辆车对每个未分配任务出价
            # 结构: {task_id: [(bid, vehicle_id, needs_charge_station), ...]}
            auction_bids = {t.id: [] for t in unassigned}

            for v in vehicle_list:
                if v_battery[v.id] < v.max_battery * 0.05:
                    continue  # 电量极低的车辆不参与

                for t in unassigned:
                    can_reach, charge_station = vehicle_can_reach(v, t)
                    if can_reach:
                        bid = compute_bid(v, t, charge_station)
                        if bid != float('inf'):
                            auction_bids[t.id].append((bid, v.id, charge_station))

            # 对每个任务选出最佳出价者
            round_assignments = []  # [(vehicle_id, task_id, charge_station)]
            for tid, bids in auction_bids.items():
                if not bids:
                    continue
                # 按出价排序（低了=更好）
                bids.sort(key=lambda x: x[0])

                # 检查是否有冲突（一辆车被多个任务选中）
                # 使用贪心: 每个任务选最佳竞价车，冲突时取bid更低的任务
                best_bid, best_vid, charge_sid = bids[0]

                # 检查这辆车是否在本轮已被分配
                already_got = [a for a in round_assignments if a[0] == best_vid]
                if not already_got:
                    round_assignments.append((best_vid, tid, charge_sid))
                else:
                    # 冲突解决: 比较bid值，保留更低（更好）的
                    existing = already_got[0]
                    existing_tid = existing[1]
                    existing_bids_for_v = auction_bids.get(existing_tid, [])
                    existing_bid_val = next(
                        (b[0] for b in existing_bids_for_v if b[1] == best_vid),
                        float('inf')
                    )
                    if best_bid < existing_bid_val:
                        # 新的bid更好，替换
                        round_assignments.remove(existing)
                        round_assignments.append((best_vid, tid, charge_sid))

            # 执行本轮分配
            for vid, tid, charge_sid in round_assignments:
                if tid in assigned_tasks:
                    continue

                v = next(v for v in vehicle_list if v.id == vid)
                t = task_map[tid]
                dist_to_task = fast_dist(v_location[vid], t.location_node)
                cons_rate = get_cons_rate(v)

                plan_items = []
                if charge_sid:
                    plan_items.append(f"CHARGE:{charge_sid}")
                plan_items.append(f"TASK:{tid}")

                assignments[str(vid)].extend(plan_items)
                assigned_tasks.add(tid)

                # 更新车辆状态
                if dist_to_task is not None:
                    v_battery[vid] -= dist_to_task * cons_rate
                v_location[vid] = t.location_node
                v_payload_used[vid] += t.weight
                v_at_depot[vid] = False

            if not round_assignments:
                break  # 没有新分配，提前结束

        # 标记失败任务
        for t in task_list:
            if t.id not in assigned_tasks:
                t.status = "FAILED"

        print(f"  [Auction] 竞标完成 ({self.num_rounds}轮): "
              f"{len(assigned_tasks)}/{len(task_list)} 任务已分配")

        return assignments
