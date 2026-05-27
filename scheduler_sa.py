"""
模拟退火 (Simulated Annealing) 调度器 —— 元启发式方法
使用模拟退火优化车辆路线，包括邻域结构设计（交换/插入/反转/重分配）、
温度调度、Metropolis接受准则。

适用于从贪心初始解出发，通过受控随机扰动逐步优化调度方案。
"""

import random
import math
import copy
import time


class SimulatedAnnealingScheduler:
    """基于模拟退火的EVRP调度器"""

    def __init__(self, initial_temp=1000.0, cooling_rate=0.95,
                 min_temp=0.1, iterations_per_temp=50,
                 time_limit=10):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.iterations_per_temp = iterations_per_temp
        self.time_limit = time_limit

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

        # 预计算
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

        # ===== 生成贪心初始解 =====
        def build_greedy_solution():
            """贪心构建初始解"""
            sol = {str(v.id): [] for v in vehicle_list}
            v_bat = {v.id: v.current_battery for v in vehicle_list}
            v_loc = {v.id: v.current_location for v in vehicle_list}
            assigned = set()

            # 按距离排序的贪心分配
            unassigned = list(task_list)
            # 按 (weight, deadline) 排序: 重的和紧急的优先
            unassigned.sort(key=lambda t: (-t.weight, t.deadline))

            for t in unassigned:
                best_v = None
                best_cost = float('inf')
                best_charge = None

                for v in vehicle_list:
                    if t.weight > v.max_payload:
                        continue

                    dist_to = fast_dist(v_loc[v.id], t.location_node)
                    if dist_to is None:
                        continue

                    cons_rate = get_cons_rate(v)
                    min_to_s = task_min_station_dist.get(t.id, 50000)
                    req_direct = (dist_to + min_to_s) * cons_rate

                    if v_bat[v.id] >= req_direct:
                        cost = dist_to
                        if cost < best_cost:
                            best_cost = cost
                            best_v = v.id
                            best_charge = None
                    else:
                        # 尝试充电中转
                        batt_to = dist_to * cons_rate
                        if v_bat[v.id] >= batt_to:
                            continue
                        for s in stations_list:
                            d_to_s = fast_dist(v_loc[v.id], s.location_node)
                            d_s_to_t = fast_dist(s.location_node, t.location_node)
                            if d_to_s is None or d_s_to_t is None:
                                continue
                            if (v_bat[v.id] >= d_to_s * cons_rate and
                                v.max_battery >= (d_s_to_t + min_to_s) * cons_rate):
                                cost = d_to_s + d_s_to_t
                                if cost < best_cost:
                                    best_cost = cost
                                    best_v = v.id
                                    best_charge = s.id

                if best_v is not None:
                    items = []
                    if best_charge:
                        items.append(f"CHARGE:{best_charge}")
                    items.append(f"TASK:{t.id}")
                    sol[str(best_v)].extend(items)
                    assigned.add(t.id)
                    dist_to = fast_dist(v_loc[best_v], t.location_node)
                    if dist_to is not None:
                        cons_rate = get_cons_rate(
                        next(vv for vv in vehicle_list if vv.id == best_v)
                    )
                    v_bat[best_v] -= dist_to * cons_rate
                    v_loc[best_v] = t.location_node

            return sol, assigned

        # ===== 评估解的质量 =====
        def evaluate(solution):
            """计算解的总成本（越低越好）"""
            total_cost = 0.0
            v_bat = {v.id: v.current_battery for v in vehicle_list}
            v_loc = {v.id: v.current_location for v in vehicle_list}
            completed = 0

            for v in vehicle_list:
                route = solution.get(str(v.id), [])
                prev_loc = v_loc[v.id]
                for item in route:
                    if item.startswith("TASK:"):
                        tid = item.split(":", 1)[1]
                        t = task_map.get(tid)
                        if t is None:
                            total_cost += 100000  # 无效任务严重惩罚
                            continue
                        d = fast_dist(prev_loc, t.location_node)
                        cons_rate = get_cons_rate(v)
                        if d is not None and d != float('inf'):
                            total_cost += d * 0.1  # 距离成本
                            energy_cost = d * cons_rate
                            if v_bat[v.id] < energy_cost:
                                total_cost += 50000  # 电量违规严重惩罚
                            else:
                                v_bat[v.id] -= energy_cost
                            prev_loc = t.location_node
                            completed += 1
                        else:
                            total_cost += 100000  # 不可达惩罚
                    elif item.startswith("CHARGE:"):
                        sid = item.split(":", 1)[1]
                        s = next((s for s in stations_list if s.id == sid), None)
                        if s:
                            d = fast_dist(prev_loc, s.location_node)
                            if d is not None:
                                total_cost += d * 0.1
                                cons_rate = get_cons_rate(v)
                                v_bat[v.id] -= d * cons_rate
                                v_bat[v.id] = v.max_battery  # 充电恢复
                                prev_loc = s.location_node

            # 完成的任务越多，成本越低
            total_cost -= completed * 1000
            return total_cost, completed

        # ===== 邻域操作 =====
        def neighbor_swap_tasks(solution):
            """随机交换两个不同车辆的TASK指令"""
            new_sol = copy.deepcopy(solution)

            # 找有任务的车辆
            vec_with_tasks = [
                vid for vid, route in new_sol.items()
                if any(item.startswith("TASK:") for item in route)
            ]
            if len(vec_with_tasks) < 2:
                return new_sol

            v1, v2 = random.sample(vec_with_tasks, 2)
            route1 = new_sol[v1]
            route2 = new_sol[v2]

            # 找TASK位置
            t1_indices = [i for i, item in enumerate(route1) if item.startswith("TASK:")]
            t2_indices = [i for i, item in enumerate(route2) if item.startswith("TASK:")]

            if not t1_indices or not t2_indices:
                return new_sol

            i1 = random.choice(t1_indices)
            i2 = random.choice(t2_indices)

            route1[i1], route2[i2] = route2[i2], route1[i1]
            return new_sol

        def neighbor_reorder(solution):
            """随机重排某辆车内的任务顺序"""
            new_sol = copy.deepcopy(solution)

            vec_with_tasks = [
                vid for vid, route in new_sol.items()
                if len([item for item in route if item.startswith("TASK:")]) >= 2
            ]
            if not vec_with_tasks:
                return new_sol

            vid = random.choice(vec_with_tasks)
            route = new_sol[vid]
            task_indices = [i for i, item in enumerate(route) if item.startswith("TASK:")]

            if len(task_indices) < 2:
                return new_sol

            # 随机选择一段并反转
            a, b = sorted(random.sample(task_indices, 2))
            segment = route[a:b + 1]
            random.shuffle(segment)
            route[a:b + 1] = segment
            return new_sol

        def neighbor_reassign(solution):
            """将一个任务从一辆车移到另一辆车"""
            new_sol = copy.deepcopy(solution)

            vec_with_tasks = [
                vid for vid, route in new_sol.items()
                if any(item.startswith("TASK:") for item in route)
            ]
            if len(vec_with_tasks) < 1:
                return new_sol

            # 源车辆
            src_vid = random.choice(vec_with_tasks)
            src_route = new_sol[src_vid]
            task_indices = [i for i, item in enumerate(src_route) if item.startswith("TASK:")]

            if not task_indices:
                return new_sol

            ti = random.choice(task_indices)
            task_item = src_route.pop(ti)
            # 同时移除可能的前置CHARGE指令
            if ti > 0 and src_route[ti - 1].startswith("CHARGE:"):
                src_route.pop(ti - 1)

            # 目标车辆（随机选另一辆）
            dst_vid = random.choice([v for v in new_sol.keys() if v != src_vid])
            # 随机插入位置
            dst_route = new_sol[dst_vid]
            if dst_route:
                insert_pos = random.randint(0, len(dst_route))
            else:
                insert_pos = 0
            dst_route.insert(insert_pos, task_item)

            return new_sol

        def neighbor_add_charge(solution):
            """随机在某任务前插入充电站"""
            new_sol = copy.deepcopy(solution)
            if not stations_list:
                return new_sol

            vec_with_tasks = [
                vid for vid, route in new_sol.items()
                if any(item.startswith("TASK:") for item in route)
            ]
            if not vec_with_tasks:
                return new_sol

            vid = random.choice(vec_with_tasks)
            route = new_sol[vid]
            task_indices = [i for i, item in enumerate(route) if item.startswith("TASK:")]

            if not task_indices:
                return new_sol

            ti = random.choice(task_indices)
            s = random.choice(stations_list)
            route.insert(ti, f"CHARGE:{s.id}")
            return new_sol

        def neighbor_remove_charge(solution):
            """随机移除一个充电站"""
            new_sol = copy.deepcopy(solution)
            vec_with_charges = [
                vid for vid, route in new_sol.items()
                if any(item.startswith("CHARGE:") for item in route)
            ]
            if not vec_with_charges:
                return new_sol

            vid = random.choice(vec_with_charges)
            route = new_sol[vid]
            charge_indices = [i for i, item in enumerate(route) if item.startswith("CHARGE:")]
            if charge_indices:
                ci = random.choice(charge_indices)
                route.pop(ci)
            return new_sol

        neighbors = [
            neighbor_swap_tasks,
            neighbor_reorder,
            neighbor_reassign,
            neighbor_add_charge,
            neighbor_remove_charge,
        ]

        # ===== SA主循环 =====
        current_sol, current_assigned = build_greedy_solution()
        current_cost, current_completed = evaluate(current_sol)

        best_sol = copy.deepcopy(current_sol)
        best_cost = current_cost
        best_completed = current_completed

        temp = self.initial_temp
        iteration = 0
        accepted = 0
        improved = 0

        while temp > self.min_temp:
            if time.time() - start_time > self.time_limit:
                break

            for _ in range(self.iterations_per_temp):
                op = random.choice(neighbors)
                new_sol = op(current_sol)
                new_cost, new_completed = evaluate(new_sol)

                delta = new_cost - current_cost

                if delta < 0:
                    # 改进，接受
                    current_sol = new_sol
                    current_cost = new_cost
                    current_completed = new_completed
                    accepted += 1
                    improved += 1

                    if new_cost < best_cost:
                        best_sol = copy.deepcopy(new_sol)
                        best_cost = new_cost
                        best_completed = new_completed
                else:
                    # 以概率接受更差的解
                    if random.random() < math.exp(-delta / temp):
                        current_sol = new_sol
                        current_cost = new_cost
                        current_completed = new_completed
                        accepted += 1

                iteration += 1

            temp *= self.cooling_rate

        # 标记未分配的任务
        assigned_in_best = set()
        for route in best_sol.values():
            for item in route:
                if item.startswith("TASK:"):
                    assigned_in_best.add(item.split(":", 1)[1])

        for t in task_list:
            if t.id not in assigned_in_best:
                t.status = "FAILED"

        print(f"  [SA] 退火完成: 温度={temp:.2f}, 迭代={iteration}, "
              f"接受率={accepted/max(iteration,1)*100:.0f}%, "
              f"完成={best_completed}/{len(task_list)}, "
              f"成本={best_cost:.1f}")

        return best_sol
