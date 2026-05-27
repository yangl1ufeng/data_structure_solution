"""
遗传算法 (Genetic Algorithm) 调度器 —— 元启发式方法
用于求解带时间窗和电量约束的新能源物流车队EVRP问题。

编码: 任务排列染色体 + 贪心分裂解码器
选择: 锦标赛选择 + 精英保留
交叉: 顺序交叉 (OX)
变异: 交换变异 + 车辆重分配变异
"""

import random
import copy
import math
import time


class GeneticEVRPScheduler:
    """基于遗传算法的EVRP调度器"""

    def __init__(self, population_size=80, generations=100,
                 crossover_rate=0.85, mutation_rate=0.15, elite_size=5,
                 time_limit=10):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.time_limit = time_limit

        # 缓存距离查询
        self._dist_cache = {}
        self._cons_rate_cache = {}

    def solve_assignment(self, vehicles, pending_tasks, stations, dist_helper, graph):
        if not pending_tasks:
            return {str(v.id): [] for v in vehicles}
        if not vehicles:
            return {}

        start_time = time.time()

        # 构建快速距离查询
        self._dist_cache = {}
        self._cons_rate_cache = {}

        def fast_dist(loc1, loc2):
            key = (loc1, loc2)
            if key not in self._dist_cache:
                self._dist_cache[key] = dist_helper.get_distance(loc1, loc2)
            return self._dist_cache[key]

        def get_cons_rate(v):
            if v.id not in self._cons_rate_cache:
                self._cons_rate_cache[v.id] = (
                    v.consumption_rate_dist + v.max_payload * v.consumption_rate_payload
                )
            return self._cons_rate_cache[v.id]

        stations_list = list(stations)
        vehicle_list = list(vehicles)
        task_list = list(pending_tasks)
        n_tasks = len(task_list)
        n_vehicles = len(vehicle_list)

        # 预计算每个任务到最近充电站的距离
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

        # ---------- 解码器: 染色体 → 车辆调度方案 ----------
        def decode(chromosome):
            """
            chromosome: 任务ID的排列列表
            贪心地将每个任务按顺序分配给能完成它的最佳车辆
            返回: {vehicle_id_str: [plan_items]}
            """
            assignments = {str(v.id): [] for v in vehicle_list}
            # 车辆虚拟状态追踪
            v_battery = {v.id: v.current_battery for v in vehicle_list}
            v_location = {v.id: v.current_location for v in vehicle_list}
            v_payload = {v.id: v.max_payload for v in vehicle_list}
            v_at_depot = {v.id: (v.current_location == v.depot_node) for v in vehicle_list}

            assigned = set()
            task_map = {t.id: t for t in task_list}

            for tid in chromosome:
                if tid in assigned:
                    continue
                t = task_map.get(tid)
                if t is None:
                    continue

                # 找到最佳车辆
                best_vid = None
                best_cost = float('inf')
                best_plan_items = None

                for v in vehicle_list:
                    if t.weight > v.max_payload:
                        continue

                    dist_to_task = fast_dist(v_location[v.id], t.location_node)
                    dist_task_to_depot = fast_dist(t.location_node, v.depot_node)
                    if dist_to_task is None or dist_task_to_depot is None:
                        continue

                    cons_rate = get_cons_rate(v)
                    min_to_station = task_min_station_dist.get(t.id, dist_task_to_depot)

                    # 检查电量可行性
                    req_battery_direct = (dist_to_task + min_to_station) * cons_rate
                    needs_charge = False
                    charge_sid = None

                    if v_battery[v.id] >= req_battery_direct:
                        # 可以直接完成
                        pass
                    else:
                        # 需要中途充电
                        batt_to_task = dist_to_task * cons_rate
                        if v_battery[v.id] < batt_to_task:
                            # 找最近可到达的充电站
                            best_station_d = float('inf')
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
                                        total = d_to_s + d_s_to_task
                                        if total < best_station_d:
                                            best_station_d = total
                                            charge_sid = s.id
                            if charge_sid is None:
                                continue
                            needs_charge = True
                        else:
                            continue

                    # 计算成本
                    cost = dist_to_task
                    if needs_charge:
                        cost *= 1.5  # 充电有额外时间成本

                    if cost < best_cost:
                        best_cost = cost
                        best_vid = v.id
                        if needs_charge:
                            best_plan_items = [f"CHARGE:{charge_sid}", f"TASK:{tid}"]
                        else:
                            best_plan_items = [f"TASK:{tid}"]

                if best_vid is not None and best_plan_items:
                    assignments[str(best_vid)].extend(best_plan_items)
                    assigned.add(tid)
                    t_obj = task_map[tid]
                    cons_rate = get_cons_rate(
                        next(v for v in vehicle_list if v.id == best_vid)
                    )
                    # 更新虚拟状态
                    dist_to = fast_dist(v_location[best_vid], t_obj.location_node)
                    if dist_to is None:
                        dist_to = 0
                    v_battery[best_vid] -= dist_to * cons_rate
                    v_location[best_vid] = t_obj.location_node
                    v_payload[best_vid] -= t_obj.weight
                    v_at_depot[best_vid] = False

            return assignments, assigned

        # ---------- 适应度函数 ----------
        def fitness(chromosome):
            assignments, assigned = decode(chromosome)
            # 完成的任务数越多越好
            completed = len(assigned)
            # 也考虑路线总距离（越短越好）
            total_dist = 0
            for v in vehicle_list:
                route = assignments.get(str(v.id), [])
                prev_loc = v.current_location
                for item in route:
                    if item.startswith("TASK:"):
                        tid = item.split(":", 1)[1]
                        t = next((t for t in task_list if t.id == tid), None)
                        if t:
                            d = fast_dist(prev_loc, t.location_node)
                            if d is not None:
                                total_dist += d
                            prev_loc = t.location_node
                    elif item.startswith("CHARGE:"):
                        sid = item.split(":", 1)[1]
                        s = next((s for s in stations_list if s.id == sid), None)
                        if s:
                            d = fast_dist(prev_loc, s.location_node)
                            if d is not None:
                                total_dist += d
                            prev_loc = s.location_node
            # 适应度: 完成数 * 10000 - 距离惩罚
            return completed * 10000 - total_dist * 0.1

        # ---------- 初始种群 ----------
        task_ids = [t.id for t in task_list]

        def create_individual():
            # 贪心启发式初始化: 先按距离排序，再加入随机扰动
            if random.random() < 0.5:
                # 基于贪心的初始化
                ref_v = vehicle_list[0]
                sorted_tasks = sorted(
                    task_list,
                    key=lambda t: fast_dist(ref_v.current_location, t.location_node)
                    if fast_dist(ref_v.current_location, t.location_node) is not None
                    else float('inf')
                )
                chromo = [t.id for t in sorted_tasks]
            else:
                # 随机初始化
                chromo = task_ids[:]
                random.shuffle(chromo)
            return chromo

        population = [create_individual() for _ in range(self.population_size)]

        # ---------- 选择 ----------
        def tournament_select(pop, k=3):
            candidates = random.sample(pop, k)
            return max(candidates, key=fitness)

        # ---------- 交叉 (OX) ----------
        def crossover(parent1, parent2):
            if random.random() > self.crossover_rate:
                return parent1[:]

            size = len(parent1)
            if size < 2:
                return parent1[:]

            a, b = sorted(random.sample(range(size), 2))
            child = [None] * size
            # 保留 parent1 的 [a, b] 段
            child[a:b + 1] = parent1[a:b + 1]
            # 从 parent2 中按顺序填充剩余位置
            p2_idx = 0
            for i in range(size):
                if child[i] is None:
                    while p2_idx < size and parent2[p2_idx] in child:
                        p2_idx += 1
                    if p2_idx < size:
                        child[i] = parent2[p2_idx]
                        p2_idx += 1
            # 填充任何遗留的 None
            used = set(x for x in child if x is not None)
            missing = [x for x in parent1 if x not in used]
            for i in range(size):
                if child[i] is None:
                    child[i] = missing.pop(0) if missing else parent1[i]
            return child

        # ---------- 变异 ----------
        def mutate(chromosome):
            if random.random() > self.mutation_rate:
                return chromosome

            mutated = chromosome[:]
            size = len(mutated)
            if size < 2:
                return mutated

            op = random.choice(['swap', 'invert', 'scramble'])
            if op == 'swap':
                i, j = random.sample(range(size), 2)
                mutated[i], mutated[j] = mutated[j], mutated[i]
            elif op == 'invert':
                i, j = sorted(random.sample(range(size), 2))
                mutated[i:j + 1] = reversed(mutated[i:j + 1])
            elif op == 'scramble':
                i, j = sorted(random.sample(range(size), 2))
                segment = mutated[i:j + 1]
                random.shuffle(segment)
                mutated[i:j + 1] = segment

            return mutated

        # ---------- GA 主循环 ----------
        best_chromosome = None
        best_fitness_val = float('-inf')
        stagnation = 0

        for gen in range(self.generations):
            # 检查时间限制
            if time.time() - start_time > self.time_limit:
                break

            # 评估种群
            pop_fitness = [(ind, fitness(ind)) for ind in population]
            pop_fitness.sort(key=lambda x: x[1], reverse=True)

            current_best = pop_fitness[0]
            if current_best[1] > best_fitness_val:
                best_fitness_val = current_best[1]
                best_chromosome = current_best[0][:]
                stagnation = 0
            else:
                stagnation += 1

            # 早停: 连续20代无改进
            if stagnation > 20:
                break

            # 构建下一代
            new_population = []

            # 精英保留
            for i in range(min(self.elite_size, len(pop_fitness))):
                new_population.append(pop_fitness[i][0][:])

            # 填充剩余
            while len(new_population) < self.population_size:
                p1 = tournament_select(population)
                p2 = tournament_select(population)
                child = crossover(p1, p2)
                child = mutate(child)
                new_population.append(child)

            population = new_population

        # ---------- 最终解码 ----------
        if best_chromosome is None:
            best_chromosome = population[0]

        assignments, assigned = decode(best_chromosome)

        # 标记未分配任务为失败
        for t in task_list:
            if t.id not in assigned:
                t.status = "FAILED"

        print(f"  [GA] 进化完成: {len(assigned)}/{n_tasks} 任务已分配, "
              f"适应度={best_fitness_val:.0f}")

        return assignments
