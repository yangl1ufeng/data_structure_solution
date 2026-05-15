class GreedyEVRPScheduler:
    """改进版贪心调度器：支持仓库充电、电池排序、中途充电接力"""

    def __init__(self, strategy_type="nearest"):
        self.strategy_type = strategy_type

    def solve_assignment(self, vehicles, pending_tasks, stations, dist_helper, graph):
        dist_cache = {}
        def fast_get_dist(loc1, loc2):
            key = (loc1, loc2)
            d = dist_cache.get(key)
            if d is not None:
                return d
            d = dist_helper.get_distance(loc1, loc2)
            dist_cache[key] = d
            return d

        stations_list = list(stations)
        assignments = {str(v.id): [] for v in vehicles}
        assigned_tasks = set()
        impossible_tasks = set()

        def get_consumption(v):
            return v.consumption_rate_dist + v.max_payload * v.consumption_rate_payload

        # === 预计算 1: 每个任务到最近充电站的距离（只算一次） ===
        task_to_min_station_dist = {}
        for task in pending_tasks:
            min_d = float('inf')
            for s in stations_list:
                d = fast_get_dist(task.location_node, s.location_node)
                if d is not None and d < min_d:
                    min_d = d
            if min_d == float('inf'):
                d = fast_get_dist(task.location_node, vehicles[0].depot_node) if vehicles else None
                if d is not None:
                    min_d = d
            task_to_min_station_dist[task.id] = min_d

        # === 预计算 2: 每个车辆到所有充电站的距离 ===
        vehicle_station_dists = {}
        for v in vehicles:
            vehicle_station_dists[v.id] = {}
            for s in stations_list:
                d = fast_get_dist(v.current_location, s.location_node)
                vehicle_station_dists[v.id][s.id] = d

        # 关键改进 1: 按电量升序排列
        vehicles_sorted = sorted(vehicles, key=lambda v: v.current_battery)

        for v in vehicles_sorted:
            cons_rate = get_consumption(v)
            best_plan = None
            best_score_val = float('inf') if self.strategy_type == "nearest" else -1
            at_depot = (v.current_location == v.depot_node)
            v_station_dists = vehicle_station_dists[v.id]

            for task in pending_tasks:
                if task.id in assigned_tasks or task.id in impossible_tasks:
                    continue
                if task.weight > v.max_payload:
                    continue

                dist_to_task = fast_get_dist(v.current_location, task.location_node)
                dist_to_depot = fast_get_dist(task.location_node, v.depot_node)
                if dist_to_task is None or dist_to_depot is None:
                    continue

                min_dist_task_to_station = task_to_min_station_dist.get(task.id, dist_to_depot)
                if min_dist_task_to_station == float('inf'):
                    min_dist_task_to_station = dist_to_depot

                # 检测 1: 任务从根本上不可能？
                best_case_battery = min_dist_task_to_station * cons_rate
                if best_case_battery > v.max_battery:
                    impossible_tasks.add(task.id)
                    continue

                req_battery_direct = (dist_to_task + min_dist_task_to_station) * cons_rate

                # 检测 2: 当前电量能否直达？
                if v.current_battery >= req_battery_direct:
                    score = dist_to_task if self.strategy_type == "nearest" else task.weight
                    better = (self.strategy_type == "nearest" and score < best_score_val) or \
                             (self.strategy_type == "largest" and score > best_score_val)
                    if better:
                        best_score_val = score
                        best_plan = ["TASK:%s" % task.id]
                    continue

                # 检测 3: 中途充电接力（外部充电站）?
                batt_to_task = dist_to_task * cons_rate
                if v.current_battery < batt_to_task:
                    best_station = None
                    best_total = float('inf')
                    for s in stations_list:
                        d_to_s = v_station_dists.get(s.id)
                        d_s_to_task = fast_get_dist(s.location_node, task.location_node)
                        if d_to_s is None or d_s_to_task is None:
                            continue
                        batt_to_s = d_to_s * cons_rate
                        batt_from_s = (d_s_to_task + min_dist_task_to_station) * cons_rate
                        if v.current_battery >= batt_to_s and v.max_battery >= batt_from_s:
                            total = d_to_s + d_s_to_task
                            if total < best_total:
                                best_total = total
                                best_station = s

                    if best_station:
                        plan = ["CHARGE:%s" % best_station.id, "TASK:%s" % task.id]
                        score = best_total if self.strategy_type == "nearest" else task.weight
                        better = (self.strategy_type == "nearest" and score < best_score_val) or \
                                 (self.strategy_type == "largest" and score > best_score_val)
                        if better:
                            best_score_val = score
                            best_plan = plan
                        continue

                # 检测 4: 仓库充电接力
                if at_depot and v.max_battery >= req_battery_direct:
                    plan = ["TASK:%s" % task.id]
                    score = dist_to_task if self.strategy_type == "nearest" else task.weight
                    better = (self.strategy_type == "nearest" and score < best_score_val) or \
                             (self.strategy_type == "largest" and score > best_score_val)
                    if better:
                        best_score_val = score
                        best_plan = plan

            # --- 分配最佳计划 ---
            if best_plan:
                assignments[str(v.id)] = best_plan
                for cmd in best_plan:
                    if cmd.startswith("TASK:"):
                        assigned_tasks.add(cmd.split(":", 1)[1])
            else:
                # 无法分配任务 → 找最近充电站或返回仓库
                has_pending = any(
                    t.id not in assigned_tasks and t.id not in impossible_tasks
                    for t in pending_tasks
                )
                if not has_pending:
                    continue

                is_low = v.current_battery < v.max_battery * 0.3
                if is_low and not at_depot:
                    best_station = None
                    min_s_dist = float('inf')
                    for s in stations_list:
                        s_dist = v_station_dists.get(s.id)
                        if s_dist is not None and s_dist < min_s_dist:
                            req = s_dist * cons_rate
                            if v.current_battery >= req:
                                min_s_dist = s_dist
                                best_station = s
                    if best_station:
                        assignments[str(v.id)] = ["CHARGE:%s" % best_station.id]
                    else:
                        dist_to_depot = fast_get_dist(v.current_location, v.depot_node)
                        if dist_to_depot is not None:
                            batt_to_depot = dist_to_depot * cons_rate
                            if v.current_battery >= batt_to_depot:
                                assignments[str(v.id)] = []
                elif is_low and at_depot:
                    assignments[str(v.id)] = []

        # 标记物理上不可能完成的任务
        if impossible_tasks:
            for t in pending_tasks:
                if t.id in impossible_tasks and t.status == "PENDING":
                    t.status = "FAILED"

        return assignments
