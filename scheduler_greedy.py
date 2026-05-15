class GreedyEVRPScheduler:
    """改进版贪心调度器：支持仓库充电、电池排序、中途充电接力"""

    def __init__(self, strategy_type="nearest"):
        self.strategy_type = strategy_type

    def solve_assignment(self, vehicles, pending_tasks, stations, dist_helper, graph):
        dist_cache = {}
        def fast_get_dist(loc1, loc2):
            if (loc1, loc2) in dist_cache:
                return dist_cache[(loc1, loc2)]
            dist = dist_helper.get_distance(loc1, loc2)
            dist_cache[(loc1, loc2)] = dist
            return dist

        stations_list = list(stations)
        assignments = {str(v.id): [] for v in vehicles}
        assigned_tasks = set()
        impossible_tasks = set()

        def get_consumption(v):
            return v.consumption_rate_dist + v.max_payload * v.consumption_rate_payload

        # 关键改进 1: 按电量升序排列 — 低电量车优先获得充电计划，防止被高电量车抢走任务后无电可用
        vehicles_sorted = sorted(vehicles, key=lambda v: v.current_battery)

        for v in vehicles_sorted:
            best_plan = None
            best_score_val = float('inf') if self.strategy_type == "nearest" else -1
            rejection_reasons = []
            at_depot = (v.current_location == v.depot_node)

            for task in pending_tasks:
                if task.id in assigned_tasks or task.id in impossible_tasks:
                    continue
                if task.weight > v.max_payload:
                    rejection_reasons.append("T%s(超重:%d>%d)" % (task.id, task.weight, v.max_payload))
                    continue

                cons_rate = get_consumption(v)
                dist_to_task = fast_get_dist(v.current_location, task.location_node)
                dist_to_depot = fast_get_dist(task.location_node, v.depot_node)

                if dist_to_task is None or dist_to_depot is None:
                    rejection_reasons.append("T%s(无路径)" % task.id)
                    continue

                # 任务点到最近充电站的距离
                min_dist_task_to_station = float('inf')
                for s in stations_list:
                    d = fast_get_dist(task.location_node, s.location_node)
                    if d is not None and d < min_dist_task_to_station:
                        min_dist_task_to_station = d
                if min_dist_task_to_station == float('inf'):
                    min_dist_task_to_station = dist_to_depot

                # 检测 1: 任务从根本上不可能？
                best_case_battery = min_dist_task_to_station * cons_rate
                if best_case_battery > v.max_battery:
                    impossible_tasks.add(task.id)
                    print("  [诊断] 任务 %s 超出物理续航上限 (需%.1f>电池%d)，标记不可完成。"
                          % (task.id, best_case_battery, v.max_battery))
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
                        d_to_s = fast_get_dist(v.current_location, s.location_node)
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

                # 检测 4: 关键改进 2 — 仓库充电接力。车辆在仓库时，仓库充电后可满足直达需求
                if at_depot and v.max_battery >= req_battery_direct:
                    plan = ["TASK:%s" % task.id]
                    score = dist_to_task if self.strategy_type == "nearest" else task.weight
                    better = (self.strategy_type == "nearest" and score < best_score_val) or \
                             (self.strategy_type == "largest" and score > best_score_val)
                    if better:
                        best_score_val = score
                        best_plan = plan
                    continue

                rejection_reasons.append("T%s(缺电:需%.1f剩%.1f)" % (task.id, req_battery_direct, v.current_battery))

            # --- 分配最佳计划 ---
            if best_plan:
                assignments[str(v.id)] = best_plan
                for cmd in best_plan:
                    if cmd.startswith("TASK:"):
                        assigned_tasks.add(cmd.split(":", 1)[1])
                print("  [%s] 车辆 %d 分配: %s (电量%.1fkWh)"
                      % (self.strategy_type.upper(), v.id, best_plan, v.current_battery))
            else:
                if pending_tasks and rejection_reasons:
                    print("  [分析] 车辆 %d(电%.1f) 无法接单: %s"
                          % (v.id, v.current_battery, ", ".join(rejection_reasons[:5])))

                has_battery_issue = any("缺电" in r for r in rejection_reasons)
                is_low = v.current_battery < v.max_battery * 0.3

                if has_battery_issue or is_low:
                    if at_depot:
                        # 关键改进 3: 在仓库低电量 → 由仓库充电处理，不派去外部充电站
                        print("  [%s] 车辆 %d 电量不足(%.1fkWh)，在仓库等待充电。"
                              % (self.strategy_type.upper(), v.id, v.current_battery))
                        assignments[str(v.id)] = []
                    else:
                        # 不在仓库 → 找最近可到达的充电站
                        best_station = None
                        min_s_dist = float('inf')
                        for s in stations_list:
                            s_dist = fast_get_dist(v.current_location, s.location_node)
                            if s_dist is not None and s_dist < min_s_dist:
                                req = s_dist * get_consumption(v)
                                if v.current_battery >= req:
                                    min_s_dist = s_dist
                                    best_station = s
                        if best_station:
                            assignments[str(v.id)] = ["CHARGE:%s" % best_station.id]
                            print("  [%s] 车辆 %d 电量不足，分配去充电站 %s"
                                  % (self.strategy_type.upper(), v.id, best_station.id))
                        else:
                            # 无法到达任何充电站 → 尝试回仓库
                            dist_to_depot = fast_get_dist(v.current_location, v.depot_node)
                            if dist_to_depot is not None:
                                batt_to_depot = dist_to_depot * get_consumption(v)
                                if v.current_battery >= batt_to_depot:
                                    assignments[str(v.id)] = []
                                    print("  [%s] 车辆 %d 无法到达充电站，返回仓库等待充电。"
                                          % (self.strategy_type.upper(), v.id))

        # 关键改进 4: 标记物理上不可能完成的任务，防止死循环
        if impossible_tasks:
            for t in pending_tasks:
                if t.id in impossible_tasks and t.status == "PENDING":
                    t.status = "FAILED"
                    print("  [系统] 任务 %s 超出所有车辆续航上限，标记为 FAILED。" % t.id)

        return assignments
