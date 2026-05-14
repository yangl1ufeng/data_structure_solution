class GreedyEVRPScheduler:
    """基于启发式贪心原则的基线调度器 (适用于对比实验)"""
    def __init__(self, strategy_type="nearest"):
        # strategy_type: "nearest" (最近优先) 或 "largest" (最大载重优先)
        self.strategy_type = strategy_type

    def solve_assignment(self, vehicles, pending_tasks, stations, dist_helper, graph):
        # ---> 新增: 距离查询本地缓存字典，极大幅度提升重复查询速度
        dist_cache = {}
        def fast_get_dist(loc1, loc2):
            if (loc1, loc2) in dist_cache:
                return dist_cache[(loc1, loc2)]
            dist = dist_helper.get_distance(loc1, loc2)
            dist_cache[(loc1, loc2)] = dist
            return dist

        assignments = {str(v.id): [] for v in vehicles}
        assigned_tasks = set()

        for v in vehicles:
            best_task = None
            best_score = float('inf') if self.strategy_type == "nearest" else -1
            
            rejection_reasons = []  # <--- 新增：记录当前车辆拒绝各个任务的原因

            # 遍历寻找最适合当前车辆的任务
            for task in pending_tasks:
                if task.id in assigned_tasks:
                    continue
                if task.weight > v.max_payload:
                    rejection_reasons.append(f"T{task.id}(超重:重{task.weight:.0f}>限{v.max_payload})")
                    continue

                # ---> 使用缓存的 fast_get_dist 替换 dist_helper.get_distance
                dist_to_task = fast_get_dist(v.current_location, task.location_node)
                dist_to_depot = fast_get_dist(task.location_node, v.depot_node)
                if dist_to_task is None or dist_to_depot is None:
                    rejection_reasons.append(f"T{task.id}(无路径寻路失败)")
                    continue

                # --- 智能中转修改：只需要能扛到任务点，再从任务点去往最近的充电站即可 ---
                min_dist_to_station_from_task = float('inf')
                for s in stations:
                    d_s = fast_get_dist(task.location_node, s.location_node)
                    if d_s is not None and d_s < min_dist_to_station_from_task:
                        min_dist_to_station_from_task = d_s
                
                # 如果任务点周围根本没有充电站，只能保守要求它有足够电回仓库
                if min_dist_to_station_from_task == float('inf'):
                    min_dist_to_station_from_task = dist_to_depot

                # 实际所需最低电量 = 去任务点 + 去最近充电站
                consumption_rate = v.consumption_rate_dist + v.max_payload * v.consumption_rate_payload
                req_battery = (dist_to_task + min_dist_to_station_from_task) * consumption_rate
                
                if v.current_battery >= req_battery:
                    if self.strategy_type == "nearest":
                        if dist_to_task < best_score:
                            best_score = dist_to_task
                            best_task = task
                    elif self.strategy_type == "largest":
                        if task.weight > best_score:
                            best_score = task.weight
                            best_task = task
                else:
                    rejection_reasons.append(f"T{task.id}(缺电:需{req_battery:.1f}剩{v.current_battery:.1f})")

            if best_task:
                assignments[str(v.id)].append(f"TASK:{best_task.id}")
                assigned_tasks.add(best_task.id)
                print(f"  [{self.strategy_type.upper()} 贪心策略] 任务 {best_task.id} 分配给车辆 {v.id}")
            else:
                # --- 新增：如果这辆车一个任务都没接，打印它拒绝别的任务的具体原因 ---
                if pending_tasks and rejection_reasons:
                    print(f"  [分析] 车辆 {v.id} 无法接单，原因明细: {', '.join(rejection_reasons)}")
                    
                # 没有任何任务能接，如果是电量不足导致，强制分配最近的充电站进行中转
                # 无论是否低于80%，只要因为缺电没有可用任务，就必须去充电！
                is_failed_due_to_battery = any("缺电" in r for r in rejection_reasons)
                
                if is_failed_due_to_battery or v.current_battery < v.max_battery * 0.3:
                    best_station = None
                    min_s_dist = float('inf')
                    for s in stations:
                        # ---> 使用缓存的 fast_get_dist
                        s_dist = fast_get_dist(v.current_location, s.location_node)
                        if s_dist is not None and s_dist < min_s_dist:
                            req_s_batt = s_dist * v.consumption_rate_dist
                            if v.current_battery >= req_s_batt:
                                min_s_dist = s_dist
                                best_station = s
                    if best_station:
                        assignments[str(v.id)].append(f"CHARGE:{best_station.id}")
                        print(f"  [{self.strategy_type.upper()} 贪心策略] 车辆 {v.id} 电量不足，分配去充电站 {best_station.id}")

        return assignments