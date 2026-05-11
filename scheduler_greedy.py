class GreedyEVRPScheduler:
    """基于启发式贪心原则的基线调度器 (适用于对比实验)"""
    def __init__(self, strategy_type="nearest"):
        # strategy_type: "nearest" (最近优先) 或 "largest" (最大载重优先)
        self.strategy_type = strategy_type

    def solve_assignment(self, vehicles, pending_tasks, stations, dist_helper, graph):
        assignments = {str(v.id): [] for v in vehicles}
        assigned_tasks = set()

        for v in vehicles:
            best_task = None
            best_score = float('inf') if self.strategy_type == "nearest" else -1

            # 遍历寻找最适合当前车辆的任务
            for task in pending_tasks:
                if task.id in assigned_tasks:
                    continue
                if task.weight > v.max_payload:
                    continue

                dist_to_task = dist_helper.get_distance(v.current_location, task.location_node)
                dist_to_depot = dist_helper.get_distance(task.location_node, v.depot_node)
                if dist_to_task is None or dist_to_depot is None:
                    continue

                # 基础电量校验 (前往任务 + 返回仓库)
                req_battery = (dist_to_task + dist_to_depot) * (v.consumption_rate_dist + v.max_payload * v.consumption_rate_payload)
                
                if v.current_battery >= req_battery:
                    if self.strategy_type == "nearest":
                        if dist_to_task < best_score:
                            best_score = dist_to_task
                            best_task = task
                    elif self.strategy_type == "largest":
                        if task.weight > best_score:
                            best_score = task.weight
                            best_task = task

            if best_task:
                assignments[str(v.id)].append(f"TASK:{best_task.id}")
                assigned_tasks.add(best_task.id)
                print(f"  [{self.strategy_type.upper()} 贪心策略] 任务 {best_task.id} 分配给车辆 {v.id}")
            else:
                # 没有任何任务能接，如果是电量不足导致，尝试去最近的充电站
                if v.current_battery < v.max_battery * 0.8:
                    best_station = None
                    min_s_dist = float('inf')
                    for s in stations:
                        s_dist = dist_helper.get_distance(v.current_location, s.location_node)
                        if s_dist is not None and s_dist < min_s_dist:
                            req_s_batt = s_dist * v.consumption_rate_dist
                            if v.current_battery >= req_s_batt:
                                min_s_dist = s_dist
                                best_station = s
                    if best_station:
                        assignments[str(v.id)].append(f"CHARGE:{best_station.id}")
                        print(f"  [{self.strategy_type.upper()} 贪心策略] 车辆 {v.id} 电量不足，分配去充电站 {best_station.id}")

        return assignments