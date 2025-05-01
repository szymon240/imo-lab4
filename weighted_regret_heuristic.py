import numpy as np
from utils import initialize_cycles, calculate_regret, insert_into_cycle


def weighted_regret_heuristic(distance_matrix: np.ndarray, w1: float = 1.0, w2: float = -1.0) -> tuple[
    list[int], list[int]]:
    cycle1, cycle2, remaining_nodes = initialize_cycles(distance_matrix)

    target_size1 = (len(distance_matrix) + 1) // 2
    target_size2 = len(distance_matrix) // 2

    while remaining_nodes:
        best_candidate = None
        best_cycle = None
        best_weighted_value = -float('inf')

        for candidate in remaining_nodes:
            regret1, increase1, _ = calculate_regret(distance_matrix, cycle1, candidate)
            regret2, increase2, _ = calculate_regret(distance_matrix, cycle2, candidate)

            weighted_value1 = w1 * regret1 + w2 * increase1
            weighted_value2 = w1 * regret2 + w2 * increase2

            if weighted_value1 > best_weighted_value and len(cycle1) < target_size1:
                best_candidate, best_cycle = candidate, cycle1
                best_weighted_value = weighted_value1
            if weighted_value2 > best_weighted_value and len(cycle2) < target_size2:
                best_candidate, best_cycle = candidate, cycle2
                best_weighted_value = weighted_value2

        insert_into_cycle(distance_matrix, best_cycle, best_candidate)
        remaining_nodes.remove(best_candidate)

    return cycle1, cycle2