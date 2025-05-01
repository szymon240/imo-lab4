import time
import swaps as og_swaps
from utils import cycle_length

def steepest_original(distance_matrix, first_cycle, second_cycle):
    start_time = time.time()
    best_length = cycle_length(first_cycle, distance_matrix) + cycle_length(second_cycle, distance_matrix)

    improved = True
    while improved:
        improved = False
        best_delta = 0
        best_move = None

        # Sprawdzanie zamian między cyklami
        for new_first, new_second, delta in og_swaps.swap_nodes_between_cycles(first_cycle, second_cycle, distance_matrix):
            if delta < best_delta:
                best_delta = delta
                best_move = ('both', new_first, new_second)

        # Sprawdzanie zamian krawędzi w obrębie pierwszego cyklu (2-opt move)
        for new_cycle, delta in og_swaps.swap_edges_within_cycle(first_cycle, distance_matrix):
            if delta < best_delta:
                best_delta = delta
                best_move = ('first', new_cycle)

        # Sprawdzanie zamian krawędzi w obrębie drugiego cyklu (2-opt move)
        for new_cycle, delta in og_swaps.swap_edges_within_cycle(second_cycle, distance_matrix):
            if delta < best_delta:
                best_delta = delta
                best_move = ('second', new_cycle)

        if best_move is not None:
            if best_move[0] == 'both':
                first_cycle, second_cycle = best_move[1], best_move[2]
            elif best_move[0] == 'first':
                first_cycle = best_move[1]
            elif best_move[0] == 'second':
                second_cycle = best_move[1]

            best_length += best_delta
            improved = True

    execution_time = time.time() - start_time
    return (first_cycle, second_cycle), best_length, execution_time