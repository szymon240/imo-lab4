import time
from utils import initialize_random_cycles
from local_search import steepest_original


def multiple_start_local_search(distance_matrix, num_starts=200):
    print(f"[MSLS] Starting MSLS with {num_starts} restarts...")
    best_cycles = None
    best_length = float('inf')
    start_time = time.time()

    for i in range(1, num_starts + 1):
        # Generate random initial solution
        cycle1, cycle2, init_time = initialize_random_cycles(distance_matrix)
        print(f"[MSLS] Restart {i}: generated random cycles, running local search...")

        # Execute local search
        (sol1, sol2), length, ls_time = steepest_original(distance_matrix, cycle1, cycle2)
        print(f"[MSLS] Restart {i}: local search complete. Length = {length:.2f}, LS time = {ls_time:.2f}s")

        # Track best
        if length < best_length:
            best_length = length
            best_cycles = (sol1, sol2)
            print(f"[MSLS] New best solution at restart {i}: Length = {best_length:.2f}")

    total_time = time.time() - start_time
    print(f"[MSLS] Finished MSLS. Best length = {best_length:.2f}, Total time = {total_time:.2f}s")
    return best_cycles, best_length, total_time