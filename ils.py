import time
import random
import local_search
import swaps as og_swaps


def iterated_local_search(distance_matrix, cycle1, cycle2, max_time, perturbation_size):

    print(f"[ILS] Starting ILS with perturbation size={perturbation_size}, time limit={max_time:.2f}s...")
    # 1) Initial local search na podanych cyklach
    (c1, c2), best_length, ls_time = local_search.steepest_original(distance_matrix, cycle1.copy(), cycle2.copy())
    best_cycles = (c1, c2)
    print(f"[ILS] Initial LS: Length = {best_length:.2f}, time = {ls_time:.2f}s")

    start_time = time.time()
    iter_count = 0
    # 2) Pętla do przekroczenia czasu
    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_time:
            break
        iter_count += 1
        #print(f"[ILS] Iteration {iter_count}: elapsed {elapsed:.2f}s")

        # Kopia
        y1 = best_cycles[0].copy()
        y2 = best_cycles[1].copy()

        # Perturbacja
        t0 = time.time()
        y1, y2 = perturb_solution(y1, y2, distance_matrix, perturbation_size)
        #print(f"[ILS] Perturb done (k={perturbation_size}) in {time.time()-t0:.2f}s")

        # LS
        t1 = time.time()
        (new1, new2), new_length, _ = local_search.steepest_original(distance_matrix, y1, y2)
        #print(f"[ILS] LS on perturbed: Length = {new_length:.2f}, time = {time.time()-t1:.2f}s")

        # Akceptacja
        if new_length < best_length:
            best_length = new_length
            best_cycles = (new1, new2)
            #print(f"[ILS] New best at iter {iter_count}: Length = {best_length:.2f}")
    total = time.time() - start_time
    #print(f"[ILS] Finished ILS after {iter_count} iterations, time = {total:.2f}s, best length = {best_length:.2f}")
    return best_cycles, best_length, total


def perturb_solution(cycle1, cycle2, distance_matrix, k):

    for _ in range(k):
        r = random.random()
        if r < 1/3:
            moves = list(og_swaps.swap_nodes_between_cycles(cycle1, cycle2, distance_matrix))
            cycle1, cycle2, _ = random.choice(moves)
        elif r < 2/3:
            if random.random() < 0.5:
                moves = list(og_swaps.swap_nodes_within_cycle(cycle1, distance_matrix))
                cycle1, _ = random.choice(moves)
            else:
                moves = list(og_swaps.swap_nodes_within_cycle(cycle2, distance_matrix))
                cycle2, _ = random.choice(moves)
        else:
            if random.random() < 0.5:
                moves = list(og_swaps.swap_edges_within_cycle(cycle1, distance_matrix))
                cycle1, _ = random.choice(moves)
            else:
                moves = list(og_swaps.swap_edges_within_cycle(cycle2, distance_matrix))
                cycle2, _ = random.choice(moves)
    return cycle1, cycle2