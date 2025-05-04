import time
import random
import local_search
from weighted_regret_heuristic import weighted_regret_heuristic
from utils import calculate_regret, cycle_length


def large_neighborhood_search(distance_matrix, cycle1, cycle2, max_time, destroy_ratio):

    print(f"[LNS] Starting LNS with time limit={max_time:.2f}s, destroy ratio={destroy_ratio*100:.0f}%...")
    # 1) Initial optional local search
    (x1, x2), best_length, ls_time = local_search.steepest_original(distance_matrix, cycle1.copy(), cycle2.copy())
    best_cycles = (x1, x2)
    print(f"[LNS] Initial LS: Length = {best_length:.2f}, time = {ls_time:.2f}s")

    start_time = time.time()
    iter_count = 0
    while True:
        elapsed = time.time() - start_time
        if elapsed >= max_time:
            break
        iter_count += 1
        #print(f"[LNS] Iteration {iter_count}: elapsed {elapsed:.2f}s")

        # Copy current best
        y1, y2 = best_cycles[0].copy(), best_cycles[1].copy()

        # Destroy: usuń ~destroy_ratio wierzchołków
        t0 = time.time()
        y1, y2, removed = destroy_solution(y1, y2, destroy_ratio)
        #print(f"[LNS] Destroy: removed {len(removed)} nodes in {time.time()-t0:.2f}s")

        # Repair: wstaw brakujace wierzcholki heurystyką
        t1 = time.time()
        y1, y2 = repair_solution(distance_matrix, y1, y2, removed)
        #print(f"[LNS] Repair: reinserted nodes in {time.time()-t1:.2f}s")

        # Optional: local search on repaired solution
        t2 = time.time()
        (y1, y2), y_length, _ = local_search.steepest_original(distance_matrix, y1, y2)
        #print(f"[LNS] LS on repaired: Length = {y_length:.2f}, time = {time.time()-t2:.2f}s")

        # Acceptance
        if y_length < best_length:
            best_length = y_length
            best_cycles = (y1, y2)
            #print(f"[LNS] New best at iter {iter_count}: Length = {best_length:.2f}")

    total = time.time() - start_time
    #print(f"[LNS] Finished LNS after {iter_count} iterations, time = {total:.2f}s, best length = {best_length:.2f}")
    return best_cycles, best_length, total


def destroy_solution(cycle1, cycle2, ratio):

    all_nodes = cycle1[1:-1] + cycle2[1:-1]
    total = len(all_nodes)
    to_remove = set(random.sample(all_nodes, int(total * ratio)))

    new_c1 = [n for n in cycle1 if n not in to_remove or n == cycle1[0]]
    new_c2 = [n for n in cycle2 if n not in to_remove or n == cycle2[0]]
    removed = list(to_remove)
    # Close cycles
    if new_c1[-1] != new_c1[0]: new_c1.append(new_c1[0])
    if new_c2[-1] != new_c2[0]: new_c2.append(new_c2[0])
    return new_c1, new_c2, removed


def repair_solution(distance_matrix, cycle1, cycle2, removed_nodes):
    """
    Naprawia rozwiązanie po fazie "destroy" za pomocą heurystyki 2-żal.
    """
    # Przywracamy usunięte wierzchołki do cykli w kolejności malejącego żalu
    for node in removed_nodes:
        # Oblicz żal dla wstawienia wierzchołka do cyklu 1
        regret1, increase1, pos1 = calculate_regret(distance_matrix, cycle1, node)
        # Oblicz żal dla wstawienia wierzchołka do cyklu 2
        regret2, increase2, pos2 = calculate_regret(distance_matrix, cycle2, node)

        # Wybierz cykl z większym żalem (lub mniejszym wzrostem długości w przypadku remisu)
        if regret1 > regret2 or (regret1 == regret2 and increase1 < increase2):
            cycle1.insert(pos1, node)
        else:
            cycle2.insert(pos2, node)

    # Zamknij cykle, jeśli nie są zamknięte
    if cycle1[-1] != cycle1[0]:
        cycle1.append(cycle1[0])
    if cycle2[-1] != cycle2[0]:
        cycle2.append(cycle2[0])

    return cycle1, cycle2