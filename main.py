import utils
import local_search
from utils import load_from_tsp
from msls import multiple_start_local_search
from ils import iterated_local_search

if __name__ == "__main__":
    kroa200_matrix, kroa200_coords = load_from_tsp('datasets/kroA200.tsp')
    krob200_matrix, krob200_coords = load_from_tsp('datasets/kroB200.tsp')

    kroa200_cycle1_random, kroa200_cycle2_random, _ = utils.initialize_random_cycles(kroa200_matrix)
    krob200_cycle1_random, krob200_cycle2_random, _ = utils.initialize_random_cycles(krob200_matrix)

    def msls_wrapper(matrix, cycle1, cycle2):
        best_cycles, best_length, total_time = multiple_start_local_search(matrix, num_starts=5)
        return best_cycles, best_length, total_time


    def ils_wrapper(matrix, c1, c2):
        return iterated_local_search(matrix, c1, c2, max_time=40, perturbation_size=3)

    utils.run_test_lab2(
        "kroA: ILS",
        kroa200_matrix,
        kroa200_coords,
        kroa200_cycle1_random,
        kroa200_cycle2_random,
        ils_wrapper
    )

    utils.run_test_lab2(
        "kroB: ILS",
        krob200_matrix,
        krob200_coords,
        krob200_cycle1_random,
        krob200_cycle2_random,
        ils_wrapper
    )

    # 1) MSLS: Multiple Start Local Search (10 uruchomień, każdorazowo 200 startów LS)
    utils.run_test_lab2(
        "kroA: MSLS (200 starts)",
        kroa200_matrix,
        kroa200_coords,
        kroa200_cycle1_random,
        kroa200_cycle2_random,
        msls_wrapper
    )
    utils.run_test_lab2(
        "kroB: MSLS (200 starts)",
        krob200_matrix,
        krob200_coords,
        krob200_cycle1_random,
        krob200_cycle2_random,
        msls_wrapper
    )

    # 2) Steepest descent (oryginalne przeszukiwanie lokalne)
    utils.run_test_lab2(
        "kroA: Steepest search (original)",
        kroa200_matrix,
        kroa200_coords,
        kroa200_cycle1_random,
        kroa200_cycle2_random,
        local_search.steepest_original
    )
    utils.run_test_lab2(
        "kroB: Steepest search (original)",
        krob200_matrix,
        krob200_coords,
        krob200_cycle1_random,
        krob200_cycle2_random,
        local_search.steepest_original
    )