# Funkcja do zamiany wierzchołków w obrębie cyklu (obliczanie delty)
def swap_nodes_within_cycle(cycle, distance_matrix):
    n = len(cycle) - 1
    for i in range(1, n):
        for j in range(i + 1, n):
            new_cycle = cycle[:]
            new_cycle[i], new_cycle[j] = new_cycle[j], new_cycle[i]

            delta = (distance_matrix[new_cycle[i - 1]][new_cycle[i]] +
                     distance_matrix[new_cycle[i]][new_cycle[i + 1]] +
                     distance_matrix[new_cycle[j - 1]][new_cycle[j]] +
                     distance_matrix[new_cycle[j]][new_cycle[j + 1]]) - (
                            distance_matrix[cycle[i - 1]][cycle[i]] +
                            distance_matrix[cycle[i]][cycle[i + 1]] +
                            distance_matrix[cycle[j - 1]][cycle[j]] +
                            distance_matrix[cycle[j]][cycle[j + 1]])

            yield new_cycle, delta


# Funkcja do zamiany wierzchołków między cyklami (obliczanie delty)
def swap_nodes_between_cycles(first_cycle, second_cycle, distance_matrix):
    n1, n2 = len(first_cycle) - 1, len(second_cycle) - 1
    for i in range(1, n1):
        for j in range(1, n2):
            new_first = first_cycle[:]
            new_second = second_cycle[:]
            new_first[i], new_second[j] = new_second[j], new_first[i]

            delta = (distance_matrix[new_first[i - 1]][new_first[i]] +
                     distance_matrix[new_first[i]][new_first[i + 1]] +
                     distance_matrix[new_second[j - 1]][new_second[j]] +
                     distance_matrix[new_second[j]][new_second[j + 1]]) - (
                            distance_matrix[first_cycle[i - 1]][first_cycle[i]] +
                            distance_matrix[first_cycle[i]][first_cycle[i + 1]] +
                            distance_matrix[second_cycle[j - 1]][second_cycle[j]] +
                            distance_matrix[second_cycle[j]][second_cycle[j + 1]])

            yield new_first, new_second, delta


# Funkcja do zamiany krawędzi w obrębie cyklu (obliczanie delty)
def swap_edges_within_cycle(cycle, distance_matrix):
    n = len(cycle) - 1
    for i in range(1, n - 1):
        for j in range(i + 2, n):
            new_cycle = cycle[:]
            new_cycle[i:j + 1] = reversed(new_cycle[i:j + 1])

            delta = (distance_matrix[new_cycle[i - 1]][new_cycle[i]] +
                     distance_matrix[new_cycle[j]][new_cycle[j + 1]]) - (
                            distance_matrix[cycle[i - 1]][cycle[i]] +
                            distance_matrix[cycle[j]][cycle[j + 1]])

            yield new_cycle, delta