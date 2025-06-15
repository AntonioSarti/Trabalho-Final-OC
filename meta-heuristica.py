
# -------------------------------------------- #
# INF05010 - Otimização Combinatória - 2025/1
# Trabalho Final - Etapa 1 - Implementação
# Refatorada para ler uma única instância, 
# otimizada para avaliação diferencial
# -------------------------------------------- #

import random
import time
import sys
import copy

# --- Funções Auxiliares ---

def read_instance(file_path):
    """Lê o arquivo de entrada da instância.
    Retorna:
    n_criminals: número de criminosos
    alliances: conjunto de pares de alianças
    adj: lista de adjacência para cada criminoso
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
        n_criminals, n_alliances = map(int, lines[0].strip().split())
        alliances = set()
        adj = [[] for _ in range(n_criminals)]
        for i in range(1, n_alliances + 1):
            c1, c2 = map(int, lines[i].strip().split())
            c1 -= 1
            c2 -= 1
            alliances.add(tuple(sorted((c1, c2))))
            adj[c1].append(c2)
            adj[c2].append(c1)
    return n_criminals, alliances, adj

def calculate_objective(solution):
    """Retorna o número de penitenciárias usadas na solução."""

    if not solution or -1 in solution:
        return float('inf')
    return max(solution) + 1

def is_feasible(solution, alliances, n_criminals):
    """Verifica se uma solução é viável: nenhum par de aliados está na mesma penitenciária."""

    if not solution or -1 in solution:
        return False
    num_pens = calculate_objective(solution)
    penitentiary_map = [set() for _ in range(num_pens)]
    for criminal, pen in enumerate(solution):
        if pen < 0 or pen >= num_pens:
            return False
        penitentiary_map[pen].add(criminal)
    for c1, c2 in alliances:
        if solution[c1] == solution[c2]:
            return False
    return True

def get_conflicts(solution, alliances):
    """Conta o número de conflitos (pares aliados na mesma penitenciária)."""

    conflicts = 0
    if not solution or -1 in solution:
        return float('inf')
    for c1, c2 in alliances:
        if solution[c1] == solution[c2]:
            conflicts += 1
    return conflicts

# --- Componentes do ILS ---

def generate_initial_solution(n_criminals, alliances, adj, random_seed):
    """Gera uma solução inicial gulosa baseada no grau dos vértices."""

    random.seed(random_seed)
    solution = [-1] * n_criminals
    criminals_ordered = list(range(n_criminals))
    criminals_ordered.sort(key=lambda c: len(adj[c]), reverse=True)
    num_pens = 0
    for criminal in criminals_ordered:
        possible_pens = list(range(num_pens))
        random.shuffle(possible_pens)
        assigned = False
        for pen in possible_pens:
            if all(solution[neighbor] != pen for neighbor in adj[criminal] if solution[neighbor] != -1):
                solution[criminal] = pen
                assigned = True
                break
        if not assigned:
            solution[criminal] = num_pens
            num_pens += 1
    return solution

def delta_conflicts(solution, alliances, criminal, new_pen, adj):
    """Calcula a variação de conflitos ao mover um criminoso."""

    delta = 0
    for neighbor in adj[criminal]:
        if solution[neighbor] == solution[criminal]:
            delta -= 1
        if solution[neighbor] == new_pen:
            delta += 1
    return delta

def local_search(current_solution, alliances, adj, n_criminals, max_iterations=1000):
    """Executa busca local com avaliação diferencial para melhorar a solução."""

    best_solution = copy.deepcopy(current_solution)
    best_conflicts = get_conflicts(best_solution, alliances)
    num_pens = calculate_objective(best_solution)

    for _ in range(max_iterations):
        improved = False
        criminals_to_try = list(range(n_criminals))
        random.shuffle(criminals_to_try)
        for criminal in criminals_to_try:
            original_pen = best_solution[criminal]
            possible_moves = list(range(num_pens)) + [num_pens]
            random.shuffle(possible_moves)
            for target_pen in possible_moves:
                if target_pen == original_pen:
                    continue
                delta = delta_conflicts(best_solution, alliances, criminal, target_pen, adj)
                new_conflicts = best_conflicts + delta
                temp_num_pens = max(num_pens, target_pen + 1)
                if delta < 0 or (delta == 0 and temp_num_pens <= num_pens):
                    best_solution[criminal] = target_pen
                    best_conflicts = new_conflicts
                    num_pens = temp_num_pens
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    pen_map = {}
    next_pen_id = 0
    cleaned_solution = [-1] * n_criminals
    for i in range(n_criminals):
        old_pen = best_solution[i]
        if old_pen not in pen_map:
            pen_map[old_pen] = next_pen_id
            next_pen_id += 1
        cleaned_solution[i] = pen_map[old_pen]

    return cleaned_solution if is_feasible(cleaned_solution, alliances, n_criminals) else current_solution

def perturb_solution(solution, alliances, adj, n_criminals, perturbation_strength, random_seed):
    """Perturba a solução movendo aleatoriamente alguns criminosos."""

    random.seed(random_seed)
    perturbed_solution = copy.deepcopy(solution)
    num_pens = calculate_objective(perturbed_solution)
    criminals_to_move = random.sample(range(n_criminals), min(perturbation_strength, n_criminals))

    for criminal in criminals_to_move:
        original_pen = perturbed_solution[criminal]
        possible_pens = list(range(num_pens + 1))
        if original_pen in possible_pens:
            possible_pens.remove(original_pen)
        random.shuffle(possible_pens)
        for target_pen in possible_pens:
            if all(perturbed_solution[neighbor] != target_pen for neighbor in adj[criminal]):
                perturbed_solution[criminal] = target_pen
                num_pens = max(num_pens, target_pen + 1)
                break

    pen_map = {}
    next_pen_id = 0
    final_perturbed_solution = [-1] * n_criminals
    for i in range(n_criminals):
        old_pen = perturbed_solution[i]
        if old_pen not in pen_map:
            pen_map[old_pen] = next_pen_id
            next_pen_id += 1
        final_perturbed_solution[i] = pen_map[old_pen]
    return final_perturbed_solution

def acceptance_criterion(current_objective, new_objective, best_objective_so_far):
    """Define o critério de aceitação da nova solução no ILS."""

    return new_objective < best_objective_so_far

def iterated_local_search(file_path, max_ils_iterations, perturbation_strength, random_seed):
    """Implementação completa do ILS: gera solução, aplica perturbação, busca local e aceita melhorias."""

    start_time = time.time()
    n_criminals, alliances, adj = read_instance(file_path)
    current_seed = random_seed
    s0 = generate_initial_solution(n_criminals, alliances, adj, current_seed)
    s_best = local_search(s0, alliances, adj, n_criminals)
    s_best_obj = calculate_objective(s_best)
    s_current = copy.deepcopy(s_best)

    for i in range(max_ils_iterations):
        current_seed += 1
        s_perturbed = perturb_solution(s_current, alliances, adj, n_criminals, perturbation_strength, current_seed)
        s_candidate = local_search(s_perturbed, alliances, adj, n_criminals)
        s_candidate_obj = calculate_objective(s_candidate)
        if is_feasible(s_candidate, alliances, n_criminals) and acceptance_criterion(calculate_objective(s_current), s_candidate_obj, s_best_obj):
            s_best = copy.deepcopy(s_candidate)
            s_best_obj = s_candidate_obj
            s_current = copy.deepcopy(s_candidate)
        else:
            s_current = copy.deepcopy(s_candidate)

    end_time = time.time()
    print(f"\nILS finalizado após {i+1} iterações.")
    print(f"Tempo total: {end_time - start_time:.2f} segundos")
    print(f"Melhor solução encontrada: {s_best}")
    print(f"Melhor valor objetivo (penitenciárias): {s_best_obj}")
    print(f"É viável: {is_feasible(s_best, alliances, n_criminals)}")
    return s_best, s_best_obj

# --- Execução do Script ---
if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Uso: python3 meta-heuristica.py <max_iteracoes_ils> <semente_aleatoria> <forca_perturbacao> <instancia.txt>")
        sys.exit(1)

    max_iterations_param = int(sys.argv[1])
    random_seed_param = int(sys.argv[2])
    perturbation_strength_param = int(sys.argv[3])
    instance_file = sys.argv[4]

    print(f"Executando ILS para Separação de Comparsas")
    print(f"Instância: {instance_file}")
    print(f"Máximo de Iterações ILS: {max_iterations_param}")
    print(f"Semente Aleatória Inicial: {random_seed_param}")
    print(f"Força da Perturbação: {perturbation_strength_param}")
    print("-" * 30)

    best_solution_found, best_objective_found = iterated_local_search(
        instance_file,
        max_iterations_param,
        perturbation_strength_param,
        random_seed_param
    )

    print("-" * 30 + "\n\n")
