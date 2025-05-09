# -------------------------------------------- #
# INF05010 - Otimização Combinatória - 2025/1
#  Trabalho Final - Etapa 1 - Implementação
#  Versão não refatorada e muuuuito verbosa!
# -------------------------------------------- #

# --- Bibliotecas ---

import random
import time
import sys
import copy
from glob import glob
# --- Funções Auxiliares ---

def read_instance(file_path):
    """Lê o arquivo de instância do problema."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
        n_criminals, n_alliances = map(int, lines[0].strip().split())
        alliances = set()
        adj = [[] for _ in range(n_criminals)]
        for i in range(1, n_alliances + 1):
            c1, c2 = map(int, lines[i].strip().split())
            # Ajusta para índice 0
            c1 -= 1
            c2 -= 1
            alliances.add(tuple(sorted((c1, c2)))) # Garante ordem (c1, c2) e evita duplicatas
            adj[c1].append(c2) # Lista de adjacência para acesso rápido aos vizinhos
            adj[c2].append(c1)
    return n_criminals, alliances, adj

def calculate_objective(solution):
    """Calcula o número de penitenciárias usadas (valor da função objetivo)."""
    if not solution or -1 in solution: # Se a solução não está completa ou vazia
        return float('inf') # Retorna infinito se a solução for inválida/incompleta
    # +1 porque as penitenciárias são 0-indexadas
    return max(solution) + 1

def is_feasible(solution, alliances, n_criminals):
    """Verifica se uma solução é viável (sem comparsas na mesma penitenciária)."""
    if not solution or -1 in solution: # Uma solução incompleta não é viável
        return False
    num_pens = calculate_objective(solution)
    # Cria um mapa para verificar rapidamente quem está em cada penitenciária
    penitentiary_map = [set() for _ in range(num_pens)]
    for criminal, pen in enumerate(solution):
        # Verifica se o índice da penitenciária é válido
        if pen < 0 or pen >= num_pens:
            # Isso pode acontecer se calculate_objective retornou infinito 
            return False
        penitentiary_map[pen].add(criminal)

    # Verifica cada aliança
    for c1, c2 in alliances:
        # Garante que ambos os criminosos estão alocados antes de checar
        if solution[c1] == solution[c2]:
            # Encontrou uma aliança na mesma penitenciária
            return False
    # Nenhuma aliança violada
    return True

def get_conflicts(solution, alliances):
    """Conta o número de conflitos (alianças na mesma penitenciária)."""
    conflicts = 0
    if not solution or -1 in solution: # Não calcula conflitos para soluções incompletas
        return float('inf')
    for c1, c2 in alliances:
        # Garante que ambos os criminosos estão alocados antes de checar
        if solution[c1] == solution[c2]:
            conflicts += 1
    return conflicts

# --- Componentes do ILS (Iterated Local Search) ---

def generate_initial_solution(n_criminals, alliances, adj, random_seed):
    """Gera uma solução inicial gulosa e aleatorizada."""
    random.seed(random_seed) # Usa a semente para reprodutibilidade
    solution = [-1] * n_criminals # -1 significa não alocado
    criminals_ordered = list(range(n_criminals))
    # Ordena por grau (mais conexões primeiro) para tentar alocar os mais restritos
    # Isso é uma heurística comum em coloração de grafos
    criminals_ordered.sort(key=lambda c: len(adj[c]), reverse=True)

    num_pens = 0 # Contador de penitenciárias usadas
    for criminal in criminals_ordered:
        # Lista de penitenciárias existentes para tentar alocar
        possible_pens = list(range(num_pens))
        random.shuffle(possible_pens) # Aleatoriza a ordem de tentativa

        assigned = False
        # Tenta alocar em uma penitenciária existente
        for pen in possible_pens:
            # Verifica se algum vizinho (comparsa) já está nesta penitenciária
            can_assign = True
            for neighbor in adj[criminal]:
                # Se o vizinho já foi alocado e está na penitenciária 'pen'
                if solution[neighbor] != -1 and solution[neighbor] == pen:
                    can_assign = False
                    break # Não pode usar esta penitenciária
            if can_assign:
                solution[criminal] = pen
                assigned = True
                break # Alocado com sucesso

        # Se não pode alocar em nenhuma penitenciária existente, cria uma nova
        if not assigned:
            solution[criminal] = num_pens # Aloca na nova penitenciária
            num_pens += 1 # Incrementa o contador de penitenciárias

    # Verificação final de viabilidade (embora a construção deva garantir)
    if not is_feasible(solution, alliances, n_criminals):
        print("Aviso: A solução inicial gerada é inviável.", file=sys.stderr)
        # Fallback (Plano B): alocar cada um em uma nova penitenciária (pior caso, mas viável)
        # solution = list(range(n_criminals)) # Descomente se precisar de um fallback robusto

    return solution

def local_search(current_solution, alliances, adj, n_criminals, max_iterations=1000):
    """
    Busca Local (Hill Climbing com First Improvement - Primeira Melhora):
    Tenta mover um criminoso para outra penitenciária (existente ou nova)
    se isso reduzir o número de penitenciárias ou mantiver o número e for viável.
    Prioriza movimentos que reduzem conflitos se a solução for inviável.
    """
    best_solution = copy.deepcopy(current_solution)
    best_objective = calculate_objective(best_solution)
    best_conflicts = get_conflicts(best_solution, alliances)

    num_pens = calculate_objective(best_solution)
    if num_pens == float('inf'): # Se a solução inicial é inválida
        print("Aviso: Busca local iniciada com solução inválida.", file=sys.stderr)
        # Tenta estimar um número de penitenciárias razoável ou retorna erro
        num_pens = n_criminals # Estimativa pessimista


    for _ in range(max_iterations): # Limita iterações para evitar loops infinitos
        improved = False # Flag para indicar se houve melhora nesta iteração
        criminals_to_try = list(range(n_criminals))
        random.shuffle(criminals_to_try) # Tenta mover criminosos em ordem aleatória

        for criminal in criminals_to_try:
            original_pen = best_solution[criminal]
            if original_pen == -1: continue # Pula criminosos não alocados (se houver)

            # Lista de penitenciárias para as quais tentar mover o criminoso
            possible_moves = list(range(num_pens))
            random.shuffle(possible_moves)
            # Considera criar uma nova penitenciária também
            if num_pens not in possible_moves:
                possible_moves.append(num_pens)

            for target_pen in possible_moves:
                # Não tenta mover para a mesma penitenciária
                if target_pen == original_pen:
                    continue

                # Cria uma cópia temporária para testar o movimento
                temp_solution = copy.deepcopy(best_solution)
                temp_solution[criminal] = target_pen

                # Verifica a viabilidade *local* do movimento (apenas para o criminoso movido)
                move_feasible_locally = True
                for neighbor in adj[criminal]:
                    # Se um vizinho está na penitenciária de destino
                    if temp_solution[neighbor] == target_pen:
                        move_feasible_locally = False
                        break

                if move_feasible_locally:
                    # Recalcula objetivo e conflitos para a nova solução *completa*
                    # Otimização: Poderia calcular delta, mas recalcular é mais simples aqui
                    current_objective = calculate_objective(temp_solution)
                    current_conflicts = get_conflicts(temp_solution, alliances)

                    # Critérios de Melhoria:
                    # 1. Menos conflitos é sempre melhor.
                    # 2. Se os conflitos são os mesmos, menos penitenciárias é melhor.
                    if current_conflicts < best_conflicts or \
                        (current_conflicts == best_conflicts and current_objective < best_objective):

                        best_solution = temp_solution # Aceita a melhora
                        best_objective = current_objective
                        best_conflicts = current_conflicts
                        # Atualiza o número de penitenciárias se mudou
                        num_pens = calculate_objective(best_solution)
                        improved = True
                        # Estratégia First Improvement: para de procurar movimentos para este criminoso
                        break

            if improved:
                # Estratégia First Improvement: para de procurar outros criminosos e reinicia a busca
                break

        # Se nenhuma melhora foi encontrada após tentar todos os criminosos
        if not improved:
            break # Ótimo local alcançado

    # Limpeza: Reenumera as penitenciárias de forma contígua (0, 1, 2...) se possível
    pen_map = {} # Mapeia IDs antigos para novos
    next_pen_id = 0
    cleaned_solution = [-1] * n_criminals
    for i in range(n_criminals):
        old_pen = best_solution[i]
        if old_pen == -1: continue # Ignora não alocados
        if old_pen not in pen_map:
            pen_map[old_pen] = next_pen_id
            next_pen_id += 1
        cleaned_solution[i] = pen_map[old_pen]

    # Dupla checagem de viabilidade após a limpeza
    if not is_feasible(cleaned_solution, alliances, n_criminals):
        print("Aviso: Busca local produziu solução inviável após limpeza.", file=sys.stderr)
        # Retorna o último estado bom conhecido se a limpeza falhar
        return current_solution

    # Retorna a solução localmente ótima encontrada
    return cleaned_solution


def perturb_solution(solution, alliances, adj, n_criminals, perturbation_strength, random_seed):
    """Perturba a solução movendo 'perturbation_strength' criminosos aleatoriamente."""
    random.seed(random_seed) # Usa semente para aleatoriedade da perturbação
    perturbed_solution = copy.deepcopy(solution)
    num_pens = calculate_objective(perturbed_solution)
    if num_pens == float('inf'): # Lida com solução inválida vinda da iteração anterior
        print("Aviso: Perturbação iniciada com solução inválida.", file=sys.stderr)
        num_pens = n_criminals # Estimativa

    # Seleciona quais criminosos mover
    criminals_to_move = random.sample(range(n_criminals), min(perturbation_strength, n_criminals))

    for criminal in criminals_to_move:
        original_pen = perturbed_solution[criminal]
        if original_pen == -1: continue # Pula não alocados

        # Opções de penitenciárias para mover (todas existentes + uma nova)
        possible_pens = list(range(num_pens + 1))
        # Não considera mover para a mesma penitenciária
        if original_pen in possible_pens:
            possible_pens.remove(original_pen)

        moved = False
        if possible_pens:
            random.shuffle(possible_pens) # Tenta as opções em ordem aleatória
            for target_pen in possible_pens:
                # Verifica se o movimento é válido em relação aos vizinhos
                can_move = True
                for neighbor in adj[criminal]:
                    # Se um vizinho já está na penitenciária alvo
                    if perturbed_solution[neighbor] == target_pen:
                        can_move = False
                        break
                if can_move:
                    perturbed_solution[criminal] = target_pen
                    # Atualiza num_pens caso uma nova penitenciária tenha sido criada
                    num_pens = max(num_pens, target_pen + 1)
                    moved = True
                    break # Moveu com sucesso

        # Se nenhum movimento válido foi encontrado (muito improvável, a menos que o grafo seja denso)
        # podemos manter a posição original. Uma perturbação mais sofisticada poderia ser necessária.

    # Reenumera as penitenciárias de forma contígua após a perturbação
    pen_map = {}
    next_pen_id = 0
    final_perturbed_solution = [-1] * n_criminals
    for i in range(n_criminals):
        old_pen = perturbed_solution[i]
        if old_pen == -1: continue
        if old_pen not in pen_map:
            pen_map[old_pen] = next_pen_id
            next_pen_id += 1
        final_perturbed_solution[i] = pen_map[old_pen]


    # É possível que a perturbação tenha tornado a solução inviável.
    # A busca local subsequente deve tentar corrigir isso.
    # if not is_feasible(final_perturbed_solution, alliances, n_criminals):
    #    print("Aviso: Perturbação resultou em uma solução inviável.", file=sys.stderr)

    # Retorna a solução perturbada e reenumerada
    return final_perturbed_solution

def acceptance_criterion(current_objective, new_objective, best_objective_so_far):
    """Critério de Aceitação: Aceita a nova solução se for melhor que a melhor encontrada até agora."""
    # O ILS tipicamente aplica a busca local *depois* da perturbação.
    # A decisão de aceitação é frequentemente sobre se o *resultado* da
    # busca local na solução perturbada substitui a melhor atual (s_best).
    # Esta versão compara o resultado direto da (perturbação + busca local)
    # contra a melhor solução encontrada até agora (s_best).
    return new_objective < best_objective_so_far

# --- Algoritmo Principal do ILS ---

def iterated_local_search(file_path, max_ils_iterations, perturbation_strength, random_seed):
    """Executa o algoritmo Iterated Local Search."""
    start_time = time.time() # Marca o tempo de início

    # Lê a instância
    n_criminals, alliances, adj = read_instance(file_path)

    # --- Fase 1: Geração da Solução Inicial ---
    current_seed = random_seed # Usa a semente principal para a geração inicial
    s0 = generate_initial_solution(n_criminals, alliances, adj, current_seed)
    s0_obj = calculate_objective(s0)
    if not is_feasible(s0, alliances, n_criminals):
        print(f"Erro: Solução inicial é inviável. Objetivo: {s0_obj}, Conflitos: {get_conflicts(s0, alliances)}", file=sys.stderr)
        # Lidar com início inviável se necessário (ex: tentar gerar de novo ou sair)
        # Por agora, continuamos, esperando que a Busca Local corrija.

    # Imprime a solução inicial (formato: tempo, valor, representação)
    print(f"{time.time() - start_time:.2f}, {s0_obj}, {s0}")

    # --- Fase 2: Melhoria Inicial com Busca Local ---
    # Aplica busca local na solução inicial
    s_best = local_search(s0, alliances, adj, n_criminals)
    s_best_obj = calculate_objective(s_best)
    # Verifica se a busca local inicial resultou em algo viável
    if not is_feasible(s_best, alliances, n_criminals):
        print(f"Erro: Busca local inicial resultou em solução inviável. Objetivo: {s_best_obj}, Conflitos: {get_conflicts(s_best, alliances)}", file=sys.stderr)
        # Fallback (Plano B) ou tratamento de erro necessário
        s_best = s0 # Reverte para a inicial se a busca local falhar gravemente
        s_best_obj = s0_obj

    # Imprime a primeira solução melhorada (melhor global até agora)
    print(f"{time.time() - start_time:.2f}, {s_best_obj}, {s_best}")

    # s_current rastreia a solução base para a próxima perturbação
    s_current = copy.deepcopy(s_best)

    # --- Fase 3: Loop Principal do ILS ---
    for i in range(max_ils_iterations): # Critério de parada: número de iterações do ILS
        current_seed += 1 # Incrementa a semente para perturbação/busca local determinísticas

        # 1. Perturbação: Gera s_perturbed a partir de s_current
        s_perturbed = perturb_solution(s_current, alliances, adj, n_criminals, perturbation_strength, current_seed)

        # 2. Busca Local: Aplica busca local em s_perturbed para obter s_candidate
        s_candidate = local_search(s_perturbed, alliances, adj, n_criminals)
        s_candidate_obj = calculate_objective(s_candidate)

        # Verifica viabilidade da solução candidata
        candidate_is_feasible = is_feasible(s_candidate, alliances, n_criminals)
        if not candidate_is_feasible:
            print(f"Iteração {i}: Aviso - Busca local na solução perturbada resultou em estado inviável. Objetivo: {s_candidate_obj}, Conflitos: {get_conflicts(s_candidate, alliances)}", file=sys.stderr)
            # Opção: Pular esta iteração, reverter s_candidate, ou tentar reparar.
            # Por agora, vamos comparar os objetivos de qualquer forma, mas sinalizar o problema.

        # 3. Critério de Aceitação (Atualização de s_best):
        # Compara s_candidate com a melhor solução global (s_best)
        # Aceita apenas se for uma melhoria E viável
        if candidate_is_feasible and acceptance_criterion(calculate_objective(s_current), s_candidate_obj, s_best_obj):
            s_best = copy.deepcopy(s_candidate) # Atualiza a melhor solução global
            s_best_obj = s_candidate_obj
            s_current = copy.deepcopy(s_candidate) # Atualiza a base para a próxima perturbação
            # Imprime a nova melhor solução encontrada
            print(f"{time.time() - start_time:.2f}, {s_best_obj}, {s_best}")
        # Else: Opcionalmente, poderia ter um critério para aceitar piores soluções às vezes (como no SA)
        #       ou para decidir se `s_current` deve ser resetado para `s_best`.
        #       A implementação básica do ILS geralmente continua a partir de `s_candidate`
        #       ou volta para `s_best`. Vamos continuar de `s_candidate` como base para a próxima perturbação
        #       para maior exploração, mesmo que não seja o melhor global.

        # 4. Critério de Aceitação (Atualização de s_current - Base para próxima iteração):
        # A referência sugere que a busca local é aplicada a s_perturbed,
        # resultando em s'. s_current (a base para a próxima iteração) é atualizada
        # para s' baseado em um critério de aceitação (comparando s' com s_current).
        # s_best é atualizado se s' for melhor que s_best.
        # Lógica Refinada: A decisão de qual solução usar como base (s_current) para
        # a *próxima* perturbação pode variar. Uma estratégia comum é sempre
        # usar o resultado da busca local (s_candidate), mesmo que não seja melhor
        # que s_best, para explorar mais. Outra é voltar para s_best se s_candidate
        # não for melhor. Adotamos a primeira estratégia (usar s_candidate):
        s_current = copy.deepcopy(s_candidate)


    # --- Fim da Execução ---
    end_time = time.time()
    print(f"\nILS finalizado após {i+1} iterações.")
    print(f"Tempo total: {end_time - start_time:.2f} segundos")
    print(f"Melhor solução encontrada: {s_best}")
    print(f"Melhor valor objetivo (penitenciárias): {s_best_obj}")
    # Verifica a viabilidade final da melhor solução
    final_feasibility = is_feasible(s_best, alliances, n_criminals)
    print(f"É viável: {final_feasibility}")
    if not final_feasibility:
        print(f"Conflitos: {get_conflicts(s_best, alliances)}") # Mostra conflitos se for inviável

    return s_best, s_best_obj

# --- Execução do Script ---
if __name__ == "__main__":
    # Verifica se os argumentos mínimos foram passados
    if len(sys.argv) < 4:
        print("Uso: python iterated_local_search.py <pasta_instancias> <max_iteracoes_ils> <semente_aleatoria> [forca_perturbacao]")
        print("Exemplo: python iterated_local_search.py inputs 1000 1 5")
        sys.exit(1) # Termina se os argumentos estiverem incorretos
        
    # O critério de parada principal é o número de iterações do ILS
    max_iterations_param = int(sys.argv[2])
    # Semente para garantir reprodutibilidade
    random_seed_param = int(sys.argv[3])    
    # Parâmetro opcional para força da perturbação (quantos criminosos mover)
    perturbation_strength_param = 5 # Valor padrão se não for fornecido
    if len(sys.argv) > 4:
        perturbation_strength_param = int(sys.argv[4])
    
    files = glob(sys.argv[1]+r"\*.txt")
    
    for file in files:
        # Lê os argumentos da linha de comando
        instance_file = file
        # Imprime informações da execução
        print(f"Executando ILS para Separação de Comparsas")
        print(f"Instância: {instance_file}")
        print(f"Máximo de Iterações ILS: {max_iterations_param}")
        print(f"Semente Aleatória Inicial: {random_seed_param}")
        print(f"Força da Perturbação: {perturbation_strength_param}")
        print("-" * 30)

        # Chama a função principal do ILS
        best_solution_found, best_objective_found = iterated_local_search(
            instance_file,
            max_iterations_param,
            perturbation_strength_param,
            random_seed_param
        )

        # Imprime um separador final kkkkkkkk
        print("-" * 30 + "\n\n")