#import Pkg; Pkg.add("JuMP");Pkg.add("HiGHS");
using JuMP
using HiGHS

file = open(ARGS[1])

# Número de criminosos e aliancas
string_aux = readline(file)
numero_dlm = findfirst(' ', string_aux)
n_criminals = parse(Int32, SubString(string_aux,1,numero_dlm))
n_allies = parse(Int32, SubString(string_aux,numero_dlm,lastindex(string_aux)))

# quantidade de penitenciarias
#inserir aqui um codigo para calcular um numero de penitenciarias base
n_pen = n_criminals
if n_criminals > 39
    n_pen = 39
end

# array de aliancas
allies = Tuple{Int32, Int32}[]

for line in readlines(file)
    numero_dlm = findfirst(' ', line)
    primeiro = parse(Int32, SubString(line,1,numero_dlm))
    segundo = parse(Int32, SubString(line,numero_dlm+1,lastindex(line)))
    tuple = (primeiro, segundo)
    push!(allies, tuple)
end

# Criando o modelo de otimização
model = Model(HiGHS.Optimizer)

# Variáveis de decisão: matriz booleana indicando se cada criminoso vai para cada penitenciaria
@variable(model, x[1:n_criminals, 1:n_pen], Bin)

# Variáveis auxiliares para representar se cada penitenciaria é usada ou não
@variable(model, y[1:n_pen], Bin)

# Função objetivo: minimizar o número de penitenciarias usadas
@objective(model, Min, sum(y))

# Restrições
# Cada criminoso deve ser alocado em uma única penitenciaria
@constraint(model, c_alloc_unique[i = 1:n_criminals], sum(x[i, j] for j in 1:n_pen) == 1)

# Se um criminoso é alocado em uma penitenciária 'j', a penitenciária 'j' deve ser marcada como usada (y[j]=1)
# Esta restrição liga a variável 'x' à variável 'y'
@constraint(model, c_pen_used[j = 1:n_pen], sum(x[i, j] for i in 1:n_criminals) <= n_criminals * y[j])

# verificar se nenhuma penitenciaria contem aliancas
@constraint(model, c_no_allies_in_pen[k = 1:n_allies, j = 1:n_pen],
    x[first(allies[k]), j] + x[last(allies[k]), j] <= 1
)

set_time_limit_sec(model, 300.0)
set_attribute(model, "random_seed", 10)

# Resolvendo o modelo
optimize!(model)

# Exibe resultados
println(solution_summary(model))

#Exibindo a solução
if termination_status(model) == MOI.OPTIMAL
    println("\n--- Solução Ótima ---")
    println("Número mínimo de penitenciarias usadas: ", objective_value(model))

    # Mostrando a alocação dos criminosos nas penitenciarias
    for j = 1:n_pen
        # Verifica se a penitenciária j foi realmente utilizada pelo modelo
        if value(y[j]) > 0.5 # Apenas para penitenciárias que foram ativadas
            criminals_in_pen = [i for i = 1:n_criminals if value(x[i, j]) > 0.5]
            if !isempty(criminals_in_pen)
                println("Penitenciaria $j: Criminosos ", criminals_in_pen)
            end
        end
    end
else
    println("\n--- Nenhuma solução ótima encontrada ---")
    println("Status de terminação: ", termination_status(model))
    # Para depuração, você pode querer exibir o status detalhado se não for ótimo
    # println("Primal Status: ", primal_status(model))
    # println("Dual Status: ", dual_status(model))
end
