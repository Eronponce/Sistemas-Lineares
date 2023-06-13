import streamlit as st
import numpy as np
import pandas as pd


def permutation_matrix(matrix):
    n = matrix.shape[0]
    matriz_permutacao = np.eye(n)
    sorted_indices = np.argsort(matrix[:, 0])  # Organiza as linhas da matriz de acordo com o primeiro elemento
    matriz_permutacao = matriz_permutacao[sorted_indices]
    return matriz_permutacao


def gaussian_elimination(matrix, n):
    # Cópia da matriz original (matriz inicial)
    matriz_original = np.copy(matrix)

    # Fase de eliminação
    matriz_L = np.eye(n)  # Inicializa a matriz L como uma matriz identidade
    for i in range(n):
        if matrix[i, i] == 0:
            # Verifica se o elemento diagonal é zero
            # e permuta a linha com outra linha não nula
            for j in range(i+1, n):
                if matrix[j, i] != 0:
                    # Realiza a permutação da linha i com a linha j
                    matrix[[i, j]] = matrix[[j, i]]
                    matriz_L[[i, j]] = matriz_L[[j, i]]  # Atualiza a matriz L com a permutação
                    break
            else:
                # Se não houver uma linha não nula para permutar, a matriz não é invertível
                return None, None
        for j in range(i+1, n):
            # Calcula a razão entre a linha atual (j) e a linha pivot (i)
            pivo = matrix[j, i] / matrix[i, i]
            matriz_L[j, i] = pivo  # Armazena a razão na matriz L
            for k in range(n+1):
                # Executa a operação de eliminação: subtrai a linha
                # pivot multiplicada pela razão da linha atual
                matrix[j, k] -= pivo * matrix[i, k]
        st.dataframe(matrix)
    
    # Exibe a matriz U (matriz resultante da eliminação de Gauss)
    # Fase de substituição
    solution = np.zeros(n)
    solution[n-1] = matrix[n-1, n] / matrix[n-1, n-1]
    for i in range(n-2, -1, -1):
        # Calcula a solução para a variável atual (i)
        solution[i] = matrix[i, n]
        for j in range(i+1, n):
            # Subtrai as variáveis já calculadas multiplicadas
            #  pelos coeficientes correspondentes
            solution[i] -= matrix[i, j] * solution[j]
        # Divide pelo coeficiente da variável atual
        #  para obter o valor da solução
        solution[i] /= matrix[i, i]

    return solution, matriz_L, matriz_original, matrix


def gauss_seidel(matrix, n, epsilon, initial_guesses):
    iteration = 0  # Inicializa o contador de iterações
    solution = np.array(initial_guesses)  # Inicializa o vetor solução
    
    while True:
        prev_solution = solution.copy()  # Faz uma cópia da solução anterior
        
        # Itera através de cada equação
        for i in range(n):
            sum1 = np.dot(matrix[i, :i], solution[:i])  # Calcula o produto escalar das soluções anteriores com os coeficientes correspondentes
            sum2 = np.dot(matrix[i, i + 1:-1], solution[i + 1:])  # Calcula o produto escalar das soluções anteriores com os coeficientes correspondentes
            solution[i] = (matrix[i, -1] - sum1 - sum2) / matrix[i, i]  # Atualiza a solução atual
            
        iteration += 1  # Incrementa o contador de iterações
        # Exemplo de DataFrame
        df = pd.DataFrame({'coluna': solution.tolist()  })

        # Crie um dicionário de mapeamento para renomear os índices
        indice_mapeamento = {i: f'x{i+1}' for i in range(len(df))}

        # Renomeie os índices usando o método rename
        df = df.rename(index=indice_mapeamento)
        st.dataframe(df)
        # Verifica a convergência com base na diferença entre a solução atual e a solução anterior
        if np.linalg.norm(solution - prev_solution) < epsilon:
            st.write("Convergiu em", iteration, "iterações")  # Exibe informações sobre a convergência
            return solution.tolist()  # Retorna a solução como uma lista
        
        # Verifica o número máximo de iterações
        if iteration > 100:
            st.write("Não convergiu após 1000 iterações")  # Exibe informações sobre a falha na convergência
            return None  # Retorna None para indicar falha na convergência


def main():
    escolha = st.sidebar.radio("Escolha o método:", ("Eliminação de Gauss", "Eliminação de Gauss-Seidel"))
    st.title("Eliminação de Gauss")
    interacao = st.number_input("Qual o tamanho da matriz:", min_value=1)
    matriz = np.zeros((interacao, interacao + 1))
    st.subheader("Coloque a matriz abaixo:")
    cols = st.columns(interacao + 1)

    for i in range(interacao):
        for j in range(interacao + 1):
            if j == interacao:
                variable_name = "="
            else:
                variable_name = f'x{j+1}'
            matriz[i, j] = cols[j].number_input(
                f"{variable_name} Elemento ({i}, {j})",
                key=f"element_{i}_{j}", step=0.1
            )

    if escolha == "Eliminação de Gauss-Seidel":
        epsilon = st.sidebar.number_input("Qual o valor de epsilon:", min_value=0.0, step=0.1)
        inputs = []
        for i in range(interacao):
            input_value = st.sidebar.text_input(f"Chutes X{i+1}")
            inputs.append(float(input_value) if input_value != "" else 0.0)

        if st.button("Calcular"):
            result = gauss_seidel(matriz, interacao, epsilon, inputs)
            if result is not None:
                st.subheader("Result:")
                st.code(result)

    if escolha == "Eliminação de Gauss":
        if st.button("Calcular"):
            result, matriz_L, matriz_original, matriz_U = gaussian_elimination(np.copy(matriz), interacao)

            if result is not None:
                st.subheader("Result:")
                st.code(result)
                st.subheader("Matriz L:")
                st.dataframe(matriz_L)
                st.subheader("Matriz U:")
                st.dataframe(matriz_U)
                st.subheader("Matriz de Permutação:")
                st.dataframe(permutation_matrix(matriz_original))
                matriz_permutacao = permutation_matrix(matriz_original)
                matriz_mult = np.matmul(matriz_L, matriz_U)
               

                st.subheader("Matriz Original L * U * Permutação:")
                st.dataframe(matriz_mult)

main()
