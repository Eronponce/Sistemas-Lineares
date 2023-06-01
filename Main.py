import streamlit as st
import numpy as np


def permutation_matrix(matrix):
    n = matrix.shape[0]
    matriz_permutacao = np.eye(n)
    sorted_indices = np.argsort(matrix[:, 0])  # Organiza as linhas da matriz de acordo com o primeiro elemento
    matriz_permutacao = matriz_permutacao[sorted_indices]
    return matriz_permutacao


def gaussian_elimination(matrix, n):
    # Fase de eliminação
    st.code(permutation_matrix(matrix))
    for i in range(n):
        if matrix[i, i] == 0:
            # Verifica se o elemento diagonal é zero
            # e permuta a linha com outra linha não nula
            for j in range(i+1, n):
                if matrix[j, i] != 0:
                    # Realiza a permutação da linha i com a linha j
                    matrix[[i, j]] = matrix[[j, i]]
                    break
            else:
                # Se não houver uma linha não nula para permutar, a matriz não é invertível
                return None
        for j in range(i+1, n):
            # Calcula a razão entre a linha atual (j) e a linha pivot (i)
            pivo = matrix[j, i] / matrix[i, i]
            for k in range(n+1):
                # Executa a operação de eliminação: subtrai a linha
                # pivot multiplicada pela razão da linha atual
                matrix[j, k] -= pivo * matrix[i, k]
        st.dataframe(matrix)
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

    return solution




def main():
    escolha = st.sidebar.radio("Escolha o método:", ("Eliminação de Gauss", "Eliminação de Gauss-Seidel"))
    
        

    st.title("Eliminação de Gauss")
    interacao = st.number_input("Qual o tamanho da matriz:", min_value=1, step=1)
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
                key=f"element_{i}_{j}", step=1.0
                )

    if escolha == "Eliminação de Gauss-Seidel":
        epsilon = st.sidebar.number_input("Qual o valor de epsilon:", min_value=0.0, step=0.1)
        inputs = []
        for i in range(interacao):
            input_value = st.sidebar.text_input(f"Chutes X{i+1}")
            inputs.append(input_value)
      
    if st.button("Calcular"):
        if escolha == "Eliminação de Gauss":
            result = gaussian_elimination(matriz, interacao)
            if result is not None:
                st.subheader("Result:")
                st.code(result)
        elif escolha == "Eliminação de Gauss-Seidel":
            result = gauss_seidel()
            if result is not None:
                st.subheader("Result:")
                st.code(result)


main()
