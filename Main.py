import streamlit as st
import numpy as np


def gaussian_elimination(matrix, n):
    # Fase de eliminação
    for i in range(n):
        if matrix[i, i] == 0:
            # Verifica se o elemento diagonal é zero,
            # o que torna a eliminação de Gauss inviável
            st.error("Cannot perform Gaussian elimination. Diagonal element is zero.")
            return None
        for j in range(i+1, n):
            # Calcula a razão entre a linha atual (j) e a linha pivot (i)
            ratio = matrix[j, i] / matrix[i, i]
            for k in range(n+1):
                # Executa a operação de eliminação: subtrai a linha
                # pivot multiplicada pela razão da linha atual
                matrix[j, k] -= ratio * matrix[i, k]
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
    st.sidebar.radio("Escolha o método:", ("Eliminação de Gauss", "Eliminação de Gauss-Jordan"))
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

    if st.button("Calcular"):
        result = gaussian_elimination(matriz, interacao)
        if result is not None:
            st.subheader("Result:")
            st.code(result)


main()
