import streamlit as st
import numpy as np

def gaussian_elimination(matriz,interacao):

    return matriz 
def main():
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
            matriz[i, j] = cols[j].number_input(f"{variable_name} Elemento ({i}, {j})", key=f"element_{i}_{j}",step = 1)
    
    if st.button("Calcular"):
        result = gaussian_elimination(matriz,interacao)
        st.subheader("Result:")
        st.write(result)
    
main()
 