import streamlit as st
import numpy as np
import pandas as pd
from src.gauss_sp import scaled_partial_pivot_gauss

# Load custom CSS for highlighting
def load_css():
    with open('assets/custom.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Display a matrix A and vector b with row highlighting based on the current step
def highlight_and_show(A, b, step):
    n = A.shape[0]
    df = pd.DataFrame(A, columns=[f"x{j}" for j in range(n)])
    df['b'] = b
    def highlight_row(row):
        styles = [''] * len(row)
        idx = row.name
        action = step['step']
        # Pivot or swap step: highlight pivot row
        if action in ('pivot', 'swap'):
            pr = step['pivot_row']
            if idx == pr:
                styles = ['background-color: lightblue'] * len(row)
        # Elimination step: highlight pivot and target rows differently
        elif action == 'elimination':
            k = step['k']; i = step['i']
            if idx == k:
                styles = ['background-color: lightgreen'] * len(row)
            elif idx == i:
                styles = ['background-color: lightcoral'] * len(row)
        # Back substitution: highlight the row being solved
        elif action == 'back_substitution':
            if idx == step['i']:
                styles = ['background-color: lightyellow'] * len(row)
        return styles
    st.dataframe(df.style.apply(highlight_row, axis=1))

# Render a single example with all its steps
def render_example(A, b, title='Example'):
    st.header(title)
    x, steps = scaled_partial_pivot_gauss(A, b, return_steps=True)
    for idx, step in enumerate(steps, 1):
        st.subheader(f"Step {idx}: {step['step'].replace('_', ' ').title()}")
        highlight_and_show(step['A'], step['b'], step)
    st.subheader('Solution')
    st.write(x)

# Fixed walkthrough page: show two predetermined examples
def render_walkthrough():
    st.title('Fixed Walkthrough')
    # 3×3 example
    A3 = np.array([[3.0,1.0,2.0],[1.0,2.0,0.0],[0.0,1.0,1.0]])
    b3 = np.array([10.0,8.0,3.0])
    render_example(A3, b3, title='3×3 Example')
    # 4×4 example
    A4 = np.array([[2.0,1.0,1.0,0.0],[4.0,3.0,3.0,1.0],[8.0,7.0,9.0,5.0],[6.0,7.0,9.0,8.0]])
    b4 = np.array([4.0,10.0,28.0,29.0])
    render_example(A4, b4, title='4×4 Example')

# Interactive playground page: user can enter or randomize a system
def render_playground():
    st.title('Interactive Playground')
    n = st.selectbox('Matrix size', [3, 4])
    st.write('Enter matrix A:')
    A = np.zeros((n, n), float)
    for i in range(n):
        cols = st.columns(n)
        for j, col in enumerate(cols):
            A[i, j] = col.number_input(f'A[{i},{j}]', key=f'A-{i}-{j}')
    st.write('Enter vector b:')
    b = np.zeros(n, float)
    cols = st.columns(n)
    for i, col in enumerate(cols):
        b[i] = col.number_input(f'b[{i}]', key=f'b-{i}')
    if st.button('Solve'):
        render_example(A, b, title='Playground Solution')

# Main entrypoint
if __name__ == '__main__':
    load_css()
    page = st.sidebar.selectbox('Page', ['Walkthrough', 'Playground'])
    if page == 'Walkthrough':
        render_walkthrough()
    else:
        render_playground()