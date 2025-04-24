import streamlit as st
import numpy as np
import pandas as pd
from src.gauss_sp import scaled_partial_pivot_gauss

# Add page config
st.set_page_config(
    page_title="Gaussian Elimination Demo",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS for highlighting
def load_css():
    with open('assets/custom.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Display a matrix A and vector b with row highlighting based on the current step
def highlight_and_show(A, b, step):
    # Build a DataFrame and a parallel CSS-class map
    n = A.shape[0]
    df = pd.DataFrame(A, columns=[f"x{j}" for j in range(n)])
    df['b'] = b
    # Prepare class names for each cell
    classes = pd.DataFrame('', index=df.index, columns=df.columns)
    typ = step.get('step')
    # Pivot or swap step: highlight entire pivot row
    if typ in ('pivot', 'swap'):
        pr = step['pivot_row']
        classes.loc[pr, :] = 'highlight-pivot-row'
    # Elimination: highlight pivot row green and target row red
    elif typ == 'elimination':
        k = step['k']; i = step['i']
        classes.loc[k, :] = 'highlight-pivot-green'
        classes.loc[i, :] = 'highlight-target'
    # Back substitution: highlight the row being solved
    elif typ == 'back_substitution':
        i = step['i']
        classes.loc[i, :] = 'highlight-backsub'
    # Render styled DataFrame with CSS classes
    st.write(df.style.set_td_classes(classes))

# Format a human-readable comment for each step
def format_step_comment(step):
    typ = step.get('step')
    if typ == 'swap':
        return (f"Column {step['k']}: pivot row {step['pivot_row']} selected "
                f"with scaled ratio {step['ratio']:.3f}. Swapped row {step['k']} and {step['pivot_row']}.")
    elif typ == 'pivot':
        return (f"Column {step['k']}: pivot row {step['pivot_row']} selected "
                f"with scaled ratio {step['ratio']:.3f}. No swap needed.")
    elif typ == 'elimination':
        num = step.get('mult_num')
        den = step.get('mult_den')
        m = step.get('multiplier')
        return (f"Row {step['i']}: eliminate A[{step['i']},{step['k']}] using "
                f"multiplier {num}/{den} = {m:.3f}.")
    elif typ == 'back_substitution':
        return (f"Back substitute for x[{step['i']}]: "
                f"x[{step['i']}] = {step['value']:.6g}.")
    return ""

# Render a single example with all its steps
def render_example(A, b, title='Example'):
    st.header(title)
    x, steps = scaled_partial_pivot_gauss(A, b, return_steps=True)
    num_steps = len(steps)
    selected = st.slider(
        "Select step", min_value=1, max_value=num_steps,
        value=1, key=f"step-{title}"
    )
    for idx, step in enumerate(steps, 1):
        with st.expander(
            f"Step {idx}: {step['step'].replace('_', ' ').title()}",
            expanded=(idx == selected)
        ):
            highlight_and_show(step['A'], step['b'], step)
            # Show a formatted comment under the step
            comment = format_step_comment(step)
            if comment:
                st.markdown(f"**{comment}**")
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
    A4 = np.array([
        [3.0,  -13.0,  9.0,   3.0],
        [-6.0,   4.0,  1.0, -18.0],
        [6.0,   -2.0,  2.0,   4.0],
        [12.0,  -8.0,  6.0,  10.0]
    ])
    b4 = np.array([-19.0, -34.0, 16.0, 26.0])
    render_example(A4, b4, title='4×4 Example (Solution: 3, 1, -2, 1)')

# Interactive playground page: user can enter or randomize a system
def render_playground():
    st.title('Interactive Playground')
    n = st.selectbox('Matrix size', [3, 4])
    # Randomize button for quick testing
    if st.button('Randomize & Solve'):
        A_rand = np.random.randint(-10, 11, size=(n, n)).astype(float)
        b_rand = np.random.randint(-10, 11, size=n).astype(float)
        with st.spinner('Solving random system...'):
            render_example(A_rand, b_rand, title=f'Random {n}×{n} System')
    with st.form('input_form'):
        st.write('Enter the augmented matrix [A | b]:')
        # Build A and b side by side
        A = np.zeros((n, n), float)
        b = np.zeros(n, float)
        for i in range(n):
            cols = st.columns(n + 1)
            for j in range(n):
                A[i, j] = cols[j].number_input(f'A[{i},{j}]', key=f'A-{i}-{j}')
            b[i] = cols[n].number_input(f'b[{i}]', key=f'b-{i}')
        submitted = st.form_submit_button('Solve')
    if submitted:
        with st.spinner('Solving system...'):
            render_example(A, b, title='Playground Solution')

# Main entrypoint
if __name__ == '__main__':
    load_css()
    st.markdown(
        '## Gaussian Elimination with Scaled Partial Pivoting\n\n'
        'Use the sidebar to switch between the Walkthrough of fixed examples and the Interactive Playground.',
        unsafe_allow_html=True
    )
    page = st.sidebar.selectbox('Page', ['Walkthrough', 'Playground'])
    if page == 'Walkthrough':
        render_walkthrough()
    else:
        render_playground()