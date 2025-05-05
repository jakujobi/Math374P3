import streamlit as st
import numpy as np
import pandas as pd
from src.gauss_sp import scaled_partial_pivot_gauss
import io
import markdown
import uuid
import os
import inspect
import time
# Try to import WeasyPrint or FPDF for PDF export; disable if unavailable
try:
    import weasyprint
    PDF_ENGINE = 'weasyprint'
    HTML = weasyprint.HTML
except ImportError:
    try:
        from fpdf import FPDF
        PDF_ENGINE = 'fpdf'
    except ImportError:
        PDF_ENGINE = None
has_pdf = PDF_ENGINE is not None

# Add page config
st.set_page_config(
    page_title="Gaussian Elimination Demo",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS for highlighting
def load_css():
    # Define CSS directly in the app
    st.markdown("""
    <style>
    /* Fade-in animation for dataframes */
    @keyframes fadeIn {
      0% { opacity: 0; }
      100% { opacity: 1; }
    }

    /* Apply fade-in to all Streamlit dataframes */
    [data-testid="stDataFrame"] {
      animation: fadeIn 0.3s ease-in-out;
    }

    /* Default highlight colors (for potential class-based styling) */
    .highlight-pivot {
      background-color: lightblue !important;
    }

    .highlight-pivot-row {
      background-color: lightblue !important;
    }

    .highlight-pivot-green {
      background-color: lightgreen !important;
    }

    .highlight-target {
      background-color: lightcoral !important;
    }

    .highlight-backsub {
      background-color: lightyellow !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Display a matrix A and vector b with row highlighting based on the current step
def highlight_and_show(A, b, step):
    # Build a DataFrame and a parallel CSS-class map
    n = A.shape[0]
    df = pd.DataFrame(A, columns=[f"x{j}" for j in range(n)])
    df['b'] = b
    
    # Apply styling directly to the DataFrame
    def highlight_row(row_idx, color):
        return ['background-color: ' + color if i == row_idx else '' for i in range(len(df))]
    
    # Determine which row to highlight based on step type
    typ = step.get('step')
    styled_df = df.copy()
    
    # Apply highlighting based on step type
    if typ in ('pivot', 'swap'):
        pr = step['pivot_row']
        styled_df = df.style.apply(lambda x: highlight_row(pr, 'lightblue'), axis=0)
    elif typ == 'elimination':
        k = step['k']
        i = step['i']
        styled_df = df.style.apply(lambda x: highlight_row(k, 'lightgreen'), axis=0).apply(
            lambda x: highlight_row(i, 'lightcoral'), axis=0)
    elif typ == 'back_substitution':
        i = step['i']
        styled_df = df.style.apply(lambda x: highlight_row(i, 'lightyellow'), axis=0)
    
    # Display the styled DataFrame
    st.write(styled_df)

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

# Generate Markdown report for a solved system
def generate_report_md(A, b, steps, x):
    lines = []
    lines.append('# Gaussian Elimination Report')
    lines.append('## Problem Statement')
    lines.append('\n**Matrix A:**\n')
    lines.append('```')
    lines.append(str(A))
    lines.append('```')
    lines.append('\n**Vector b:**\n')
    lines.append('```')
    lines.append(str(b))
    lines.append('```')
    lines.append('\n## Step-by-step Details')
    for idx, step in enumerate(steps, 1):
        lines.append(f"\n### Step {idx}: {step['step'].replace('_', ' ').title()}")
        # human-readable comment
        comment = format_step_comment(step)
        if comment:
            lines.append(f"**{comment}**\n")
        # matrix state
        Aug = np.hstack((step['A'], step['b'].reshape(-1, 1)))
        lines.append('```')
        for row in Aug:
            lines.append(str(row.tolist()))
        lines.append('```')
    lines.append('\n## Solution')
    lines.append('```')
    lines.append(str(tuple(x)))
    lines.append('```')
    return '\n'.join(lines)

# Convert Markdown string to PDF bytes
def convert_md_to_pdf(md_str):
    if not has_pdf:
        raise RuntimeError("PDF export is unavailable: no PDF engine installed (weasyprint or fpdf).")
    try:
        if PDF_ENGINE == 'weasyprint':
            html_body = markdown.markdown(md_str)
            html = f"<html><body>{html_body}</body></html>"
            pdf = HTML(string=html).write_pdf()
            return pdf
        elif PDF_ENGINE == 'fpdf':
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            for line in md_str.split('\n'):
                pdf.multi_cell(0, 8, line)
            return pdf.output(dest='S').encode('latin-1')
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        raise RuntimeError(f"PDF generation failed: {str(e)}")

# Render a single example with all its steps
def render_example(A, b, title='Example'):
    origA = A.copy()
    origb = b.copy()
    x, steps = scaled_partial_pivot_gauss(origA.copy(), origb.copy(), return_steps=True)
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
    # Phase 6: Solution Verification
    st.subheader('Solution Verification')
    residual = origA.dot(x) - origb
    st.write(f'Residual (Ax - b): {residual}')
    st.write(f'Infinity Norm of Residual: {np.linalg.norm(residual, np.inf):.3e}')
    # Export buttons
    md_report = generate_report_md(origA, origb, steps, x)
    st.subheader('Report Preview')
    st.markdown(md_report)
    # Create a unique slug from title with a random UUID suffix for widget keys
    slug_base = ''.join(c.lower() if c.isalnum() else '_' for c in title)
    slug = f"{slug_base}_{uuid.uuid4().hex}"
    # Download Markdown report with unique key
    st.download_button(
        label='Download report (Markdown)',
        data=md_report,
        file_name='gauss_elimination_report.md',
        mime='text/markdown',
        key=f'download_md_{slug}'
    )
    # Download PDF report if available, with unique key
    if has_pdf:
        try:
            pdf_bytes = convert_md_to_pdf(md_report)
            st.download_button(
                label='Download report (PDF)',
                data=pdf_bytes,
                file_name='gauss_elimination_report.pdf',
                mime='application/pdf',
                key=f'download_pdf_{slug}'
            )
        except Exception as e:
            st.error(f"PDF export failed: {str(e)}")
            st.warning('PDF export disabled: WeasyPrint encountered an error. Try reinstalling it.')
    else:
        st.warning('PDF export disabled: install WeasyPrint or FPDF to enable this feature.')
    # Phase 8: References & Notes
    st.subheader('References & Notes')
    st.markdown("""
    - [Gaussian elimination – Wikipedia](https://en.wikipedia.org/wiki/Gaussian_elimination)
    - Burden & Faires, *Numerical Analysis*, Ch. 3
    - Uses scaled partial pivoting for numerical stability.
    """, unsafe_allow_html=True)

# Fixed walkthrough page: show two predetermined examples
def render_walkthrough():
    st.title('Fixed Walkthrough')
    # 3×3 example
    A3 = np.array([[3.0,1.0,2.0],[1.0,2.0,0.0],[0.0,1.0,1.0]])
    b3 = np.array([10.0,8.0,3.0])
    # Phase 1: Metadata & Problem Statement
    st.title('3×3 Example Report')
    st.markdown("""
**John Akujobi**  
MATH 374: Scientific Computation (Spring 2025), South Dakota State University  
Professor: Dr Kimn, Dept. of Math & Statistics  
GitHub: [jakujobi](https://github.com/jakujobi)
""", unsafe_allow_html=True)
    st.subheader('Problem Statement')
    st.markdown('Solve the linear system **A x = b**, where:')
    st.code(str(A3), language='text')
    st.code(str(b3), language='text')
    # Phase 2: Algorithm Overview
    st.subheader('Algorithm Overview')
    st.markdown("""
- Compute scale factors s[i] = max_j |A[i,j]|  
- For each column k:
  1. Compute ratio |A[i,k]|/s[i] for i=k..n-1  
  2. Select pivot row with max ratio, swap if needed  
  3. Eliminate A[i,k] for i>k  
- Back-substitution to find solution vector x  

```python
for k in range(n-1):
    s = [max(abs(A[i,:])) for i in range(n)]
    ratios = [abs(A[i,k])/s[i] for i in range(k,n)]
    pivot = k + int(np.argmax(ratios))
    if pivot != k:
        A[[k,pivot]] = A[[pivot,k]]
        b[k], b[pivot] = b[pivot], b[k]
    for i in range(k+1,n):
        m = A[i,k]/A[k,k]
        A[i,k:] -= m * A[k,k:]
# Back-substitution...
```  
""", unsafe_allow_html=True)
    # Phase 3: Step-by-Step Summary
    origA3 = A3.copy()
    x3, steps3 = scaled_partial_pivot_gauss(A3.copy(), b3.copy(), return_steps=True)
    st.subheader('Step-by-Step Summary')
    records3 = []
    for idx, step in enumerate(steps3, start=1):
        records3.append({
            'Step #': idx,
            'Type': step['step'],
            'k': step.get('k', ''),
            'i': step.get('i', ''),
            'pivot_row': step.get('pivot_row', ''),
            'multiplier': step.get('multiplier', ''),
            'ratio': step.get('ratio', ''),
            'value': step.get('value', '')
        })
    df3 = pd.DataFrame(records3)
    st.table(df3)
    # Phase 4: Intermediate Matrices
    st.subheader('Intermediate Matrices')
    n3 = origA3.shape[0]
    for k in range(n3 - 1):
        pivot_step = next(s for s in steps3 if s['step'] in ('pivot', 'swap') and s['k'] == k)
        elim_steps = [s for s in steps3 if s['step'] == 'elimination' and s['k'] == k]
        last_elim = elim_steps[-1] if elim_steps else None
        cols = st.columns(3)
        cols[0].subheader(f'Original Matrix (k={k})')
        cols[0].write(pd.DataFrame(origA3, columns=[f"x{j}" for j in range(n3)]))
        cols[1].subheader(f'After Pivot (k={k})')
        cols[1].write(pd.DataFrame(pivot_step['A'], columns=[f"x{j}" for j in range(n3)]))
        if last_elim:
            cols[2].subheader(f'After Elimination (k={k})')
            cols[2].write(pd.DataFrame(last_elim['A'], columns=[f"x{j}" for j in range(n3)]))
        else:
            cols[2].write('No elimination steps')
    render_example(A3, b3, title='3×3 Example')
    # 4×4 example
    A4 = np.array([
        [3.0,  -13.0,  9.0,   3.0],
        [-6.0,   4.0,  1.0, -18.0],
        [6.0,   -2.0,  2.0,   4.0],
        [12.0,  -8.0,  6.0,  10.0]
    ])
    b4 = np.array([-19.0, -34.0, 16.0, 26.0])
    # Phase 1: Metadata & Problem Statement
    st.title('4×4 Example Report')
    st.markdown("""
**John Akujobi**  
MATH 374: Scientific Computation (Spring 2025), South Dakota State University  
Professor: Dr Kimn, Dept. of Math & Statistics  
GitHub: [jakujobi](https://github.com/jakujobi)
""", unsafe_allow_html=True)
    st.subheader('Problem Statement')
    st.markdown('Solve the linear system **A x = b**, where:')
    st.code(str(A4), language='text')
    st.code(str(b4), language='text')
    # Phase 2: Algorithm Overview
    st.subheader('Algorithm Overview')
    st.markdown("""
- Compute scale factors s[i] = max_j |A[i,j]|  
- For each column k:
  1. Compute ratio |A[i,k]|/s[i] for i=k..n-1  
  2. Select pivot row with max ratio, swap if needed  
  3. Eliminate A[i,k] for i>k  
- Back-substitution to find solution vector x  

```python
for k in range(n-1):
    s = [max(abs(A[i,:])) for i in range(n)]
    ratios = [abs(A[i,k])/s[i] for i in range(k,n)]
    pivot = k + int(np.argmax(ratios))
    if pivot != k:
        A[[k,pivot]] = A[[pivot,k]]
        b[k], b[pivot] = b[pivot], b[k]
    for i in range(k+1,n):
        m = A[i,k]/A[k,k]
        A[i,k:] -= m * A[k,k:]
# Back-substitution...
```  
""", unsafe_allow_html=True)
    # Phase 3: Step-by-Step Summary
    origA4 = A4.copy()
    x4, steps4 = scaled_partial_pivot_gauss(A4.copy(), b4.copy(), return_steps=True)
    st.subheader('Step-by-Step Summary')
    records4 = []
    for idx, step in enumerate(steps4, start=1):
        records4.append({
            'Step #': idx,
            'Type': step['step'],
            'k': step.get('k', ''),
            'i': step.get('i', ''),
            'pivot_row': step.get('pivot_row', ''),
            'multiplier': step.get('multiplier', ''),
            'ratio': step.get('ratio', ''),
            'value': step.get('value', '')
        })
    df4 = pd.DataFrame(records4)
    st.table(df4)
    # Phase 4: Intermediate Matrices
    st.subheader('Intermediate Matrices')
    n4 = origA4.shape[0]
    for k in range(n4 - 1):
        pivot_step = next(s for s in steps4 if s['step'] in ('pivot', 'swap') and s['k'] == k)
        elim_steps = [s for s in steps4 if s['step'] == 'elimination' and s['k'] == k]
        last_elim = elim_steps[-1] if elim_steps else None
        cols = st.columns(3)
        cols[0].subheader(f'Original Matrix (k={k})')
        cols[0].write(pd.DataFrame(origA4, columns=[f"x{j}" for j in range(n4)]))
        cols[1].subheader(f'After Pivot (k={k})')
        cols[1].write(pd.DataFrame(pivot_step['A'], columns=[f"x{j}" for j in range(n4)]))
        if last_elim:
            cols[2].subheader(f'After Elimination (k={k})')
            cols[2].write(pd.DataFrame(last_elim['A'], columns=[f"x{j}" for j in range(n4)]))
        else:
            cols[2].write('No elimination steps')
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
        'Use the sidebar to switch between examples and the Interactive Playground.',
        unsafe_allow_html=True
    )
    # Sidebar navigation
    page = st.sidebar.radio('Page', ['3×3 Example', '4×4 Example', 'Playground'])
    st.sidebar.markdown('---')
    st.sidebar.markdown('**Version**: 1.0.0')
    st.sidebar.markdown('[GitHub Repository](https://github.com/jakujobi/Math374P3)')
    if page == '3×3 Example':
        A3 = np.array([[3.0, 1.0, 2.0], [1.0, 2.0, 0.0], [0.0, 1.0, 1.0]])
        b3 = np.array([10.0, 8.0, 3.0])
        # Phase 1: Metadata & Problem Statement
        st.title('3×3 Example Report')
        st.markdown("""
**John Akujobi**  
MATH 374: Scientific Computation (Spring 2025), South Dakota State University  
Professor: Dr Kimn, Dept. of Math & Statistics  
GitHub: [jakujobi](https://github.com/jakujobi)
""", unsafe_allow_html=True)
        st.subheader('Problem Statement')
        st.markdown('Solve the linear system **A x = b**, where:')
        st.code(str(A3), language='text')
        st.code(str(b3), language='text')
        # Phase 2: Algorithm Overview
        st.subheader('Algorithm Overview')
        st.markdown("""
- Compute scale factors s[i] = max_j |A[i,j]|  
- For each column k:
  1. Compute ratio |A[i,k]|/s[i] for i=k..n-1  
  2. Select pivot row with max ratio, swap if needed  
  3. Eliminate A[i,k] for i>k  
- Back-substitution to find solution vector x  

```python
for k in range(n-1):
    s = [max(abs(A[i,:])) for i in range(n)]
    ratios = [abs(A[i,k])/s[i] for i in range(k,n)]
    pivot = k + int(np.argmax(ratios))
    if pivot != k:
        A[[k,pivot]] = A[[pivot,k]]
        b[k], b[pivot] = b[pivot], b[k]
    for i in range(k+1,n):
        m = A[i,k]/A[k,k]
        A[i,k:] -= m * A[k,k:]
# Back-substitution...
```  
""", unsafe_allow_html=True)
        # Phase 3: Step-by-Step Summary
        origA3 = A3.copy()
        x3, steps3 = scaled_partial_pivot_gauss(A3.copy(), b3.copy(), return_steps=True)
        st.subheader('Step-by-Step Summary')
        records3 = []
        for idx, step in enumerate(steps3, start=1):
            records3.append({
                'Step #': idx,
                'Type': step['step'],
                'k': step.get('k', ''),
                'i': step.get('i', ''),
                'pivot_row': step.get('pivot_row', ''),
                'multiplier': step.get('multiplier', ''),
                'ratio': step.get('ratio', ''),
                'value': step.get('value', '')
            })
        df3 = pd.DataFrame(records3)
        st.table(df3)
        # Phase 4: Intermediate Matrices
        st.subheader('Intermediate Matrices')
        n3 = origA3.shape[0]
        for k in range(n3 - 1):
            pivot_step = next(s for s in steps3 if s['step'] in ('pivot', 'swap') and s['k'] == k)
            elim_steps = [s for s in steps3 if s['step'] == 'elimination' and s['k'] == k]
            last_elim = elim_steps[-1] if elim_steps else None
            cols = st.columns(3)
            cols[0].subheader(f'Original Matrix (k={k})')
            cols[0].write(pd.DataFrame(origA3, columns=[f"x{j}" for j in range(n3)]))
            cols[1].subheader(f'After Pivot (k={k})')
            cols[1].write(pd.DataFrame(pivot_step['A'], columns=[f"x{j}" for j in range(n3)]))
            if last_elim:
                cols[2].subheader(f'After Elimination (k={k})')
                cols[2].write(pd.DataFrame(last_elim['A'], columns=[f"x{j}" for j in range(n3)]))
            else:
                cols[2].write('No elimination steps')
        # Phase 5: Core Solver Code
        source = inspect.getsource(scaled_partial_pivot_gauss)
        st.subheader('Core Solver Code')
        st.code(source, language='python')
        # Phase 6: Performance Metrics
        n3 = A3.shape[0]
        start = time.perf_counter()
        _ = scaled_partial_pivot_gauss(A3.copy(), b3.copy(), return_steps=False)
        elapsed = time.perf_counter() - start
        flops3 = int((2/3) * n3**3)
        st.subheader('Performance Metrics')
        st.write(f"Execution Time: {elapsed:.6f} seconds")
        st.write(f"Estimated Floating-point Operations: {flops3:,}")
        render_example(A3, b3, title='3×3 Example')
    elif page == '4×4 Example':
        A4 = np.array([
            [3.0, -13.0, 9.0, 3.0],
            [-6.0, 4.0, 1.0, -18.0],
            [6.0, -2.0, 2.0, 4.0],
            [12.0, -8.0, 6.0, 10.0]
        ])
        b4 = np.array([-19.0, -34.0, 16.0, 26.0])
        # Phase 1: Metadata & Problem Statement
        st.title('4×4 Example Report')
        st.markdown("""
**John Akujobi**  
MATH 374: Scientific Computation (Spring 2025), South Dakota State University  
Professor: Dr Kimn, Dept. of Math & Statistics  
GitHub: [jakujobi](https://github.com/jakujobi)
""", unsafe_allow_html=True)
        st.subheader('Problem Statement')
        st.markdown('Solve the linear system **A x = b**, where:')
        st.code(str(A4), language='text')
        st.code(str(b4), language='text')
        # Phase 2: Algorithm Overview
        st.subheader('Algorithm Overview')
        st.markdown("""
- Compute scale factors s[i] = max_j |A[i,j]|  
- For each column k:
  1. Compute ratio |A[i,k]|/s[i] for i=k..n-1  
  2. Select pivot row with max ratio, swap if needed  
  3. Eliminate A[i,k] for i>k  
- Back-substitution to find solution vector x  

```python
for k in range(n-1):
    s = [max(abs(A[i,:])) for i in range(n)]
    ratios = [abs(A[i,k])/s[i] for i in range(k,n)]
    pivot = k + int(np.argmax(ratios))
    if pivot != k:
        A[[k,pivot]] = A[[pivot,k]]
        b[k], b[pivot] = b[pivot], b[k]
    for i in range(k+1,n):
        m = A[i,k]/A[k,k]
        A[i,k:] -= m * A[k,k:]
# Back-substitution...
```  
""", unsafe_allow_html=True)
        # Phase 3: Step-by-Step Summary
        origA4 = A4.copy()
        x4, steps4 = scaled_partial_pivot_gauss(A4.copy(), b4.copy(), return_steps=True)
        st.subheader('Step-by-Step Summary')
        records4 = []
        for idx, step in enumerate(steps4, start=1):
            records4.append({
                'Step #': idx,
                'Type': step['step'],
                'k': step.get('k', ''),
                'i': step.get('i', ''),
                'pivot_row': step.get('pivot_row', ''),
                'multiplier': step.get('multiplier', ''),
                'ratio': step.get('ratio', ''),
                'value': step.get('value', '')
            })
        df4 = pd.DataFrame(records4)
        st.table(df4)
        # Phase 4: Intermediate Matrices
        st.subheader('Intermediate Matrices')
        n4 = origA4.shape[0]
        for k in range(n4 - 1):
            pivot_step = next(s for s in steps4 if s['step'] in ('pivot', 'swap') and s['k'] == k)
            elim_steps = [s for s in steps4 if s['step'] == 'elimination' and s['k'] == k]
            last_elim = elim_steps[-1] if elim_steps else None
            cols = st.columns(3)
            cols[0].subheader(f'Original Matrix (k={k})')
            cols[0].write(pd.DataFrame(origA4, columns=[f"x{j}" for j in range(n4)]))
            cols[1].subheader(f'After Pivot (k={k})')
            cols[1].write(pd.DataFrame(pivot_step['A'], columns=[f"x{j}" for j in range(n4)]))
            if last_elim:
                cols[2].subheader(f'After Elimination (k={k})')
                cols[2].write(pd.DataFrame(last_elim['A'], columns=[f"x{j}" for j in range(n4)]))
            else:
                cols[2].write('No elimination steps')
        # Phase 5: Core Solver Code
        source = inspect.getsource(scaled_partial_pivot_gauss)
        st.subheader('Core Solver Code')
        st.code(source, language='python')
        # Phase 6: Performance Metrics
        n4 = A4.shape[0]
        start = time.perf_counter()
        _ = scaled_partial_pivot_gauss(A4.copy(), b4.copy(), return_steps=False)
        elapsed = time.perf_counter() - start
        flops4 = int((2/3) * n4**3)
        st.subheader('Performance Metrics')
        st.write(f"Execution Time: {elapsed:.6f} seconds")
        st.write(f"Estimated Floating-point Operations: {flops4:,}")
        render_example(A4, b4, title='4×4 Example (Solution: 3, 1, -2, 1)')
    else:
        render_playground()