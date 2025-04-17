# MATH 373 Project 3 — *100 points*

**Due:** `10:50 AM, May 9th, 2025`

Please submit your:

* Report
* Source code
* Environment information (so the TA can run your code)

---

## Problem Description

Consider the linear system of equations:

$$
\begin{pmatrix}
3 & -13 & 9 & 3 \\
-6 & 4 & 1 & -18 \\
6 & -2 & 2 & 4 \\
12 & -8 & 6 & 10
\end{pmatrix}
\begin{pmatrix}
x_1 \\
x_2 \\
x_3 \\
x_4
\end{pmatrix}
=
\begin{pmatrix}
-19 \\
-34 \\
16 \\
26
\end{pmatrix}
$$

---

## Tasks

1. Solve the system using  **Gaussian elimination with scaled partial pivoting** .
2. Show **intermediate matrices** at each step of the elimination process.

---

## Project Structure

- **Modules/**: core solver (`gauss_scaled.py`), visualization, report utilities
- **tests/**: pytest suite verifying each function and end-to-end solver
- **docs/**:
  - `Project3.md` (this document)
  - `requirements.md` (specifications & dependencies)
  - `tasks.md` (test-first plan)
- `requirements.txt`: Python dependencies
- `README.md`: project overview & usage
- `streamlit_app.py`: optional interactive demo

## Algorithm & Pseudocode

Scaled Partial Pivot Gaussian Elimination:

```text
# Input: A ∈ ℝⁿˣⁿ, b ∈ ℝⁿ
compute scales[i] = max_j |A[i,j]| for each row i
for k in 0..n-2:
  pivot_i = argmax_{i=k..n-1} (|A[i,k]| / scales[i])
  swap rows A[k] ↔ A[pivot_i]; swap b[k] ↔ b[pivot_i]
  for i in k+1..n-1:
    factor = A[i,k] / A[k,k]
    A[i,k..n-1] -= factor * A[k,k..n-1]
    b[i]      -= factor * b[k]
# U = A (upper‐triangular)
# back_substitution to solve Ux = b
```

## Dependencies & Setup

- Python 3.8+
- `numpy >=1.24.0`
- `pandas >=2.0.0`
- `matplotlib >=3.7.0`
- `pytest >=7.0.0`
- (Optional) `streamlit >=1.24.1`

Setup:

```bash
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Usage

```bash
# Run solver module directly
python -m Modules.gauss_scaled

# For interactive walkthrough
streamlit run streamlit_app.py
```

## Testing

Run all tests:

```bash
pytest --maxfail=1 --disable-warnings -q
```

## Timeline

- **Apr 17–20**: finalize docs & scaffolding
- **Apr 21–24**: implement compute_scales & select_pivot
- **Apr 25–28**: implement elimination & back_substitution
- **Apr 29–May 1**: full solver & tests
- **May 2–4**: visualization & report generator
- **May 5–7**: optional UI & polishing
- **May 8**: final review & packaging
- **May 9, 10:50 AM**: submission

## Deliverables

- `Modules/gauss_scaled.py`
- `Modules/visualization.py`
- `Modules/report.py`
- `tests/test_gauss_scaled.py`
- `docs/Project3.md`
- `docs/requirements.md`
- `docs/tasks.md`
- `requirements.txt`
- `README.md`
- `streamlit_app.py`
