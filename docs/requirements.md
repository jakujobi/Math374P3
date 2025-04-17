# Project 3 Requirements

## 1. Functional Requirements
- Implement Gaussian elimination with **scaled partial pivoting**
- Display intermediate matrices at each elimination step
- Solve the specified 4×4 linear system and output the solution vector
- Provide a back‑substitution routine for final solution

## 2. Non‑Functional Requirements
- Use **Python 3.8+**
- Write clear, modular code with docstrings for every function
- Ensure outputs are formatted cleanly (console or file)
- Include **unit tests** via `pytest`
- (Optional) Provide an interactive step‑through via **Streamlit**

## 3. Dependencies
- `numpy >= 1.24.0`
- `pandas >= 2.0.0`
- `matplotlib >= 3.7.0`
- `pytest >= 7.0.0`
- (Optional) `streamlit >= 1.24.1`

## 4. Environment Setup
1. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 5. Deliverables
- `Modules/gauss_scaled.py` (core solver)
- `Modules/visualization.py` (matrix display, plotting)
- `Modules/report.py` (Markdown/LaTeX report generator)
- `tests/test_gauss_scaled.py`
- `docs/Project3.md` (project report)
- `docs/requirements.md` (this document)
- `requirements.txt`
- `README.md`

## 6. Testing Strategy
- Unit tests for each subroutine: scale computation, pivot selection, elimination, back‑sub
- Validate final solution against `numpy.linalg.solve`
- Test edge cases: zero or near‑zero pivots

## 7. Documentation
- Function-level docstrings
- Usage examples in `README.md` or report
- Inline comments for algorithm steps

## 8. Usage

- Run solver module:

  ```bash
  python -m Modules.gauss_scaled
  ```

- Interactive walkthrough (optional):

  ```bash
  streamlit run streamlit_app.py
  ```

## 9. Timeline

- **Apr 17–20**: finalize docs & scaffolding
- **Apr 21–24**: implement compute_scales & select_pivot
- **Apr 25–28**: implement elimination & back_substitution
- **Apr 29–May 1**: full solver & tests
- **May 2–4**: visualization & report
- **May 5–7**: optional UI & polish
- **May 8**: final review & packaging
- **May 9, 10:50 AM**: submission

## 10. Contact

- **Developer**: John Akujobi
- **Course**: MATH 374 – Computational Theory
- **Due**: May 9, 2025, 10:50 AM
