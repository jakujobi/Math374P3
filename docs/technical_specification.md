# Comprehensive Technical Specification

## 1 Introduction

* ****Project title:** MATH 374 Project 3: Gaussian Elimination (Streamlit App)**
* **Author:** John Akujobi, Computer Science student at South Dakota State University
* **GitHub:**[jakujobi](https://github.com/jakujobi)
* **Affiliation:** Jerome J. Lohr College of Engineering, Department of Mathematics and Statistics, SDSU
* **Supervisor:** Dr. Kimn
* **Course:** CSC 374 Scientific Computation
* **Purpose:** Deliver an interactive and static demonstrator of Gaussian elimination with scaled partial pivoting, showing every internal arithmetic ste.
* **Motivation:** Reliable linear solvers underpin machine learning and scientific computing. We learned partial pivoting in class and scaled partial pivoting further guards against numerical instability by normalizing row scales—*like choosing the clearest voice in a noisy crowd*.

## 2 Stakeholder & Audience

* **Primary graders:** Instructors assessing numerical-analysis or scientific computing assignments.
* **Secondary reviewers:** Employers evaluating my GitHub portfolio at `github.com/jakujobi` for coding and documentation quality.
* **End-users:** Me and other students exploring Gaussian elimination visually.

## 3 Goals & Objectives

* **Correctness:** Need to produce the exact solution and match `<span>numpy.linalg.solve</span>` within a tolerance of $10^{-12}$.
* **Transparency:** Expose every arithmetic update (row swaps, multipliers, row eliminations, back-substitution) and intermediate steps with concise comments. Dr. Kimn stressed how important this during class.
* **Interactivity:** Provide both fixed example walkthroughs. Create a user-driven playground for 3×3 and 4×4 systems (this is an optional self driven enhancement)
* **Reusability:** Offer a modular solver (`<span>src/gauss_sp.py</span>`) and a self-contained Streamlit app (`<span>streamlit_app.py</span>`).
* **Exportability:** Allow users to download full reports including all steps and discussions in Markdown and PDF formats. This makes it easier for Dr.Kimn to grade the report while his TA reviews the source code.
* **Testing:** Taking from what I learnt in SE306 Software Engineering class, this project includes a pytest suite ensuring solver correctness for example systems.

## 4 Scope

### In Scope

* Dense systems of size $n\le4$ solved with scaled partial pivoting (row swaps only).
* A Streamlit application with two pages: fixed walkthrough and interactive playground.
* Export functionality for Markdown and PDF.
* Python module encapsulating the solver logic in `<span>src/gauss_sp.py</span>`, plus automated tests.

### Out of Scope

* Complete pivoting or LU decompositions.
* Sparse or large ($n>4$) systems.
* Performance benchmarking or GPU acceleration. (Lets look at this in the future since the class's aim is to learn how to efficiently do computations on computers)

## 5 Functional Requirements

1. **R1 – Solver module:**`<span>scaled_partial_pivot_gauss(A, b, return_steps=False)</span>` in `<span>src/gauss_sp.py</span>` returns solution and, if requested, detailed step logs including `<span>A_k</span>`, `<span>b_k</span>`, and metadata (`<span>pivot_row</span>`, `<span>multiplier</span>`, `<span>swap</span>`).
2. **R2 – Streamlit UI:** Two pages (“Walkthrough” and “Playground”) selectable via sidebar.
3. **R3 – Fixed Walkthrough:** Sequentially display every step for a predetermined 4×4 example and a 3×3 example, with inline comments under each step.
4. **R4 – Interactive Playground:** Allow entry or randomization of a 3×3 or 4×4 system; upon request, compute and display all steps.
5. **R5 – Highlighting:** CSS-based row highlighting for pivot and target rows in each displayed matrix.
6. **R6 – Export:** Buttons for downloading the full report (all steps, notes, discussions) as Markdown (`<span>.md</span>`) and as PDF (`<span>.pdf</span>`).
7. **R7 – Testing:** Pytest tests covering the 4×4 and 3×3 examples, verifying solution accuracy and residual norms.

## 6 Non-Functional Requirements

* **NFR-1 – Technologies:** Python ≥3.10, Streamlit, NumPy, pytest, and a PDF library (`<span>pdfkit</span>` or `<span>weasyprint</span>`).
* **NFR-2 – Style:** Use default Streamlit theme; custom CSS in `<span>assets/custom.css</span>` for highlighting and animations.
* **NFR-3 – Licensing:** MIT license for all code to facilitate portfolio sharing.
* **NFR-4 – Repository:** Public GitHub repo, with clear `<span>.gitignore</span>` and CI support for tests.

## 7 Algorithmic Design

1. Compute scaling factors $s_i = \max_j |a_{ij}|$.
2. For each elimination column $k$: select pivot row $p\ge k$ maximizing $|a_{pk}|/s_p$.
3. Swap row $k$ with row $p$ if needed; log the swap.
4. For each row $i>k$, compute multiplier $m = a_{ik}/a_{kk}$, update $a_{i,k:}\leftarrow a_{i,k:} - m,a_{k,k:}$ and $b_i\leftarrow b_i - m,b_k$; log each update.
5. Perform back-substitution from $i=n-1$ down to $0$, logging each solved variable.

## 8 Software Architecture

* **Modules:**
  * `<span>src/gauss_sp.py</span>`: core solver and metadata logging.
  * `<span>streamlit_app.py</span>`: Streamlit app orchestrating UI, solver calls, highlighting, and exports.
  * `<span>assets/custom.css</span>`: styles for row highlighting and fade-in animations.
* **Helpers:**
  * `<span>render_example()</span>`, `<span>matrix_input_grid()</span>`, `<span>highlight_and_show()</span>`, `<span>format_info()</span>`, `<span>export_markdown()</span>`, `<span>export_pdf()</span>`.

## 9 Directory Layout

```
Math374P3/
├─ streamlit_app.py
├─ src/
│   └─ gauss_sp.py
├─ assets/
│   └─ custom.css
├─ tests/
│   └─ test_gauss_sp.py
├─ requirements.txt
├─ streamlit.toml         # Streamlit page config
├─ pyproject.toml         # (or setup.py) for packaging (learned from Compiler Construction Class)
├─ .gitignore
├─ reports/               # Generated Markdown and PDF reports
│   ├─ Gaussian Elimination Solver 3x3.pdf
│   ├─ Gaussian Elimination Solver 4x4.pdf
│   ├─ Project 3 - 3x3 Matrix.md
│   ├─ Project 3 - 3x3 Matrix.pdf
│   ├─ Project 3 - 4x4 matrix.md
│   ├─ Project 3 - 4x4 matrix.pdf
│   ├─ Screenshot - Project 3 - 3x3 Matrix.pdf
│   └─ Screenshot - Project 3 - 4x4 Matrix.pdf
└─ README.md
```

## 10 User Interface & Visualization

* **Linear top-to-bottom flow:** each section (title, problem statement, steps, discussion) appears in sequence.
* **Step containers:**`<span>st.container</span>` or `<span>st.subheader</span>` per step, followed by `<span>st.dataframe</span>` and a concise `<span>st.markdown</span>` note.
* **CSS highlighting:** pivot and target rows colored via custom CSS variables and fade-in keyframe animations.
* **Export UI:** sidebar buttons triggering downloads of full Markdown and PDF reports including every step and discussion.

## 11 Testing Strategy

* **Unit tests** in `<span>tests/test_gauss_sp.py</span>`:
  * 4×4 example matches NumPy and residual < 1e-11.
  * 3×3 example matches NumPy and residual < 1e-11.
* **Continuous integration:** GitHub Actions to run pytest on each push.

## 12 Documentation & Reporting

* **README.md**: Setup instructions, usage examples for Streamlit app.
* **Inline comments**: Every arithmetic operation in `<span>src/gauss_sp.py</span>` documented.
* **App exports**: Fully self-contained report outputs for offline grading and portfolio display.

## 13 Tools & Technologies

| Tool         | Purpose                    |
| ------------ | -------------------------- |
| Python 3.10+ | Core language              |
| Streamlit    | Web UI framework           |
| NumPy        | Numerical computations     |
| pytest       | Unit testing               |
| pdfkit/Weasy | Markdown to PDF conversion |
| Git/GitHub   | Version control & hosting  |

## 14 Installation & Usage

```bash
# Clone the repository
git clone https://github.com/jakujobi/Math374P3.git
cd Math374P3

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

## 15 Deliverables

* `<span>streamlit_app.py</span>` – Streamlit application.
* `<span>src/gauss_sp.py</span>` – solver module with detailed logging.
* `<span>assets/custom.css</span>` – Highlighting and animation styles.
* `<span>tests/test_gauss_sp.py</span>` – Pytest suite.
* README, `<span>.gitignore</span>`, `<span>requirements.txt</span>`, and packaging config.

## 16 Timeline (Suggested)

| Week | Milestone                                               |
| ---- | ------------------------------------------------------- |
| 1    | Setup repo, implement solver, write tests               |
| 2    | Build streamlit_app.py structure, render fixed examples |
| 3    | Implement CSS highlighting, export functions            |
| 4    | User testing, CI setup, polishing documentation         |

## 17 Future Enhancements

* Add pivot-strategy toggle (standard vs. scaled vs. none).
* Extend to larger matrices with performance optimizations.
* Deploy via Streamlit Cloud or Docker container.

## 18 References & Resources

* Burden & Faires, *Numerical Analysis* (Ch. 2).
* MIT 18.06 Linear Algebra lectures (YouTube).
* Jake VanderPlas, *Python Data Science Handbook* (NumPy).
* Cheney & Kincaid, *Numerical Mathematics and Computing*, 7th Edition
* Debugging assistance from Qwen 3 locally run

---

*End of specification*
