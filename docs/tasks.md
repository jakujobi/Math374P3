# Test-First Task List for Math374P3

Before writing implementation code, ensure each feature passes its corresponding test.

1. compute_scales
   - Implement `compute_scales(A)` to return the max absolute value of each row
   - Confirm `test_compute_scales_simple` passes

2. select_pivot
   - Implement `select_pivot(A, scales, k)` using scaled ratios
   - Confirm `test_select_pivot_scaled` passes

3. eliminate_step & back_substitution
   - Implement `eliminate_step(A, b, pivot_row, k)` for row swapping and elimination
   - Implement `back_substitution(U, f)` to solve upper-triangular system
   - Confirm `test_eliminate_and_back_sub` passes

4. gauss_scaled orchestration
   - Combine routines into `gauss_scaled(A, b)` that collects intermediate matrices
   - Confirm `test_gauss_scaled_solution` passes

5. Visualization utilities
   - Implement `print_matrix` and `plot_matrix` in `Modules/visualization.py`
   - (Optional) Add tests in `tests/test_visualization.py`

6. Report generation
   - Implement `generate_report(intermediates, solution, filepath)` in `Modules/report.py`
   - (Optional) Test report output structure

7. Integration & docs
   - Update `streamlit_app.py` for interactive walkthrough (optional)
   - Finalize documentation in `docs/Project3.md`, `docs/requirements.md`, `docs/tasks.md`

8. Final validation
   - Ensure all tests pass:
     ```bash
     pytest --maxfail=1 --disable-warnings -q
     ```
