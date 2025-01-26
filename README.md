# Dissertation

**Solving Heston's Partial Differential Equation by Finite Difference Methods**

---

# Code

## Introduction

This program provides an implementation of finite difference methods to price European call options under the Heston stochastic volatility model by solving its partial differential equation (PDE). The methodology includes:

- Discretising the PDE over **non-uniform grids** for better accuracy.
- Leveraging the **ubiquitous Kronecker product** to optimise computations.
- Applying **Alternating Direction Implicit (ADI)** time-stepping schemes for stability and efficiency.

Additionally, the program contains functions to generate plots for results, the Greeks and error analysis.

---

## How to Use My Program

### Prerequisites:
- **Python Version**: Recommend using Python 3.9
- **Development Environment**: PyCharm Community Edition or any Python IDE.

### Steps to Run:
1. Extract the zip file containing the code.
2. Install the necessary dependencies listed in the `import` section of the file.
3. Open the `FDMheston.py` file in your IDE.

---

### Program Options:
Inside `FDMheston.py`, you can perform the following tasks:

1. **Run the Heston PDE Solver**:
   - Execute the function `FDMheston()` with default starting parameters.
   - Customise the arguments to adjust the Heston model parameters.
   - Update the initial stock price (`S0`) and variance (`v0`) by modifying:
     ```python
     S_current = ...
     V_current = ...
     ```

2. **Enable Plotting**:
   - Activate surface plotting by setting:
     ```python
     plot_all = True
     ```
   - Plot discretisation errors:
     - Spatial error: `plot_spatial = True`
     - Temporal error: `plot_temporal = True`

3. **Refine Time-Stepping Schemes**:
   - Enable damping for the **Douglas scheme**:
     ```python
     damping_Do = True
     ```
   - Enable damping for the **Craig-Sneyd scheme**:
     ```python
     damping_CS = True
     ```

4. **Measure Performance**:
   - Uncomment the following lines to print computation times:
     ```python
     # print("Computation Time (ADI loop):", time)
     # print("Computation Time 2 (A matrix):", time2)
     # print("Computation Time (All):", time4)
     ```

5. **Visualise Sparse Matrices**:
   - Uncomment the following lines to plot the structure of the `A` matrices:
     ```python
     # plot_sparse_matrix(A_0, 'A_0')
     # plot_sparse_matrix(A_1, "A_1")
     # plot_sparse_matrix(A_2, "A_2")
     # plot_sparse_matrix(A, "A")
     ```

---

## Summary

This program serves as a robust tool for exploring finite difference methods in derivative pricing, providing valuable insights into efficient solutions for multi-dimensional PDE solvers.
