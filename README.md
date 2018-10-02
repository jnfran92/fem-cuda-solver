# Acoustic Vibration of a Fluid in a Three-Dimensional Cavity: Finite Element Method Simulation using CUDA and MATLAB

This work describes an implementation of a FEM acoustic application on a GPU using C/C++ and CUDA libraries.[![Article](https://img.shields.io/badge/conference-article-blue.svg)](https://www.google.com) 

## Methods
### FEM Model
The acoustic model is a rectangular rigid-walled cavity with dimensions `Lx × Ly × Lz = 0.414m × 0.314m × 0.360m` filled with air, for which `ρ0 = 1.21kg m−3` and the speed of sound `c = 342ms−1`.

### Simulation Procedure
The CUDA and MATLAB implementations execute the following procedure:
1) Inertia and stiffness matrices are created based on the cavity features. These matrices give the behavior of a single acoustic element.
2) According to the number of the chosen elements, the global matrix assembly process is computed using inertia and stiffness matrices for each element.
3) For CUDA,` Divide an conquer (cusolverDn<t>sygvd()) and (cusolverDn<t>sygvj())` method are applied to solve the global matrices of the model (Jacobi method with tolerance value: 1e−3 and the maximum number of sweeps: 15).
4) For MATLAB (CPU), `eig() function` is used (`QR method`) to solve the global matrices.
  
## Performance Test
The rectangular cavity is divided into regular elements, thus the number of nodes corresponds to the number of divisions in any side plus one raised to the cube. For example: if the `number of divisions is 3` the global matrices are of size `n×n` where `n = (number of divisions +1)3 = 43 = 64 (grid size of 4×4×4)`.<br>

Tests were performed with different values of n in ascending order: 64, 512, 1728, 4096 and 8000. Computation time was measured five times for each test (an individual test was performed for each method in single and double precision). Results are shown bellow:


<b>*</b> **CPU:** MacBook Pro (early 2015) with a 2.9GHz Intel Core i5 processor and 8 GB 1867 MHz DDR3 RAM memory.<br> 
<b>**</b> **GPU:** CUDA libraries running on a `TITAN X (Pascal) GPU`.<br>

### Single precision implementation:
Execution Mean Time in Seconds

| Matrix Size n | MATLAB<sup>*</sup> | Divide and Conquer<sup>**</sup> | Jacobi<sup>**</sup> |
| --- | --- |  --- |  --- | 
|64 | 0.002 | 0.004 | 0.001|
|512 | 0.022 | 0.078 | 0.015|
|1728 | 0.657 | 1.101 |0.259|
|4096 | 8.423 | 11.399 | 2.088|
|8000 | 56.729 | 91.792 | 11.519|

### Double precision implementation:
Execution Mean Time in Seconds

| Matrix Size n | MATLAB<sup>*</sup> | Divide and Conquer<sup>**</sup> | Jacobi<sup>**</sup> |
| --- | --- |  --- |  --- | 
| 64 |0.002 |0.007 |0.003|
|512 |0.044 |0.118 |0.09|
|1728 |1.103 |1.615 |1.765|
|4096 |15.433 |19.780 |19.566|
|8000 |122.548 |167.761 |127.929|

## Accuracy Test
`Eigenvalues` were computed with a high number of nodes `n = 8000`, in order to improve the accuracy. Test was performed for each method in single and double precision. `Mean absolute error` was calculated using the first fifteen modes when comparing with the exact solution. Results are shown bellow:

### Accuracy test for single and double precision
Mean Absolute Error

| Method | Single | Double |
| --- | --- |  --- |
| MATLAB|2.356 |1.356 |
|Divide and Conquer |1.306 |1.304 |
|Jacobi |1.787 |1.731 |

## Eigenvectors distribution
Using `GPU-only implementation in single precision`, eigenvectors were computed with the highest number of nodes allowed by the device memory capacity: `12GB` (n = 24389, grid size = 29 × 29 × 29).`Computation time was roughly four minutes`. Arbitrary eigenvectors were used to graph the sound pressure distribution for low and high frequencies in the cavity and they are shown bellow:

![Acoustic FEM Model GPU](https://i.imgur.com/8tDzzqq.png)


## Final notes
- In the best case, execution time of Dense Matrix Solver was shortened from 56.73 seconds (CPU: MATLAB) to 11.52 seconds (GPU: Jacobi method) with n = 8000 and single precision. 
- Jacobi method in single precision is the fastest method (almost five times faster) to solve the eigenvalue problem. 
- Divide and conquer method finds the most accurate solution. As a result it has the lowest mean absolute error.


