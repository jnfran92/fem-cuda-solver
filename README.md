# Acoustic Vibration of a Fluid in a Three-Dimensional Cavity: Finite Element Method Simulation using CUDA and MATLAB

This work describes an implementation of a FEM acoustic application on a GPU using C/C++ and CUDA libraries. The acoustic model is a rigid-walled cavity with enclosed fluid and rectangular faces. Three-dimensional acoustic elements are used to model the geometric form of the cavity. Natural frequencies were computed using inertia and stiffness matrices in a general eigenvalue problem. These matrices are symmetric, dense and grow in a cubic ratio from the number of divisions in the grid. The model was implemented using cuSOLVER libraries to solve the eigenvalue problem in single and double precision. The MATLAB implementation was performed for CPU in order to compare the results of GPU implementation. The GPU-based Jacobi method in single precision gives the best results, this method is five times faster than the MATLAB implementation. The divide and conquer method in double precision for GPU is the most accurate implementation when comparing with the exact solution of the model. Lastly, the sound pressure distribution in the cavity was graphed using eigenvectors.

**Article Link**

## Methods
### FEM Model
The acoustic model is a rectangular rigid-walled cavity with dimensions `Lx × Ly × Lz = 0.414m × 0.314m × 0.360m` filled with air, for which `ρ0 = 1.21kg m−3` and the speed of sound `c = 342ms−1`.

### Simulation Procedure
The CUDA and MATLAB implementations execute the following procedure:
1) Inertia and stiffness matrices are created based on the cavity features. These matrices give the behavior of a single acoustic element.
2) According to the number of the chosen elements, the global matrix assembly process is computed using iner- tia and stiffness matrices for each element.
3) For CUDA, Divide an conquer (cusolverDn<t>sygvd()) and (cusolverDn<t>sygvj()) method are applied to solve the global matrices of the model (Jacobi method with tolerance value: 1e−3 and the maximum number of sweeps: 15).
4) For MATLAB (CPU), eig() function is used (QR method) to solve the global matrices.
  
## Performance Tests
The rectangular cavity is divided into regular elements, thus the number of nodes corresponds to the number of divisions in any side plus one raised to the cube. For example: if the `number of divisions is 3` the global matrices are of size `n×n` where `n = (number of divisions +1)3 = 43 = 64 (grid size of 4×4×4)`.<br>

Computation time was measured five times for each test (an individual test was performed for each method in single and double precision). Results are shown bellow:


<b>*</b> **CPU:** MacBook Pro (early 2015) with a 2.9GHz Intel Core i5 processor and 8 GB 1867 MHz DDR3 RAM memory.<br> 
<b>**</b> **GPU:** CUDA libraries running on a `TITAN X (Pascal) GPU`.<br>

### Single precision implementation:

| Matrix Size n | MATLAB<sup>*</sup> | Divide and Conquer<sup>**</sup> | Jacobi<sup>**</sup> |
| --- | --- |  --- |  --- | 
|64 | 0.002 | 0.004 | 0.001|
|512 | 0.022 | 0.078 | 0.015|
|1728 | 0.657 | 1.101 |0.259|
|4096 | 8.423 | 11.399 | 2.088|
|8000 | 56.729 | 91.792 | 11.519|

### Double precision implementation:

| Matrix Size n | MATLAB<sup>*</sup> | Divide and Conquer<sup>**</sup> | Jacobi<sup>**</sup> |
| --- | --- |  --- |  --- | 
| 64 |0.002 |0.007 |0.003|
|512 |0.044 |0.118 |0.09|
|1728 |1.103 |1.615 |1.765|
|4096 |15.433 |19.780 |19.566|
|8000 |122.548 |167.761 |127.929|







