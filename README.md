# Acoustic Vibration of a Fluid in a Three-Dimensional Cavity: Finite Element Method Simulation using CUDA and MATLAB

This work describes an implementation of a FEM acoustic application on a GPU using C/C++ and CUDA libraries. The acoustic model is a rigid-walled cavity with enclosed fluid and rectangular faces. Three-dimensional acoustic elements are used to model the geometric form of the cavity. Natural frequencies were computed using inertia and stiffness matrices in a general eigenvalue problem. These matrices are symmetric, dense and grow in a cubic ratio from the number of divisions in the grid. The model was implemented using cuSOLVER libraries to solve the eigenvalue problem in single and double precision. The MATLAB implementation was performed for CPU in order to compare the results of GPU implementation. The GPU-based Jacobi method in single precision gives the best results, this method is five times faster than the MATLAB implementation. The divide and conquer method in double precision for GPU is the most accurate implementation when comparing with the exact solution of the model. Lastly, the sound pressure distribution in the cavity was graphed using eigenvectors.

**Article Link**

## Results:

**CPU:** MacBook Pro (early 2015) with a 2.9GHz Intel Core i5 processor and 8 GB 1867 MHz DDR3 RAM memory.<br>
**GPU:** CUDA libraries running on a TITAN X (Pascal) GPU.<br>

Single precision implementation (time in seconds):

| Matrix Size n | MATLAB<sup>*</sup> | Divide and Conquer<sup>**</sup> | Jacobi<sup>**</sup> |
| --- | --- |  --- |  --- | 
|64 | 0.002 | 0.004 | 0.001|
|512 | 0.022 | 0.078 | 0.015|
|1728 | 0.657 | 1.101 |0.259|
|4096 | 8.423 | 11.399 | 2.088|
|8000 | 56.729 | 91.792 | 11.519|

<small>* CPU, ** GPU </small>







Matrix Size n
Execution Mean Time in Seconds
MATLAB
Divide and Conquer
Jacobi
|64 | 0.002 | 0.004 | 0.001|
|512 | 0.022 | 0.078 | 0.015|
|1728 | 0.657 | 1.101 |0.259|
|4096 | 8.423 | 11.399 | 2.088|
|8000 | 56.729 | 91.792 | 11.519|
