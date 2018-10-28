# Acoustic Vibration of a Fluid in a Three-Dimensional Cavity: Finite Element Method Simulation using CUDA and MATLAB

An implementation of a FEM acoustic model on a GPU using C/C++ and CUDA libraries.

[![Article](https://img.shields.io/badge/ieee-article-blue.svg)](https://www.google.com) 

## Quick steps
Just do `make`:

```
  cd CUDA
  make
  ./fem_solver A B C
 ```
 
 A: Number of nodes n in grid `n x n x n`
 
 B: Method type. `0` for Conquer and Divide, `1` for Jacobi. 
 
 C: Precision type. `0` for Single, `1` for Double. 

## Notes
CUDA 9.2 is needed(also cuSolver)

NVIDIA Tesla Titan X, P100 and V100 were tested.
