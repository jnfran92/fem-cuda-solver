

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

//#include "acoustic_matrices.h"



long idx(long i, long j, long s) {
	return (i * s + j);
}

long dof_n(long i, long j, long k, long n) {
	return i + n * (j) + n * n * (k);
}


void get_indexes(long i, long j, long k, long n, int n_dofs, long* out) {
	out[0] = dof_n(i, j, k, n);
	out[1] = dof_n(i + 1, j, k, n);
	out[2] = dof_n(i + 1, j + 1, k, n);
	out[3] = dof_n(i, j + 1, k, n);
	out[4] = dof_n(i, j, k + 1, n);
	out[5] = dof_n(i + 1, j, k + 1, n);
	out[6] = dof_n(i + 1, j + 1, k + 1, n);
	out[7] = dof_n(i, j + 1, k + 1, n);
}

void create_fem_matrix_double(double* H, double*Q, double* Hg, double* Qg,  int n_dofs, long n){

	long e = n - 1; // elements number
	long n_total = n * n * n;
	long* indexes = new long[n_dofs];

	long i, j, k;
	int x, y;

	for (i = 0; i < e; i++) {
		for (j = 0; j < e; j++) {
			for (k = 0; k < e; k++) {
				get_indexes(i, j, k, n, n_dofs, indexes);
			for (x = 0; x < 8; x++) {
					for (y = 0; y < 8; y++) {
						//						printf("%ld  %ld\n",idx(indexes[x],indexes[y],n_total), idx(x,y,n_dofs));
						Hg[idx(indexes[x], indexes[y], n_total)] += H[idx(x, y,
								n_dofs)];
						Qg[idx(indexes[x], indexes[y], n_total)] += Q[idx(x, y,
								n_dofs)];
					}
				}
			}
		}
	}
}


void create_fem_matrix_float(float* H, float*Q, float* Hg, float* Qg,  int n_dofs, long n){


	long e = n - 1; // elements number
	long n_total = n * n * n;
	long* indexes = new long[n_dofs];

	long i, j, k;
	int x, y;

	for (i = 0; i < e; i++) {
		for (j = 0; j < e; j++) {
			for (k = 0; k < e; k++) {
				get_indexes(i, j, k, n, n_dofs, indexes);
				for (x = 0; x < 8; x++) {
					for (y = 0; y < 8; y++) {
						//						printf("%ld  %ld\n",idx(indexes[x],indexes[y],n_total), idx(x,y,n_dofs));
						Hg[idx(indexes[x], indexes[y], n_total)] += H[idx(x, y,
								n_dofs)];
						Qg[idx(indexes[x], indexes[y], n_total)] += Q[idx(x, y,
								n_dofs)];
					}
				}
			}
		}
	}
}


