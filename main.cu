#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "eigen_solver.h"
#include "ElementMatrices/acoustic_matrices_float.h"
#include "ElementMatrices/acoustic_matrices.h"
#include "fem_builder.h"

using namespace std;


int main(int argc, char **argv) {


	// ------------------------------------
	// Input Variables
	long n = 4; // number of nodes by side
	// Divide and Conquer (0) or Jacobi(1)
	int method_type = 1;
	// Single (0) or Double(1)
	int precision_type = 0;
	// ------------------------------------


	int n_dofs = 8;
	long n_total = n * n * n;
	long n_total_g = n_total * n_total;



	cout << "--------------------------------------------------------------"<< endl;
	cout << "FEM CUDA SOLVER" << endl;
	cout << "	Number of DOFs per element: " << n_dofs << endl;
	cout << "	Nodes in any side: " << n << "  Total Nodes in Model: "<<n_total << endl;
	cout << "" << endl;


	if (precision_type == 0){

		cout << "                       Single Precision" << endl;
		cout << "" << endl;

		float* Hg1 = new float[n_total_g];
		float* Qg1 = new float[n_total_g];

		create_fem_matrix_float(H1, Q1, Hg1, Qg1, n_dofs, n);

		if (method_type == 0){

			cout << "	Divide and Conquer Method" << endl;
					cout << "" << endl;

			float_eig_solve(Hg1, Qg1, n_total);
		}else{

			cout << "	Jacobi Method" << endl;
					cout << "" << endl;

			float_eig_solve_jacobi(Hg1, Qg1,n_total);

//			float_eig_solve_jacobi_UMA(Hg1,Qg1,n_total);

		}

		delete[] Hg1;
		delete[] Qg1;

	}else{

		cout << "                       Double Precision" << endl;
		cout << "" << endl;

		double* Hg = new double[n_total_g];
		double* Qg = new double[n_total_g];

		create_fem_matrix_double(H, Q, Hg, Qg, n_dofs, n);

		cout << "Compute cuSolver Double" << endl;

		if (method_type == 0){

			cout << "	Divide and Conquer Method" << endl;
					cout << "" << endl;

			double_eig_solve(Hg, Qg, n_total);
		}else{

			cout << "	Jacobi Method" << endl;
					cout << "" << endl;

			double_eig_solve_jacobi(Hg, Qg, n_total);
		}

		delete[] Hg;
		delete[] Qg;
	}

	cout << "--------------------------------------------------------------"
			<< endl;



	return 0;
}
