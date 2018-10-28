#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "eigen_solver.h"
#include "acoustic_matrices/acoustic_matrices_float.h"
#include "acoustic_matrices/acoustic_matrices.h"
#include "fem_builder.h"

using namespace std;


int main(int argc, char **argv) {


	if (argc !=4){
		printf("Error args, [1]n, [2]method_type, [3]precision_type\n");
		return 0;
	}


	// ------------------------------------
	// Input Variables
	long n = atoi(argv[1]); // number of nodes by side
	// Divide and Conquer (0) or Jacobi(1)
	int method_type = atoi(argv[2]);
	// Single (0) or Double(1)
	int precision_type = atoi(argv[3]);
	// ------------------------------------


	int n_dofs = 8;
	int n_total = n * n * n;
	long n_total_g = n_total * n_total;



	cout << "--------------------------------------------------------------"<< endl;
	cout << "FEM CUDA SOLVER" << endl;
	cout << "Grid dim: " << n <<"x"<< n <<"x"<< n <<  endl;
	cout << "Total Nodes in Model: "<<n_total << endl;
	cout << "" << endl;


	if (precision_type == 0){
		cout <<"Single Precision" << endl;

		float* Hg1 = new float[n_total_g];
		float* Qg1 = new float[n_total_g];

		create_fem_matrix_float(H1, Q1, Hg1, Qg1, n_dofs, n);

		if (method_type == 0){

			cout <<"Divide and Conquer Method" << endl;
					cout << "" << endl;

			float_eig_solve(Hg1, Qg1, n_total);
		}else{

			cout <<"Jacobi Method" << endl;
					cout << "" << endl;

			float_eig_solve_jacobi(Hg1, Qg1,n_total);


		}

		delete[] Hg1;
		delete[] Qg1;

	}else{

		cout << "Double Precision" << endl;

		double* Hg = new double[n_total_g];
		double* Qg = new double[n_total_g];

		create_fem_matrix_double(H, Q, Hg, Qg, n_dofs, n);

		if (method_type == 0){

			cout << "Divide and Conquer Method" << endl;
					cout << "" << endl;

			double_eig_solve(Hg, Qg, n_total);
		}else{

			cout << "Jacobi Method" << endl;
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
