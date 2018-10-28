
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <assert.h>

#include "csvfile.h"

using namespace std;



void write_csv_eigens_float(float *V, float *W, int lda ){


	long w_max = 15;

	try {
		csvfile csv("./data_out/eigenvalues.csv"); // throws exceptions!
		for (long i = 0; i < w_max; i++) {
			csv<< W[i];
			csv << endrow;
		}
	} catch (const std::exception &ex) {
		std::cout << "Exception was thrown: " << ex.what() << std::endl;
	}

	printf("First 15 Eigenvalues saved in: ./data_out/eigenvalues.csv \n");


	long v_max = 15;

	try {
		csvfile csv("./data_out/eigenvectors.csv"); // throws exceptions!
		for (long i = 0; i < lda; i++) {
			for (long j = 0; j < v_max; j++) {
				csv<< V[lda * j + i];
			}
			csv << endrow;
		}
	} catch (const std::exception &ex) {
		std::cout << "Exception was thrown: " << ex.what() << std::endl;
	}

	printf("First 15 Eigenvectors saved in: ./data_out/eigenvectors.csv \n");

}

void write_csv_eigens_double(double *V, double *W, int lda){

	long w_max = 15;

	try {
		csvfile csv("./CUDA/data_out/eigenvalues.csv"); // throws exceptions!
		for (long i = 0; i < w_max; i++) {
			csv<< W[i];
			csv << endrow;
		}
	} catch (const std::exception &ex) {
		std::cout << "Exception was thrown: " << ex.what() << std::endl;
	}

	printf("First 15 Eigenvalues saved in: ./CUDA/data_out/eigenvalues.csv \n");


	long v_max = 15;

	try {
		csvfile csv("./CUDA/data_out/eigenvectors.csv"); // throws exceptions!
		for (long i = 0; i < lda; i++) {
			for (long j = 0; j < v_max; j++) {
				csv<< V[lda * j + i];
			}
			csv << endrow;
		}
	} catch (const std::exception &ex) {
		std::cout << "Exception was thrown: " << ex.what() << std::endl;
	}

	printf("First 15 Eigenvectors saved in: ./CUDA/data_out/eigenvectors.csv  \n");

}


void float_eig_solve(float *Hg, float *Qg, int n_total) {

	//Time Metrics
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float milliseconds = 0;

	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;

	const int m = n_total;
	const int lda = m;


//	printf("m * lda =    %d \n", m*lda);

	float* V = new float[lda * m]; // eigenvectors
	float* W = new float[m]; // eigenvalues

	float *d_A = NULL;
	float *d_B = NULL;
	float *d_W = NULL;
	int *devInfo = NULL;
	float *d_work = NULL;
	int lwork = 0;
	int info_gpu = 0;

	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);


	cudaStat1 = cudaMalloc((void**) &d_A, sizeof(float) * lda * m);
	cudaStat2 = cudaMalloc((void**) &d_B, sizeof(float) * lda * m);
	cudaStat3 = cudaMalloc((void**) &d_W, sizeof(float) * m);
	cudaStat4 = cudaMalloc((void**) &devInfo, sizeof(int));
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);

	// A = Hg; B = Qg;
	cudaStat1 = cudaMemcpy(d_A, Hg, sizeof(float) * lda * m,
			cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(d_B, Qg, sizeof(float) * lda * m,
			cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);

	// step 3: query working space of sygvd
	cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	cusolver_status = cusolverDnSsygvd_bufferSize(cusolverH, itype, jobz, uplo,
			m, d_A, lda, d_B, lda, d_W, &lwork);

	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

	cudaStat1 = cudaMalloc((void**) &d_work, sizeof(float) * lwork);

	assert(cudaSuccess == cudaStat1);


//	cout << "Compute Eigenvectors and eigenvalues" << endl;

	cudaEventRecord(start);
	cusolver_status = cusolverDnSsygvd(cusolverH, itype, jobz, uplo, m, d_A,
			lda, d_B, lda, d_W, d_work, lwork, devInfo);
	cudaStat1 = cudaDeviceSynchronize();

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time Elapsed:%f\n", milliseconds);

	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat1);

	cudaStat1 = cudaMemcpy(W, d_W, sizeof(float) * m, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(V, d_A, sizeof(float) * lda * m,
			cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int),
			cudaMemcpyDeviceToHost);

	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

//	printf("after sygvd: info_gpu = %d\n", info_gpu);
	assert(0 == info_gpu);

	write_csv_eigens_float(V, W, lda);

	// free resources
	if (d_A)
		cudaFree(d_A);
	if (d_B)
		cudaFree(d_B);
	if (d_W)
		cudaFree(d_W);
	if (devInfo)
		cudaFree(devInfo);
	if (d_work)
		cudaFree(d_work);
	if (cusolverH)
		cusolverDnDestroy(cusolverH);
	cudaDeviceReset();
}

void float_eig_solve_jacobi(float *Hg, float *Qg, int n_total) {

	//Time Metrics
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float milliseconds = 0;

	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;

	cudaStream_t stream = NULL;
	syevjInfo_t syevj_params = NULL;

	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;

	const int m = n_total;
	const int lda = m;

	float* V = new float[lda * m]; // eigenvectors
	float* W = new float[m]; // eigenvalues

	float *d_A = NULL;
	float *d_B = NULL;
	float *d_W = NULL;
	int *d_info = NULL;
	float *d_work = NULL;
	int lwork = 0;
	int info = 0;

	/* configuration of sygvj  */
	const float tol = 1.e-3;
	const int max_sweeps = 15;
	const cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
	const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
	const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

	/* numerical results of syevj  */
	double residual = 0;
	int executed_sweeps = 0;

//	printf("example of sygvj Float \n");
//	printf("tol = %E, default value is machine zero \n", tol);
//	printf("max. sweeps = %d, default value is 100\n", max_sweeps);

	/* step 1: create cusolver handle, bind a stream  */
	status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	assert(cudaSuccess == cudaStat1);

	status = cusolverDnSetStream(cusolverH, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* step 2: configuration of syevj */
	status = cusolverDnCreateSyevjInfo(&syevj_params);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* default value of tolerance is machine zero */
	status = cusolverDnXsyevjSetTolerance(syevj_params, tol);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* default value of max. sweeps is 100 */
	status = cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps);
	assert(CUSOLVER_STATUS_SUCCESS == status);


	/* step 3: copy A and B to device */
	cudaStat1 = cudaMalloc((void**) &d_A, sizeof(float) * lda * m);
	cudaStat2 = cudaMalloc((void**) &d_B, sizeof(float) * lda * m);
	cudaStat3 = cudaMalloc((void**) &d_W, sizeof(float) * m);
	cudaStat4 = cudaMalloc((void**) &d_info, sizeof(int));
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);

	// A = Hg; B = Qg;
	cudaStat1 = cudaMemcpy(d_A, Hg, sizeof(float) * lda * m,
			cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(d_B, Qg, sizeof(float) * lda * m,
			cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);


	status = cusolverDnSsygvj_bufferSize(cusolverH, itype, jobz, uplo,
			m, d_A, lda, d_B, lda, d_W, &lwork, syevj_params);

	assert(status == CUSOLVER_STATUS_SUCCESS);

	cudaStat1 = cudaMalloc((void**) &d_work, sizeof(float) * lwork);

	assert(cudaSuccess == cudaStat1);

	// step 4: compute spectrum of (A,B)

//	cout << "Compute Eigenvectors and eigenvalues" << endl;

	cudaEventRecord(start);

	status = cusolverDnSsygvj(cusolverH, itype, jobz, uplo, m, d_A,
			lda, d_B, lda, d_W, d_work, lwork, d_info, syevj_params);

	cudaStat1 = cudaDeviceSynchronize();

	assert(CUSOLVER_STATUS_SUCCESS == status);
	assert(cudaSuccess == cudaStat1);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time Elapsed:%f\n", milliseconds);

	cudaStat1 = cudaMemcpy(W, d_W, sizeof(float) * m, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(V, d_A, sizeof(float) * lda * m,
			cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(&info, d_info, sizeof(int),
			cudaMemcpyDeviceToHost);

	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

	if (0 == info) {
//		printf("------------------------>sygvj converges \n");
	} else if (0 > info) {
		printf("Error: %d-th parameter is wrong \n", -info);
		exit(1);
	} else if (m >= info) {
		printf(
				"Error: leading minor of order %d of B is not positive definite\n",
				-info);
		exit(1);
	} else { /* info = m+1 */
		printf("WARNING: info = %d : sygvj does not converge \n", info);
	}

//	printf("eigenvalue = (matlab base-1), ascending order\n");
//
//	for (int i = 0; i < 15; i++) {
////		printf("W[%d] = %f\n", i + 1, pow(W[i], 0.5) / (2 * 3.1415926535897));
//		printf("%f\n", pow(W[i], 0.5) / (2 * 3.1415926535897));
////		printf("W[%d] = %f\n", i + 1, W[i]);
//	}
//
//	printf("-----------------------Vectors ------------------------**  \n");
////	long w_index = 1;
////	long v_max = 90;
////
////
////	for (long i = 0; i < v_max; i++) {
////		printf("%f\n", V[i + lda*w_index]);
////	}
//
//	printf("  \n");
//
//	printf("Saving Data Vector!\n");
//
////
////	long v_max = 90;
////
////
//////	long w_index = 1;
////	int w_index_max = 50;
////
////
////
////	try {
////		csvfile csv("../Data/EigenVectors50.csv"); // throws exceptions!
////		for (long i = 0; i < lda; i++) {
////
////			for (int var = 0; var < w_index_max; ++var) {
////
////
////				csv << V[i + lda * var];
////
////			}
////			csv << endrow;
////
//////			csv << V[i + lda * w_index] << endrow;
////		}
////	} catch (const std::exception &ex) {
////		std::cout << "Exception was thrown: " << ex.what() << std::endl;
////	}
//


	write_csv_eigens_float(V, W, lda);

	 status = cusolverDnXsyevjGetSweeps(cusolverH, syevj_params,
			&executed_sweeps);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	status = cusolverDnXsyevjGetResidual(cusolverH, syevj_params, &residual);
	assert(CUSOLVER_STATUS_SUCCESS == status);

//	printf("residual |M - V*W*V**H|_F = %E \n", residual);
//	printf("number of executed sweeps = %d \n", executed_sweeps);



	// free resources
	if (d_A)
		cudaFree(d_A);
	if (d_B)
		cudaFree(d_B);
	if (d_W)
		cudaFree(d_W);
	if (d_info)
		cudaFree (d_info);
	if (d_work)
		cudaFree(d_work);
	if (cusolverH)
		cusolverDnDestroy(cusolverH);

	if (stream)
		cudaStreamDestroy(stream);
	if (syevj_params)
		cusolverDnDestroySyevjInfo(syevj_params);
	cudaDeviceReset();
}

void double_eig_solve(double *Hg, double *Qg, int n_total) {

	//Time Metrics
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float milliseconds = 0;

	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;

	const int m = n_total;
	const int lda = m;

	double* V = new double[lda * m]; // eigenvectors
	double* W = new double[m]; // eigenvalues

	double *d_A = NULL;
	double *d_B = NULL;
	double *d_W = NULL;
	int *devInfo = NULL;
	double *d_work = NULL;
	int lwork = 0;
	int info_gpu = 0;


	cusolver_status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);


	cudaStat1 = cudaMalloc((void**) &d_A, sizeof(double) * lda * m);
	cudaStat2 = cudaMalloc((void**) &d_B, sizeof(double) * lda * m);
	cudaStat3 = cudaMalloc((void**) &d_W, sizeof(double) * m);
	cudaStat4 = cudaMalloc((void**) &devInfo, sizeof(int));
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);


	cudaStat1 = cudaMemcpy(d_A, Hg, sizeof(double) * lda * m,
			cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(d_B, Qg, sizeof(double) * lda * m,
			cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);


	cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
	cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvalues and eigenvectors.
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	cusolver_status = cusolverDnDsygvd_bufferSize(cusolverH, itype, jobz, uplo,
			m, d_A, lda, d_B, lda, d_W, &lwork);
	assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
	cudaStat1 = cudaMalloc((void**) &d_work, sizeof(double) * lwork);
	assert(cudaSuccess == cudaStat1);


	cudaEventRecord(start);
	cusolver_status = cusolverDnDsygvd(cusolverH, itype, jobz, uplo, m, d_A,
			lda, d_B, lda, d_W, d_work, lwork, devInfo);

	cudaStat1 = cudaDeviceSynchronize();

	assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
	assert(cudaSuccess == cudaStat1);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time Elapsed:%f\n", milliseconds);



	cudaStat1 = cudaMemcpy(W, d_W, sizeof(double) * m, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(V, d_A, sizeof(double) * lda * m,
			cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(&info_gpu, devInfo, sizeof(int),
			cudaMemcpyDeviceToHost);

	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);

//	printf("after sygvd: info_gpu = %d\n", info_gpu);
	assert(0 == info_gpu);

	write_csv_eigens_double(V, W, lda);


	// free resources
	if (d_A)
		cudaFree(d_A);
	if (d_B)
		cudaFree(d_B);
	if (d_W)
		cudaFree(d_W);
	if (devInfo)
		cudaFree(devInfo);
	if (d_work)
		cudaFree(d_work);
	if (cusolverH)
		cusolverDnDestroy(cusolverH);
	cudaDeviceReset();

}

void double_eig_solve_jacobi(double *Hg, double *Qg, int n_total) {

	//Time Metrics
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	float milliseconds = 0;

	cusolverDnHandle_t cusolverH = NULL;
	cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;

	cudaStream_t stream = NULL;
	syevjInfo_t syevj_params = NULL;

	cudaError_t cudaStat1 = cudaSuccess;
	cudaError_t cudaStat2 = cudaSuccess;
	cudaError_t cudaStat3 = cudaSuccess;
	cudaError_t cudaStat4 = cudaSuccess;

	const int m = n_total;
	const int lda = m;

	double* V = new double[lda * m]; // eigenvectors
	double* W = new double[m]; // eigenvalues

	double *d_A = NULL;
	double *d_B = NULL;
	double *d_W = NULL;
	int *d_info = NULL;
	double *d_work = NULL;
	int lwork = 0;
	int info = 0;

	/* configuration of sygvj  */
	const float tol = 1.e-3;
	const int max_sweeps = 15;
	const cusolverEigType_t itype = CUSOLVER_EIG_TYPE_1; // A*x = (lambda)*B*x
	const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR; // compute eigenvectors.
	const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

//	/* numerical results of syevj  */
//	double residual = 0;
//	int executed_sweeps = 0;

//	printf("example of sygvj Double\n");
//	printf("tol = %E, default value is machine zero \n", tol);
//	printf("max. sweeps = %d, default value is 100\n", max_sweeps);

	/* step 1: create cusolver handle, bind a stream  */
	status = cusolverDnCreate(&cusolverH);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	assert(cudaSuccess == cudaStat1);

	status = cusolverDnSetStream(cusolverH, stream);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* step 2: configuration of syevj */
	status = cusolverDnCreateSyevjInfo(&syevj_params);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* default value of tolerance is machine zero */
	status = cusolverDnXsyevjSetTolerance(syevj_params, tol);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* default value of max. sweeps is 100 */
	status = cusolverDnXsyevjSetMaxSweeps(syevj_params, max_sweeps);
	assert(CUSOLVER_STATUS_SUCCESS == status);

	/* step 3: copy A and B to device */
	cudaStat1 = cudaMalloc((void**) &d_A, sizeof(double) * lda * m);
	cudaStat2 = cudaMalloc((void**) &d_B, sizeof(double) * lda * m);
	cudaStat3 = cudaMalloc((void**) &d_W, sizeof(double) * m);
	cudaStat4 = cudaMalloc((void**) &d_info, sizeof(int));
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);
	assert(cudaSuccess == cudaStat4);

	// A = Hg; B = Qg;
	cudaStat1 = cudaMemcpy(d_A, Hg, sizeof(double) * lda * m,
			cudaMemcpyHostToDevice);
	cudaStat2 = cudaMemcpy(d_B, Qg, sizeof(double) * lda * m,
			cudaMemcpyHostToDevice);
	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);

	status = cusolverDnDsygvj_bufferSize(cusolverH, itype, jobz, uplo,
			m, d_A, lda, d_B, lda, d_W, &lwork, syevj_params);

	assert(status == CUSOLVER_STATUS_SUCCESS);

	cudaStat1 = cudaMalloc((void**) &d_work, sizeof(double) * lwork);

	assert(cudaSuccess == cudaStat1);

	// step 4: compute spectrum of (A,B)

//	cout << "Compute Eigenvectors and eigenvalues" << endl;

	cudaEventRecord(start);

	status = cusolverDnDsygvj(cusolverH, itype, jobz, uplo, m, d_A,
			lda, d_B, lda, d_W, d_work, lwork, d_info, syevj_params);

	cudaStat1 = cudaDeviceSynchronize();

	assert(CUSOLVER_STATUS_SUCCESS == status);
	assert(cudaSuccess == cudaStat1);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time Elapsed:%f\n", milliseconds);

	cudaStat1 = cudaMemcpy(W, d_W, sizeof(double) * m, cudaMemcpyDeviceToHost);
	cudaStat2 = cudaMemcpy(V, d_A, sizeof(double) * lda * m,
			cudaMemcpyDeviceToHost);
	cudaStat3 = cudaMemcpy(&info, d_info, sizeof(int),
			cudaMemcpyDeviceToHost);

	assert(cudaSuccess == cudaStat1);
	assert(cudaSuccess == cudaStat2);
	assert(cudaSuccess == cudaStat3);


	if (0 == info) {
//		printf("------------------------>sygvj converges \n");
	} else if (0 > info) {
		printf("Error: %d-th parameter is wrong \n", -info);
		exit(1);
	} else if (m >= info) {
		printf(
				"Error: leading minor of order %d of B is not positive definite\n",
				-info);
		exit(1);
	} else { /* info = m+1 */
		printf("WARNING: info = %d : sygvj does not converge \n", info);
	}

	//	printf("eigenvalue = (matlab base-1), ascending order\n");
	//
	//	for (int i = 0; i < 15; i++) {
	////		printf("W[%d] = %f\n", i + 1, pow(W[i], 0.5) / (2 * 3.1415926535897));
	//		printf("%f\n", i + 1, pow(W[i], 0.5) / (2 * 3.1415926535897));
	//	}

	write_csv_eigens_double(V, W, lda);

	// free resources
	if (d_A)
		cudaFree(d_A);
	if (d_B)
		cudaFree(d_B);
	if (d_W)
		cudaFree(d_W);
	if (d_info)
		cudaFree (d_info);
	if (d_work)
		cudaFree(d_work);
	if (cusolverH)
		cusolverDnDestroy(cusolverH);

	if (stream)
		cudaStreamDestroy(stream);
	if (syevj_params)
		cusolverDnDestroySyevjInfo(syevj_params);


	cudaDeviceReset();
}



