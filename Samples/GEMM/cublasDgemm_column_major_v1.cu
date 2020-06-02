/* This program is for the matrix multiplication using cublas lib in column major format (v1)
 Input Data		: From Host to Device, 
 Operation		: on Device, 
 Output Result	: From Device to Host
 Coded by: PUNEET DHEER*/

#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <iostream>
#include <cassert>
#include <cmath> //fabs
#include <ctime>

using namespace std;

// Fortran-style indexing column-major
int cm(int column, int row, int nRows)
{
	return column*nRows + row;
}


void fill_mat_A(double *mat, int rows, int columns)
{
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < columns; j++)
		{
			//mat[cm(j, i, rows)] = (i * 3) + (j * 2);
			mat[cm(j, i, rows)] = rand();
		}
	}
}


void fill_mat_B(double *mat, int rows, int columns)
{
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < columns; j++)
		{
			//mat[cm(j, i, rows)] = (i * 5) + (j * 9);
			mat[cm(j, i, rows)] = rand();
		}
	}
}


void show_mat(double *mat, int rows, int columns)
{
	cout << endl;

	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < columns; j++)
		{

			cout << mat[cm(j, i, rows)] << " ";

			cout << "[" << cm(j, i, rows) << "]" << " ";

			cout << "[" << &mat[cm(j, i, rows)] << "]" << " ";
		}

		cout << "\n";
	}
	cout << "\n";
}


void cross_check_result(double *A, double *B, double *C, int rowsA, int columnsB, int rowsB)
{
	cout << endl;

	double calc;
	int count = 1;
	int total = rowsA*columnsB;
	
	for (size_t i = 0; i < rowsA; i++)
	{
		for (size_t j = 0; j < columnsB; j++)
		{
			calc = 0;

			for (size_t x = 0; x < rowsB; x++)
			{
							
				calc += A[x*rowsA + i] * B[j*rowsB + x];		
			
				//cout << A[x*rowsA + i] << "  " << B[j*rowsB + x] << "  "<< calc <<"\n";
			}
			
			cout << "\r...CHECKING...Element No: " << count << "/" << total;
			count++;

			assert(fabs(C[j*rowsA + i] - calc) <= 0.000001);
			
			/*cout << endl;
			cout << "-----------------------------------------------" << endl;
			cout << "[ Diff: "<<fabs(C[j*rowsA + i] - calc)<< " ] " <<"[ Resultant GPU Mat:  "<< C[j*rowsA + i] <<" ] "<< "[ Resultant CPU Mat: "<<calc<<" ] "<< "\n";
			cout << "-----------------------------------------------" << endl;*/
		}
	}
	
}


int main()
{
	int row_A = 100;
	int col_A = 100;
	int row_B = col_A;
	int col_B = 100;
	int row_C = row_A;
	int col_C = col_B;

	clock_t start, end, tstart, tend;
	double secs, msecs;

	cout << fixed;

	tstart = clock();

	// Using "thrust"
	cout << "Initialization of Host Matrices... ";
	start = clock();
	thrust::host_vector<double> h_A(row_A * col_A); // on the host side (CPU) 
	thrust::host_vector<double> h_B(row_B * col_B); // on the host side (CPU)
	thrust::host_vector<double> h_C(row_C * col_C); // on the host side (CPU)
	end = clock();
	secs = (double(end - start)) / CLOCKS_PER_SEC;
	msecs= (double(end - start)) / (CLOCKS_PER_SEC/1000);
	cout << "...Done..." << " Execution Time: " << secs << " secs, " << msecs << " msecs" << "\n" << "\n";

	// fill the matrices on the host with some simple values for further cross check
	cout << "Filling Host Matrices... ";
	start = clock();
	fill_mat_A(thrust::raw_pointer_cast(&h_A[0]), row_A, col_A);
	fill_mat_B(thrust::raw_pointer_cast(&h_B[0]), row_B, col_B);
	//show_mat(thrust::raw_pointer_cast(&h_A[0]), row_A, col_A);
	//show_mat(thrust::raw_pointer_cast(&h_B[0]), row_B, col_B);
	end = clock();
	secs = (double(end - start)) / CLOCKS_PER_SEC;
	msecs = (double(end - start)) / (CLOCKS_PER_SEC / 1000);
	cout << "...Done..." << " Execution Time: " << secs << " secs, " << msecs << " msecs" << "\n" << "\n";


	// move the data from host to device
	cout << "Moving Host Matrices to Device... ";
	start = clock();
	thrust::device_vector<double> d_A = h_A; // on the device side (GPU)
	thrust::device_vector<double> d_B = h_B; // on the device side (GPU)
	thrust::device_vector<double> d_C(row_C * col_C); // on the device side (GPU)
	end = clock();
	secs = (double(end - start)) / CLOCKS_PER_SEC;
	msecs = (double(end - start)) / (CLOCKS_PER_SEC / 1000);
	cout << "...Done..." << " Execution Time: " << secs << " secs, " << msecs << " msecs" << "\n" << "\n";


	// Initialize CUBLAS
	cublasHandle_t handle;
	cublasStatus_t status = cublasCreate(&handle);

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		cerr << "!!! CUBLAS initialization error !!!\n";
	}

	double alpha = 1.0f;
	double beta = 0.0f;

	// C = (alpha*A) * B + (beta*C)
	// A(m*k)*B(k*n) = C(m*n)
	/* cublasDgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
	int m, int n, int k,
	const double *alpha,
	const double *A, int lda,
	const double *B, int ldb,
	const double *beta,
	double *C, int ldc)*/

	cout << "Multiplication is running on Device... ";
	start = clock();
	status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, row_A, col_B, col_A, &alpha,
		thrust::raw_pointer_cast(&d_A[0]), row_A,
		thrust::raw_pointer_cast(&d_B[0]), row_B,
		&beta,
		thrust::raw_pointer_cast(&d_C[0]), row_C);
	end = clock();
	secs = (double(end - start)) / CLOCKS_PER_SEC;
	msecs = (double(end - start)) / (CLOCKS_PER_SEC / 1000);
	cout << "...Done..." << " Execution Time: " << secs << " secs, " << msecs << " msecs" << "\n" << "\n";


	if (status != CUBLAS_STATUS_SUCCESS)
	{
		cerr << "!!! kernel execution error !!!\n";
	}

	// move the result from device to host
	cout << "Moving Result from Device to Host... ";
	start = clock();
	h_C = d_C; 
	end = clock();
	secs = (double(end - start)) / CLOCKS_PER_SEC;
	msecs = (double(end - start)) / (CLOCKS_PER_SEC / 1000);
	cout << "...Done..." << " Execution Time: " << secs << " secs, " << msecs << " msecs" << "\n" << "\n";

	//show_mat(thrust::raw_pointer_cast(&h_C[0]), row_C, col_C);

	//cross check the result
	cout << "SANITY check is running on Host... ";
	start = clock();
	cross_check_result(thrust::raw_pointer_cast(&h_A[0]), thrust::raw_pointer_cast(&h_B[0]), thrust::raw_pointer_cast(&h_C[0]), row_A, col_B, row_B);
	end = clock();
	secs = (double(end - start)) / CLOCKS_PER_SEC;
	msecs = (double(end - start)) / (CLOCKS_PER_SEC / 1000);
	cout << "...Done..." << " Execution Time: " << secs << " secs, " << msecs << " msecs" << "\n" << "\n";


	status = cublasDestroy(handle);

	if (status != CUBLAS_STATUS_SUCCESS)
	{
		cerr << "!!! shutdown error !!!\n";
	}

	tend = clock();
	secs = (double(tend - tstart)) / CLOCKS_PER_SEC;
	msecs = (double(tend - tstart)) / (CLOCKS_PER_SEC / 1000);
	cout << "Total Execution Time: " << secs << " secs, " << msecs << " msecs" << "\n" << "\n";

	return 0;
}