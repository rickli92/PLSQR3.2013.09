/**
 *  @file aprod_impl.c
 *
 *  @date May 14, 2012
 *  @author He Huang
 *  @brief Implementation of aprod function of PLSQR2/3
 */

#include "aprod_impl.h"
#include "FileIO.h"
#include <assert.h>
//#define DEBUG

/**
 * @brief y<-y+A*x
 * @param procRank
 * @param numProc
 * @param iteration
 * @param isFinalIteration 0 indicates false, 1 indicate true
 * @param p data
 */
void aprod_mode_1_tridiagonal(int procRank, int numProc, int iteration, int isFinalIteration, PPDataV3_TRI * p) {
	int i, rank, err;
//Kernel SpMV	<<almost the same as one-diagonal except vector x's index

	/*Memory initialization*/
	int numRow_k = p->matCSC_k->numRow;
#if GPUALL==1
	static MY_FLOAT_TYPE * temp_vec_y_Ax_k_d=NULL;
	static MY_FLOAT_TYPE * temp_vec_y_Ax_d_d=NULL;
	if(NULL==temp_vec_y_Ax_k_d){
		cudaMalloc_w((void **)&temp_vec_y_Ax_k_d, sizeof(MY_FLOAT_TYPE) * numRow_k);
		assert(temp_vec_y_Ax_k_d);
	}
	if(NULL==temp_vec_y_Ax_d_d){
		cudaMalloc_w((void**)&temp_vec_y_Ax_d_d, sizeof(MY_FLOAT_TYPE) * p->vec_y_d_len);
		assert(temp_vec_y_Ax_d_d);
	}
	cudaMemset_w(temp_vec_y_Ax_k_d, 0, sizeof(MY_FLOAT_TYPE) * numRow_k);
	cudaMemset_w(temp_vec_y_Ax_d_d, 0, sizeof(MY_FLOAT_TYPE) * p->vec_y_d_len);
#endif

	static MY_FLOAT_TYPE * temp_vec_y_Ax_k_allreduce=NULL;
	static MY_FLOAT_TYPE * temp_vec_y_Ax_k = NULL;
	static MY_FLOAT_TYPE * temp_vec_y_Ax_d = NULL;

	// only allocate memory at the first time
	if(temp_vec_y_Ax_k_allreduce==NULL){
		temp_vec_y_Ax_k_allreduce = (MY_FLOAT_TYPE*) calloc(numRow_k, sizeof(MY_FLOAT_TYPE)); //record reduced temporal result
		assert(temp_vec_y_Ax_k_allreduce);
	}
	if(temp_vec_y_Ax_k==NULL){
		temp_vec_y_Ax_k = (MY_FLOAT_TYPE*) calloc(numRow_k, sizeof(MY_FLOAT_TYPE));
		assert(temp_vec_y_Ax_k);
	}
	if(temp_vec_y_Ax_d==NULL){
		temp_vec_y_Ax_d = (MY_FLOAT_TYPE *) calloc(p->vec_y_d_len, sizeof(MY_FLOAT_TYPE));
		assert(temp_vec_y_Ax_d);
	}
	//clear memory each time
	memset(temp_vec_y_Ax_k_allreduce, 0, numRow_k * sizeof(MY_FLOAT_TYPE));
	memset(temp_vec_y_Ax_k, 0, numRow_k * sizeof(MY_FLOAT_TYPE));
	memset(temp_vec_y_Ax_d, 0, p->vec_y_d_len * sizeof(MY_FLOAT_TYPE));

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("spmv_ker");
#endif
#ifdef USEPAPI
        /* Setup PAPI library and begin collecting data from the counters */
        float real_time1=0.0, proc_time1=0.0, mflops1=0.0;         long long flpops1=0;     int retval1=0;
        if(isFinalIteration){
                if((retval1=PAPI_flops( &real_time1, &proc_time1, &flpops1, &mflops1))<PAPI_OK)
                        test_fail(__FILE__, __LINE__, "PAPI_flops", retval1);
        }
#endif

#if GPUALL==1
	SpMatCSR * mat = p->gpu->matCSR_k_d;
	MY_FLOAT_TYPE * vec = p->gpu->vec_x_d;
	cusparseXcsrmv_w(mat->numRow, mat->numCol, mat->numNonzero, mat->val, mat->row_ptr, mat->col_idx,
			vec, temp_vec_y_Ax_k_d);
#else
	cpuSpMV_CSC(p->matCSC_k, p->vec_x, temp_vec_y_Ax_k);
#endif

#ifdef USEPAPI
	/* Collect the data into the variables passed in */
	if(isFinalIteration){
		float real_time=0.0, proc_time=0.0, mflops=0.0; 	long long flpops=0;	int retval=0;
		if((retval=PAPI_flops( &real_time, &proc_time, &flpops, &mflops))<PAPI_OK)
			test_fail(__FILE__, __LINE__, "PAPI_flops", retval);
		float mflops_all = 0;
		MPI_Reduce(&mflops, &mflops_all, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
		if(procRank==0) {
			printf("%d: \"spmv_ker\" real_time=%f, proc_time=%f \n", procRank, real_time-real_time1, proc_time-proc_time1);
			printf("%d: \"spmv_ker\" total MFLOPS = %f for %d processes\n", procRank, mflops_all, numProc);
		}
	}
#endif
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstop("spmv_ker");
#endif
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("allreduce");
#endif
#if GPUALL==1
	cudaMemcpy_w(temp_vec_y_Ax_k, temp_vec_y_Ax_k_d, numRow_k * sizeof(MY_FLOAT_TYPE), mycudaMemcpyDeviceToHost);//device to host
	MPI_Allreduce(temp_vec_y_Ax_k, temp_vec_y_Ax_k_allreduce, numRow_k,
			MY_MPI_FLOAT_TYPE, MPI_SUM, MPI_COMM_WORLD);
	cudaMemcpy_w(temp_vec_y_Ax_k_d, temp_vec_y_Ax_k_allreduce, numRow_k * sizeof(MY_FLOAT_TYPE), mycudaMemcpyHostToDevice);//host to device
#else
	MPI_Allreduce(temp_vec_y_Ax_k, temp_vec_y_Ax_k_allreduce, numRow_k,
			MY_MPI_FLOAT_TYPE, MPI_SUM, MPI_COMM_WORLD);
#endif
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstop("allreduce");
#endif

#if GPUALL==1
	MY_FLOAT_TYPE alpha = 1.0;
	//p->gpu->vec_y_k_d <-- temp_vec_y_Ax_k_d + p->gpu->vec_y_k_d
	cublasSaxpy_w(numRow_k, &alpha, temp_vec_y_Ax_k_d, 1, p->gpu->vec_y_k_d, 1);
#else
	int chunk= (numRow_k+32*5-1)/(32*5);
#pragma omp parallel for private(i) schedule(dynamic, chunk)
	for (i = 0; i < numRow_k; i++) {
		p->vec_y_k[i] += temp_vec_y_Ax_k_allreduce[i];
	}
#endif //GPUALL
//Kernel SpMV	>>

//	Damping SpMV <<
	int extended_vec_x_startIdx = p->extended_vec_x_startIdx;
	int extended_vec_x_len = p->extended_vec_x_len;
	static MY_FLOAT_TYPE * extended_vec_x=NULL;
	if(extended_vec_x==NULL){//allocate in the first iteration
		extended_vec_x = (MY_FLOAT_TYPE *) malloc(extended_vec_x_len * sizeof(MY_FLOAT_TYPE));
		assert(extended_vec_x);
	}
	memset(extended_vec_x, 0, extended_vec_x_len * sizeof(MY_FLOAT_TYPE));
#if GPUALL==1//update p->vec_x, device to host, must do this first before memcpy
	cudaMemcpy_w(p->vec_x, p->gpu->vec_x_d, p->vec_x_len * sizeof(MY_FLOAT_TYPE), mycudaMemcpyDeviceToHost);
#endif
	/*assign local vector x to corresponding location of extended vector x*/
	if(p->vec_x_begin_gloIdx >= extended_vec_x_startIdx
			&& (p->vec_x_begin_gloIdx - extended_vec_x_startIdx + p->vec_x_len) <= extended_vec_x_len)
		memcpy(extended_vec_x + (p->vec_x_begin_gloIdx - extended_vec_x_startIdx), p->vec_x, p->vec_x_len * sizeof(MY_FLOAT_TYPE));
	else;
/* else // (p->vec_x_begin_gloIdx - extended_vec_x_startIdx + p->vec_x_len) > extended_vec_x_len
 * This happens when there is no element in local matrix that corresponding to local vector x, so no need to copy local vector*/

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("aprod1_gather_x");
#endif
/*Copy receive list to extended vector*/
#ifdef USE_COMPRESSED_VEC
	GatherVecXCompressed(procRank, numProc, p, extended_vec_x_startIdx, extended_vec_x, isFinalIteration);
#else
	gather_vec_x(procRank, numProc, p, extended_vec_x_startIdx, extended_vec_x);
#endif
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstop("aprod1_gather_x");
#endif

#if GPUALL==1	//update extended_vec_x_d, host to device
	static MY_FLOAT_TYPE * extended_vec_x_d=NULL;
	if(NULL==extended_vec_x_d){
		cudaMalloc_w((void **)&extended_vec_x_d, p->extended_vec_x_len * sizeof(MY_FLOAT_TYPE));
		assert(extended_vec_x_d);
	}
//	cudaMemset_w(extended_vec_x_d, 0, size);
	cudaMemcpy_w(extended_vec_x_d, extended_vec_x, p->extended_vec_x_len * sizeof(MY_FLOAT_TYPE), mycudaMemcpyHostToDevice);
#endif

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("spmv_damp");
#endif
#ifdef USEPAPI
	/* Setup PAPI library and begin collecting data from the counters */
	real_time1=0.0, proc_time1=0.0, mflops1=0.0;         flpops1=0;     retval1=0;
	if(isFinalIteration){
		if((retval1=PAPI_flops( &real_time1, &proc_time1, &flpops1, &mflops1))<PAPI_OK)
			test_fail(__FILE__, __LINE__, "PAPI_flops", retval1);
	}
#endif

#if GPUALL==1
	SpMatCSR * mat2 = p->gpu->matCSR_d_d;
	if(iteration==1){
//		mat2->col_idx[i] - extended_vec_x_startIdx, adjust column index
		int_x_plus_a_w(mat2->col_idx, mat2->numNonzero, -extended_vec_x_startIdx);
	}
	cusparseXcsrmv_w(mat2->numRow, mat2->numCol, mat2->numNonzero,  mat2->val, mat2->row_ptr, mat2->col_idx,
			extended_vec_x_d, temp_vec_y_Ax_d_d);
	alpha=1.0;
	// p->gpu->vec_y_d_d <-- temp_vec_y_Ax_d_d + p->gpu->vec_y_d_d
	cublasSaxpy_w(p->vec_y_d_len, &alpha, temp_vec_y_Ax_d_d, 1, p->gpu->vec_y_d_d, 1);

#else
	cpuSpMV_CSR_v2(p->matCSR_d, extended_vec_x, extended_vec_x_startIdx, temp_vec_y_Ax_d);
	for (i = 0; i < p->vec_y_d_len; i++) {
		p->vec_y_d[i] += temp_vec_y_Ax_d[i];
	}
#endif


#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstop("spmv_damp");
#endif
//	Damping SpMV >>

// Make this a persistent vector during the main iterations
	if(isFinalIteration==1){
#ifdef USEPAPI
	/* Collect the data into the variables passed in */
		float real_time2=0.0, proc_time2=0.0, mflops2=0.0; 	long long flpops2=0;	int retval2=0;
		if((retval2=PAPI_flops( &real_time2, &proc_time2, &flpops2, &mflops2))<PAPI_OK)
			test_fail(__FILE__, __LINE__, "PAPI_flops", retval2);
		float mflops_all = 0;
		MPI_Reduce(&mflops2, &mflops_all, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
		if(procRank==0) {
			printf("%d: \"spmv_damp\" real_time=%f, proc_time=%f \n", procRank, real_time2-real_time1, proc_time2-proc_time1);
			printf("%d: \"spmv_damp\" total MFLOPS = %f for %d processes\n", procRank, mflops_all, numProc);
		}
#endif

		//free memory for temprial variable in the last iteration
		free(temp_vec_y_Ax_k);temp_vec_y_Ax_k=NULL;
		free(temp_vec_y_Ax_k_allreduce);temp_vec_y_Ax_k_allreduce=NULL;
		free(temp_vec_y_Ax_d);temp_vec_y_Ax_d=NULL;
		free(extended_vec_x);extended_vec_x=NULL;
#if GPUALL ==1
		cudaFree_w(extended_vec_x_d);extended_vec_x_d=NULL;
		cudaFree_w(temp_vec_y_Ax_k_d);temp_vec_y_Ax_k_d=NULL;
		cudaFree_w(temp_vec_y_Ax_d_d);temp_vec_y_Ax_d_d=NULL;
#endif
	}





}
/**
 * @brief y <- A(transpose)*x + y
 * @param procRank
 * @param numProc
 * @param iteration
 * @param isFinalIteration
 * @param p
 */
void aprod_mode_2_tridiagonal(int procRank, int numProc, int iteration,
		int isFinalIteration, PPDataV3_TRI * p)
{
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("spmvT_other1");
#endif

	int i=0, rank=0, err=0;
/*Memory initialization*/
#if GPUALL==1
	static MY_FLOAT_TYPE * temp_vec_x_tAy_k_d = NULL;
	static MY_FLOAT_TYPE * temp_vec_x_tAy_d_d = NULL;
	if(NULL==temp_vec_x_tAy_k_d){
		cudaMalloc_w((void **)&temp_vec_x_tAy_k_d, sizeof(MY_FLOAT_TYPE)*  p->vec_x_len);
		assert(temp_vec_x_tAy_k_d);
	}
	if(NULL==temp_vec_x_tAy_d_d){
		cudaMalloc_w((void**)&temp_vec_x_tAy_d_d, sizeof(MY_FLOAT_TYPE) * p->vec_x_len);
		assert(temp_vec_x_tAy_d_d);
	}

	if(iteration > 0){// does not participate computation in 0th iteration
		cudaMemset_w(temp_vec_x_tAy_k_d, 0, sizeof(MY_FLOAT_TYPE)*  p->vec_x_len);
		cudaMemset(temp_vec_x_tAy_d_d, 0, sizeof(MY_FLOAT_TYPE) * p->vec_x_len);
	}
#endif

	static MY_FLOAT_TYPE * temp_vec_x_tAy_k = NULL;
	static MY_FLOAT_TYPE * temp_vec_x_tAy_d = NULL;

	if(temp_vec_x_tAy_k==NULL){
		temp_vec_x_tAy_k = (MY_FLOAT_TYPE *) malloc(p->vec_x_len * sizeof(MY_FLOAT_TYPE));
		assert(temp_vec_x_tAy_k);
	}
	if (temp_vec_x_tAy_d == NULL){
		temp_vec_x_tAy_d = (MY_FLOAT_TYPE *) malloc(p->vec_x_len * sizeof(MY_FLOAT_TYPE));
		assert(temp_vec_x_tAy_d);
	}
	memset(temp_vec_x_tAy_k, 0, p->vec_x_len * sizeof(MY_FLOAT_TYPE));
	memset(temp_vec_x_tAy_d, 0, p->vec_x_len * sizeof(MY_FLOAT_TYPE));

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstop("spmvT_other1");
#endif

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("spmvT_ker");
#endif

#ifdef USEPAPI
        /* Setup PAPI library and begin collecting data from the counters */
        float real_time1=0.0, proc_time1=0.0, mflops1=0.0;         long long flpops1=0;     int retval1=0;
        if(isFinalIteration){
                if((retval1=PAPI_flops( &real_time1, &proc_time1, &flpops1, &mflops1))<PAPI_OK)
                        test_fail(__FILE__, __LINE__, "PAPI_flops", retval1);
        }
#endif
//kernel SpMV:
	if(iteration ==0 ){//0th iteration is computed on CPU always
		cpuSpMVTranspose_CSC(p->matCSC_k, p->vec_y_k, temp_vec_x_tAy_k);
	}else{
#if GPUALL==1
		SpMatCSC * mat = p->gpu->matCSC_k_d;
		MY_FLOAT_TYPE * vec = p->gpu->vec_y_k_d;
		if(iteration==0)
			cpuSpMVTranspose_CSC(p->matCSC_k, p->vec_y_k, temp_vec_x_tAy_k);
		else if(iteration > 0)
			cusparseXcsrmv_w( mat->numCol, mat->numRow, mat->numNonzero, mat->val, mat->col_ptr, mat->row_idx, vec, temp_vec_x_tAy_k_d);
#else
		cpuSpMVTranspose_CSC(p->matCSC_k, p->vec_y_k, temp_vec_x_tAy_k);
#endif
	}

#ifdef USEPAPI
	/* Collect the data into the variables passed in */
	if(isFinalIteration){
		float real_time=0.0, proc_time=0.0, mflops=0.0; 	long long flpops=0;	int retval=0;
		if((retval=PAPI_flops( &real_time, &proc_time, &flpops, &mflops))<PAPI_OK)
			test_fail(__FILE__, __LINE__, "PAPI_flops", retval);
		float mflops_all = 0;
		MPI_Reduce(&mflops, &mflops_all, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
		if(procRank==0) {
			printf("%d: \"spmvT_ker\" real_time=%f, proc_time=%f \n", procRank, real_time-real_time1, proc_time-proc_time1);
			printf("%d: \"spmvT_ker\" total MFLOPS = %f for %d processes\n", procRank, mflops_all, numProc);
		}
	}
#endif
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstop("spmvT_ker");
#endif

//damping SpMV, borrow vector y from neighbors
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("spmvT_other2");
#endif
	int extended_vec_y_d_startIdx = p->extended_vec_y_d_startIdx;
	int extended_vec_y_d_len = p->extended_vec_y_d_len;
	static MY_FLOAT_TYPE * extended_vec_y_d=NULL;
	if(extended_vec_y_d==NULL){
		extended_vec_y_d = (MY_FLOAT_TYPE *) malloc(extended_vec_y_d_len * sizeof(MY_FLOAT_TYPE));
		assert(extended_vec_y_d);
	}
	memset(extended_vec_y_d, 0, extended_vec_y_d_len * sizeof(MY_FLOAT_TYPE));
#if GPUALL==1//update p->vec_y_d, device to host, must do this first before memcpy
	if(iteration>0) cudaMemcpy_w(p->vec_y_d, p->gpu->vec_y_d_d, p->vec_y_d_len * sizeof(MY_FLOAT_TYPE), mycudaMemcpyDeviceToHost);
#endif
/* copy local vector y to corresponding location of extended vector y*/
	if( p->vec_y_d_begin>=extended_vec_y_d_startIdx
			&&(p->vec_y_d_begin - extended_vec_y_d_startIdx + p->vec_y_d_len) <= extended_vec_y_d_len){
		memcpy(extended_vec_y_d + (p->vec_y_d_begin - extended_vec_y_d_startIdx), p->vec_y_d,
			p->vec_y_d_len * sizeof(MY_FLOAT_TYPE));
	}else;
	/* else // (p->vec_y_d_begin - extended_vec_y_d_startIdx + p->vec_y_d_len) > extended_vec_y_d_len
	 * This happens when there is no element in local matrix that corresponding to local vector y, so no need to copy local vector*/

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstop("spmvT_other2");
#endif

//	pull vector y damping part from neighbors
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("aprod2_gather_y");
#endif
#ifdef USE_COMPRESSED_VEC
	GatherVecYCompressed(procRank, numProc, p, extended_vec_y_d_startIdx, extended_vec_y_d, isFinalIteration);
#else
	gather_vec_y (procRank, numProc, p, extended_vec_y_d_startIdx, extended_vec_y_d);
#endif

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstop("aprod2_gather_y");
#endif

#if GPUALL==1	//update extended_vec_y_d_d, host to device
	static MY_FLOAT_TYPE * extended_vec_y_d_d=NULL;
	if(NULL==extended_vec_y_d_d){
		cudaMalloc_w((void **)&extended_vec_y_d_d, sizeof(MY_FLOAT_TYPE) * p->extended_vec_y_d_len);
		assert(extended_vec_y_d_d);
	}
	if(iteration>0){
//		cudaMemset_w(extended_vec_y_d_d, 0, size);
		cudaMemcpy_w(extended_vec_y_d_d, extended_vec_y_d, sizeof(MY_FLOAT_TYPE) * p->extended_vec_y_d_len, mycudaMemcpyHostToDevice);
	}

#endif

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("spmvT_damp");
#endif
#ifdef USEPAPI
        /* Setup PAPI library and begin collecting data from the counters */
        real_time1=0.0, proc_time1=0.0, mflops1=0.0;         flpops1=0;     retval1=0;
        if(isFinalIteration){
                if((retval1=PAPI_flops( &real_time1, &proc_time1, &flpops1, &mflops1))<PAPI_OK)
                        test_fail(__FILE__, __LINE__, "PAPI_flops", retval1);
        }

#endif
	if(iteration ==0 ){//0th iteration is computed on CPU always
		cpuSpMVTranspose_CSC_v2(p->matCSC_d_bycol, extended_vec_y_d, extended_vec_y_d_startIdx,
				temp_vec_x_tAy_d);
		//add current vector x's result with previous vector x
		for (i = 0; i < p->vec_x_len; i++) {
			p->vec_x[i] += (temp_vec_x_tAy_k[i] + temp_vec_x_tAy_d[i]);
		}
	}else if (iteration>0) {
#if GPUALL==1
		SpMatCSC * mat = p->gpu->matCSC_d_bycol_d;
		if(iteration==1){
			int_x_plus_a_w(mat->row_idx, mat->numNonzero, -extended_vec_y_d_startIdx);
		}
		cusparseXcsrmv_w( mat->numCol, mat->numRow, mat->numNonzero, mat->val, mat->col_ptr, mat->row_idx,
				extended_vec_y_d_d, temp_vec_x_tAy_d_d);
		//p->vec_x[i] += (temp_vec_x_tAy_k[i] + temp_vec_x_tAy_d[i]);
		MY_FLOAT_TYPE alpha=1.0;
		//temp_vec_x_tAy_d_d <-- temp_vec_x_tAy_k_d + temp_vec_x_tAy_d_d
		cublasSaxpy_w(p->vec_x_len, &alpha, temp_vec_x_tAy_k_d, 1, temp_vec_x_tAy_d_d, 1);

		//p->gpu->vec_x_d <-- temp_vec_x_tAy_d_d + p->gpu->vec_x_d
		cublasSaxpy_w(p->vec_x_len, &alpha, temp_vec_x_tAy_d_d, 1, p->gpu->vec_x_d, 1);
#else
		cpuSpMVTranspose_CSC_v2(p->matCSC_d_bycol, extended_vec_y_d, extended_vec_y_d_startIdx,
				temp_vec_x_tAy_d);
		//add current vector x's result with previous vector x
		for (i = 0; i < p->vec_x_len; i++) {
			p->vec_x[i] += (temp_vec_x_tAy_k[i] + temp_vec_x_tAy_d[i]);
		}
#endif
	}

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstop("spmvT_damp");
#endif


	if(isFinalIteration){
#ifdef USEPAPI
	/* Collect the data into the variables passed in */
		float real_time=0.0, proc_time=0.0, mflops=0.0; 	long long flpops=0;	int retval=0;
		if((retval=PAPI_flops( &real_time, &proc_time, &flpops, &mflops))<PAPI_OK)
			test_fail(__FILE__, __LINE__, "PAPI_flops", retval);
		float mflops_all = 0;
		MPI_Reduce(&mflops, &mflops_all, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
		if(procRank==0) {
			printf("%d: \"spmvT_damp\" real_time=%f, proc_time=%f \n", procRank, real_time-real_time1, proc_time-proc_time1);
			printf("%d: \"spmvT_damp\" total MFLOPS = %f for %d processes\n", procRank, mflops_all, numProc);
		}
#endif

		free(temp_vec_x_tAy_k);temp_vec_x_tAy_k=NULL;
		free(temp_vec_x_tAy_d);temp_vec_x_tAy_d=NULL;
		free(extended_vec_y_d);extended_vec_y_d=NULL;
#if GPUALL==1
		cudaFree_w(extended_vec_y_d_d);
		cudaFree_w(temp_vec_x_tAy_k_d);
		cudaFree_w(temp_vec_x_tAy_d_d);
#endif
	}
}



#ifdef ONEDIAGONAL
/*A C variable has one of the following linkages: external linkage, internal linkage, or no linkage. Variables with block scope or prototype scope have no linkage. That means they are private to the block or prototype in which they are defined. A variable with file scope can have either internal or external linkage. A variable with external linkage can be used anywhere in a multifile program. A variable with internal linkage can be used anywhere in a single file.*/
static int * overlap_offset_gloIdx1/*len is 1*/;
static MPI_Win win_overlap_offset1;
static int overlap_len1=0;
static MY_FLOAT_TYPE * overlap_array1;
static MPI_Win win_overlap_array1;

static int * overlap_offset_gloIdx2; //len is 1
static MPI_Win win_overlap_offset2;
static int overlap_len2=0;
static MY_FLOAT_TYPE * overlap_array2;
static MPI_Win win_overlap_array2;
#endif

#ifdef PLSQR2
void aprod_plsqr2(int version, int procRank, int numProc, int mode,
		int m /*row*/, int n /*column*/, MY_FLOAT_TYPE x[]/*v len=n*/,
		MY_FLOAT_TYPE y[] /*u=perPorcVectorB len=m, assign NULL if PLSQRV3*/,
		void *UsrWork, int isFinalIteration, int iteration) {

	int i;
	Per_Proc_Data_v2 *perProcData = (Per_Proc_Data_v2*) UsrWork;

	if (mode == 1) { //y[m,1] = y[m,1] + A[m,n]*x[n,1]
		int loc_numOfRows = perProcData->matrix_block->numRow;
		/*			for (i = 0; i < loc_numOfRows; i++)//can be optimized with memset ...
		 perProcData->vec_y_Ax_temp[i] = 0.0;*/ ///5/20/11
#ifdef BARRIER
		MPI_Barrier(MPI_COMM_WORLD);
#endif
		GPTLstart("spmv");
#if GPUALL==1
		cusparseSpMVpV_w(perProcData->matrix_block, x, y);
#else
		memset(perProcData->vec_y_Ax_temp, 0,
				sizeof(MY_FLOAT_TYPE) * (perProcData->matrix_block->numRow));
		cpuSpMV_CSR(perProcData->matrix_block, x, perProcData->vec_y_Ax_temp);
		//update y[]
#pragma ivdep
		for (i = 0; i < loc_numOfRows; i++) {
			y[i] += perProcData->vec_y_Ax_temp[i];
		}
#endif
#ifdef GPTL
		GPTLstop("spmv");
#endif
	}

	else if (mode == 2) //mode =2
			{ //x[n,1] = x[n,1] + transpose(A)[n,m]*y[m,1]

		if (numProc == 1) { //serial code
#ifdef BARRIER
			MPI_Barrier(MPI_COMM_WORLD);
#endif
			GPTLstart("spmvT");
			if (iteration == 0) {
				memset(perProcData->vec_x_tAy_temp, 0,
						sizeof(MY_FLOAT_TYPE) * perProcData->totalNumOfColumns);
				cpuSpMVTranspose_CSR(perProcData->matrix_block, y,
						perProcData->vec_x_tAy_temp);
				for (i = 0; i < perProcData->totalNumOfColumns; i++)
					x[i] += perProcData->vec_x_tAy_temp[i];

			} else {
#if GPUALL==1
#if TRANSPOSE_AVOID==1
				myspmvpvTransKer_w(perProcData->matrix_block, y/*len=row*/, x);
#else
				cusparseSpMVTranspV_w(perProcData->matrix_block, y /*len=row*/, x); //y and x are located in GPU
#endif
#else
				memset(perProcData->vec_x_tAy_temp, 0,
						sizeof(MY_FLOAT_TYPE) * perProcData->totalNumOfColumns);
				cpuSpMVTranspose_CSR(perProcData->matrix_block, y,
						perProcData->vec_x_tAy_temp);
				for (i = 0; i < perProcData->totalNumOfColumns; i++)
					x[i] += perProcData->vec_x_tAy_temp[i];
#endif
			}
#ifdef GPTL
			GPTLstop("spmvT");
#endif
		}

		else { // np > 1, parallel code

			/*#pragma ivdep
			 for (i = 0; i < perProcData->totalNumOfColumns; i++) {
			 perProcData->vec_x_tAy_temp[i] = 0.0;//clear up every iteration
			 }*/ //5/20/11
#ifdef BARRIER
			MPI_Barrier(MPI_COMM_WORLD);
#endif
			GPTLstart("spmvT_other");
			MY_FLOAT_TYPE* xx_temp; //sum vector
			xx_temp = (MY_FLOAT_TYPE*) calloc(perProcData->totalNumOfColumns,
					sizeof(MY_FLOAT_TYPE));
			if (NULL == xx_temp) {
				fprintf(stderr, "xx_temp calloc fails \n");
				exit(-1);
			}
			memset(xx_temp, 0,
					sizeof(MY_FLOAT_TYPE) * (perProcData->totalNumOfColumns));
			GPTLstop("spmvT_other");

			//transpose(A)[n,m]*y[m,1]
			if (iteration == 0) {
#ifdef BARRIER
				MPI_Barrier(MPI_COMM_WORLD);
#endif
				GPTLstart("spmvT_init");
				memset(perProcData->vec_x_tAy_temp, 0,
						sizeof(MY_FLOAT_TYPE) * perProcData->totalNumOfColumns);
				cpuSpMVTranspose_CSR(perProcData->matrix_block, y,
						perProcData->vec_x_tAy_temp);
				MPI_Allreduce(perProcData->vec_x_tAy_temp, xx_temp,
						perProcData->totalNumOfColumns, MY_MPI_FLOAT_TYPE,
						MPI_SUM, MPI_COMM_WORLD);
				GPTLstop("spmvT_init");

			} else { //itn>=1
#ifdef BARRIER
				MPI_Barrier(MPI_COMM_WORLD);
#endif
				GPTLstart("spmvT");
#if GPUALL==1
#if TRANSPOSE_AVOID==1//clear 0 is done within
				myspmvTransKer_w(perProcData->matrix_block, y, perProcData->tAy_gpu);
#else//clear 0 is done within  cusparseSpMVTrans_w
				cusparseSpMVTrans_w(perProcData->matrix_block, y/* m,*/, perProcData->tAy_gpu);
#endif
				GPTLstart("cuda_memcpy_vecx1");
				cudaMemcpy_w(perProcData->vec_x_tAy_temp, perProcData->tAy_gpu, sizeof(MY_FLOAT_TYPE)*(perProcData->totalNumOfColumns), mycudaMemcpyDeviceToHost);
				GPTLstop("cuda_memcpy_vecx1");
#else//CPU
				memset(perProcData->vec_x_tAy_temp, 0,
						sizeof(MY_FLOAT_TYPE) * perProcData->totalNumOfColumns);
				cpuSpMVTranspose_CSR(perProcData->matrix_block, y,
						perProcData->vec_x_tAy_temp);
#endif
#ifdef GPTL
				GPTLstop("spmvT");
#endif
#ifdef BARRIER
				MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
				GPTLstart("Allreduce");
#endif
				MPI_Allreduce(perProcData->vec_x_tAy_temp, xx_temp,
						perProcData->totalNumOfColumns, MY_MPI_FLOAT_TYPE,
						MPI_SUM, MPI_COMM_WORLD);
				GPTLstop("Allreduce");
			}

			//copy x back to GPU

			if (iteration == 0) { //update x[]
#ifdef BARRIER
				MPI_Barrier(MPI_COMM_WORLD);
#endif
				GPTLstart("init_update_x");
#pragma ivdep
				for (i = 0; i < perProcData->totalNumOfColumns; i++)
					x[i] += xx_temp[i];
				GPTLstop("init_update_x");
			} else { //itn>1
#ifdef BARRIER
				MPI_Barrier(MPI_COMM_WORLD);
#endif
				GPTLstart("update_x");
#if GPUALL==1
				GPTLstart("cuda_memcpy_vecx2");
				cudaMemcpy_w(perProcData->tAy_gpu, xx_temp, sizeof(MY_FLOAT_TYPE)*(perProcData->totalNumOfColumns), mycudaMemcpyHostToDevice);
				GPTLstop("cuda_memcpy_vecx2");
				// x<-x+tAy  5/29/2011 16:45
				MY_FLOAT_TYPE alpha1 = 1.0f;
				cublasSaxpy_w(perProcData->totalNumOfColumns, &alpha1, perProcData->tAy_gpu, 1, x, 1);
#else
				//update x[]
#pragma ivdep
				for (i = 0; i < perProcData->totalNumOfColumns; i++)
					x[i] += xx_temp[i];
#endif
				GPTLstop("update_x");
			}
			free(xx_temp);
		}

	}

}
#endif // PLSQR2

//-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#ifdef ONEDIAGONAL
void aprod_mode_1_onediagonal(int procRank, int numProc, int iteration,
		int isFinalIteration, PerProcDataV3 * p) {
	int i;

	MY_FLOAT_TYPE * X = p->vec_x;
	/*kernel part CSC*/
	int loc_numRow_k = p->matCSC_k->numRow;

	MY_FLOAT_TYPE * temp_vec_y_Ax_k = (MY_FLOAT_TYPE*) calloc(loc_numRow_k,
			sizeof(MY_FLOAT_TYPE));
	if (temp_vec_y_Ax_k == NULL) {
		fprintf(stderr, "can not allocate temp_vec_y_Ax_k in aprod\n");
		exit(EXIT_FAILURE);
	}
	MY_FLOAT_TYPE * temp_vec_y_Ax_k_allreduce = (MY_FLOAT_TYPE*) calloc(
			loc_numRow_k, sizeof(MY_FLOAT_TYPE)); //record reduced temporal result
	if (temp_vec_y_Ax_k_allreduce == NULL) {
		fprintf(stderr,
				"can not allocate  temp_vec_y_Ax_k_allreduce  in aprod\n");
		exit(EXIT_FAILURE);
	}
	//set to 0
	memset(temp_vec_y_Ax_k_allreduce, 0, loc_numRow_k * sizeof(MY_FLOAT_TYPE));
	memset(temp_vec_y_Ax_k, 0, loc_numRow_k * sizeof(MY_FLOAT_TYPE));

	int firstIdxInVecX = p->aj_k - p->vec_x_begin_gloIdx;//local index of vector x

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("spmv_ker");
#endif
	cpuSpMV_CSC(p->matCSC_k, &X[firstIdxInVecX], temp_vec_y_Ax_k);
#ifdef GPTL
	GPTLstop("spmv_ker");
#endif

	//MPI_Allreduce
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("allreduce_ker");
#endif
	MPI_Allreduce(temp_vec_y_Ax_k, temp_vec_y_Ax_k_allreduce, loc_numRow_k,
			MY_MPI_FLOAT_TYPE, MPI_SUM, MPI_COMM_WORLD);
#ifdef GPTL
	GPTLstop("allreduce_ker");
#endif
	for (i = 0; i < loc_numRow_k; i++) {
		p->vec_y_k[i] += temp_vec_y_Ax_k_allreduce[i];
	}

#ifdef DEBUG	//aprod1_tempOutput
	char fname0[1024];
	sprintf(fname0, "aprod1_tempOutput_itn_%d_rank_%d" , iteration, procRank);
	FILE * aprod1=fopen(fname0, "w");
	if(aprod1==NULL) {
		fprintf(stderr, "cannot open %s\n", fname0);
		exit(EXIT_FAILURE);
	}
	fprintf(aprod1, "temp_vec_y_Ax_k\n");
	for(i=0; i<loc_numRow_k; i++) {
		fprintf(aprod1, "[%d]=%e ", i, temp_vec_y_Ax_k[i]);
	}
	fprintf(aprod1, "\n\ntemp_vec_y_Ax_k_allreduce\n");
	for(i=0; i<loc_numRow_k; i++) {
		fprintf(aprod1, "[%d]=%e ", i, temp_vec_y_Ax_k_allreduce[i]);
	}
	fprintf(aprod1, "\n\np->vec_y_k\n");
	for(i=0; i<loc_numRow_k; i++) {
		fprintf(aprod1, "[%d]=%e ", i, p->vec_y_k[i]);
	}
#endif

	free(temp_vec_y_Ax_k);
	free(temp_vec_y_Ax_k_allreduce);
	/*end of kernel part CSC*/

	/*damping part CSR*/
	int loc_numRow_d = p->matCSR_d->numRow;
	MY_FLOAT_TYPE * temp_vec_y_Ax_d = (MY_FLOAT_TYPE *) calloc(loc_numRow_d,
			sizeof(MY_FLOAT_TYPE));
	if (temp_vec_y_Ax_d == NULL) {
		fprintf(stderr, "can not allocate temp_vec_y_Ax_d in aprod\n");
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < loc_numRow_d; i++) {
		temp_vec_y_Ax_d[i] = 0.0;
	}
//#ifdef  ONEDIAGONAL
	firstIdxInVecX = p->aj_d - p->vec_x_begin_gloIdx;
//#endif

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("spmv_damp");
#endif
	cpuSpMV_CSR(p->matCSR_d, &X[firstIdxInVecX], temp_vec_y_Ax_d);
#ifdef GPTL
	GPTLstop("spmv_damp");
#endif

	for (i = 0; i < loc_numRow_d; i++) {
		p->vec_y_d[i] += temp_vec_y_Ax_d[i];
	}
#ifdef DEBUG	//aprod1_tempOutput
	fprintf(aprod1, "\n\ntemp_vec_y_Ax_d\n");
	for(i=0; i<loc_numRow_d; i++) {
		fprintf(aprod1, "[%d]/[%d]=%e ", i, i+p->vec_y_begin_gloIdx_d, temp_vec_y_Ax_d[i]);
	}
	fprintf(aprod1, "\n\np->vec_y_d\n");
	for(i=0; i<loc_numRow_d; i++) {
		fprintf(aprod1, "[%d]/[%d]=%e ", i, i+p->vec_y_begin_gloIdx_d, p->vec_y_d[i]);
	}
	fclose(aprod1);
#endif
	free(temp_vec_y_Ax_d);
	/*end of damping part CSR*/

}
void aprod_mode_2_onediagonal(int procRank, int numProc, int iteration,
		int isFinalIteration, PerProcDataV3 * p) {
	int i;

	/*kernel part*/
	int loc_vec_x_len = p->vec_x_end_gloIdx - p->vec_x_begin_gloIdx + 1; //the length of vector is local length
	/*trans(A)*y, kernel part */MY_FLOAT_TYPE * temp_vec_x_tAy_k =
	(MY_FLOAT_TYPE *) calloc(loc_vec_x_len, sizeof(MY_FLOAT_TYPE));
	if (temp_vec_x_tAy_k == NULL) {
		fprintf(stderr, "cannot allocate temp_vec_x_ATy_k\n");
		exit(EXIT_FAILURE);
	}
	/*trans(A)*y, damping part */MY_FLOAT_TYPE * temp_vec_x_tAy_d =
	(MY_FLOAT_TYPE *) calloc(loc_vec_x_len, sizeof(MY_FLOAT_TYPE));
	if (temp_vec_x_tAy_d == NULL) {
		fprintf(stderr, "cannot allocate temp_vec_x_ATy_d\n");
		exit(EXIT_FAILURE);
	}
	/*trans(A)*y, whole part : kernel + damping */MY_FLOAT_TYPE * temp_vec_x_ATy =
	(MY_FLOAT_TYPE *) calloc(loc_vec_x_len, sizeof(MY_FLOAT_TYPE));
	if (temp_vec_x_ATy == NULL) {
		fprintf(stderr, "cannot allocate temp_vec_x_ATy\n");
		exit(EXIT_FAILURE);
	}
	for (i = 0; i < loc_vec_x_len; i++) {
		temp_vec_x_ATy[i] = temp_vec_x_tAy_d[i] = temp_vec_x_tAy_k[i] = 0.0;
	}
	int firstIdxInVecX = p->aj_k - p->vec_x_begin_gloIdx; //convert to local index
	//kernel SpMV
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("spmvT_ker");
#endif
	cpuSpMVTranspose_CSC(p->matCSC_k, p->vec_y_k,
			&temp_vec_x_tAy_k[firstIdxInVecX]);
#ifdef GPTL
	GPTLstop("spmvT_ker");
#endif

	firstIdxInVecX = p->aj_d - p->vec_x_begin_gloIdx; //convert to local index
	//damping SpMV
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("spmvT_damp");
#endif
	cpuSpMVTranspose_CSR(p->matCSR_d, p->vec_y_d,
			&temp_vec_x_tAy_d[firstIdxInVecX]);
#ifdef GPTL
	GPTLstop("spmvT_damp");
#endif

	for (i = 0; i < loc_vec_x_len; i++) { //combine kernel and damping
		temp_vec_x_ATy[i] = temp_vec_x_tAy_k[i] + temp_vec_x_tAy_d[i];
	}
#ifdef DEBUG	//aprod2_tempOutput
	char fname1[1024];
	sprintf(fname1, "aprod2_tempOutput_itn_%d_rank_%d", iteration, procRank);
	FILE * aprod2 = fopen(fname1, "w");
	if(aprod2==NULL) {
		fprintf(stderr, "cannot open %s\n", fname1);
		exit(EXIT_FAILURE);
	}
	fprintf(aprod2, "vec_y_k\n");
	for(i=0; i<p->matCSC_k->numRow; i++) {
		fprintf(aprod2, "[%d]=%e ", i, p->vec_y_k[i]);
	}
	fprintf(aprod2, "\n\nvec_y_d\n");
	for(i=0; i<p->vec_y_end_gloIdx_d - p->vec_y_begin_gloIdx_d + 1; i++) {
		fprintf(aprod2, "[%d/%d]=%e ", i, i+p->vec_y_begin_gloIdx_d, p->vec_y_d[i]);
	}
	fprintf(aprod2, "\n\ntemp_vec_x_ATy_k\n");
	for(i=0; i<loc_vec_x_len; i++) {
		fprintf(aprod2, "[%d/%d]=%e ",i, i+p->vec_x_begin_gloIdx, temp_vec_x_tAy_k[i]);
	}
	fprintf(aprod2, "\n\ntemp_vec_x_ATy_d\n");
	for(i=0; i<loc_vec_x_len; i++) {
		fprintf(aprod2, "[%d/%d]=%e ",i, i+p->vec_x_begin_gloIdx, temp_vec_x_tAy_d[i]);
	}
	fprintf(aprod2, "\n\ntemp_vec_x_ATy\n");
	for(i=0; i<loc_vec_x_len; i++) {
		fprintf(aprod2, "[%d/%d]=%e ",i, i+p->vec_x_begin_gloIdx, temp_vec_x_ATy[i]);
	}
#endif

	free(temp_vec_x_tAy_k);
	free(temp_vec_x_tAy_d);

//two phase reduction

	//compute overlap ONLY at the 0th iteration
	//exchange overlap offset index with neighboring process((((
	if (iteration == 0) {
#ifdef BARRIER
		MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
		GPTLstart("init2reduce");
#endif
		MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &overlap_offset_gloIdx1);
		*overlap_offset_gloIdx1 = 0;
		MPI_Win_create(overlap_offset_gloIdx1, sizeof(int), sizeof(int),
				MPI_INFO_NULL, MPI_COMM_WORLD, &win_overlap_offset1);
		MPI_Win_fence(0, win_overlap_offset1);
		if (procRank % 2 == 0) { //even rank, send to right neighbor
			if (procRank + 1 < numProc) { //put
				MPI_Put(&(p->vec_x_end_gloIdx), 1, MPI_INT, procRank + 1, 0, 1,
						MPI_INT, win_overlap_offset1);
			} else { // the boundary process, do nothing
				MPI_Put(&(p->vec_x_end_gloIdx), 1, MPI_INT, MPI_PROC_NULL, 0, 1,
						MPI_INT, win_overlap_offset1);
			}
		} else { //if (procRank%2==1) //odd rank, send to left neighbor
			MPI_Put(&(p->vec_x_begin_gloIdx), 1, MPI_INT, procRank - 1, 0, 1,
					MPI_INT, win_overlap_offset1);
		}
		MPI_Win_fence(0, win_overlap_offset1);

//					compute overlap length
		if (procRank % 2 == 0) {
			if (procRank + 1 < numProc) {
				overlap_len1 = p->vec_x_end_gloIdx - *overlap_offset_gloIdx1
				+ 1;
			} else
			//the boundary process, do nothing
			overlap_len1 = 0;
		} else { //if (procRank%2==1)
			overlap_len1 = *overlap_offset_gloIdx1 - p->vec_x_begin_gloIdx + 1;
		}
		//exchange overlap offset index with neighboring process))))
#ifdef DEBUG
		printf("%d: 1st phase of reduction, overlap_offset1=%d, overlap_len1=%d\n", procRank, *overlap_offset_gloIdx1, overlap_len1);
#endif

//					help info
		int min_overlap_len1, max_overlap_len1;
		MPI_Reduce(&overlap_len1, &min_overlap_len1, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
		MPI_Reduce(&overlap_len1, &max_overlap_len1, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
		if (procRank == 0) {
			printf("%d:1st phase of reduction: min_overlap_len1=%d %s(%d)-%s\n",
					procRank, min_overlap_len1, __FILE__, __LINE__,
					__FUNCTION__);
			printf("%d:1st phase of reduction: max_overlap_len1=%d %s(%d)-%s\n",
					procRank, max_overlap_len1, __FILE__, __LINE__,
					__FUNCTION__);
		}

		//exchange overlap array with neighboring process((((

		MPI_Alloc_mem(sizeof(MY_FLOAT_TYPE) * overlap_len1, MPI_INFO_NULL,
				&overlap_array1);
		MPI_Win_create(overlap_array1, overlap_len1 * sizeof(MY_FLOAT_TYPE),
				sizeof(MY_FLOAT_TYPE), MPI_INFO_NULL, MPI_COMM_WORLD,
				&win_overlap_array1);

		MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &overlap_offset_gloIdx2);
		*overlap_offset_gloIdx2 = 0;
		MPI_Win_create(overlap_offset_gloIdx2, sizeof(int), sizeof(int),
				MPI_INFO_NULL, MPI_COMM_WORLD, &win_overlap_offset2);
		MPI_Win_fence(0, win_overlap_offset2);
		if (procRank % 2 == 0) {
			if (procRank - 1 >= 0) {
				MPI_Put(&(p->vec_x_begin_gloIdx), 1, MPI_INT, procRank - 1, 0,
						1, MPI_INT, win_overlap_offset2);
			} else {
				MPI_Put(&(p->vec_x_begin_gloIdx), 1, MPI_INT, MPI_PROC_NULL, 0,
						1, MPI_INT, win_overlap_offset2);
			}
		} else {
			if (procRank + 1 < numProc)
			MPI_Put(&(p->vec_x_end_gloIdx), 1, MPI_INT, procRank + 1, 0, 1,
					MPI_INT, win_overlap_offset2);
			else
			MPI_Put(&(p->vec_x_end_gloIdx), 1, MPI_INT, MPI_PROC_NULL, 0, 1,
					MPI_INT, win_overlap_offset2);
		}
		//compute overlapped length
		MPI_Win_fence(0, win_overlap_offset2);
		if (procRank % 2 == 0) {
			if (procRank - 1 >= 0) {
				overlap_len2 = *overlap_offset_gloIdx2 - p->vec_x_begin_gloIdx
				+ 1;
			} else {
				overlap_len2 = 0;
			}
		} else {
			if (procRank + 1 < numProc)
			overlap_len2 = p->vec_x_end_gloIdx - *overlap_offset_gloIdx2
			+ 1;
			else
			overlap_len2 = 0;
		}
		//exchange overlap offset index with neighboring process>>>
#ifdef DEBUG
		printf("%d: second set of reduction, overlap_offset2=%d, overlap_len=%d\n", procRank, *overlap_offset_gloIdx2, overlap_len2);
#endif
		int min_overlap_len2, max_overlap_len2;
		MPI_Reduce(&overlap_len2, &min_overlap_len2, 1, MPI_INT, MPI_MIN, 0,
				MPI_COMM_WORLD);
		MPI_Reduce(&overlap_len2, &max_overlap_len2, 1, MPI_INT, MPI_MAX, 0,
				MPI_COMM_WORLD);
		if (procRank == 0) {
			printf("%d:2nd phase of reduction: min_overlap_len2=%d %s(%d)-%s\n",
					procRank, min_overlap_len2, __FILE__, __LINE__,
					__FUNCTION__);
			printf("%d:2nd phase of reduction: max_overlap_len2=%d %s(%d)-%s\n",
					procRank, max_overlap_len2, __FILE__, __LINE__,
					__FUNCTION__);
		}
		//exchange overlap array with neighboring process<<<
		MPI_Alloc_mem(sizeof(MY_FLOAT_TYPE) * overlap_len2, MPI_INFO_NULL,
				&overlap_array2);
		MPI_Win_create(overlap_array2, overlap_len2 * sizeof(MY_FLOAT_TYPE),
				sizeof(MY_FLOAT_TYPE), MPI_INFO_NULL, MPI_COMM_WORLD,
				&win_overlap_array2);
		GPTLstop("init2reduce");
	} //end of iteration=0
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	//first phase of reduce(((
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	GPTLstart("2reduce");
	GPTLstart("1st_reduce");
#if MPICOMMTYPE==1
	MPI_Win_fence(0, win_overlap_array1);
#elif MPICOMMTYPE==2
	MPI_Status sta1, sta2; MPI_Request reqs1, reqs2;
#endif
	if (procRank % 2 == 0) { //even rank
		if (procRank + 1 < numProc) { //*overlap_offset - p->vec_x_begin_gloIdx: convert to local index
#if MPICOMMTYPE==1
			MPI_Put(&(temp_vec_x_ATy[*overlap_offset_gloIdx1 - p->vec_x_begin_gloIdx]), overlap_len1,
					MY_MPI_FLOAT_TYPE, procRank + 1, 0, overlap_len1, MY_MPI_FLOAT_TYPE, win_overlap_array1);
#elif MPICOMMTYPE==2
			MPI_Irecv(overlap_array1, overlap_len1, MY_MPI_FLOAT_TYPE, procRank+1, 2, MPI_COMM_WORLD, &reqs2);
			MPI_Isend(&(temp_vec_x_ATy[*overlap_offset_gloIdx1 - p->vec_x_begin_gloIdx]), overlap_len1,
					MY_MPI_FLOAT_TYPE, procRank+1, 1, MPI_COMM_WORLD, &reqs1);
			//??overlap_array1 allocated by mpi, OK? 9/16/2011
			MPI_Wait(&reqs1, &sta1);
			MPI_Wait(&reqs2, &sta2);
#endif
		} else { //the boundary process, do nothing
#if MPICOMMTYPE==1
			MPI_Put(&(temp_vec_x_ATy[*overlap_offset_gloIdx1 - p->vec_x_begin_gloIdx]), overlap_len1,
					MY_MPI_FLOAT_TYPE, MPI_PROC_NULL, 0, overlap_len1, MY_MPI_FLOAT_TYPE, win_overlap_array1);
#endif
		}
	} else { //if (procRank%2==1) //odd rank
#if MPICOMMTYPE==1
//				if(procRank-1>=0)
		MPI_Put(&(temp_vec_x_ATy[0]), overlap_len1, MY_MPI_FLOAT_TYPE,
				procRank - 1, 0, overlap_len1, MY_MPI_FLOAT_TYPE,
				win_overlap_array1);
//				else	MPI_Put(&(temp_vec_x_ATy[0]), overlap_len, MY_MPI_FLOAT_TYPE, MPI_PROC_NULL, 0,overlap_len, MY_MPI_FLOAT_TYPE, win_overlap_array);
#elif MPICOMMTYPE==2
		MPI_Irecv(overlap_array1, overlap_len1, MY_MPI_FLOAT_TYPE, procRank-1, 1, MPI_COMM_WORLD, &reqs1);
		MPI_Isend(&(temp_vec_x_ATy[0]), overlap_len1, MY_MPI_FLOAT_TYPE, procRank-1, 2, MPI_COMM_WORLD, &reqs2);
		MPI_Wait(&reqs1, &sta1);
		MPI_Wait(&reqs2, &sta2);
#endif
	}
#if MPICOMMTYPE==1
	MPI_Win_fence(0, win_overlap_array1);
#elif MPICOMMTYPE==2
//			MPI_Wait(&reqs1, &sta1);
//			MPI_Wait(&reqs2, &sta2);
#endif
	GPTLstop("1st_reduce");
	//exchange overlap array with neighboring process))))

	//added up with overlap
	int k;
	if (procRank % 2 == 0) {
		for (k = 0; k < overlap_len1; k++) { //vector uses local index
			temp_vec_x_ATy[*overlap_offset_gloIdx1 - p->vec_x_begin_gloIdx + k] +=
			overlap_array1[k];
		}
	} else {
		for (k = 0; k < overlap_len1; k++) { //vector uses local index
			temp_vec_x_ATy[0 + k] += overlap_array1[k];
		}
	}
//end of first set of reduce)))

//second phase of reduce (((
	//exchange overlap offset index with neighboring process<<<
#if MPICOMMTYPE==1
	MPI_Win_fence(0, win_overlap_array2);
#endif
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	GPTLstart("2nd_reduce");
	if (procRank % 2 == 0) { //even rank
		if (procRank - 1 >= 0) {
#if MPICOMMTYPE==1
			MPI_Put(&(temp_vec_x_ATy[0]), overlap_len2, MY_MPI_FLOAT_TYPE,
					procRank - 1, 0, overlap_len2, MY_MPI_FLOAT_TYPE,
					win_overlap_array2);
#elif MPICOMMTYPE==2
			MPI_Irecv(overlap_array2, overlap_len2, MY_MPI_FLOAT_TYPE, procRank-1, 2, MPI_COMM_WORLD, &reqs2);
			MPI_Isend(&(temp_vec_x_ATy[0]), overlap_len2, MY_MPI_FLOAT_TYPE, procRank-1, 1, MPI_COMM_WORLD, &reqs1);
			MPI_Wait(&reqs1, &sta1);
			MPI_Wait(&reqs2, &sta2);
#endif
		} else {
#if MPICOMMTYPE==1
			MPI_Put(&(temp_vec_x_ATy[0]), overlap_len2, MY_MPI_FLOAT_TYPE,
					MPI_PROC_NULL, 0, overlap_len2, MY_MPI_FLOAT_TYPE,
					win_overlap_array2);
#endif
		}
	} else { //odd reank
		if (procRank + 1 < numProc) {
#if MPICOMMTYPE==1
			MPI_Put(
					&(temp_vec_x_ATy[*overlap_offset_gloIdx2
							- p->vec_x_begin_gloIdx]), overlap_len2,
					MY_MPI_FLOAT_TYPE, procRank + 1, 0, overlap_len2,
					MY_MPI_FLOAT_TYPE, win_overlap_array2);
#elif MPICOMMTYPE==2
			MPI_Irecv(overlap_array2, overlap_len2, MY_MPI_FLOAT_TYPE, procRank+1, 1, MPI_COMM_WORLD, &reqs1);
			MPI_Isend(&(temp_vec_x_ATy[*overlap_offset_gloIdx2 - p->vec_x_begin_gloIdx]), overlap_len2, MY_MPI_FLOAT_TYPE, procRank+1, 2, MPI_COMM_WORLD, &reqs2);
			MPI_Wait(&reqs1, &sta1);
			MPI_Wait(&reqs2, &sta2);
#endif
		} else {
#if MPICOMMTYPE==1
			MPI_Put(
					&(temp_vec_x_ATy[*overlap_offset_gloIdx2
							- p->vec_x_begin_gloIdx]), overlap_len2,
					MY_MPI_FLOAT_TYPE, MPI_PROC_NULL, 0, overlap_len2,
					MY_MPI_FLOAT_TYPE, win_overlap_array2);
#endif
		}
	}
#if MPICOMMTYPE==1
	MPI_Win_fence(0, win_overlap_array2);
#elif MPICOMMTYPE==2
//			MPI_Wait(&reqs1, &sta1);
//			MPI_Wait(&reqs2, &sta2);
#endif

	GPTLstop("2nd_reduce");
	if (procRank % 2 == 0) {
		for (k = 0; k < overlap_len2; k++) {
			temp_vec_x_ATy[0 + k] += overlap_array2[k];
		}
	} else {
		for (k = 0; k < overlap_len2; k++) {
			temp_vec_x_ATy[*overlap_offset_gloIdx2 - p->vec_x_begin_gloIdx + k] +=
			overlap_array2[k];
		}
	}
//>>>end of second set of reduction

	//add with vector x's previous value
	for (i = 0; i < loc_vec_x_len; i++) {
		p->vec_x[i] += temp_vec_x_ATy[i];
	}

#ifdef	DEBUG	//aprod2_tempOutput
	fprintf(aprod2, "\nafter 2phase reduction \np->vec_x \n");
	for(i=0; i<loc_vec_x_len; i++) {
		fprintf(aprod2, "[%d/%d]=%e ",i, i+p->vec_x_begin_gloIdx, p->vec_x[i]);
		//for matlab print
//				fprintf(aprod2, "p_vec_x(%d)=%e ", i+p->vec_x_begin_gloIdx+1, p->vec_x[i]);
	}
	fclose(aprod2);
#endif

	GPTLstop("2reduce");
	if (isFinalIteration == 1) {
		MPI_Win_free(&win_overlap_offset2);
		MPI_Win_free(&win_overlap_array2);
		MPI_Free_mem(overlap_offset_gloIdx2);
		MPI_Free_mem(overlap_array2);
		MPI_Win_free(&win_overlap_offset1);
		MPI_Win_free(&win_overlap_array1);
		MPI_Free_mem(overlap_offset_gloIdx1);
		MPI_Free_mem(overlap_array1);
	}

	free(temp_vec_x_ATy);
}
#endif

//--------------------------------------------------------------------------------------------------------------------------



/*
 int vecXLen = p->vec_x_lenp->matCSC_k->numCol;// full length of vector x
 //trans(A)*y, kernel
 MY_FLOAT_TYPE * temp_vec_x_tAy_k = (MY_FLOAT_TYPE *) malloc(vecXLen * sizeof(MY_FLOAT_TYPE));
 //trans(A)*y, damping
 MY_FLOAT_TYPE * temp_vec_x_tAy_d = (MY_FLOAT_TYPE *) malloc(vecXLen * sizeof(MY_FLOAT_TYPE));
 if(temp_vec_x_tAy_k==NULL || temp_vec_x_tAy_d==NULL){
 fprintf(stderr, "%d: %s(%d)-%s: cannot allocate TEMP_VEC_X_tAy_* \n ", procRank, __FILE__,__LINE__,__FUNCTION__); exit(-1);
 }
 for(i=0; i<vecXLen; i++ ){
 temp_vec_x_tAy_k[i]=temp_vec_x_tAy_d[i]=0.0; //assign to 0.0, later will compare with 0.0 in VEC_POS
 }

 GPTLstart("spmvT_ker");
 cpuSpMVTranspose_CSC(p->matCSC_k, p->vec_y_k , &temp_vec_x_tAy_k[p->vec_x_pos.kerStart]);
 GPTLstop("spmvT_ker");

 GPTLstart("spmvT_damp");
 cpuSpMVTranspose_CSR(p->matCSR_d, p->vec_y_d , temp_vec_x_tAy_d );
 GPTLstop("spmvT_damp");
 //determine vector x position information
 if(iteration==0){
 //projecting damping part of matrix A's column index to a vector, if the element of the vector is 0, no projecting; otherwise, projecting
 int * projectedVecX = (int *) malloc(n*sizeof(int));
 memset(projectedVecX, 0, n*sizeof(int));
 projectingAColumnToX(p->matCSR_d, projectedVecX);
 determineVEC_POS( procRank, numProc, &p->vec_x_pos, projectedVecX, vecXLen);

 free(projectedVecX);
 }


 //combine kernel and damping, added with vec_x's previous value
 for(i=0; i<vecXLen; i++){
 p->vec_x[i] += (temp_vec_x_tAy_k[i] + temp_vec_x_tAy_d[i]);
 }
 #ifdef DEBUG	//aprod2_tempOutput
 int loc_vec_x_len = vecXLen;
 char fname1[1024];
 sprintf(fname1, "aprod2_tempOutput_itn_%d_rank_%d", iteration, procRank);
 FILE * aprod2 = fopen(fname1, "w");
 if(aprod2==NULL){
 fprintf(stderr, "cannot open %s\n", fname1);
 exit(EXIT_FAILURE);
 }
 fprintf(aprod2, "vec_y_k\n");
 for(i=0; i<p->matCSC_k->numRow; i++){
 fprintf(aprod2, "[%d]=%e ", i, p->vec_y_k[i]);
 }
 fprintf(aprod2, "\n\nvec_y_d\n");
 for(i=0; i<p->vec_y_end_gloIdx_d - p->vec_y_begin_gloIdx_d + 1; i++){
 fprintf(aprod2, "[%d/%d]=%e ", i, i+p->vec_y_begin_gloIdx_d, p->vec_y_d[i]);
 }
 fprintf(aprod2, "\n\ntemp_vec_x_ATy_k\n");
 for(i=0; i<loc_vec_x_len; i++){
 fprintf(aprod2, "[%d/%d]=%e ",i, i, temp_vec_x_tAy_k[i]);
 }
 fprintf(aprod2, "\n\ntemp_vec_x_ATy_d\n");
 for(i=0; i<loc_vec_x_len; i++){
 fprintf(aprod2, "[%d/%d]=%e ",i, i, temp_vec_x_tAy_d[i]);
 }
 fprintf(aprod2, "\n\np->vec_x\n");
 for(i=0; i<loc_vec_x_len; i++){
 fprintf(aprod2, "[%d/%d]=%e ",i, i, p->vec_x[i]);
 }
 #endif
 free(temp_vec_x_tAy_k);
 free(temp_vec_x_tAy_d);

 //			.... add more functions here
 */

/*

 void printaprod(const SpMatCSR * A, const MY_FLOAT_TYPE x[], const MY_FLOAT_TYPE y[]) {
 int i, j;
 for (i = 0; i < A->numRow; i++) {
 int firstEntityInRow = A->row_ptr[i];
 int nextFirstEntityInRow = A->row_ptr[i + 1];
 for (j = firstEntityInRow; j < nextFirstEntityInRow; j++) {
 printf("A->data[%d][%d]=%e\n", i, j, A->val[j]);
 }
 }
 for (i = 0; i < A->numCol; i++) {
 printf("x[%d]=\t%e\n", i, x[i]);
 }
 for (i = 0; i < A->numRow; i++) {
 printf("y[%d]=\t%e\n", i, y[i]);
 }
 }
 */
