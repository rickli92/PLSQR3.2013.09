//#define CUDA is defined in aprod.c, uncomment this if using CUDA
//#define DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

#include "aprod_s.h"
#include "../common.h"
#include "../CPUSpMV.h"
//#include "../GPUComputeSpMV.cuh"
//#include "../tool/gptl.h"
#include "../FileIO.h"

/*
 *
 * */
void aprod_s(int mode, int m /*row*/,int n /*column*/, MY_FLOAT_TYPE x[]/*v len=n*/,
		MY_FLOAT_TYPE y[] , void *UsrWork)
{
	int i;

//	if (version == PLSQRV2) {

		Per_Proc_Data_v2 *perProcData = (Per_Proc_Data_v2*) UsrWork;

		if (mode == 1) {//y[m,1] = y[m,1] + A[m,n]*x[n,1]
			//GPTLstart("SpMvM A");
			int loc_numOfRows = perProcData->matrix_block->numRow;
			for (i = 0; i < loc_numOfRows; i++)//can be optimized with memcpy ...
				perProcData->vec_y_Ax_temp[i] = 0.0;

#ifdef CUDA
			gpuSpMV_CSR(perProcData->matrix_block,x,perProcData->vec_y_Ax_temp, procRank, isFinalIteration);
#else
			cpuSpMV_CSR(perProcData->matrix_block, x,perProcData->vec_y_Ax_temp);
#endif
//			GPTLstop("SpMvM A");

			//update y[]
#pragma ivdep
			for (i = 0; i < loc_numOfRows; i++) {
				y[i] += perProcData->vec_y_Ax_temp[i];
			}

		} else if (mode == 2)//mode =2
		{//x[n,1] = x[n,1] + transpose(A)[n,m]*y[m,1]
			//GPTLstart("SpMvM At");
			for (i = 0; i < perProcData->totalNumOfColumns; i++) {
				perProcData->vec_x_tAy_temp[i] = 0.0;//clear up every iteration
			}
			//transpose(A)[n,m]*y[m,1]
#ifdef CUDA
			gpuSpMVTranspose_CSR(perProcData->matrix_block,y, m, perProcData->vec_x_tAy_temp, procRank, isFinalIteration);
#else
			cpuSpMVTranspose_CSR(perProcData->matrix_block,	y, perProcData->vec_x_tAy_temp );
#endif


			//update x[]
			int k;
			for (k = 0; k < perProcData->totalNumOfColumns; k++)
				x[k] += /*xx_temp[k];*/ perProcData->vec_x_tAy_temp[k];
		}
//	}
///////////////////////////////////////PLSQRv3///////////////////////////////////////////////////////////////////////////////////////////////
/*	else if (version==PLSQRV3)
	{
		PerProcDataV3 * p = (PerProcDataV3 *) UsrWork;
		if (mode == 1) {//y <- y+Ax
			MY_FLOAT_TYPE * X= p->vec_x;
			kernel part CSC
			int loc_numRow_k=p->block_k->numRow;
			MY_FLOAT_TYPE * temp_vec_y_Ax_k= (MY_FLOAT_TYPE* )calloc(loc_numRow_k, sizeof(MY_FLOAT_TYPE));
			if(temp_vec_y_Ax_k==NULL){
				fprintf(stderr, "can not allocate temp_vec_y_Ax_k in aprod\n"); exit(EXIT_FAILURE);
			}
			MY_FLOAT_TYPE * temp_vec_y_Ax_k_allreduce= (MY_FLOAT_TYPE* )calloc(loc_numRow_k, sizeof(MY_FLOAT_TYPE));//record reduced temporal result
			if(temp_vec_y_Ax_k_allreduce==NULL){
				fprintf(stderr, "can not allocate  temp_vec_y_Ax_k_allreduce  in aprod\n"); exit(EXIT_FAILURE);
			}
			for(i=0; i< loc_numRow_k; i++){
				temp_vec_y_Ax_k_allreduce[i]=temp_vec_y_Ax_k[i]=0.0;
			}
			int firstIdxInVecX =p->aj_k - p->vec_x_begin_gloIdx;
			cpuSpMV_CSC(p->block_k, &X[firstIdxInVecX], temp_vec_y_Ax_k);
			//MPI_Allreduce
			MPI_Allreduce (temp_vec_y_Ax_k , temp_vec_y_Ax_k_allreduce, loc_numRow_k, MY_MPI_FLOAT_TYPE, MPI_SUM, MPI_COMM_WORLD );
			for(i=0; i< loc_numRow_k; i++){
				p->vec_y_k[i] += temp_vec_y_Ax_k_allreduce[i];
			}
			free(temp_vec_y_Ax_k);
			free(temp_vec_y_Ax_k_allreduce);

			damping part CSR
			int loc_numRow_d=p->block_d->numRow;
			MY_FLOAT_TYPE * temp_vec_y_Ax_d = (MY_FLOAT_TYPE *) calloc(loc_numRow_d, sizeof(MY_FLOAT_TYPE));
			if(temp_vec_y_Ax_d==NULL){
				fprintf(stderr, "can not allocate temp_vec_y_Ax_d in aprod\n"); exit(EXIT_FAILURE);
			}
			for(i=0; i<loc_numRow_d; i++){
				temp_vec_y_Ax_d[i]=0.0;
			}
			firstIdxInVecX = p->aj_d - p->vec_x_begin_gloIdx;
			cpuSpMV_CSR(p->block_d, &X[firstIdxInVecX], temp_vec_y_Ax_d);
			for(i=0; i<loc_numRow_d; i++){
				p->vec_y_d[i] += temp_vec_y_Ax_d[i];
			}
			free(temp_vec_y_Ax_d);

		} else if (mode == 2){//x<-x+AT*y
			kernel part
			int loc_vec_x_len=p->vec_x_end_gloIdx - p->vec_x_begin_gloIdx +1;
			MY_FLOAT_TYPE * temp_vec_x_ATy_k = (MY_FLOAT_TYPE *) calloc(loc_vec_x_len, sizeof(int));
			MY_FLOAT_TYPE * temp_vec_x_ATy_d = (MY_FLOAT_TYPE *) calloc(loc_vec_x_len, sizeof(int));
			MY_FLOAT_TYPE * temp_vec_x_ATy = (MY_FLOAT_TYPE *) calloc(loc_vec_x_len, sizeof(int));
			for(i=0; i<loc_vec_x_len; i++){
				temp_vec_x_ATy[i]=temp_vec_x_ATy_d[i]=temp_vec_x_ATy_k[i]=0.0;
			}
			int firstIdxInVecX =p->aj_k - p->vec_x_begin_gloIdx;
			cpuSpMVTranspose_CSC(p->block_k, p->vec_y_k ,  &temp_vec_x_ATy_k[firstIdxInVecX]);
			firstIdxInVecX = p->aj_d - p->vec_x_begin_gloIdx;
			cpuSpMVTranspose_CSR(p->block_d, p->vec_y_d , &temp_vec_x_ATy_d[firstIdxInVecX] );
			for(i=0; i<loc_vec_x_len; i++){
				temp_vec_x_ATy[i]=temp_vec_x_ATy_k[i]+temp_vec_x_ATy_d[i];
			}
			free(temp_vec_x_ATy_k);
			free(temp_vec_x_ATy_d);
//two sets of reduce
//first set of reduce
			//exchange overlap offset index with neighboring process((((
			MPI_Barrier(MPI_COMM_WORLD);
			printf("%d:pos1\n", procRank);

			int *  overlap_offset=(int *)malloc(sizeof(int));
			MPI_Alloc_mem(sizeof(int), MPI_INFO_NULL, &overlap_offset);
			MPI_Win win_overlap_offset;
			MPI_Win_create(overlap_offset, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_overlap_offset);

			MPI_Win_fence(0MPI_MODE_NOPRECEDE, win_overlap_offset);
			if(procRank%2==0  ){
				if(procRank+1<numProc){//put
					MPI_Put(&(p->vec_x_end_gloIdx), 1, MPI_INT, procRank+1, 0, 1, MPI_INT, win_overlap_offset);
				}
				else{
					MPI_Put(&(p->vec_x_end_gloIdx), 1, MPI_INT, MPI_PROC_NULL, 0, 1, MPI_INT, win_overlap_offset);
				}
			}
			else //if (procRank%2==1)
			{
				MPI_Put(&(p->vec_x_begin_gloIdx), 1, MPI_INT, procRank-1, 0, 1, MPI_INT, win_overlap_offset);
			}
			MPI_Win_fence(0(MPI_MODE_NOSTORE | MPI_MODE_NOSUCCEED), win_overlap_offset);

			int overlap_len;
			if(procRank%2==0){
				if(procRank+1<numProc){
					overlap_len= p->vec_x_end_gloIdx - *overlap_offset;
				}else
					overlap_len=0;
			}else //if (procRank%2==1)
			{
				overlap_len=*overlap_offset - p->vec_x_begin_gloIdx;
			}
			//exchange overlap offset index with neighboring process))))
#ifdef DEBUG
			printf("%d: first set of reduction, overlap_len=%d\n", procRank, overlap_len);
#endif
			//exchange overlap array with neighboring process((((
			MY_FLOAT_TYPE * overlap_array;
			MPI_Alloc_mem(sizeof(MY_FLOAT_TYPE)*overlap_len, MPI_INFO_NULL, &overlap_array);
			MPI_Win win_overlap_array;
			MPI_Win_create(overlap_array, overlap_len*sizeof(MY_FLOAT_TYPE), sizeof(MY_FLOAT_TYPE), MPI_INFO_NULL, MPI_COMM_WORLD, &win_overlap_array);
			MPI_Win_fence(0, win_overlap_array);
			if(procRank%2==0){
				if(procRank+1<=numProc-1){
					MPI_Put(&(temp_vec_x_ATy[*overlap_offset - p->vec_x_begin_gloIdx]), overlap_len,
							MY_MPI_FLOAT_TYPE, procRank+1, 0, overlap_len, MY_MPI_FLOAT_TYPE, win_overlap_array);
				}
				else//do nthing
					MPI_Put(&(temp_vec_x_ATy[*overlap_offset - p->vec_x_begin_gloIdx]), overlap_len,
							MY_MPI_FLOAT_TYPE, MPI_PROC_NULL, 0, overlap_len, MY_MPI_FLOAT_TYPE, win_overlap_array);
			}else //if (procRank%2==1)
			{
				if(procRank-1>=0)
					MPI_Put(&(temp_vec_x_ATy[0]), overlap_len, MY_MPI_FLOAT_TYPE, procRank-1, 0,overlap_len, MY_MPI_FLOAT_TYPE, win_overlap_array);
				else	MPI_Put(&(temp_vec_x_ATy[0]), overlap_len, MY_MPI_FLOAT_TYPE, MPI_PROC_NULL, 0,overlap_len, MY_MPI_FLOAT_TYPE, win_overlap_array);
			}
			MPI_Win_fence(0, win_overlap_array);
			//exchange overlap array with neighboring process))))
			//add with overlap
			int k;
			for(k=0; k<overlap_len; k++){
				if(procRank%2==0){
					temp_vec_x_ATy[*overlap_offset - p->vec_x_begin_gloIdx + k]+=overlap_array[k];
				}else{
					temp_vec_x_ATy[0+k]+=overlap_array[k];
				}
			}
//end of first set of reduce

//second set of reduce
			//exchange overlap offset index with neighboring process((((
			MPI_Win_fence(0MPI_MODE_NOPRECEDE, win_overlap_offset);
			if(procRank%2==0){
				if(procRank-1>=0){
					MPI_Put(&(p->vec_x_begin_gloIdx), 1, MPI_INT, procRank-1, 0, 1, MPI_INT, win_overlap_offset);
				}else{
					MPI_Put(&(p->vec_x_begin_gloIdx), 1, MPI_INT, MPI_PROC_NULL, 0, 1, MPI_INT, win_overlap_offset);
				}
			}else{
				if(procRank+1<=numProc-1)
					MPI_Put(&(p->vec_x_end_gloIdx), 1, MPI_INT, procRank+1, 0, 1, MPI_INT, win_overlap_offset);
				else
					MPI_Put(&(p->vec_x_end_gloIdx), 1, MPI_INT, MPI_PROC_NULL, 0, 1, MPI_INT, win_overlap_offset);
			}
			MPI_Win_fence(0(MPI_MODE_NOSTORE | MPI_MODE_NOSUCCEED), win_overlap_offset);
			if(procRank%2==0){
				if(procRank-1>=0){
					overlap_len=*overlap_offset - p->vec_x_begin_gloIdx;
				}else{
					overlap_len=0;
				}
			}else{
				if(procRank+1<=numProc-1)
					overlap_len=p->vec_x_end_gloIdx - *overlap_offset;
				else overlap_len=0;
			}
#ifdef DEBUG
			printf("%d: second set of reduction, overlap_len=%d\n", procRank, overlap_len);
#endif
			MPI_Win_fence(0, win_overlap_array);
			if(procRank%2==0){
				if(procRank-1>=0){
					MPI_Put(&(temp_vec_x_ATy[0]), overlap_len, MY_MPI_FLOAT_TYPE,
						procRank-1, 0, overlap_len, MY_MPI_FLOAT_TYPE, win_overlap_array);
				}else
					MPI_Put(&(temp_vec_x_ATy[0]), overlap_len, MY_MPI_FLOAT_TYPE,
							MPI_PROC_NULL, 0, overlap_len, MY_MPI_FLOAT_TYPE, win_overlap_array);
			}else{
				if(procRank+1<=numProc-1)
					MPI_Put(&(temp_vec_x_ATy[*overlap_offset - p->vec_x_begin_gloIdx]), overlap_len, MY_MPI_FLOAT_TYPE,	procRank+1, 0, overlap_len, MY_MPI_FLOAT_TYPE, win_overlap_array);
				else
					MPI_Put(&(temp_vec_x_ATy[*overlap_offset - p->vec_x_begin_gloIdx]), overlap_len, MY_MPI_FLOAT_TYPE,	MPI_PROC_NULL, 0, overlap_len, MY_MPI_FLOAT_TYPE, win_overlap_array);
			}
			MPI_Win_fence(0, win_overlap_array);
			for(k=0; k<overlap_len; k++){
				if(procRank%2==0){
					temp_vec_x_ATy[0+k] += overlap_array[k];
				}else{
					temp_vec_x_ATy[*overlap_offset - p->vec_x_begin_gloIdx + k] += overlap_array[k];
				}
			}
//end of second set of reduction

			for(i=0; i<loc_vec_x_len; i++){
				p->vec_x[i] += temp_vec_x_ATy[i];
			}
			MPI_Win_free(&win_overlap_offset);
			MPI_Win_free(&win_overlap_array);//cannot free when np=3
			MPI_Free_mem(overlap_offset);
			MPI_Free_mem(overlap_array);
			free(temp_vec_x_ATy);

			//accumulate vector x's value in every process to one process, and assign it to vector v
			MPI_Win win_vec;
			MPI_Win_create(x, n*sizeof(MY_FLOAT_TYPE), sizeof(MY_FLOAT_TYPE), MPI_INFO_NULL, MPI_COMM_WORLD, &win_vec);
			MPI_Win_fence(0, win_vec);
			int count=p->vec_x_end_gloIdx - p->vec_x_begin_gloIdx +1;
			MPI_Accumulate(p->vec_x, count, MY_MPI_FLOAT_TYPE, 0, p->vec_x_begin_gloIdx, count, MY_MPI_FLOAT_TYPE, MPI_SUM, win_vec);
			MPI_Win_fence(0, win_vec);
			MPI_Bcast(x, n, MY_MPI_FLOAT_TYPE, 0, MPI_COMM_WORLD);
			MPI_Win_free(&win_vec);
		}//end of mode =2
	}*/
}
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
