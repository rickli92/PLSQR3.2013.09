/**
 * @file main.h
 * @author He Huang
 * @brief Control compilation and execution of SPLSQR
 *
 */
#ifndef _MAIN_H_
#define _MAIN_H_

#include <mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#ifdef _OPENMP
#include <omp.h>
#endif

//BEGIN OF OPTION

/**Barrier for timer. None of barrier is necessary. Disable this if used in production. */
#define BARRIER

/** Option: Single or Double: choose only one. single or double version: 1 single; 2 double*/
#define SDVERSION 1  //
#if SDVERSION!=1
	#define SDVERSION 2  //double
#endif
#if SDVERSION==1
#define MY_FLOAT_TYPE float
#define MY_MPI_FLOAT_TYPE MPI_FLOAT
#elif SDVERSION == 2
#define MY_FLOAT_TYPE double
#define MY_MPI_FLOAT_TYPE MPI_DOUBLE
#endif

/** Option: PAPI */
//#define USEPAPI
#ifdef USEPAPI
#include "papi.h"
/*static*/ void test_fail(char *file, int line, char *call, int retval);
#endif

/** Option: GPTL timer */
#define GPTL
#ifdef GPTL
#include "./tool/gptl.h"
#endif

#define PLSQR3
#ifdef PLSQR3 ///*PLSQR3 options*/
/*Enable indicates use BOTH row AND column permutation; otherwise use row permutation ONLY*/
	//#define COL_PERM

/*mpi communication type
 * ONLY works for ONEDIAGONAL
 * 1: 1-sided
 * 2: point to point*/
//	#define MPICOMMTYPE 1 //1-sided
	#if MPICOMMTYPE!=1
		#define MPICOMMTYPE 2 //point to point
	#endif

/*determine use one diagonal or tri-diagonal*/
	#define TRIDIAGONAL

/*Use sparse vector to transfer necessary message volume during vector reconstruction*/
	#define USE_COMPRESSED_VEC

/** GPU options enable if 1, 0 otherwise*/
//	#define GPUALL 1

/** Define number of process per node for GPU*/
	#define PPN 1 // 1 (blue waters) 2 (yellowstone) 3 (keeneland) or 7 (seismic)
#endif //end of PLSQR3
//--------------------------------------------------------------------------------------------------------------------


#define BUFSIZE 1024
#define PAGESIZE 4096

typedef enum {X_D, Y_D, AX_D, TAY_D} DEV_PTR;


///////////////////////////////////////////






//------------------------------------------------


#define MAXREADNUM 1024

typedef enum {CSR, CSC} MATRIXTYPE;

typedef enum  {IOASCII, IOBINARY}IOTYPE;
/* PLSQRV2,
 * PLSQRV3: One main diagonal
 * PLSQRV3TRI: Tri-diagonal
 * */
typedef enum  {PLSQRV2, PLSQRV3, PLSQRV3ONE, PLSQRV3TRI} VERSION;

typedef enum {
	mycudaMemcpyHostToHost,
	mycudaMemcpyHostToDevice,
	mycudaMemcpyDeviceToHost,
	mycudaMemcpyDeviceToDevice
} mycudaMemcpyKind;

typedef struct{
	int r, c;
	double val;
}NONZERO;
/*
 * compressed sparse row
 * http://netlib.org/linalg/html_templates/node91.html
*/

typedef struct tagSpMatCSR{
	int numRow;
	int numCol;
	int numNonzero;
	int* row_ptr;//array
	int* col_idx;//array
	MY_FLOAT_TYPE* val;//array
} SpMatCSR;

//csr format, useless, to be delete
typedef struct tagGPUSpMatCSR{
	int numRow;
	int numCol;
	int numNonzero;
	int* row_ptr;//array
	int* col_idx;//array
	MY_FLOAT_TYPE* val;//array
} GPUSparseMatrixCSR;

/*tagPaddedSparseMatrixCSR*/
typedef struct tagPadSpMatCSR{
	int numRow;
	int numCol;
	int numNonzero;
	int* row_ptr;//array
	int* col_idx;//array
	MY_FLOAT_TYPE* val;//array
} PadSpMatCSR;




/* Compressed Sparse Column
 * http://netlib.org/linalg/html_templates/node92.html
 * */
typedef struct tagSpMatCSC{
	int numRow;
	int numCol;
	int numNonzero;
	int* col_ptr;//array
	int* row_idx;//array, global index
	MY_FLOAT_TYPE* val;//array
}SpMatCSC;



/*typedef struct {
    int i;
	int j;
    MY_FLOAT_TYPE val;
} matrix_entry_t;*/



typedef struct tagMatNonzeroEntry{
    int rowIndex;
	int columnIndex;
    MY_FLOAT_TYPE value;
} MatNonzeroEntry;

//COO format, tagBasicSparseMatrix
typedef struct tagBasicSpMat {
	int numRow;
	int numCol;
	int numNonzero;
	MatNonzeroEntry * nonzeroEntry; //array
	int count;
} BasicSpMat;


/* used in PLSQR3*/
typedef struct tagBasicSpMat2{
	int numRow;
	int numCol;
	int numNonzero;
	int * rowIdx; //array
	int * colIdx;//array
	MY_FLOAT_TYPE * val;//array
	int count;
}BasicSpMat2;


typedef struct tagPer_Proc_Data_v2{
	VERSION version;
	SpMatCSR * matrix_block;
	MY_FLOAT_TYPE * vec_y_Ax_temp; //vector Ax;array len =row; temporary result for each process
	MY_FLOAT_TYPE * vec_x_tAy_temp;//vector ATy;array len=column,  vec_x_ATy_temp in processes except 0 are reduced to process 0

	MY_FLOAT_TYPE * Ax_gpu;//vec_y_Ax_temp in gpu, memory allocated in lsqr.c
	MY_FLOAT_TYPE * tAy_gpu;//vec_x_tAy_temp in gpu, memory allocated in lsqr.c

	int* procReceiveInd; // index when gathing data from all processors. The data on each processor has its own index.
	int* perProcNumRow; // array of num of rows on each process, also corresponds to the dimension of vectorB in each process
	int totalNumOfRows;
	int totalNumOfColumns;
/*	int rowStartPositionInEntireMatrix;*/
} Per_Proc_Data_v2;






/* create on 08/24/2010
 * New version of data distributed among different process, include both kernel and damping matrix
 * *_k: kernel related data
 * *_d: damping related data
 * It also contains both matrix and corresponding vector x and vector y
 *
 * a stands for top left point of the block
 * c stands for bottom right point of the block
 * i stands for global row index, j stands for global column index,
 * 		i and j indicates the indices in the original matrix not the transformed matrix
 * ai_k, aj_k, ci_k, cj_k and ai_d, aj_d, ci_d, cj_d are only used once
 * for compute overlapped index of vector x and y. They are all global index,
 * when computing, global index is converted into local index of individual process
 *
 * Note 12/6/2011 This is only for one main diagonal
 * */
typedef struct tagPerProcDataV3{
	VERSION version; //PLSQRV3
	VERSION subversion; //PLSQRV3ONE:  one main diagonal
	//kernel part, local index
	SpMatCSC * matCSC_k;//kernel data matrix, Compressed sparse column (CSC or CCS)
	//global indices
	int ai_k, aj_k, ci_k, cj_k;

	//damping part, local index
	SpMatCSR * matCSR_d;
	//global indices in entire matrix:kernel + damping
	int ai_d, aj_d, ci_d, cj_d;

/*	vector part, vector y has two parts: kernel part and damping part,
 * kernel part's length is fixed, which is equal to number of row in kernel,
 * damping part's length is evenly partitioned, and is not overlapped
 * Reference: PLSQR3_NCAR3.4.pdf pp.14/25
 * */
	MY_FLOAT_TYPE * vec_y_k;// dimension is the same as block->numRow
	MY_FLOAT_TYPE * vec_y_d;//array, Temporal vector y result obtained from A*x
	int vec_y_begin_gloIdx_d;//global index (kernel + damping)
	int vec_y_end_gloIdx_d;//global index (kernel + damping)

	//VECTOR X
	MY_FLOAT_TYPE * vec_x;//array, Temporal vector x result obtained from A^T*y, the length is equal to the difference between begin and end index
	//vector x begin index in global index of vector x, for computing overlap
	int vec_x_begin_gloIdx;
	int vec_x_end_gloIdx;

}PerProcDataV3;



/* Created 12/6/2011
 * define the internal structure of vector x, the starting and ending position of different pieces of vector x
 * All Start and End position are global indices
 *
 * Position information is equal to -1 indicates the value is disabled, nonnegative indicates it is enabled
 * */
/*
typedef struct tagVEC_POS{
//the number of pieces : e.g. could be one kernel piece and three damping pieces
//	int count;


 Initialized before LSQR computing in  initPPDataV3_TRI
	int kerStart;
	int kerEnd;

 damp1, damp2, damp3 correspond to left, main, and right diagonal of the tridiagonal in the matrix
 * These are initialized when the computing damping part of vector x in A^T*y
 * (before combine the kernel part of the vector x) at the 0th iteration aprod.c: line 670
 * 			GPTLstart("spmvT_ker");
			cpuSpMVTranspose_CSC(p->matCSC_k, p->vec_y_k , &temp_vec_x_tAy_k[p->vec_x_pos.kerStart]);
			GPTLstop("spmvT_ker");

			GPTLstart("spmvT_damp");
			cpuSpMVTranspose_CSR(p->matCSR_d, p->vec_y_d , temp_vec_x_tAy_d );
			GPTLstop("spmvT_damp");
			//determine vector x position information
			if(iteration==0){

			}


			//combine kernel and damping
			for(i=0; i<vecXLen; i++){
				p->vec_x[i] = temp_vec_x_tAy_k[i] + temp_vec_x_tAy_d[i];
			}
 *
	int damp1Start;
	int damp1End;

	int damp2Start;
	int damp2End;

	int damp3Start;
	int damp3End;

}VEC_POS;
*/

/* Created on 12/6/2011
 * Each MPI task's communication set, only used for local MPI task
 * Each element in the communication set has overlap with the local MPI task
 * */
/*
typedef struct tagCommunicationSet{
	//number of associated MPI tasks
	int count;
	//the list of MPI tasks by rank ID, array
	int * rankList;

	// The following two variables record the starting and ending overlap position of each element in rankList
	int * rankOverlapPosStartList;

	int * rankOverlapPosEndList;
}CommunicationSet;
*/

/** Compressed Sparse Vector format is used for vector  that has several pieces of nonzeros.
 * The gaps between pieces are all zeros.
 * E.g. a 10-dimension sparse vector [1 2 0 0 3 4 5 0 0 6 ]
 * npiece=3
 * nnz=6
 * ptr=[0 4 9]
 * len=[2 3 1]
 * val=[1 2 3 4 5 6]
 * */
typedef struct tagCompressedVec{
	/*number of nonzero pieces*/
	int npiece;
	/*number of nonzeros of the vectors*/
	int nnz;
	/*Starting Index Array, ptr[npiece]: starting index of each nonzero piece.
	 * ptr[i] is the ith nonzeros starting index in the original vector*/
	int * ptr;
	/*Array len[npiece] The length of each nonzero piece*/
	int * len;
	/*Nonzero Value Array: val[nnz]*/
	MY_FLOAT_TYPE * val;
}CompressedVec;


/** @brief GPU version of tagPPDataV3_TRI
 * The postfix _d indicates memory is allocated in GPU device memory*/
typedef struct tagPPDataV3_TRI_GPU{
	SpMatCSC * matCSC_k_d;
	SpMatCSR * matCSR_k_d;// transpose of matCSC_k_d

	SpMatCSR * matCSR_d_d;
	SpMatCSC * matCSC_d_bycol_d;


	MY_FLOAT_TYPE * vec_y_k_d;
	MY_FLOAT_TYPE * vec_y_d_d;
	MY_FLOAT_TYPE * vec_x_d;

	/*Temporal variable in lsqr.c*/
	MY_FLOAT_TYPE * lsqr_x_d;//x[]
	MY_FLOAT_TYPE * lsqr_w_d;//w[]
	MY_FLOAT_TYPE * lsqr_se_d;//se[]
	MY_FLOAT_TYPE * lsqr_t_d;//t is not an array

}PPDataV3_TRI_GPU;


/* Created on 12/6/2011
 * For tri-diagonal version, extends from PerProcDataV3
 *
 * 1/9/2012
 * add explicit storage of damping part of matrix A
 * */
#define COMM_SET_LEN  100
/** @brief data structure used in SPLSQR */
typedef struct tagPPDataV3_TRI{
	VERSION version;
	VERSION subversion; /**<PLSQRV3TRI*/

/** @berif Matrix(kernel): use global row indices, Compressed sparse column (CSC or CCS)*/
	const SpMatCSC * matCSC_k;
	int ai_k, aj_k, ci_k, cj_k;///position index of top left corner "a", and bottom right corner "c"

/** @berif Matrix(Damping): damping matrix, use global column indices*/
	const SpMatCSR * matCSR_d;

/** @berif Matrix(Damping):
 * explicit storage of damping part of matrix A
 * From the view of the original matrix, the damping transpose is decomposed by column,
 * exactly the same as the way of decomposing kernel
 * The matrix's row idx is damping global, not including kernel, column index is local
 * */
	const SpMatCSC * matCSC_d_bycol;

/** @berif Vector: vector y has two parts: kernel part and damping part,
 * kernel part's length is fixed, which is equal to number of row in kernel,
 * damping part's length is evenly partitioned, no overlap
 * Reference: PLSQR3_NCAR3.4.pdf pp.14/25
 * */
	MY_FLOAT_TYPE * vec_y_k; /// dimension is the same as block->numRow

	MY_FLOAT_TYPE * vec_y_d; ///array, Temporal vector y result obtained from A*x
	int vec_y_d_begin; /// index of damping global(exclude kernel index)
	int vec_y_d_end;
	int vec_y_d_len; ///p->vec_y_d_len=p->vec_y_d_end - p->vec_y_d_begin + 1;

	/**@berif vector x is now a partition across MPI tasks,//1/12/2012*/
	MY_FLOAT_TYPE * vec_x;
	int vec_x_begin_gloIdx;
	int vec_x_end_gloIdx;
	int vec_x_len; ///	p->vec_x_len = endCol_perProc_k - startCol_perProc_k + 1;


/** @berif Receive List
 * communication rank, start index, and length: -1 means no value,
 * length is limited by COMM_SET_LEN
 * determined determined by gather_vec_x_init() and gather_vec_y_init() in vector_comm.c
 * do not include local rank
 * */
	int vec_x_recv_list_rank[COMM_SET_LEN];
	int vec_x_recv_list_startIdx[COMM_SET_LEN];
	int vec_x_recv_list_len[COMM_SET_LEN];
	int vec_x_recv_list_size;// the size of the above three arrays

	int vec_y_d_recv_list_rank[COMM_SET_LEN];
	int vec_y_d_recv_list_startIdx[COMM_SET_LEN];
	int vec_y_d_recv_list_len[COMM_SET_LEN]; //for simpliclity, currently set len to be the entire length of decomposed vector in neighbors
	int vec_y_d_recv_list_size;// actual size of array vec_y_d_recv_list_*

/**	@berif Send List
	list of receiver's rank, used in source rank, length is limited by COMM_SET_LEN
	each element stores receiver's rank, -1 means blank.
	determined by gather_vec_x_init() and gather_vec_y_init() in vector_comm.c
	*/
	int vec_x_send_list_rank[COMM_SET_LEN];
	int vec_x_send_list_size;

	int vec_y_d_send_list_rank[COMM_SET_LEN];
	int vec_y_d_send_list_size;


/** @brief Extended Vector {x, y}, local vector {x, y} plus vectors from neighbors
 * used in during damping matrix multiplication
 * determined at the 0 or 1 iteration  3/8/2012	*/
	int extended_vec_x_startIdx;///global index
	int extended_vec_x_len;///the sum of length of its neighbors

	int extended_vec_y_d_startIdx;///damping global index
	int extended_vec_y_d_len;


#ifdef PLSQR3
/**@berif window for vec_x, which is MPI_Alloc_mem at initPPDataV3_TRI()*/
	MPI_Win win_vec_x;
/**@berif window for vec_y_d, which is MPI_Alloc_mem at initPPDataV3_TRI()*/
	MPI_Win win_vec_y_d;
#endif

/**@berif 	Sparse Vector*/
#ifdef USE_COMPRESSED_VEC
/**@berif array to receive sparse vector from neighbors,
 * the length of array is vec_x_recv_list_size and vec_y_d_recv_list_size
 * allocated by GatherVec{X,Y}InitByProjY*/
	CompressedVec * sp_vec_x_recv_list; ///length = vec_x_recv_list_size
	CompressedVec * sp_vec_x_send_list; ///length = vec_x_send_list_size
	CompressedVec * sp_vec_y_d_recv_list; ///length = vec_y_d_recv_list_size
	CompressedVec * sp_vec_y_d_send_list;///length = vec_y_d_send_list_size
#endif
/** @berif data in GPU*/
#if GPUALL==1
	PPDataV3_TRI_GPU * gpu;
#endif
}PPDataV3_TRI;















//File Name's structure
typedef struct tagFileName{
	// file's real name, e.g.200805130707.YCH.BHR.RAYL.010.av
	char name[1024];
	// every file corresponds to one row of the entire matrix,
	//the row index is assigned by our program from 1 to N(N is the number of rows of the matrix).
	int rowIndex;
}FileName;


static void parseError( const char * err, const char *file, const int line, const char * func ) {
	printf( "%s(%d)-%s: ERROR: %s \n", file, line, func, err );
	exit( EXIT_FAILURE );
}

#define PARSE_ERROR( err ) (parseError( err, __FILE__, __LINE__, __FUNCTION__ ))

static void ParseMPIError(int err, const char *file, const int line, const char * func)
{
	fprintf(stderr, "%s(%d)-%s MPI ERROR CODE=%d", __FILE__, __LINE__, __FUNCTION__, err);

	char error_string[BUFSIZ];
	int length_of_error_string;
	MPI_Error_string(err, error_string, &length_of_error_string);
	fprintf(stderr, "%s\n",  error_string);
	exit( EXIT_FAILURE );
}
#define PARSE_MPI_ERROR(err) (ParseMPIError(err, __FILE__, __LINE__, __FUNCTION__))

#endif // _COMMON_H_




