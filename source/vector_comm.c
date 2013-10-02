/*
 * vector_comm.c
 *
 *  Created on: May 14, 2012
 *      Author: hhuang1
 * Vector reconstruction for multi-diagonal
 */

#include "vector_comm.h"
#include "sparse_vec.h"
#ifdef TRIDIAGONAL

//#define DEBUG

extern int group_size_omp;

/*3/16/13*/
void initGatherX(int procRank, int numProc, PPDataV3_TRI * p){
	if(numProc==1) return;
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
		GPTLstart("init_gather_x");
#endif
	int check_startIdx1 = 0, check_endIdx1 = 0/*, check_endIdx2=0*/;

#ifdef USE_COMPRESSED_VEC
	GatherVecXInitByProjX(procRank, numProc, p, &check_startIdx1, &check_endIdx1);
#else
	gather_vec_x_init(procRank, numProc, iteration, p, &check_startIdx1, &check_endIdx1);
#endif

	int extended_vec_x_startIdx = p->extended_vec_x_startIdx;
	int extended_vec_x_len = p->extended_vec_x_len;

//check
	int check_endIdx2 = extended_vec_x_startIdx + extended_vec_x_len - 1;
	if (extended_vec_x_startIdx <= check_startIdx1 && check_endIdx2 >= check_endIdx1) {
	} else {
		fprintf(stderr,
				"%d: ERROR %s(%d)-%s: startIdx(%d)<=check_startIdx1(%d) && check_endIdx2(%d)>=check_endIdx1(%d)\n",
				procRank, __FILE__, __LINE__, __FUNCTION__, extended_vec_x_startIdx,
				check_startIdx1, check_endIdx2, check_endIdx1);
		exit(-1);
	}
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstop("init_gather_x");
#endif
}

void initGatherY(int procRank, int numProc, PPDataV3_TRI * p){
	if(numProc==1) return;
	//pull vector y overlap with neighbors
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("init_gather_y");
#endif
	int check_startIdx1 = 0, check_endIdx1 = 0/*, check_endIdx2=0*/;

#ifdef USE_COMPRESSED_VEC
	GatherVecYInitByProjY(procRank, numProc, p, &check_startIdx1, &check_endIdx1);
#else
	gather_vec_y_init (procRank, numProc, iteration, p, &check_startIdx1, &check_endIdx1);
#endif

	int extended_vec_y_d_startIdx = p->extended_vec_y_d_startIdx;
	int extended_vec_y_d_len = p->extended_vec_y_d_len;

//check
	int check_endIdx2 = extended_vec_y_d_startIdx + extended_vec_y_d_len - 1;
	if (extended_vec_y_d_startIdx <= check_startIdx1 && check_endIdx2 >= check_endIdx1) {
	} else {
		fprintf(stderr, "%d: ERROR %s(%d)-%s: startIdx(%d)<=check_startIdx1(%d) && check_endIdx2(%d)>=check_endIdx1(%d)\n",
				procRank, __FILE__, __LINE__, __FUNCTION__, extended_vec_y_d_startIdx,
				check_startIdx1, check_endIdx2, check_endIdx1);
		exit(-1);
	}

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstop("init_gather_y");
#endif
}





/* Initialization and finalization of vector x for vector gather in multi-diagonal case
 * compute vector x overlap with neighbors, only as at iteration==1
 * @param p OUTPUT: p->vec_x_recv_list_* and p->extended_local_x_* are filled in this function
 * */
void gather_vec_x_init (int procRank, int numProc, int iteration,
		 PPDataV3_TRI * p, int * check_startIdx1, int * check_endIdx1)
{
	int i=0, rank=0, err=0;

/*Initialization*/
	if (iteration == 1 && numProc > 1) {//compute vector x overlap with neighbors, only as at iteration==1

		//the starting index of all processes
		int * vec_x_start_idx_all_array = (int *) malloc(
				(numProc + 1) * sizeof(int));
		if (vec_x_start_idx_all_array == NULL) {
			fprintf(stderr, "%s(%d)-%s:", __FILE__, __LINE__, __FUNCTION__);
			exit(-1);
		}
		MPI_Allgather(&p->vec_x_begin_gloIdx, 1, MPI_INT,
				vec_x_start_idx_all_array, 1, MPI_INT, MPI_COMM_WORLD);
		vec_x_start_idx_all_array[numProc] = p->matCSR_d->numCol;

		/*		int * projectedVectorX = (int *) malloc(n * sizeof(int));
		 if(projectedVectorX==NULL){
		 fprintf(stderr, "%s(%d)-%s:", __FILE__, __LINE__, __FUNCTION__);exit(-1);
		 }
		 projectingAColumnToX (p->matCSR_d, projectedVectorX);*/

		int projectedVectorX_startIdx = 0, projectedVectorX_len = 0;
		projectingAColumnToX_v2_countLength(p->matCSR_d, &projectedVectorX_startIdx, &projectedVectorX_len);
		*check_startIdx1 = projectedVectorX_startIdx;
		*check_endIdx1 = projectedVectorX_startIdx + projectedVectorX_len - 1;
	#ifdef DEBUG
		printf("%d: %s(%d)-%s: startIdx=%d, len=%d\n", procRank, __FILE__, __LINE__, __FUNCTION__, projectedVectorX_startIdx, projectedVectorX_len);
	#endif
		int * projectedVectorX = (int *) malloc(projectedVectorX_len * sizeof(int));
		if (NULL == projectedVectorX) {
			fprintf(stderr, "%s(%d)-%s:\n", __FILE__, __LINE__, __FUNCTION__);
			exit(-1);
		}
		memset(projectedVectorX, 0, projectedVectorX_len * sizeof(int));
		projectingAColumnToX_v2(p->matCSR_d, projectedVectorX, projectedVectorX_startIdx, projectedVectorX_len);
	#ifdef DEBUG
		/*		for(i=0; i<len; i++){
		 printf("projectedVectorX[%d]=%d\n", i, projectedVectorX[i]);
		 }*/
	#endif

		//fill receive list
		int c1 = 0;
		for (i = 0; i < projectedVectorX_len; i++) {
			if (projectedVectorX[i] == 1) {
				for (rank = 0; rank < numProc; rank++) {
					int globalIdx = i + projectedVectorX_startIdx;
					if (rank != procRank /*not myself*/
							&& vec_x_start_idx_all_array[rank] <= globalIdx && globalIdx < vec_x_start_idx_all_array[rank + 1]) {
						if (c1 >= COMM_SET_LEN) {
							fprintf(stderr, "%s(%d)-%s: comm set is greater than COMM_SET_LEN=%d\n",
									__FILE__, __LINE__, __FUNCTION__, COMM_SET_LEN);
							exit(-1);
						}
						if (c1 == 0 || (c1 > 0 && rank != p->vec_x_recv_list_rank[c1 - 1])) {//do not do repeated work
							p->vec_x_recv_list_rank[c1] = rank;
							p->vec_x_recv_list_startIdx[c1] = vec_x_start_idx_all_array[rank];
							p->vec_x_recv_list_len[c1] = vec_x_start_idx_all_array[rank + 1]
											- vec_x_start_idx_all_array[rank];
							c1++;
						}
					}
				}
			}
		}
		p->vec_x_recv_list_size = c1;

		free(projectedVectorX);
	//		3/8/2012 determine extended_vec_x_startIdx, extended_vec_x_len
		for (rank = 0; rank < numProc; rank++) {
			if (vec_x_start_idx_all_array[rank] <= *check_startIdx1
					&& *check_startIdx1 < vec_x_start_idx_all_array[rank + 1]) {
				p->extended_vec_x_startIdx = vec_x_start_idx_all_array[rank];
			}
			if (vec_x_start_idx_all_array[rank] <= *check_endIdx1
					&& *check_endIdx1 < vec_x_start_idx_all_array[rank + 1]) {
				p->extended_vec_x_len = vec_x_start_idx_all_array[rank + 1]
						- p->extended_vec_x_startIdx;
			}
		}
		free(vec_x_start_idx_all_array);

	#if MPICOMMTYPE==2
	/* fill send_candidate_x, 	used for sender, initialized in the 0st iteration
	 * send_candidate_x[i]=1 indicates this MPI task needs to send vector to the ith rank,
	 * send_candidate_x[i]=-1 indicates this MPI task does NOT needs to send vector to the ith rank
	 * */
		int * send_candidate_x;
		MPI_Win win_send_candidate_x;
		MPI_Alloc_mem(numProc * sizeof(int), MPI_INFO_NULL, &(send_candidate_x));
		if (NULL == send_candidate_x) {
			fprintf(stderr, "%d: %s(%d)-%s", procRank, __FILE__, __LINE__, __FUNCTION__);
			exit(-1);
		}
		memset(send_candidate_x, -1, numProc * sizeof(int));
		err = MPI_Win_create(send_candidate_x, (MPI_Aint) (numProc * sizeof(int)),
				sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_send_candidate_x);
		if (err != 0) {
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank, __FILE__, __LINE__, __FUNCTION__, err);
			exit(-1);
		}

		err = MPI_Win_fence(0, win_send_candidate_x);
		if (err != 0) {
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank,
					__FILE__, __LINE__, __FUNCTION__, err);
			exit(-1);
		}
		for (i = 0; i < p->vec_x_recv_list_size; i++) { //2. mark receiver's rank at sender's list
			err = MPI_Put(&procRank, 1, MPI_INT, p->vec_x_recv_list_rank[i],
					(MPI_Aint) procRank, 1, MPI_INT, win_send_candidate_x);
			if (err != 0)
				fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank,
						__FILE__, __LINE__, __FUNCTION__, err);
		}
		err = MPI_Win_fence(0, win_send_candidate_x);
		if (err != 0) {
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank,
					__FILE__, __LINE__, __FUNCTION__, err);
			exit(-1);
		}

	//count recipient list length
		p->vec_x_send_list_size = 0;
		for (i = 0; i < numProc; i++) {
			if (send_candidate_x[i] >= 0)
				p->vec_x_send_list_size++;
		}
		if (p->vec_x_send_list_size > COMM_SET_LEN) {//check
			fprintf(stderr,
					"%s(%d)-%s: comm set is greater than COMM_SET_LEN=%d\n",
					__FILE__, __LINE__, __FUNCTION__, COMM_SET_LEN);
			exit(-1);
		}

		//set send list
		int m = 0;
		for (i = 0; i < numProc; i++) {
			if (send_candidate_x[i] >= 0) {
				p->vec_x_send_list_rank[m] = send_candidate_x[i];
				m++;
			}
		}
		MPI_Win_free(&win_send_candidate_x);
		MPI_Free_mem(send_candidate_x);
	#endif
	}//ENDOF	if (iteration == 1 && numProc > 1)

}

/* The first part is the same as gather_vec_x_init, except the location of free(projectedVectorX)
 * USE_COMPRESSED_VEC
 * Fill sparse vector x in PPDataV3_Tri, i.e., sp_vec_x and sp_vec_x_recv_list
 * Call ONLY from gather_vec_x_init
 * Initialize sparse vector in PPDataV3_TRI
 * by projectedAColToX[p->vec_x_begin - startIdx]: a fake vec_x, with lenght vec_x_len
 * USE_COMPRESSED_VEC
 * */
#ifdef USE_COMPRESSED_VEC
void GatherVecXInitByProjX(int procRank, int numProc,
		 PPDataV3_TRI * p, int * check_startIdx1, int * check_endIdx1)
{
	if(numProc==1) return;
	///////////////////////////FIRST PART/////////////////////////////////////////////////////

	int i=0, j=0, rank=0, err=0;

/*Initialization*/
//compute vector x overlap with neighbors, only as at iteration==1

	//the starting index of all processes
	int * vec_x_start_idx_all_array = (int *) malloc((numProc + 1) * sizeof(int));
	if (vec_x_start_idx_all_array == NULL) {
		fprintf(stderr, "%s(%d)-%s:", __FILE__, __LINE__, __FUNCTION__);
		exit(-1);
	}
	MPI_Allgather(&p->vec_x_begin_gloIdx, 1, MPI_INT,
			vec_x_start_idx_all_array, 1, MPI_INT, MPI_COMM_WORLD);
	vec_x_start_idx_all_array[numProc] = p->matCSR_d->numCol;

	int projectedVectorX_startIdx = 0, projectedVectorX_len = 0;
	projectingAColumnToX_v2_countLength(p->matCSR_d, &projectedVectorX_startIdx, &projectedVectorX_len);
	*check_startIdx1 = projectedVectorX_startIdx;
	*check_endIdx1 = projectedVectorX_startIdx + projectedVectorX_len - 1;
#ifdef DEBUG
	printf("%d: %s(%d)-%s: startIdx=%d, len=%d\n", procRank, __FILE__, __LINE__, __FUNCTION__, projectedVectorX_startIdx, projectedVectorX_len);
#endif
	int * projectedVectorX = (int *) malloc(projectedVectorX_len * sizeof(int));
	if (NULL == projectedVectorX) {
		fprintf(stderr, "%s(%d)-%s:\n", __FILE__, __LINE__, __FUNCTION__);
		exit(-1);
	}
	memset(projectedVectorX, 0, projectedVectorX_len * sizeof(int));
	projectingAColumnToX_v2(p->matCSR_d, projectedVectorX, projectedVectorX_startIdx, projectedVectorX_len);
#ifdef DEBUG
	/*		for(i=0; i<len; i++){
	 printf("projectedVectorX[%d]=%d\n", i, projectedVectorX[i]);
	 }*/
#endif

	//fill receive list
	int c1 = 0;
	for (i = 0; i < projectedVectorX_len; i++) {
		if (projectedVectorX[i] == 1) {
			for (rank = 0; rank < numProc; rank++) {
				int globalIdx = i + projectedVectorX_startIdx;
				if (rank != procRank /*not myself*/
						&& vec_x_start_idx_all_array[rank] <= globalIdx && globalIdx < vec_x_start_idx_all_array[rank + 1]) {
					if (c1 >= COMM_SET_LEN) {
						fprintf(stderr, "%s(%d)-%s: comm set is greater than COMM_SET_LEN=%d\n",
								__FILE__, __LINE__, __FUNCTION__, COMM_SET_LEN);
						exit(-1);
					}
					if (c1 == 0 || (c1 > 0 && rank != p->vec_x_recv_list_rank[c1 - 1])) {//do not do repeated work
						p->vec_x_recv_list_rank[c1] = rank;
						p->vec_x_recv_list_startIdx[c1] = vec_x_start_idx_all_array[rank];
						p->vec_x_recv_list_len[c1] = vec_x_start_idx_all_array[rank + 1]
										- vec_x_start_idx_all_array[rank];
						c1++;
					}
				}
			}
		}
	}
	p->vec_x_recv_list_size = c1;


//		3/8/2012 determine extended_vec_x_startIdx, extended_vec_x_len
	for (rank = 0; rank < numProc; rank++) {
		if (vec_x_start_idx_all_array[rank] <= *check_startIdx1
				&& *check_startIdx1 < vec_x_start_idx_all_array[rank + 1]) {
			p->extended_vec_x_startIdx = vec_x_start_idx_all_array[rank];
		}
		if (vec_x_start_idx_all_array[rank] <= *check_endIdx1
				&& *check_endIdx1 < vec_x_start_idx_all_array[rank + 1]) {
			p->extended_vec_x_len = vec_x_start_idx_all_array[rank + 1]
					- p->extended_vec_x_startIdx;
		}
	}
	free(vec_x_start_idx_all_array);

#if MPICOMMTYPE==2
/* fill send_candidate_x, 	used for sender, initialized in the 0st iteration
 * send_candidate_x[i]=1 indicates this MPI task needs to send vector to the ith rank,
 * send_candidate_x[i]=-1 indicates this MPI task does NOT needs to send vector to the ith rank
 * */
	int * send_candidate_x=NULL;
	MPI_Win win_send_candidate_x;
	err = MPI_Alloc_mem(numProc * sizeof(int), MPI_INFO_NULL, &(send_candidate_x));
	if(err!=MPI_SUCCESS)	PARSE_MPI_ERROR(err);
	if (NULL == send_candidate_x) {
		fprintf(stderr, "%d: %s(%d)-%s", procRank, __FILE__, __LINE__, __FUNCTION__);
		exit(-1);
	}
	memset(send_candidate_x, -1, numProc * sizeof(int));
	err = MPI_Win_create(send_candidate_x, (MPI_Aint) (numProc * sizeof(int)),
			sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_send_candidate_x);
	if(err!=MPI_SUCCESS)	PARSE_MPI_ERROR(err);

	err = MPI_Win_fence(0, win_send_candidate_x);
	if(err!=MPI_SUCCESS)	PARSE_MPI_ERROR(err);
	for (i = 0; i < p->vec_x_recv_list_size; i++) { //2. mark receiver's rank at sender's list
		err = MPI_Put(&procRank, 1, MPI_INT, p->vec_x_recv_list_rank[i],
				(MPI_Aint) procRank, 1, MPI_INT, win_send_candidate_x);
		if (err != 0)
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank,
					__FILE__, __LINE__, __FUNCTION__, err);
	}
	err = MPI_Win_fence(0, win_send_candidate_x);
	if(err!=MPI_SUCCESS)	PARSE_MPI_ERROR(err);

//count recipient list length
	p->vec_x_send_list_size = 0;
	for (i = 0; i < numProc; i++) {
		if (send_candidate_x[i] >= 0)
			p->vec_x_send_list_size++;
	}
	if (p->vec_x_send_list_size > COMM_SET_LEN) {//check
		fprintf(stderr,	"%s(%d)-%s: comm set is greater than COMM_SET_LEN=%d\n",
				__FILE__, __LINE__, __FUNCTION__, COMM_SET_LEN);
		exit(-1);
	}

	//set send list
	int m = 0;
	for (i = 0; i < numProc; i++) {
		if (send_candidate_x[i] >= 0) {
			p->vec_x_send_list_rank[m] = send_candidate_x[i];
			m++;
		}
	}
	err = MPI_Win_free(&win_send_candidate_x);
	if(err!=MPI_SUCCESS)	PARSE_MPI_ERROR(err);
	err = MPI_Free_mem(send_candidate_x);
	if(err!=MPI_SUCCESS)	PARSE_MPI_ERROR(err);
#endif



//////////////////////////////SECOND PART//////////////////////////////////////////////
	if(p->vec_x_recv_list_size != 0)
		p->sp_vec_x_recv_list=(CompressedVec *)malloc(p->vec_x_recv_list_size * sizeof(CompressedVec));
	else p->sp_vec_x_recv_list=NULL;
	if(p->vec_x_send_list_size != 0)
	p->sp_vec_x_send_list=(CompressedVec *)malloc(p->vec_x_send_list_size * sizeof(CompressedVec));
	else p->sp_vec_x_send_list=NULL;
	/*Temporal receiving list, used to receive list from other neighbors and construct sparse receive list*/
	int ** temp_vec_x_recv_list=(int ** )malloc(p->vec_x_recv_list_size * sizeof(int *));
	if(temp_vec_x_recv_list==NULL){
		fprintf(stderr, "%s(%d):%s\n", __FILE__, __LINE__, __FUNCTION__); exit(-1);
	}

	int chunk=(p->vec_x_recv_list_size + group_size_omp-1) / (group_size_omp);
#pragma omp parallel for private(i) schedule(dynamic, chunk)
	for(i=0; i<p->vec_x_recv_list_size; i++){
		temp_vec_x_recv_list[i]=(int *)malloc(p->vec_x_recv_list_len[i] * sizeof(int));
		if(temp_vec_x_recv_list[i]==NULL){
			fprintf(stderr, "%s(%d):%s\n", __FILE__, __LINE__, __FUNCTION__); exit(-1);
		}
		memset(temp_vec_x_recv_list[i], 0, p->vec_x_recv_list_len[i] * sizeof(int));
	}

	int ** temp_vec_x_send_list = (int **)malloc(p->vec_x_send_list_size * sizeof(int *));
	if(temp_vec_x_send_list==NULL){
		fprintf(stderr, "%s(%d):%s\n", __FILE__, __LINE__, __FUNCTION__); exit(-1);
	}

	chunk=(p->vec_x_send_list_size + group_size_omp-1) / (group_size_omp);
#pragma omp parallel for private(i) schedule(dynamic, chunk)
	for(i=0; i < p->vec_x_send_list_size; i++){
		temp_vec_x_send_list[i]=(int *)malloc(p->vec_x_len * sizeof(int));
		if(temp_vec_x_send_list[i]==NULL){
			fprintf(stderr, "%s(%d):%s\n", __FILE__, __LINE__, __FUNCTION__); exit(-1);
		}
		memset(temp_vec_x_send_list[i], 0, p->vec_x_len * sizeof(int));
	}

/*	Loop projectedVectorX and fill temp_vec_x_recv_list,
*	and then use temp_vec_x_recv_list to fill sp_vec_x_recv_list*/

	for(i=0; i< projectedVectorX_len; i++){
		if(projectedVectorX[i]==1){
			int globalIdx= i + projectedVectorX_startIdx;
			for(j=0; j<p->vec_x_recv_list_size; j++){
				int startIdx=p->vec_x_recv_list_startIdx[j];
				int len=p->vec_x_recv_list_len[j];
				if(startIdx<=globalIdx && globalIdx<startIdx+len){
					int idx=globalIdx-startIdx;
					temp_vec_x_recv_list[j][idx]=1;
				}

			}
		}
	}
	free(projectedVectorX);

	for(j=0; j<p->vec_x_recv_list_size; j++){
		InitSparseVecFromOriginalIntVec(temp_vec_x_recv_list[j], p->vec_x_recv_list_len[j],
				&(p->sp_vec_x_recv_list[j]));
	}

/*	Inform temp_vec_x_send_list by temp_vec_x_recv_list
* Note that : p->vec_x_recv_list_size and p->vec_x_send_list_size are in REVERSE order*/
	int size_send = p->vec_x_send_list_size;
	int size_recv = p->vec_x_recv_list_size;

	MPI_Status * sta = (MPI_Status *) malloc((size_send + size_recv) * sizeof(MPI_Status));
	MPI_Request * req = (MPI_Request *) malloc((size_send + size_recv) * sizeof(MPI_Request));

	for (i = 0; i < size_send; i++) {
		err = MPI_Irecv(temp_vec_x_send_list[i], p->vec_x_len, MPI_INT,
				p->vec_x_send_list_rank[i], 0, MPI_COMM_WORLD,
				&req[i + size_recv]);
		if (err != 0) {
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", rank,
					__FILE__, __LINE__, __FUNCTION__, err);
			exit(-1);
		}
	}

	for (i = 0; i < size_recv; i++) {
		err = MPI_Isend(temp_vec_x_recv_list[i], p->vec_x_recv_list_len[i], MPI_INT,
				p->vec_x_recv_list_rank[i], 0, MPI_COMM_WORLD, &req[i]);
		if (err != 0) {
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", rank,
					__FILE__, __LINE__, __FUNCTION__, err);
			exit(-1);
		}
	}

	err = MPI_Waitall(size_send + size_recv, req, sta);
	if (err != 0) {
		fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", rank, __FILE__,
				__LINE__, __FUNCTION__, err);
		exit(-1);
	}
	free(sta);
	free(req);

/*Convert temp_vec_x_send_list to compressed sp_vec_x_send_list*/
	for(i=0; i<p->vec_x_send_list_size; i++){
		InitSparseVecFromOriginalIntVec(temp_vec_x_send_list[i], p->vec_x_len,
				&(p->sp_vec_x_send_list[i]));
	}

	//free memory
	for(i=0; i<p->vec_x_recv_list_size; i++){
		free(temp_vec_x_recv_list[i]);
	}
	free(temp_vec_x_recv_list);

	for(i=0; i<p->vec_x_send_list_size; i++){
		free(temp_vec_x_send_list[i]);
	}
	free(temp_vec_x_send_list);
}
#endif

//void Init_fake_vec_x(PPDataV3_TRI * p, int * fake_vec_x)
//{
//	SpMatCSR *A = p->matCSR_d;
//	int m = A->numRow;
//	int rowIdx, colIdx, firstEntryInRow, lastEntryInRow, entryIdx;
//	//FOR each row in A
//    for ( rowIdx = 0; rowIdx < m; rowIdx++) {
//    	 firstEntryInRow = A->row_ptr[rowIdx];
//    	 lastEntryInRow = A->row_ptr[rowIdx+1];
//#pragma ivdep
//    	 //FOR each entry in current row
//    	for (entryIdx = firstEntryInRow; entryIdx < lastEntryInRow; entryIdx++)
//    	{
//    		colIdx = A->col_idx[entryIdx];
////    		Ax[rowIdx] += A->val[entryIdx] * x[colIdx];//key part: only difference
//    		if(p->vec_x_begin_gloIdx<=colIdx && colIdx<=p->vec_x_end_gloIdx){
//    			fake_vec_x[colIdx - p->vec_x_begin_gloIdx]=1;
//    		}
//    	}
//    }
//}

/* print out help info
 * communication info: msg length indicates msg count without compression,
 * while msg nnz indicates msg count with compression*/
void PrintVecInfo(const int procRank, const int numProc, PPDataV3_TRI * p)
{
	int i;
	int min = 0, max = 0, sum = 0;
	//l* indicates local, *Tot indicates total
	int lLengthMax, lLengthMin, lLengthTot;
	//g* indicates global
	int gLengthMax, gLengthMin, gLengthTot;
	MY_FLOAT_TYPE lLengthAvg, gLengthAvg;

#ifdef USE_COMPRESSED_VEC
	// number of nonzero sent/received by compressed vectors
	int lnnz_max, lnnz_min, lnnz_tol;
	int gnnz_max, gnnz_min, gnnz_tol;
	MY_FLOAT_TYPE lnnz_ave, gnnz_ave;
#endif

	int listLength;
	if (procRank == 0)
		printf("%s(%d)-%s: communication info: msg length indicates msg count without compression, while msg nnz indicates msg count with compression\n", __FILE__, __LINE__, __FUNCTION__);
////////////////////////////////////////////////////////////////////////////////////////
	MPI_Reduce(&p->vec_x_recv_list_size, &min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&p->vec_x_recv_list_size, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&p->vec_x_recv_list_size, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	listLength = p->vec_x_recv_list_size;
	lLengthMax = max_vec(p->vec_x_recv_list_len, listLength);
	MPI_Reduce(&lLengthMax, &gLengthMax, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

	lLengthMin = min_vec(p->vec_x_recv_list_len, listLength);
	MPI_Reduce(&lLengthMin, &gLengthMin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

	lLengthTot = sum_vec(p->vec_x_recv_list_len, listLength);
	lLengthAvg = (MY_FLOAT_TYPE) lLengthTot / (MY_FLOAT_TYPE) listLength;
	MPI_Reduce(&lLengthAvg, &gLengthAvg, 1, MY_MPI_FLOAT_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
	gLengthAvg = gLengthAvg / (MY_FLOAT_TYPE) numProc;

	if (procRank == 0) {
		printf("vector x receive list count {min,avg,max}:={%d,%f,%d}\n", min,
				1.0 * sum / numProc, max);
		printf("vector x receive msg length {min,avg,max}:={%d,%f,%d}\n",
				gLengthMin, gLengthAvg, gLengthMax);
	}

#ifdef USE_COMPRESSED_VEC
	int * nnz_array = NULL;
	nnz_array = (int *)malloc(sizeof(int)*listLength);
	if (nnz_array==NULL) PARSE_ERROR("nnz_array");
	for(i=0; i<listLength; i++){
		nnz_array[i]=p->sp_vec_x_recv_list[i].nnz;
	}
	lnnz_max = max_vec(nnz_array, listLength);
	MPI_Reduce(&lnnz_max, &gnnz_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	lnnz_min = min_vec(nnz_array, listLength);
	MPI_Reduce(&lnnz_min, &gnnz_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	lnnz_tol = sum_vec(nnz_array, listLength);
	lnnz_ave = (MY_FLOAT_TYPE) lnnz_tol / (MY_FLOAT_TYPE) listLength;
	MPI_Reduce(&lnnz_ave, &gnnz_ave, 1, MY_MPI_FLOAT_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
	gnnz_ave = gnnz_ave /  (MY_FLOAT_TYPE) numProc;
	if (procRank == 0) {
		printf("vector x receive msg nnz {min,avg,max}:={%d,%f,%d}\n",
				gnnz_min, gnnz_ave, gnnz_max);
	}
	 free(nnz_array);
#endif
////////////////////////////////////////////////////////////////////////////////////////
	MPI_Reduce(&p->vec_y_d_recv_list_size, &min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&p->vec_y_d_recv_list_size, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&p->vec_y_d_recv_list_size, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
	listLength = p->vec_y_d_recv_list_size;
	lLengthMax = max_vec(p->vec_y_d_recv_list_len, listLength);
	MPI_Reduce(&lLengthMax, &gLengthMax, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

	lLengthMin = min_vec(p->vec_y_d_recv_list_len, listLength);
	MPI_Reduce(&lLengthMin, &gLengthMin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

	lLengthTot = sum_vec(p->vec_y_d_recv_list_len, listLength);
	lLengthAvg = (MY_FLOAT_TYPE) lLengthTot / (MY_FLOAT_TYPE) listLength;
	MPI_Reduce(&lLengthAvg, &gLengthAvg, 1, MY_MPI_FLOAT_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
	gLengthAvg = gLengthAvg / (MY_FLOAT_TYPE) numProc;
	if (procRank == 0) {
		printf("vector y_d receive list count {min,avg,max}:={%d,%f,%d}\n", min,
				1.0 * sum / numProc, max);
		printf("vector y_d receive msg length {min,avg,max}:={%d,%f,%d}\n",
				gLengthMin, gLengthAvg, gLengthMax);
	}
#ifdef USE_COMPRESSED_VEC

	nnz_array = (int *)malloc(sizeof(int)*listLength);
	if (nnz_array==NULL) PARSE_ERROR("nnz_array");
	for(i=0; i<listLength; i++){
		nnz_array[i]=p->sp_vec_y_d_recv_list[i].nnz;
	}
	lnnz_max = max_vec(nnz_array, listLength);
	MPI_Reduce(&lnnz_max, &gnnz_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	lnnz_min = min_vec(nnz_array, listLength);
	MPI_Reduce(&lnnz_min, &gnnz_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	lnnz_tol = sum_vec(nnz_array, listLength);
	lnnz_ave = (MY_FLOAT_TYPE) lnnz_tol / (MY_FLOAT_TYPE) listLength;
	MPI_Reduce(&lnnz_ave, &gnnz_ave, 1, MY_MPI_FLOAT_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
	gnnz_ave = gnnz_ave /  (MY_FLOAT_TYPE) numProc;
	if (procRank == 0) {
		printf("vector y_d receive msg nnz {min,avg,max}:={%d,%f,%d}\n",
				gnnz_min, gnnz_ave, gnnz_max);
	}
	free(nnz_array);

#endif
////////////////////////////////////////////////////////////////////////////////////////
	MPI_Reduce(&p->vec_x_send_list_size, &min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&p->vec_x_send_list_size, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&p->vec_x_send_list_size, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	listLength = p->vec_x_send_list_size;
	lLengthMax = p->vec_x_len;
	MPI_Reduce(&lLengthMax, &gLengthMax, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

	lLengthMin = p->vec_x_len;
	MPI_Reduce(&lLengthMin, &gLengthMin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

	lLengthTot = p->vec_x_len * listLength;
	lLengthAvg = (MY_FLOAT_TYPE) p->vec_x_len;
	MPI_Reduce(&lLengthAvg, &gLengthAvg, 1, MY_MPI_FLOAT_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
	gLengthAvg = gLengthAvg / (MY_FLOAT_TYPE) numProc;
	if (procRank == 0) {
		printf("vector x send list count {min,avg,max}:={%d,%f,%d}\n", min,
				1.0 * sum / numProc, max);
		printf("vector x send msg length {min,avg,max}:={%d,%f,%d}\n",
				gLengthMin, gLengthAvg, gLengthMax);
	}
#ifdef USE_COMPRESSED_VEC
	nnz_array = (int *)malloc(sizeof(int)*listLength);
	if (nnz_array==NULL) PARSE_ERROR("nnz_array");
	for(i=0; i<listLength; i++){
		nnz_array[i]=p->sp_vec_x_send_list[i].nnz;
	}
	lnnz_max = max_vec(nnz_array, listLength);
	MPI_Reduce(&lnnz_max, &gnnz_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	lnnz_min = min_vec(nnz_array, listLength);
	MPI_Reduce(&lnnz_min, &gnnz_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	lnnz_tol = sum_vec(nnz_array, listLength);
	lnnz_ave = (MY_FLOAT_TYPE) lnnz_tol / (MY_FLOAT_TYPE) listLength;
	MPI_Reduce(&lnnz_ave, &gnnz_ave, 1, MY_MPI_FLOAT_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
	gnnz_ave = gnnz_ave /  (MY_FLOAT_TYPE) numProc;
	if (procRank == 0) {
		printf("vector x send msg nnz {min,avg,max}:={%d,%f,%d}\n",
				gnnz_min, gnnz_ave, gnnz_max);
	}
	free(nnz_array);
#endif
////////////////////////////////////////////////////////////////////////////////////////
	MPI_Reduce(&p->vec_y_d_send_list_size, &min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	MPI_Reduce(&p->vec_y_d_send_list_size, &max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	MPI_Reduce(&p->vec_y_d_send_list_size, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	listLength = p->vec_y_d_send_list_size;
	lLengthMax = p->vec_y_d_len;
	MPI_Reduce(&lLengthMax, &gLengthMax, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

	lLengthMin = p->vec_y_d_len;
	MPI_Reduce(&lLengthMin, &gLengthMin, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

	lLengthTot = p->vec_y_d_len * listLength;
	lLengthAvg = (MY_FLOAT_TYPE) p->vec_y_d_len;
	MPI_Reduce(&lLengthAvg, &gLengthAvg, 1, MY_MPI_FLOAT_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
	gLengthAvg = gLengthAvg / (MY_FLOAT_TYPE) numProc;
	if (procRank == 0) {
		printf("vector y_d send list count {min,avg,max}:={%d,%f,%d}\n", min,
				1.0 * sum / numProc, max);
		printf("vector y_d send msg length {min,avg,max}:={%d,%f,%d}\n",
				gLengthMin, gLengthAvg, gLengthMax);
	}
#ifdef USE_COMPRESSED_VEC
	nnz_array = (int *)malloc(sizeof(int)*listLength);
	if (nnz_array==NULL) PARSE_ERROR("nnz_array");
	for(i=0; i<listLength; i++){
		nnz_array[i]=p->sp_vec_y_d_send_list[i].nnz;
	}
	lnnz_max = max_vec(nnz_array, listLength);
	MPI_Reduce(&lnnz_max, &gnnz_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
	lnnz_min = min_vec(nnz_array, listLength);
	MPI_Reduce(&lnnz_min, &gnnz_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
	lnnz_tol = sum_vec(nnz_array, listLength);
	lnnz_ave = (MY_FLOAT_TYPE) lnnz_tol / (MY_FLOAT_TYPE) listLength;
	MPI_Reduce(&lnnz_ave, &gnnz_ave, 1, MY_MPI_FLOAT_TYPE, MPI_SUM, 0, MPI_COMM_WORLD);
	gnnz_ave = gnnz_ave /  (MY_FLOAT_TYPE) numProc;
	if (procRank == 0) {
		printf("vector y_d send msg nnz {min,avg,max}:={%d,%f,%d}\n",
				gnnz_min, gnnz_ave, gnnz_max);
	}
	free(nnz_array);
#endif
}

/*
 * @param startIdx extended vector's starting index
 * */
void gather_vec_x(int procRank, int numProc, PPDataV3_TRI * p, const int startIdx, MY_FLOAT_TYPE * vec_x_plus_neighbor)
{
	int i=0, rank=0, err=0;
#if MPICOMMTYPE==1
//	pull vector x from neighbors
	err= MPI_Win_fence(0, p->win_vec_x); if(err!=0) fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank, __FILE__, __LINE__, __FUNCTION__, err);
	for(i=0; i<p->vec_x_recv_list_size; i++) {
		err=MPI_Get(&vec_x_plus_neighbor[p->vec_x_recv_list_startIdx[i]-startIdx], p->vec_x_recv_list_len[i], MY_MPI_FLOAT_TYPE,
				p->vec_x_recv_list_rank[i], (MPI_Aint)0, p->vec_x_recv_list_len[i], MY_MPI_FLOAT_TYPE, p->win_vec_x);
		if(err!=0) fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank, __FILE__, __LINE__, __FUNCTION__, err);
	}
	err=MPI_Win_fence(0, p->win_vec_x); if(err!=0) fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank, __FILE__, __LINE__, __FUNCTION__, err);
#elif MPICOMMTYPE==2
//	Isend, Irecv
	int size_send = p->vec_x_send_list_size;
	int size_recv = p->vec_x_recv_list_size;
	int bptr;

	MPI_Status * sta = (MPI_Status *) malloc((size_send + size_recv) * sizeof(MPI_Status));
	if(sta==NULL) PARSE_ERROR("sta");
	MPI_Request * req = (MPI_Request *) malloc((size_send + size_recv) * sizeof(MPI_Request));
	if(req==NULL) PARSE_ERROR("req");

	for (i = 0; i < size_recv; i++) {
		bptr=p->vec_x_recv_list_startIdx[i] - startIdx;
		err = MPI_Irecv(&vec_x_plus_neighbor[bptr],
				p->vec_x_recv_list_len[i], MY_MPI_FLOAT_TYPE,
				p->vec_x_recv_list_rank[i], 0, MPI_COMM_WORLD,
				&req[i + size_send]);
		if (err != 0) {
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank,
					__FILE__, __LINE__, __FUNCTION__, err);
			exit(-1);
		}
	}
	for (i = 0; i < size_send; i++) {
		err = MPI_Isend(p->vec_x, p->vec_x_len, MY_MPI_FLOAT_TYPE,
				p->vec_x_send_list_rank[i], 0, MPI_COMM_WORLD, &req[i]);
		if (err != 0) {
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank,
					__FILE__, __LINE__, __FUNCTION__, err);
			exit(-1);
		}
	}

	err = MPI_Waitall(size_send + size_recv, req, sta);
	if (err != 0) {
		fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank, __FILE__,
				__LINE__, __FUNCTION__, err);
		exit(-1);
	}
	free(sta);
	free(req);
#endif //End of  MPICOMMTYPE
}

/* Gather compressed sparse vector x. Similar behavior to gather_vec_x.
 * Only difference is this one use compressed sparse vector */
#ifdef USE_COMPRESSED_VEC
void GatherVecXCompressed(int procRank, int numProc, PPDataV3_TRI * p, const int startIdx, MY_FLOAT_TYPE * extended_vec_x, int isFinalIteration){
	int i=0, rank=0, err=0;

/* 1. nonblocking send and receive CompressedVec.val*/
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("step1_comm");
#endif
	int size_send = p->vec_x_send_list_size;
	int size_recv = p->vec_x_recv_list_size;

	static MPI_Status * sta = NULL;
	if(NULL==sta){
		sta = (MPI_Status *) malloc((size_send + size_recv) * sizeof(MPI_Status));
		if(sta==NULL) PARSE_ERROR("sta");
	}

	static MPI_Request * req = NULL;
	if(NULL==req){
		req = (MPI_Request *) malloc((size_send + size_recv) * sizeof(MPI_Request));
		if(req==NULL) PARSE_ERROR("req");
	}

	for (i = 0; i < size_recv; i++) {
		CompressedVec * sv = &(p->sp_vec_x_recv_list[i]);
		err = MPI_Irecv(sv->val, sv->nnz, MY_MPI_FLOAT_TYPE, p->vec_x_recv_list_rank[i],
				0, MPI_COMM_WORLD, &req[i + size_send]);
		if (err != 0) {
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank,
					__FILE__, __LINE__, __FUNCTION__, err);
			exit(-1);
		}
	}
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	for (i = 0; i < size_send; i++) {
#ifdef GPTL
		GPTLstart("CopyRegularVecToSparseVec");
#endif
		CopyRegularVecToSparseVec(p->vec_x, &(p->sp_vec_x_send_list[i]));
#ifdef GPTL
		GPTLstop("CopyRegularVecToSparseVec");
#endif
		CompressedVec * sv =  &(p->sp_vec_x_send_list[i]);
		err = MPI_Isend(sv->val, sv->nnz, MY_MPI_FLOAT_TYPE, p->vec_x_send_list_rank[i],
				0, MPI_COMM_WORLD, &req[i]);
		if (err != 0) {
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank,
					__FILE__, __LINE__, __FUNCTION__, err);
			exit(-1);
		}
	}

	err = MPI_Waitall(size_send + size_recv, req, sta);
	if (err != 0) {
		fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank, __FILE__,
				__LINE__, __FUNCTION__, err);
		exit(-1);
	}
	if(isFinalIteration){
		free(sta);
		free(req);
	}
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstop("step1_comm");
#endif
/* 2. distribute CompressedVec.val to extended vector*/
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("step2_comp");
#endif
	for (i = 0; i < size_recv; i++) {
		CompressedVec * sv = &(p->sp_vec_x_recv_list[i]);
//		int len = p->vec_x_recv_list_len[i];
//		int bptr=p->vec_x_recv_list_startIdx[i] - startIdx;
		UncompressedSparseVec(extended_vec_x + p->vec_x_recv_list_startIdx[i] - startIdx,
				p->vec_x_recv_list_len[i], sv);
	}

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstop("step2_comp");
#endif
}
#endif// USE_COMPRESSED_VEC


/* Initialization and finalization of vector y for vector gather in multi-diagonal case
 * compute vector y overlap with neighbors, only as at iteration==1
 * @param p OUTPUT: p->vec_y_recv_list_* and p->extended_local_y_* are filled in this function
 * */
void gather_vec_y_init (int procRank, int numProc, int iteration,
		PPDataV3_TRI * p, int * check_startIdx1, int * check_endIdx1)
{
	int i=0, rank=0, err=0;

	/*Initialization*/
	if (iteration == 0 && numProc > 1) {
		 //1. Initialization
		int * vec_y_d_start_idx_all_array = (int *) malloc(
				(numProc + 1) * sizeof(int));
		if (vec_y_d_start_idx_all_array == NULL) {
			fprintf(stderr, "%s(%d)-%s:", __FILE__, __LINE__, __FUNCTION__);
			exit(-1);
		}
		MPI_Allgather(&p->vec_y_d_begin, 1, MPI_INT,
				vec_y_d_start_idx_all_array, 1, MPI_INT, MPI_COMM_WORLD);
		vec_y_d_start_idx_all_array[numProc] = p->matCSC_d_bycol->numRow;

		/*		int * projectedARowToY = (int *)malloc(m_d*sizeof(int));
		 if(projectedARowToY==NULL){
		 fprintf(stderr, "%s(%d)-%s:", __FILE__, __LINE__, __FUNCTION__);exit(-1);
		 }
		 projectingARowToY(p->matCSC_d_bycol, projectedARowToY);*/

		/* 2/18/2012 make vector y scalable.
		 * Note that startIdx, len is not the same as extended_vec_x_startIdx, extended_vec_x_len  */
		int projectedARowToY_startIdx = 0, projectedARowToY_len = 0;
		projectingARowToY_v2_countLength(p->matCSC_d_bycol, &projectedARowToY_startIdx, &projectedARowToY_len);
		*check_startIdx1 = projectedARowToY_startIdx;
		*check_endIdx1 = projectedARowToY_startIdx + projectedARowToY_len - 1;
	#ifdef DEBUG
		printf("%d: %s(%d)-%s: startIdx=%d, len=%d\n", procRank, __FILE__, __LINE__, __FUNCTION__, projectedARowToY_startIdx, projectedARowToY_len);
	#endif

		int * projectedARowToY = (int *) malloc(projectedARowToY_len * sizeof(int));
		if (projectedARowToY == NULL) {
			fprintf(stderr, "%s(%d)-%s:\n", __FILE__, __LINE__, __FUNCTION__);
			exit(-1);
		}
		memset(projectedARowToY, 0, projectedARowToY_len * sizeof(int));
		projectingARowToY_v2(p->matCSC_d_bycol, projectedARowToY, projectedARowToY_startIdx, projectedARowToY_len);

		int c1 = 0; //comm counter
		for (i = 0; i < projectedARowToY_len; i++) {
			if (projectedARowToY[i] == 1) {
				for (rank = 0; rank < numProc; rank++) {
					int globalIdx = i + projectedARowToY_startIdx;
					if (rank != procRank
							&& vec_y_d_start_idx_all_array[rank] <= globalIdx
							&& globalIdx < vec_y_d_start_idx_all_array[rank + 1]) { //not include itself rank!=procRank
						if (c1 >= COMM_SET_LEN) {
							fprintf(stderr, "%s(%d)-%s: comm set is greater than COMM_SET_LEN=%d\n",
									__FILE__, __LINE__, __FUNCTION__, COMM_SET_LEN);
							exit(-1);
						}
						if (c1 == 0 || (c1 > 0 && rank != p->vec_y_d_recv_list_rank[c1 - 1])) {
							p->vec_y_d_recv_list_rank[c1] = rank;
							p->vec_y_d_recv_list_startIdx[c1] = vec_y_d_start_idx_all_array[rank];
							p->vec_y_d_recv_list_len[c1] = vec_y_d_start_idx_all_array[rank + 1]
											- vec_y_d_start_idx_all_array[rank];
							c1++;
						}
					}
				}
			}
		}
		p->vec_y_d_recv_list_size = c1;

		free(projectedARowToY);
	//		3/8/2012 determine extended_local_y_startIdx, extended_local_y_len
		for (rank = 0; rank < numProc; rank++) {
			if (vec_y_d_start_idx_all_array[rank] <= *check_startIdx1
					&& *check_startIdx1 < vec_y_d_start_idx_all_array[rank + 1]) {
				p->extended_vec_y_d_startIdx = vec_y_d_start_idx_all_array[rank];
			}
			if (vec_y_d_start_idx_all_array[rank] <= *check_endIdx1
					&& *check_endIdx1 < vec_y_d_start_idx_all_array[rank + 1]) {
				p->extended_vec_y_d_len =
						vec_y_d_start_idx_all_array[rank + 1] - p->extended_vec_y_d_startIdx;
			}
		}

		free(vec_y_d_start_idx_all_array);

	#if MPICOMMTYPE==2
	/* fill send_candidate_y, 	used for sender, initialized in the 0st iteration
	 * send_candidate_y[i]=1 indicates this MPI task needs to send vector to the ith rank,
	 * send_candidate_y[i]=-1 indicates this MPI task does NOT needs to send vector to the ith rank
	 * */
		int * send_candidate_y;
		MPI_Alloc_mem(numProc * sizeof(int), MPI_INFO_NULL, &(send_candidate_y));
		if (NULL == send_candidate_y) {
			fprintf(stderr, "%d: %s(%d)-%s", procRank, __FILE__, __LINE__,
					__FUNCTION__);
			exit(-1);
		}
		memset(send_candidate_y, -1, numProc * sizeof(int));
		MPI_Win win_send_candidate_y;
		err = MPI_Win_create(send_candidate_y, (MPI_Aint) (numProc * sizeof(int)),
				sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_send_candidate_y);
		if (0 != err) {
			fprintf(stderr, "%d: %s(%d)-%s", procRank, __FILE__, __LINE__,
					__FUNCTION__);
			exit(-1);
		}

		err = MPI_Win_fence(0, win_send_candidate_y);
		if (err != 0) {
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank,
					__FILE__, __LINE__, __FUNCTION__, err);
			exit(-1);
		}
		//2. mark receiver's rank at sender's list
		for (i = 0; i < p->vec_y_d_recv_list_size; i++) {
			err = MPI_Put(&procRank, 1, MPI_INT, p->vec_y_d_recv_list_rank[i],
					(MPI_Aint) procRank, 1, MPI_INT, win_send_candidate_y);
			if (err != 0)
				fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank,
						__FILE__, __LINE__, __FUNCTION__, err);
		}
		err = MPI_Win_fence(0, win_send_candidate_y);
		if (err != 0) {
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank,
					__FILE__, __LINE__, __FUNCTION__, err);
			exit(-1);
		}

	//count and setup send list
		p->vec_y_d_send_list_size = 0;
		for (i = 0; i < numProc; i++) {
			if (send_candidate_y[i] >= 0)
				p->vec_y_d_send_list_size++;
		}
		if (p->vec_y_d_send_list_size > COMM_SET_LEN) {
			fprintf(stderr, "%s(%d)-%s: comm set is greater than COMM_SET_LEN=%d\n",
					__FILE__, __LINE__, __FUNCTION__, COMM_SET_LEN);
			exit(-1);
		}
		int m = 0;
		for (i = 0; i < numProc; i++) {
			if (send_candidate_y[i] >= 0) {
				p->vec_y_d_send_list_rank[m] = send_candidate_y[i];
				m++;
			}
		}
		MPI_Win_free(&win_send_candidate_y);
		MPI_Free_mem(send_candidate_y);
	#endif



	}//ENDOF	if (iteration == 1 && numProc > 1)
}

#ifdef USE_COMPRESSED_VEC
/* The first part is the same as gather_vec_y_init , except the location of free(projectedVectorY)
 * Initialize sparse vector in PPDataV3_TRI
 * by projectedARowToY[p->vec_y_d_begin - startIdx]: a fake vec_y_d, with lenght vec_y_d_len
 * USE_COMPRESSED_VEC
 * */
void GatherVecYInitByProjY(int procRank, int numProc,
		PPDataV3_TRI * p, int * check_startIdx1, int * check_endIdx1)
{
	if(1==numProc) return;
///////////////////////////FIRST PART/////////////////////////////////////////////////////

	int i=0, j=0, rank=0, err=0;

	/*Initialization*/
	 //1. Initialization
	int * vec_y_d_start_idx_all_array = (int *) malloc(
			(numProc + 1) * sizeof(int));
	if (vec_y_d_start_idx_all_array == NULL) {
		fprintf(stderr, "%s(%d)-%s:", __FILE__, __LINE__, __FUNCTION__);
		exit(-1);
	}
	MPI_Allgather(&p->vec_y_d_begin, 1, MPI_INT,
			vec_y_d_start_idx_all_array, 1, MPI_INT, MPI_COMM_WORLD);
	vec_y_d_start_idx_all_array[numProc] = p->matCSC_d_bycol->numRow;

	/*		int * projectedARowToY = (int *)malloc(m_d*sizeof(int));
	 if(projectedARowToY==NULL){
	 fprintf(stderr, "%s(%d)-%s:", __FILE__, __LINE__, __FUNCTION__);exit(-1);
	 }
	 projectingARowToY(p->matCSC_d_bycol, projectedARowToY);*/

	/* 2/18/2012 make vector y scalable.
	 * Note that startIdx, len is not the same as extended_vec_x_startIdx, extended_vec_x_len  */
	int projectedARowToY_startIdx = 0, projectedARowToY_len = 0;
	projectingARowToY_v2_countLength(p->matCSC_d_bycol, &projectedARowToY_startIdx, &projectedARowToY_len);
	*check_startIdx1 = projectedARowToY_startIdx;
	*check_endIdx1 = projectedARowToY_startIdx + projectedARowToY_len - 1;
#ifdef DEBUG
	printf("%d: %s(%d)-%s: startIdx=%d, len=%d\n", procRank, __FILE__, __LINE__, __FUNCTION__, projectedARowToY_startIdx, projectedARowToY_len);
#endif

	int * projectedARowToY = (int *) malloc(projectedARowToY_len * sizeof(int));
	if (projectedARowToY == NULL) {
		fprintf(stderr, "%s(%d)-%s:\n", __FILE__, __LINE__, __FUNCTION__);
		exit(-1);
	}
	memset(projectedARowToY, 0, projectedARowToY_len * sizeof(int));
	projectingARowToY_v2(p->matCSC_d_bycol, projectedARowToY, projectedARowToY_startIdx, projectedARowToY_len);

	int c1 = 0; //comm counter
	for (i = 0; i < projectedARowToY_len; i++) {
		if (projectedARowToY[i] == 1) {
			for (rank = 0; rank < numProc; rank++) {
				int globalIdx = i + projectedARowToY_startIdx;
				if (rank != procRank
						&& vec_y_d_start_idx_all_array[rank] <= globalIdx
						&& globalIdx < vec_y_d_start_idx_all_array[rank + 1]) { //not include itself rank!=procRank
					if (c1 >= COMM_SET_LEN) {
						fprintf(stderr, "%s(%d)-%s: comm set is greater than COMM_SET_LEN=%d\n",
								__FILE__, __LINE__, __FUNCTION__, COMM_SET_LEN);
						exit(-1);
					}
					if (c1 == 0 || (c1 > 0 && rank != p->vec_y_d_recv_list_rank[c1 - 1])) {
						p->vec_y_d_recv_list_rank[c1] = rank;
						p->vec_y_d_recv_list_startIdx[c1] = vec_y_d_start_idx_all_array[rank];
						p->vec_y_d_recv_list_len[c1] = vec_y_d_start_idx_all_array[rank + 1]
										- vec_y_d_start_idx_all_array[rank];
						c1++;
					}
				}
			}
		}
	}
	p->vec_y_d_recv_list_size = c1;


//		3/8/2012 determine extended_local_y_startIdx, extended_local_y_len
	for (rank = 0; rank < numProc; rank++) {
		if (vec_y_d_start_idx_all_array[rank] <= *check_startIdx1
				&& *check_startIdx1 < vec_y_d_start_idx_all_array[rank + 1]) {
			p->extended_vec_y_d_startIdx = vec_y_d_start_idx_all_array[rank];
		}
		if (vec_y_d_start_idx_all_array[rank] <= *check_endIdx1
				&& *check_endIdx1 < vec_y_d_start_idx_all_array[rank + 1]) {
			p->extended_vec_y_d_len =
					vec_y_d_start_idx_all_array[rank + 1] - p->extended_vec_y_d_startIdx;
		}
	}

	free(vec_y_d_start_idx_all_array);

#if MPICOMMTYPE==2
/* fill send_candidate_y, 	used for sender, initialized in the 0st iteration
 * send_candidate_y[i]=1 indicates this MPI task needs to send vector to the ith rank,
 * send_candidate_y[i]=-1 indicates this MPI task does NOT needs to send vector to the ith rank
 * */
	int * send_candidate_y;
	err=MPI_Alloc_mem(numProc * sizeof(int), MPI_INFO_NULL, &(send_candidate_y));
	if(err!=MPI_SUCCESS)	PARSE_MPI_ERROR(err);
	if (NULL == send_candidate_y) {
		fprintf(stderr, "%d: %s(%d)-%s", procRank, __FILE__, __LINE__,
				__FUNCTION__);
		exit(-1);
	}
	memset(send_candidate_y, -1, numProc * sizeof(int));
	MPI_Win win_send_candidate_y;
	err = MPI_Win_create(send_candidate_y, (MPI_Aint) (numProc * sizeof(int)),
			sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &win_send_candidate_y);
	if(err!=MPI_SUCCESS)	PARSE_MPI_ERROR(err);

	err = MPI_Win_fence(0, win_send_candidate_y);
	if(err!=MPI_SUCCESS)	PARSE_MPI_ERROR(err);

	//2. mark receiver's rank at sender's list
	for (i = 0; i < p->vec_y_d_recv_list_size; i++) {
		err = MPI_Put(&procRank, 1, MPI_INT, p->vec_y_d_recv_list_rank[i],
				(MPI_Aint) procRank, 1, MPI_INT, win_send_candidate_y);
		if (err != 0)
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank,
					__FILE__, __LINE__, __FUNCTION__, err);
	}
	err = MPI_Win_fence(0, win_send_candidate_y);
	if(err!=MPI_SUCCESS)	PARSE_MPI_ERROR(err);

//count and setup send list
	p->vec_y_d_send_list_size = 0;
	for (i = 0; i < numProc; i++) {
		if (send_candidate_y[i] >= 0)
			p->vec_y_d_send_list_size++;
	}
	if (p->vec_y_d_send_list_size > COMM_SET_LEN) {
		fprintf(stderr, "%s(%d)-%s: comm set is greater than COMM_SET_LEN=%d\n",
				__FILE__, __LINE__, __FUNCTION__, COMM_SET_LEN);
		exit(-1);
	}
	int m = 0;
	for (i = 0; i < numProc; i++) {
		if (send_candidate_y[i] >= 0) {
			p->vec_y_d_send_list_rank[m] = send_candidate_y[i];
			m++;
		}
	}
	err=MPI_Win_free(&win_send_candidate_y);
	if(err!=MPI_SUCCESS)	PARSE_MPI_ERROR(err);
	err=MPI_Free_mem(send_candidate_y);
	if(err!=MPI_SUCCESS)	PARSE_MPI_ERROR(err);
#endif


//////////////////////////////SECOND PART//////////////////////////////////////////////
	if(p->vec_y_d_recv_list_size!=0)
		p->sp_vec_y_d_recv_list=(CompressedVec *)malloc(p->vec_y_d_recv_list_size * sizeof(CompressedVec));
	else p->sp_vec_y_d_recv_list=NULL;
	if(p->vec_y_d_send_list_size!=0)
		p->sp_vec_y_d_send_list=(CompressedVec *)malloc(p->vec_y_d_send_list_size * sizeof(CompressedVec));
	else p->sp_vec_y_d_send_list=NULL;
/*Temporal receiving list, used to receive list from other neighbors and construct sparse receive list*/
	int ** temp_vec_y_d_recv_list= (int **)malloc(p->vec_y_d_recv_list_size * sizeof(int *));
	if(temp_vec_y_d_recv_list==NULL){
		fprintf(stderr, "%s(%d):%s\n", __FILE__, __LINE__, __FUNCTION__); exit(-1);
	}

	int chunk=(p->vec_y_d_recv_list_size + group_size_omp-1) / (group_size_omp);
#pragma omp parallel for private(i) schedule(dynamic, chunk)
	for(i=0; i<p->vec_y_d_recv_list_size; i++){
		temp_vec_y_d_recv_list[i]=(int *)malloc(p->vec_y_d_recv_list_len[i] * sizeof(int));
		if(NULL==temp_vec_y_d_recv_list[i]){
			fprintf(stderr, "%s(%d):%s\n", __FILE__, __LINE__, __FUNCTION__); exit(-1);
		}
		memset(temp_vec_y_d_recv_list[i], 0, p->vec_y_d_recv_list_len[i] * sizeof(int));
	}

	int ** temp_vec_y_d_send_list=(int **)malloc(p->vec_y_d_send_list_size * sizeof(int *));
	if(temp_vec_y_d_send_list==NULL){
		fprintf(stderr, "%s(%d):%s\n", __FILE__, __LINE__, __FUNCTION__); exit(-1);
	}

	chunk=(p->vec_y_d_send_list_size + group_size_omp-1) / (group_size_omp);
#pragma omp parallel for private(i) schedule(dynamic, chunk)
	for(i=0; i<p->vec_y_d_send_list_size; i++){
		temp_vec_y_d_send_list[i]=(int *)malloc(p->vec_y_d_len * sizeof(int));
		if(NULL==temp_vec_y_d_send_list[i]){
			fprintf(stderr, "%s(%d):%s\n", __FILE__, __LINE__, __FUNCTION__); exit(-1);
		}
		memset(temp_vec_y_d_send_list[i], 0, p->vec_y_d_len * sizeof(int));
	}

/*	Loop projectedARowToY and fill temp_vec_y_d_recv_list,
*	and then use temp_vec_y_d_recv_list to fill sp_vec_y_d_recv_list*/
	for (i = 0; i < projectedARowToY_len; i++) {
		if (projectedARowToY[i] == 1) {
			int globalIdx = i + projectedARowToY_startIdx;
			for(j=0; j<p->vec_y_d_recv_list_size; j++){
				int startIdx=p->vec_y_d_recv_list_startIdx[j];
				int len=p->vec_y_d_recv_list_len[j];
				if(startIdx<=globalIdx && globalIdx<(startIdx+len)){
					int idx=globalIdx - startIdx;
					temp_vec_y_d_recv_list[j][idx]=1;
				}
			}
		}
	}
	free(projectedARowToY);
	for(j=0; j<p->vec_y_d_recv_list_size; j++){
		InitSparseVecFromOriginalIntVec(temp_vec_y_d_recv_list[j], p->vec_y_d_recv_list_len[j],
				&(p->sp_vec_y_d_recv_list[j]));
	}

/*	Inform temp_vec_y_d_send_list by temp_vec_y_d_recv_list
* Note that : p->vec_y_d_recv_list_size and p->vec_y_d_send_list_size are in REVERSE order*/
	int size_send=p->vec_y_d_send_list_size;
	int size_recv=p->vec_y_d_recv_list_size;

	MPI_Status * sta = (MPI_Status *) malloc((size_send + size_recv) * sizeof(MPI_Status));
	MPI_Request * req = (MPI_Request *) malloc((size_send + size_recv) * sizeof(MPI_Request));

	for (i = 0; i < size_send; i++) {
		err = MPI_Irecv(temp_vec_y_d_send_list[i], p->vec_y_d_len, MPI_INT,
				p->vec_y_d_send_list_rank[i], 0, MPI_COMM_WORLD,
				&req[i + size_recv]);
		if (err != 0) {
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", rank,
					__FILE__, __LINE__, __FUNCTION__, err);
			exit(-1);
		}
	}
	for (i = 0; i < size_recv; i++) {
		err = MPI_Isend(temp_vec_y_d_recv_list[i], p->vec_y_d_recv_list_len[i] , MPI_INT,
				p->vec_y_d_recv_list_rank[i], 0, MPI_COMM_WORLD, &req[i]);
		if (err != 0) {
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", rank,
					__FILE__, __LINE__, __FUNCTION__, err);
			exit(-1);
		}
	}

	err = MPI_Waitall(size_send + size_recv, req, sta);
	if (err != 0) {
		fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", rank, __FILE__,
				__LINE__, __FUNCTION__, err);
		exit(-1);
	}
	free(sta);
	free(req);

/*Convert temp_vec_y_d_send_list to compressed sp_vec_y_d_send_list*/
	for(i=0; i<p->vec_y_d_send_list_size; i++){
		InitSparseVecFromOriginalIntVec(temp_vec_y_d_send_list[i], p->vec_y_d_len,
				&(p->sp_vec_y_d_send_list[i]));
	}


/*free memory*/
	for(i=0; i<p->vec_y_d_recv_list_size; i++){
		if(temp_vec_y_d_recv_list[i]!=NULL){
			free(temp_vec_y_d_recv_list[i]);
		}
	}
	if(temp_vec_y_d_recv_list!=NULL) free(temp_vec_y_d_recv_list);

	for(i=0; i<p->vec_y_d_send_list_size; i++){
		if(temp_vec_y_d_send_list[i]!=NULL){
			free(temp_vec_y_d_send_list[i]);
		}
	}
	if(temp_vec_y_d_send_list!=NULL) free(temp_vec_y_d_send_list);


}
#endif //END OF USE_COMPRESSED_VEC

/*
 * @param startIdx extended vector's starting index
 * */
void gather_vec_y(int procRank, int numProc, PPDataV3_TRI * p, const int startIdx, MY_FLOAT_TYPE * vec_y_d_plus_neighbor)
{
	int i=0, rank=0, err=0;

#if MPICOMMTYPE==1
	err=MPI_Win_fence(0, p->win_vec_y_d); if(err!=0) fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank, __FILE__, __LINE__, __FUNCTION__, err);
	for(i=0; i<p->vec_y_d_recv_list_size; i++) { //2. pull vector y from neighbors
		err=MPI_Get(&vec_y_d_plus_neighbor[p->vec_y_d_recv_list_startIdx[i]-startIdx], p->vec_y_d_recv_list_len[i], MY_MPI_FLOAT_TYPE,
				p->vec_y_d_recv_list_rank[i], (MPI_Aint)0, p->vec_y_d_recv_list_len[i],MY_MPI_FLOAT_TYPE, p->win_vec_y_d);
		if(err!=0) fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank, __FILE__, __LINE__, __FUNCTION__, err);
	}
	err=MPI_Win_fence(0, p->win_vec_y_d); if(err!=0) fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank, __FILE__, __LINE__, __FUNCTION__, err);

#elif MPICOMMTYPE==2
//	Isend, Irecv
	int size_send = p->vec_y_d_send_list_size;
	int size_recv = p->vec_y_d_recv_list_size;
	int bptr;

	MPI_Status * sta = (MPI_Status *)malloc((size_send + size_recv) * sizeof(MPI_Status));
	if(sta==NULL) PARSE_ERROR("sta");
	MPI_Request * req = (MPI_Request *) malloc((size_send + size_recv) * sizeof(MPI_Request));
	if(req==NULL) PARSE_ERROR("req");

	for (i = 0; i < size_recv; i++) { //p->vec_y_comm_set_startIdx[i]-startIdx: convert global index to local index for vect y
		bptr = p->vec_y_d_recv_list_startIdx[i] - startIdx;
		err = MPI_Irecv(&vec_y_d_plus_neighbor[bptr],
				p->vec_y_d_recv_list_len[i], MY_MPI_FLOAT_TYPE,
				p->vec_y_d_recv_list_rank[i], 0, MPI_COMM_WORLD, &req[i + size_send]);
		if (err != 0) {
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank,
					__FILE__, __LINE__, __FUNCTION__, err);
			exit(-1);
		}
	}

//        MPI_Barrier(MPI_COMM_WORLD);

	for (i = 0; i < size_send; i++) {
		err = MPI_Isend(p->vec_y_d, p->vec_y_d_len, MY_MPI_FLOAT_TYPE,
				p->vec_y_d_send_list_rank[i], 0, MPI_COMM_WORLD, &req[i]);
		if (err != 0) {
			fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank,
					__FILE__, __LINE__, __FUNCTION__, err);
			exit(-1);
		}
	}
	err = MPI_Waitall(size_recv, req, sta);
	if (err != 0) {
		fprintf(stderr, "%d: %s(%d)-%s MPI ERROR CODE=%d", procRank, __FILE__,
				__LINE__, __FUNCTION__, err);
		exit(-1);
	}
	free(sta);
	free(req);
#endif

}

/* Gather compressed sparse vector y. Similar behavior to gather_vec_y_d.
 * Only difference is this one use compressed sparse vector */
#ifdef USE_COMPRESSED_VEC
void GatherVecYCompressed(int procRank, int numProc, PPDataV3_TRI * p, const int startIdx, MY_FLOAT_TYPE * extended_vec_y_d, int isFinalIteration)
{
	int i=0, rank=0, err=0;

	/* 1. nonblocking send and receive CompressedVec.val*/
	int size_send = p->vec_y_d_send_list_size;
	int size_recv = p->vec_y_d_recv_list_size;
	static MPI_Status * sta = NULL;
	static MPI_Request * req = NULL;
	if(NULL==sta){
		sta = (MPI_Status *)malloc((size_send + size_recv) * sizeof(MPI_Status));
		if(sta==NULL) PARSE_ERROR("sta");
	}
	if(NULL==req){
		req = (MPI_Request *) malloc((size_send + size_recv) * sizeof(MPI_Request));
		if(req==NULL) PARSE_ERROR("req");
	}


	for (i = 0; i < size_recv; i++) {
		CompressedVec * sv = &(p->sp_vec_y_d_recv_list[i]);
		err = MPI_Irecv(sv->val, sv->nnz, MY_MPI_FLOAT_TYPE, p->vec_y_d_recv_list_rank[i],
				0, MPI_COMM_WORLD, &req[i + size_send]);
		if(err!=MPI_SUCCESS)	PARSE_MPI_ERROR(err);
	}
	for (i = 0; i < size_send; i++) {
		CompressedVec * sv =  &(p->sp_vec_y_d_send_list[i]);
		CopyRegularVecToSparseVec(p->vec_y_d, sv);
		err = MPI_Isend(sv->val, sv->nnz, MY_MPI_FLOAT_TYPE, p->vec_y_d_send_list_rank[i],
				0, MPI_COMM_WORLD, &req[i]);
		if(err!=MPI_SUCCESS)	PARSE_MPI_ERROR(err);
	}

	err = MPI_Waitall(size_send + size_recv, req, sta);
	if(err!=MPI_SUCCESS)	PARSE_MPI_ERROR(err);

	if(isFinalIteration){
		free(sta);
		free(req);
	}

	/* 2. distribute CompressedVec.val to extended vector*/
	for(i =0; i< size_recv; i++){
		CompressedVec * sv = &(p->sp_vec_y_d_recv_list[i]);
//		int len = p->vec_y_d_recv_list_len[i];
//		int bptr=p->vec_y_d_recv_list_startIdx[i] - startIdx ;
		UncompressedSparseVec(extended_vec_y_d + (p->vec_y_d_recv_list_startIdx[i] - startIdx), p->vec_y_d_recv_list_len[i], sv);
	}
}
#endif// USE_COMPRESSED_VEC

#endif//TRIDIAGONAL
