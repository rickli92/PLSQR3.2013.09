/*
 * tridiagonal.c
 *
 *  Created on: Dec 6, 2011
 *      Author: hh
 */


#include "tridiagonal.h"

#ifdef  PLSQR3

void initPPDataV3_TRI(int procRank, int numProc, PPDataV3_TRI * p, SpMatCSC * matCSC_k,
		SpMatCSR * matCSR_d, SpMatCSC * spMatCSC_d_bycol, int numRow_k, int startCol_perProc_k, int endCol_perProc_k,
		int startRowIdx_perProc_d, int endRowIdx_perProc_d)
{
	p->version=PLSQRV3;
	p->subversion=PLSQRV3TRI;
	int i, err=0;
//kernel matrix and Boundary of kernel block
	p->matCSC_k=matCSC_k;
	p->ai_k=0;
	p->aj_k=startCol_perProc_k;
	p->ci_k=numRow_k-1;
	p->cj_k=endCol_perProc_k;

//damping matrix
	p->matCSR_d=matCSR_d;
	p->matCSC_d_bycol= spMatCSC_d_bycol;

//vector y, allocate memory. decomposed vector
//kernel
	p->vec_y_k=(MY_FLOAT_TYPE *)calloc(numRow_k, sizeof(MY_FLOAT_TYPE));
	if(p->vec_y_k==NULL){fprintf(stderr, "allocating p->vec_y_k fails\n");}
	memset(p->vec_y_k, 0, numRow_k*sizeof(MY_FLOAT_TYPE));

	p->vec_y_d_begin= startRowIdx_perProc_d;
	p->vec_y_d_end= endRowIdx_perProc_d;
	p->vec_y_d_len=p->vec_y_d_end - p->vec_y_d_begin + 1;
//damping
//	p->vec_y_d=(MY_FLOAT_TYPE *)calloc(len1,sizeof(MY_FLOAT_TYPE));
	MPI_Alloc_mem(p->vec_y_d_len*sizeof(MY_FLOAT_TYPE), MPI_INFO_NULL, &(p->vec_y_d));
	if(p->vec_y_d==NULL){fprintf(stderr, "allocating p-> vec_y_d fails\n");}
	memset(p->vec_y_d, 0, p->vec_y_d_len*sizeof(MY_FLOAT_TYPE));
	if( MPI_SUCCESS!=MPI_Win_create(p->vec_y_d, (MPI_Aint)(sizeof(MY_FLOAT_TYPE)*p->vec_y_d_len),
			sizeof(MY_FLOAT_TYPE), MPI_INFO_NULL, MPI_COMM_WORLD, &(p->win_vec_y_d))){
		fprintf(stderr, "%d: %s(%d)-%s", procRank, __FILE__, __LINE__, __FUNCTION__); exit(-1);
	}


//vector x,
	p->vec_x_begin_gloIdx=startCol_perProc_k;
	p->vec_x_end_gloIdx=endCol_perProc_k;
	p->vec_x_len = endCol_perProc_k - startCol_perProc_k + 1;
//	p->vec_x=(MY_FLOAT_TYPE *)malloc(len2*sizeof(MY_FLOAT_TYPE));
	MPI_Alloc_mem(sizeof(MY_FLOAT_TYPE)*p->vec_x_len, MPI_INFO_NULL, &(p->vec_x));
	memset(p->vec_x, 0, p->vec_x_len*sizeof(MY_FLOAT_TYPE) );
	if(MPI_SUCCESS!=MPI_Win_create(p->vec_x, (MPI_Aint)(sizeof(MY_FLOAT_TYPE)*p->vec_x_len),
			sizeof(MY_FLOAT_TYPE), MPI_INFO_NULL, MPI_COMM_WORLD, &(p->win_vec_x))){
		fprintf(stderr, "%d: %s(%d)-%s", procRank, __FILE__, __LINE__, __FUNCTION__); exit(-1);
	}


//	communication
	memset(p->vec_y_d_recv_list_rank, -1, COMM_SET_LEN*sizeof(int));
	memset(p->vec_y_d_recv_list_startIdx, -1, COMM_SET_LEN*sizeof(int));
	memset(p->vec_y_d_recv_list_len, -1, COMM_SET_LEN*sizeof(int));
	p->vec_y_d_recv_list_size=0;

	memset(p->vec_x_recv_list_rank, -1, COMM_SET_LEN*sizeof(int));
	memset(p->vec_x_recv_list_startIdx, -1, COMM_SET_LEN*sizeof(int));
	memset(p->vec_x_recv_list_len, -1, COMM_SET_LEN*sizeof(int));
	p->vec_x_recv_list_size=0;

	memset(p->vec_x_send_list_rank, -1, COMM_SET_LEN*sizeof(int));
	p->vec_x_send_list_size=0;
	memset(p->vec_y_d_send_list_rank, -1, COMM_SET_LEN*sizeof(int));
	p->vec_y_d_send_list_size=0;

	if(numProc==1)	{
		p->extended_vec_y_d_startIdx=p->vec_y_d_begin;
		p->extended_vec_y_d_len=p->vec_y_d_len;
		p->extended_vec_x_startIdx=p->vec_x_begin_gloIdx;
		p->extended_vec_x_len=p->vec_x_len;
	}

#if MPICOMMTYPE==2

#endif

#ifdef USE_COMPRESSED_VEC
	p->sp_vec_y_d_recv_list = p->sp_vec_y_d_send_list = p->sp_vec_x_recv_list = p->sp_vec_x_send_list = NULL;
#endif

}
void finalPPDataV3_TRI(PPDataV3_TRI * p)
{
	free(p->vec_y_k);
//	free(p->vec_y_d);
	MPI_Free_mem(p->vec_y_d);
//	free(p->vec_x);
	MPI_Free_mem(p->vec_x);
	MPI_Win_free(&p->win_vec_x);
	MPI_Win_free(&p->win_vec_y_d);

	/*Finalization*/
#ifdef USE_COMPRESSED_VEC
	int i;
//	FinalizeSparseVec(&(p->sp_vec_y_d));
	for(i=0; i<p->vec_y_d_recv_list_size; i++){
		FinalizeSparseVec(&(p->sp_vec_y_d_recv_list[i]));
	}
	free(p->sp_vec_y_d_recv_list);

	for(i=0; i<p->vec_y_d_send_list_size; i++){
		FinalizeSparseVec(&(p->sp_vec_y_d_send_list[i]));
	}
	free(p->sp_vec_y_d_send_list);

//	FinalizeSparseVec(&(p->sp_vec_x));
	for(i=0; i<p->vec_x_recv_list_size; i++){
		FinalizeSparseVec(&(p->sp_vec_x_recv_list[i]));
	}
	free(p->sp_vec_x_recv_list);

	for(i=0; i<p->vec_x_send_list_size; i++){
		FinalizeSparseVec(&(p->sp_vec_x_send_list[i]));
	}
	free(p->sp_vec_x_send_list);

#endif
}

/* vec: input
 * vecSize: the size of vec
 *  vp:output, is the damping position information of vector vec
 * */
/*
void determineVEC_POS(int rank, int size, VEC_POS * vp, int * vec_damp, int vecSize)
{
	set: >=0
	unset: <0
	start and end is the nonzero bound index

	int i;
	for(i=0; i<vecSize; i++){
		if(vec_damp[i]==0.0){
			if(vp->damp1Start>=0 && vp->damp1End<0)//damp1start is set, but damp1end is unset
				vp->damp1End=i-1;
			else if (vp->damp2Start>=0 && vp->damp2End<0)
				vp->damp2End=i-1;
			else if(vp->damp3Start>=0 && vp->damp3End<0)
				vp->damp3End=i-1;

		}else {//vec[i]!=0.0
			if(vp->damp1Start <0) // if damp1start is not set
				vp->damp1Start=i; //set damp1start
			else if (vp->damp1End>=0 && vp->damp2Start < 0) //if damp1end is set and damp2start is not set
				vp->damp2Start = i; //set damp2start
			else if (vp->damp2End>=0 && vp->damp3Start<0) //if damp2end is set and damp3start is not set
				vp->damp3Start = i; //set damp3start
		}

	}

	//if start is set, but end is not set until the end of the loop, then set end to be the length-1
	if(vp->damp1Start>=0 && vp ->damp1End<0)
		vp->damp1End=vecSize-1;
	else if (vp->damp2Start>=0 && vp->damp2End<0)
		vp->damp2End=vecSize-1;
	else if(vp->damp3Start>=0 && vp->damp3End<0)
		vp->damp3End=vecSize-1;

	//print out result
	fprintf(stdout, "%s(%d)-%s: rank=%d {kerStart=%d, kerEnd=%d}  {damp1Start=%d, damp1End=%d} {damp2Start=%d, damp2End=%d} {damp3Start=%d, damp3End=%d}\n",
			__FILE__,__LINE__,__FUNCTION__, rank, vp->kerStart, vp->kerEnd, vp->damp1Start, vp->damp1End, vp->damp2Start, vp->damp2End, vp->damp3Start, vp->damp3End);
}
*/






//project damping part of matrix A's column index vertically, equivalently project to vector x.
//This can help determine vector x's distribution in individual process
//return projectedVectorX, 0, means no element, 1 means there is nonzero element
void projectingAColumnToX (const SpMatCSR * matCSR_d, int * projectedVectorX)
{
	int m = matCSR_d->numRow;
	int n = matCSR_d->numCol;
	//set projectedVectorX to 0
	memset(projectedVectorX, 0, n*sizeof(int));
	int first, last, colIdx;
	int i=0, j=0;
	for(i=0; i<m; i++){
		first = matCSR_d->row_ptr[i];
		last = matCSR_d->row_ptr[i+1];
		for(j=first; j<last; j++){
			colIdx = matCSR_d->col_idx[j];
			projectedVectorX[colIdx]=1;
		}
	}
}
/* count length and startIdx for projectingAColumnToX_v2
 * @param startIdx OUTPUT
 * @param len OUTPUT
 * */
void projectingAColumnToX_v2_countLength(const SpMatCSR * matCSR_d, int * startIdx, int * len)
{
	int m = matCSR_d->numRow;
	int n = matCSR_d->numCol;
	int first, last, colIdx, i=0, j=0, endIdx;
	*startIdx=INT_MAX;
	endIdx=INT_MIN;

	for(i=0; i<m; i++){
		first = matCSR_d->row_ptr[i];
		last = matCSR_d->row_ptr[i+1];
		for(j=first; j<last; j++){
			colIdx = matCSR_d->col_idx[j];
			if(colIdx<*startIdx){
				*startIdx=colIdx;
			}
			if(colIdx>endIdx){
				endIdx=colIdx;
			}
		}
	}
	*len=endIdx-*startIdx+1;
}
/* Almost the same as projectingAColumnToX, except using startIdx for vector x
 * @param projectedVectorX OUTPUT, 0 indicates there is no nonzero element, 1 indicates there is nonzero element in this column*/
void projectingAColumnToX_v2 (SpMatCSR * matCSR_d, int * projectedVectorX, int  startIdx, int  len)
{
	int m = matCSR_d->numRow;
	int n = matCSR_d->numCol;
	int first, last, /*colIdx,*/ i=0, j=0;
	for(i=0; i<m; i++){
		first = matCSR_d->row_ptr[i];
		last = matCSR_d->row_ptr[i+1];
		for(j=first; j<last; j++){
//			colIdx = matCSR_d->col_idx[j];
			int local_col_idx = matCSR_d->col_idx[j] - startIdx;
			if(local_col_idx <len)
				projectedVectorX[local_col_idx]=1;//convert to local index
			else{
				fprintf(stderr, "%s(%d)-%s\n", __FILE__, __LINE__, __FUNCTION__); exit(-1);
			}
		}
	}
}


/*Projecting matrix A to y axis. */
void projectingARowToY(const SpMatCSC * matCSC_d, int * projectingARowToY)
{
	int m = matCSC_d->numRow;
	int n = matCSC_d->numCol;
	//set 0
	memset(projectingARowToY, 0, m*sizeof(int));
	int first, last, rowIdx, i, j;
	for(i=0; i<n; i++){
		first=matCSC_d->col_ptr[i];
		last=matCSC_d->col_ptr[i+1];
		for(j=first; j<last; j++){
			rowIdx=matCSC_d->row_idx[j];
			projectingARowToY[rowIdx]=1;
		}
	}
}

/* 2/18/2012
 * Determinate the start index and length of projectingARowToY
 * output: int * startIdx, int * len
 * */
void projectingARowToY_v2_countLength(const SpMatCSC * matCSC_d, int * startIdx, int * len){
	int m = matCSC_d->numRow;
	int n = matCSC_d->numCol;
	int first, last, rowIdx, i, j, endIdx;
	*startIdx=INT_MAX;
	endIdx=INT_MIN;
	for(i=0; i<n; i++){
		first=matCSC_d->col_ptr[i];
		last=matCSC_d->col_ptr[i+1];
		for(j=first; j<last; j++){
			rowIdx=matCSC_d->row_idx[j];
			if(rowIdx < *startIdx){
				*startIdx=rowIdx;
			}
			if(rowIdx > endIdx){
				endIdx=rowIdx;
			}
		}
	}
	*len=endIdx-*startIdx+1;
}
/* 2/18/2012
 * Projecting matrix A to y axis.
 * Almost the same as projectingARowToY, except using startIdx for projectingARowToY
 */
void projectingARowToY_v2(const SpMatCSC * matCSC_d, int * projectingARowToY, const int startIdx, const int len)
{
	int m = matCSC_d->numRow;
	int n = matCSC_d->numCol;
	int first, last, rowIdx, i, j;
	for(i=0; i<n; i++){
		first=matCSC_d->col_ptr[i];
		last=matCSC_d->col_ptr[i+1];
		for(j=first; j<last; j++){
			rowIdx=matCSC_d->row_idx[j];//global index
			if(rowIdx-startIdx < len)
				projectingARowToY[rowIdx-startIdx]=1;//convert to local index
			else{
				fprintf(stderr, "%s(%d)-%s\n", __FILE__, __LINE__, __FUNCTION__); exit(-1);
			}
		}
	}
}

//#define PRINT_CONCISE //If enable this macro, printPPDataV3_TRIToFile will only print some necessary information

void printPPDataV3_TRIToFile(int rank, int size, const PPDataV3_TRI * p, char * name)
{
	char  filename[1024];
	sprintf(filename, "PPDataV3_TRI_rank_%d_%s", rank, name);
	int colIdx, rowIdx, firstEntryInCol, lastEntryInCol, entryIdx;
	int firstEntryInRow, lastEntryInRow;
	int i, j;
	FILE * fp=fopen(filename, "w");
	if(fp==NULL){
		fprintf(stderr, "%d: cannot open %s\n", rank, filename);
	}
	fprintf(fp, "version=%d\n", p->version);
	fprintf(fp, "\nai_k=%d, aj_k=%d, ci_k=%d, cj_k=%d\n", p->ai_k, p->aj_k, p->ci_k, p->cj_k);

//	fprintf(fp, "\nvec_y_begin_gloIdx_d=%d, vec_y_end_gloIdx_d=%d\n",
//			p->vec_y_d_begin, p->vec_y_d_end);


	//block_k
	fprintf(fp, "\nblock_k, numRow=%d, numCol=%d, numNonzero=%d:\n",
			p->matCSC_k->numRow, p->matCSC_k->numCol, p->matCSC_k->numNonzero);
	for(colIdx=0; colIdx<p->matCSC_k->numCol; colIdx++){
		firstEntryInCol=p->matCSC_k->col_ptr[colIdx];
		lastEntryInCol=p->matCSC_k->col_ptr[colIdx+1];
		for(entryIdx=firstEntryInCol; entryIdx<lastEntryInCol; entryIdx++){
			rowIdx=p->matCSC_k->row_idx[entryIdx];
#ifdef PRINT_CONCISE
			if((colIdx==0||colIdx==p->matCSC_k->numCol-1) && entryIdx==firstEntryInCol)
#endif
			fprintf(fp, "%d_[%d][%d]=%e\n", entryIdx, rowIdx, colIdx, p->matCSC_k->val[entryIdx]);
		}
	}
	//block_d
	fprintf(fp, "\nblock_d, numRow=%d, numCol=%d, numNonzero=%d:\n",
			p->matCSR_d->numRow, p->matCSR_d->numCol, p->matCSR_d->numNonzero);
	for(rowIdx=0; rowIdx<p->matCSR_d->numRow; rowIdx++){
		firstEntryInRow=p->matCSR_d->row_ptr[rowIdx];
		lastEntryInRow=p->matCSR_d->row_ptr[rowIdx+1];
//		fprintf(fp, "firstEntryInRow=%d, lastEntryInRow=%d\n", firstEntryInRow, lastEntryInRow);
		for (entryIdx = firstEntryInRow; entryIdx < lastEntryInRow; entryIdx++){
			colIdx = p->matCSR_d->col_idx[entryIdx];
#ifdef PRINT_CONCISE
			if((rowIdx==0 || rowIdx == p->matCSR_d->numRow-1) && entryIdx == firstEntryInRow)
#endif
			fprintf(fp, "%d_[%d][%d]=%e\n", entryIdx, rowIdx, colIdx, p->matCSR_d->val[entryIdx]);
		}
	}

	//block_d_bycol
	fprintf(fp, "\nblock_d_bycol, numRow=%d, numCol=%d, numNonzero=%d:\n",
			p->matCSC_d_bycol->numRow, p->matCSC_d_bycol->numCol, p->matCSC_d_bycol->numNonzero);
	for(colIdx=0; colIdx<p->matCSC_d_bycol->numCol; colIdx++){
		firstEntryInCol=p->matCSC_d_bycol->col_ptr[colIdx];
		lastEntryInCol=p->matCSC_d_bycol->col_ptr[colIdx+1];
		for(entryIdx=firstEntryInCol; entryIdx<lastEntryInCol; entryIdx++){
			rowIdx=p->matCSC_d_bycol->row_idx[entryIdx];
#ifdef PRINT_CONCISE
			if((colIdx==0 || colIdx == p->matCSC_d_bycol->numCol-1) && entryIdx==firstEntryInCol)
#endif
			fprintf(fp, "%d_[%d][%d]=%e\n", entryIdx, rowIdx, colIdx, p->matCSC_d_bycol->val[entryIdx]);
		}
	}

	//vector
	fprintf(fp, "\nvec_y_k, dimension=%d:\n", p->matCSC_k->numRow);
	for(i=0; i<p->matCSC_k->numRow; i++){
#ifdef PRINT_CONCISE
			if(i==0 || i==p->matCSC_k->numRow-1)
#endif
		fprintf(fp, "[%d]=%e\n", i, p->vec_y_k[i]);
	}
	fprintf(fp, "\nvec_y_d, p->vec_y_d_begin=%d, dimension=%d:\n", p->vec_y_d_begin, p->vec_y_d_len);
	for(i=p->vec_y_d_begin; i<=p->vec_y_d_end; i++){
#ifdef PRINT_CONCISE
		if(i==p->vec_y_d_begin||i==p->vec_y_d_end)
#endif
//		if(0.0!= p->vec_y_d[i-p->vec_y_d_begin])
			fprintf(fp, "[%d/%d]=%e\n", i-p->vec_y_d_begin, i, p->vec_y_d[i-p->vec_y_d_begin]);
	}

	fprintf(fp, "\nvec_x, p->vec_x_begin_gloIdx=%d dimension=%d:\n",p->vec_x_begin_gloIdx, p->vec_x_len);
	for(i=p->vec_x_begin_gloIdx; i<=p->vec_x_end_gloIdx; i++){
#ifdef PRINT_CONCISE
		if(i==p->vec_x_begin_gloIdx || i==p->vec_x_end_gloIdx)
#endif
//		if( p->vec_x[i-p->vec_x_begin_gloIdx]!=0.0f)
			fprintf(fp, "[%d]=%e\n", i, p->vec_x[i-p->vec_x_begin_gloIdx]);
	}




//	communication set
	fprintf(fp, "\nvec_x_recv_list_size %d\n", p->vec_x_recv_list_size );
	for(i=0; i<p->vec_x_recv_list_size; i++){
		fprintf(fp, "rank=%d, startIdx=%d, length=%d\n",
				p->vec_x_recv_list_rank[i], p->vec_x_recv_list_startIdx[i], p->vec_x_recv_list_len[i]);
	}
	fprintf(fp, "\nvec_y_d_recv_list_size %d \n", p->vec_y_d_recv_list_size );
	for(i=0; i<p->vec_y_d_recv_list_size; i++){
			fprintf(fp, "rank=%d, startIdx=%d, length=%d\n",
					p->vec_y_d_recv_list_rank[i], p->vec_y_d_recv_list_startIdx[i], p->vec_y_d_recv_list_len[i]);
	}

	fprintf(fp, "\nvec_x_send_list_size %d\n", p->vec_x_send_list_size);
	for(i=0; i<p->vec_x_send_list_size; i++){
		fprintf(fp, "rank=%d\t", p->vec_x_send_list_rank[i]);
	}
	fprintf(fp, "\nvec_y_d_send_list_size %d\n", p->vec_y_d_send_list_size);
	for(i=0; i<p->vec_y_d_send_list_size; i++){
		fprintf(fp, "rank=%d\t", p->vec_y_d_send_list_rank[i]);
	}

/*check sparse vector*/
#ifdef USE_COMPRESSED_VEC
	/*local vec x*/
//	fprintf(fp, "\nsp_vec_x npiece=%d, nnz=%d\n", p->sp_vec_x.npiece, p->sp_vec_x.nnz);

	/*local vec y*/
//	fprintf(fp, "\nsp_vec_y_d npiece=%d, nnz=%d\n", p->sp_vec_y_d.npiece, p->sp_vec_y_d.nnz);
//	for(j=0; j<p->sp_vec_y_d.npiece; j++){
//		fprintf(fp, "ptr[%d]=%d len[%d]=%d\n",j, p->sp_vec_y_d.ptr[j], j, p->sp_vec_y_d.len[j]);
//	}

	/*array to receive sparse vector from neighbors,
	 * the length of array is vec_x_recv_list_size and vec_y_d_recv_list_size
	 * allocated by FillSparseVec{X,Y}InPPDataV3_Tri*/
	fprintf(fp, "\nsp_vec_x_recv_list \n");
	for(i=0; i<p->vec_x_recv_list_size; i++){
		fprintf(fp, "sp_vec_x_recv_list[%d]:  npiece=%d, nnz=%d \n",
				p->vec_x_recv_list_rank[i], p->sp_vec_x_recv_list[i].npiece, p->sp_vec_x_recv_list[i].nnz);
//		for(j=0; j<p->sp_vec_x_recv_list[i].npiece; j++){
//			fprintf(fp, "ptr[%d]=%d len[%d]=%d\n",j, p->sp_vec_x_recv_list[i].ptr[j], j, p->sp_vec_x_recv_list[i].len[j]);
//		}
	}

	fprintf(fp, "\nsp_vec_y_d_recv_list \n");
	for(i=0; i<p->vec_y_d_recv_list_size; i++){
		fprintf(fp, "sp_vec_y_d_recv_list[%d]:  npiece=%d, nnz=%d \n",
				p->vec_y_d_recv_list_rank[i], p->sp_vec_y_d_recv_list[i].npiece, p->sp_vec_y_d_recv_list[i].nnz);
	}

	fprintf(fp, "\nsp_vec_x_send_list \n");
	for(i=0; i<p->vec_x_send_list_size; i++){
		fprintf(fp, "sp_vec_x_send_list[%d]:  npiece=%d, nnz=%d \n",
				p->vec_x_send_list_rank[i], p->sp_vec_x_send_list[i].npiece, p->sp_vec_x_send_list[i].nnz);
	}

	fprintf(fp, "\nsp_vec_y_d_send_list \n");
	for(i=0; i<p->vec_y_d_send_list_size; i++){
		fprintf(fp, "sp_vec_y_d_send_list[%d]: npiece=%d, nnz=%d \n",
				p->vec_y_d_send_list_rank[i], p->sp_vec_y_d_send_list[i].npiece, p->sp_vec_y_d_send_list[i].nnz);
		if(p->sp_vec_y_d_send_list[i].npiece>0)
			fprintf(fp, "[%d, %d]\n", p->sp_vec_y_d_send_list[i].ptr[0], p->sp_vec_y_d_send_list[i].len[0]);

	}

#endif // USE_COMPRESSED_VEC
	fclose(fp);
}





#endif



//#endif

