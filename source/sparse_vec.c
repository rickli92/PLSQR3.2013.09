/*
 * sparseVec.c
 *
 *  Created on: Aug 4, 2012
 *      Author: hhuang1
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
//#include "main.h"
#include "sparse_vec.h"

/* Initialize a sparse vector from an original vector
 * @param ori_vec original vector array
 * @param len length of original vector
 * @param sparse_vec compressed sparse vector. Note that sparse_vec.val is initialized with 0
 * */
void InitSparseVecFromOriginalIntVec(const int * ori_vec, const int len, CompressedVec * sparse_vec)
{
	int i;
	sparse_vec->npiece=0;
	sparse_vec->nnz=0;
	//setup sparse_vec->npiece, sparse_vec->nnz
	int pre=0;
	for(i=0; i< len; i++){
		if(ori_vec[i]!=0){
			sparse_vec->nnz++;
			if(pre==0) {//starting of one piece
				sparse_vec->npiece++;
			}
		}
		pre=ori_vec[i];
	}

	//allocate array
	sparse_vec->ptr = (int *)malloc(sparse_vec->npiece * sizeof(int));
	if(sparse_vec->ptr ==NULL){
		fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__); exit(-1);
	}
	memset(sparse_vec->ptr, 0, sparse_vec->npiece * sizeof(int));

	sparse_vec->len = (int *)malloc(sparse_vec->npiece * sizeof(int));
	if(sparse_vec->len ==NULL){
		fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__); exit(-1);
	}
	memset(sparse_vec->len , 0, sparse_vec->npiece * sizeof(int));

	sparse_vec->val = (MY_FLOAT_TYPE *) malloc(sparse_vec->nnz * sizeof(MY_FLOAT_TYPE));
	if(sparse_vec->val==NULL){
		fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__); exit(-1);
	}
	memset(sparse_vec->val, 0, sparse_vec->nnz * sizeof(MY_FLOAT_TYPE));

	//setup ptr, len, val array
	int piece_counter=-1;
	int nnz_counter=-1;
	pre=0;
	for(i=0; i<len; i++){
		if(ori_vec[i]!=0){
			sparse_vec->val[++nnz_counter]=0.0;
			if(pre==0){
				sparse_vec->ptr[++piece_counter]= i;
			}
			sparse_vec->len[piece_counter]++;
		}
		pre=ori_vec[i];
	}

	//	check
	if((nnz_counter+1) == sparse_vec->nnz){
		//OK
	}else{
		fprintf(stderr, "%s(%d)-%s: nnz_counter=%d, sp->nnz=%d\n", __FILE__, __LINE__, __FUNCTION__, nnz_counter, sparse_vec->nnz); exit(-1);
	}
}

/*Initlize sparse vector arrays via communication*/
/*void InitSparseVecArrayViaComm(CompressedVec * sp, ){

}*/

void FinalizeSparseVec(CompressedVec * sparse_vec)
{
	free(sparse_vec->ptr);
	free(sparse_vec->len);
	free(sparse_vec->val);
}

/*Print function*/
void PrintSparseVec(CompressedVec * s){
	printf("%s(%d):%s Print SparseVec\n", __FILE__, __LINE__, __FUNCTION__);
	printf("npiece=%d, nnz=%d\n", s->npiece, s->nnz);
	int i;
	for(i=0; i<s->npiece; i++){
		printf("ptr[%d]=%d, len[%d]=%d \n", i, s->ptr[i], i, s->len[i]);
	}
	for(i=0; i<s->nnz; i++){
		printf("val[%d]=%lf \n", i, s->val[i]);
	}

}

/*Uncompress vector to ordinary vector with 'len' length*/
void UncompressedSparseVec(MY_FLOAT_TYPE * vec, const int len, const CompressedVec * sv){
	memset(vec, 0, len*sizeof(MY_FLOAT_TYPE));
	if(sv->ptr[sv->npiece-1] + sv->len[sv->npiece-1] > len){
		fprintf(stderr, "%s(%d)-%s: Ordinary vector length does not match sparse vector length \n", __FILE__, __LINE__, __FUNCTION__);exit(-1);
	}
	int i, j, nnz_counter=-1;
	for(i=0; i<sv->npiece; i++){
		for(j=0; j< sv->len[i]; j++){
			int base = sv->ptr[i]+j;
			vec[base]=sv->val[++nnz_counter];
		}
	}
}

/* Compress the vector
 * Copy regular vector to sparse vector. Vec must conform sp, e.g. size, nnz
 * @param vec INPUT
 * @param sp OUTPUT, must be initialized prior to call this function, this function only fill CompressedVec.val
 * */
void CopyRegularVecToSparseVec(MY_FLOAT_TYPE * vec, CompressedVec * sp)
{
	int i, j, nnz_counter=-1, npiece=sp->npiece;
	for(i=0; i<npiece; i++){
		int len=sp->len[i];
		int ptr=sp->ptr[i];
		for(j=0; j<len; j++){
//			int base_idx = ptr + j;
			sp->val[++nnz_counter]=vec[ptr + j];
		}
	}
//	check
	if((nnz_counter+1) == sp->nnz){
		//OK
	}else{
		fprintf(stderr, "%s(%d)-%s: nnz_counter=%d, sp->npiece=%d, sp->nnz=%d\n", __FILE__, __LINE__, __FUNCTION__, nnz_counter, sp->npiece, sp->nnz); /*exit(-1);*/
		for(i=0; i<npiece; i++){
			printf("(%d):[%d, %d]\n", i, sp->ptr[i], sp->len[i]);
		}
	}
}

/*test function*/
//int main ( int argc, char * argv [])
//{
//	int i,len = 10;
//	MY_FLOAT_TYPE  ori_vec[]= { 0.0,-1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 0.0, 7.0, 0.0};
//	printf("Original vector \n");
//	for(i=0; i<len; i++){
//		printf("ori_vec[%d]=%f\n", i, ori_vec[i]);
//	}
//
//	int  ori_vec_bone[]= {0, -1, 2, 0, 3, 4, 5, 0, 7, 0};
//	CompressedVec sparse_vec;
////	InitSparseVecFromOriginalVec(ori_vec, len, &sparse_vec);
//	InitSparseVecFromOriginalIntVec(ori_vec_bone, len, &sparse_vec);
//	PrintSparseVec(&sparse_vec);
//
//	printf("copy CopyRegularVecToSparseVec\n");
//	CopyRegularVecToSparseVec(ori_vec, &sparse_vec);
//	PrintSparseVec(&sparse_vec);
//
//	printf("after uncompressing \n");
//	MY_FLOAT_TYPE uncompressed_vec[10];
//	UncompressedSparseVec(uncompressed_vec, len, &sparse_vec);
//	for(i=0; i<len; i++){
//		printf("uncompressed_vec[%d]=%f\n", i, uncompressed_vec[i]);
//	}
//
//	FinalizeSparseVec(&sparse_vec);
//	return 0;
//}


