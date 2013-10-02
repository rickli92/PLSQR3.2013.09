/*
 * sparseVec.h
 *
 *  Created on: Aug 4, 2012
 *      Author: hhuang1
 */

#ifndef SPARSEVEC_H_
#define SPARSEVEC_H_

#include "main.h"



void InitSparseVecFromOriginalIntVec(const int * ori_vec, const int len, CompressedVec * sparse_vec);

void PrintSparseVec(CompressedVec * s);

void FinalizeSparseVec(CompressedVec * sparse_vec);

void UncompressedSparseVec(MY_FLOAT_TYPE * vec, const int len, const CompressedVec * sv);

void CopyRegularVecToSparseVec(MY_FLOAT_TYPE * vec, CompressedVec * sp);

#endif /* SPARSEVEC_H_ */
