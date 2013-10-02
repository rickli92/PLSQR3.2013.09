#ifndef COMPUTESPMV_H_
#define COMPUTESPMV_H_

#include "main.h"

void cpuSpMV_CSR(const SpMatCSR * A, const MY_FLOAT_TYPE x[], MY_FLOAT_TYPE Ax[]);

void cpuSpMV_CSR_v2(const SpMatCSR * A, const MY_FLOAT_TYPE x[], const int startIdxOfX, MY_FLOAT_TYPE Ax[]);

void cpuSpMVTranspose_CSR(const SpMatCSR *A, const MY_FLOAT_TYPE y[], MY_FLOAT_TYPE tAy[]);

void cpuSpMV_CSC(const SpMatCSC *A, const MY_FLOAT_TYPE x[], MY_FLOAT_TYPE Ax[]);

void cpuSpMVTranspose_CSC(const SpMatCSC *A, const MY_FLOAT_TYPE y[], MY_FLOAT_TYPE tAy[]);

void cpuSpMVTranspose_CSC_v2(const SpMatCSC *A, const MY_FLOAT_TYPE y[], const int startIdxOfY, MY_FLOAT_TYPE tAy[] );
#endif /*COMPUTERSPMV_H_*/
