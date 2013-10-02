#ifndef LSQRCHK_H_
#define LSQRCHK_H_

#include "math.h"
#include "cblas.h"

void acheck(int m, int n, int nout, void (*aprod)(int mode, int m, int n, MY_FLOAT_TYPE x[], MY_FLOAT_TYPE y[], 
			int leniw, int lenrw, int iw[], int rw[]),MY_FLOAT_TYPE eps, int leniw, int lenrw, int iw[], 
			MY_FLOAT_TYPE rw[], MY_FLOAT_TYPE v[], MY_FLOAT_TYPE w[], MY_FLOAT_TYPE x[], MY_FLOAT_TYPE y[], MY_FLOAT_TYPE inform);

void  xcheck(int m, int n, int nout, void (*aprod)(int mode, int m, int n, MY_FLOAT_TYPE x[], MY_FLOAT_TYPE y[], 
			 int leniw, int lenrw, int iw[], int rw[]),MY_FLOAT_TYPE anorm, MY_FLOAT_TYPE damp, MY_FLOAT_TYPE eps,int leniw, 
			 int lenrw, int iw[], MY_FLOAT_TYPE rw[], MY_FLOAT_TYPE b[], MY_FLOAT_TYPE u[],MY_FLOAT_TYPE v[], MY_FLOAT_TYPE w[], MY_FLOAT_TYPE x[], 
			 MY_FLOAT_TYPE inform, MY_FLOAT_TYPE test1,MY_FLOAT_TYPE test2, MY_FLOAT_TYPE test3);


#endif  // LSQRCHK_H_






