/* cblas.h
   $Revision: 273 $ $Date: 2006-09-04 15:59:04 -0700 (Mon, 04 Sep 2006) $

   ----------------------------------------------------------------------
   This file is part of BCLS (Bound-Constrained Least Squares).

   Copyright (C) 2006 Michael P. Friedlander, Department of Computer
   Science, University of British Columbia, Canada. All rights
   reserved. E-mail: <mpf@cs.ubc.ca>.
   
   BCLS is free software; you can redistribute it and/or modify it
   under the terms of the GNU Lesser General Public License as
   published by the Free Software Foundation; either version 2.1 of the
   License, or (at your option) any later version.
   
   BCLS is distributed in the hope that it will be useful, but WITHOUT
   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General
   Public License for more details.
   
   You should have received a copy of the GNU Lesser General Public
   License along with BCLS; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301
   USA
   ----------------------------------------------------------------------
*/
/*!
   \file
   CBLAS library header file.
*/

#ifndef _CBLAS_H
#define _CBLAS_H


#include <math.h>
#include <stddef.h>
#include "main.h"


#define OFFSET(N, incX) ((incX) > 0 ?  0 : ((N) - 1) * (-(incX)))

enum CBLAS_ORDER    {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};


// ---------------------------------------------------------------------
// d2norm  returns  sqrt( a**2 + b**2 )  with precautions
// to avoid overflow.
//
// 21 Mar 1990: First version.
// ---------------------------------------------------------------------
static MY_FLOAT_TYPE d2norm(const MY_FLOAT_TYPE a, const MY_FLOAT_TYPE b) {
	MY_FLOAT_TYPE scale;
	const MY_FLOAT_TYPE zero = 0.0;

	scale = fabs(a) + fabs(b);
	if (scale == zero)
		return zero;
	else
		return scale
				* sqrt((a / scale) * (a / scale) + (b / scale) * (b / scale));
}

static void dload(const int n, const MY_FLOAT_TYPE alpha, MY_FLOAT_TYPE x[]) {
	int i;
	for (i = 0; i < n; i++)
		x[i] = alpha;
	return;
}

void
cblas_daxpy(const int N, const MY_FLOAT_TYPE alpha, const MY_FLOAT_TYPE *X,
            const int incX, MY_FLOAT_TYPE *Y, const int incY);

void
cblas_dcopy(const int N, const MY_FLOAT_TYPE *X, const int incX, 
            MY_FLOAT_TYPE *Y, const int incY);


MY_FLOAT_TYPE
cblas_ddot(const int N, const MY_FLOAT_TYPE *X, const int incX,
           const MY_FLOAT_TYPE *Y, const int incY);

//MY_FLOAT_TYPE cblas_dnrm2_parallel_plsqr3_forVecY(PerProcDataV3 * p, const int incY );
MY_FLOAT_TYPE cblas_dnrm2_parallel_plsqr3_forVecY(/*PerProcDataV3*/void * pp,
		const int incY, int itn) ;

/*MY_FLOAT_TYPE cblas_dnrm2_parallel_plsqr_forVecX(const int procRank, const int numProc, const PerProcDataV3 * p,
		const MY_FLOAT_TYPE *vec , const int inc);*/

MY_FLOAT_TYPE cblas_dnrm2_parallel(const int X_dim, const MY_FLOAT_TYPE *X, const int incX, const int itn);

MY_FLOAT_TYPE
cblas_dnrm2(const int N, const MY_FLOAT_TYPE *X, const int incX);

MY_FLOAT_TYPE my_nrm2(const int N, const MY_FLOAT_TYPE *vec);

//void cblas_dscal_plsqr3(PerProcDataV3 * p, const MY_FLOAT_TYPE alpha, const int incVec);
void cblas_dscal_plsqr3(/*PerProcDataV3 */void * pp, const MY_FLOAT_TYPE alpha, const int incVec, int itn) ;

void
cblas_dscal(const int N, const MY_FLOAT_TYPE alpha, MY_FLOAT_TYPE *X, const int incX);

void
cblas_dgemv(const enum CBLAS_ORDER order,
            const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
            const MY_FLOAT_TYPE alpha, const MY_FLOAT_TYPE  *A, const int lda,
            const MY_FLOAT_TYPE  *X, const int incX, const MY_FLOAT_TYPE beta,
            MY_FLOAT_TYPE  *Y, const int incY);

#endif /* _CBLAS_H */
