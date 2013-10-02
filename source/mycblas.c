/*
 * cblas.c
 *
 *  Created on: Aug 24, 2012
 *      Author: hh
 */

#include "cblas.h"
#include "main.h"

//2/4/2013: add simple norm; need to modify GPU's norm accordingly in the future
//#define USE_SIMPLE_NORM



void cblas_daxpy(const int N, const MY_FLOAT_TYPE alpha, const MY_FLOAT_TYPE *X,
		const int incX, MY_FLOAT_TYPE *Y, const int incY) {
//#ifdef BARRIER
//        MPI_Barrier(MPI_COMM_WORLD);
//#endif
//JMD	GPTLstart("daxpy");

	int i;

	if (N <= 0)
		return;
	if (alpha == 0.0)
		return;

	if (incX == 1 && incY == 1) {
		const int m = N % 4;

		for (i = 0; i < m; i++)
			Y[i] += alpha * X[i];

		for (i = m; i + 3 < N; i += 4) {
			Y[i] += alpha * X[i];
			Y[i + 1] += alpha * X[i + 1];
			Y[i + 2] += alpha * X[i + 2];
			Y[i + 3] += alpha * X[i + 3];
		}
	} else {
		int ix = OFFSET(N, incX);
		int iy = OFFSET(N, incY);

		for (i = 0; i < N; i++) {
			Y[iy] += alpha * X[ix];
			ix += incX;
			iy += incY;
		}
	}
//JMD	GPTLstop("daxpy");
}

void cblas_dcopy(const int N, const MY_FLOAT_TYPE *X, const int incX,
		MY_FLOAT_TYPE *Y, const int incY) {
//#ifdef BARRIER
//        MPI_Barrier(MPI_COMM_WORLD);
//#endif
//	GPTLstart("dcopy");
	int i;
	int ix = OFFSET(N, incX);
	int iy = OFFSET(N, incY);

	for (i = 0; i < N; i++) {
		Y[iy] = X[ix];
		ix += incX;
		iy += incY;
	}
//	GPTLstop("dcopy");
}

MY_FLOAT_TYPE cblas_ddot(const int N, const MY_FLOAT_TYPE *X, const int incX,
		const MY_FLOAT_TYPE *Y, const int incY) {
//#ifdef BARRIER
//        MPI_Barrier(MPI_COMM_WORLD);
//#endif
//JMD	GPTLstart("ddot");

	MY_FLOAT_TYPE r = 0.0;
	int i;
	int ix = OFFSET(N, incX);
	int iy = OFFSET(N, incY);

	for (i = 0; i < N; i++) {
		r += X[ix] * Y[iy];
		ix += incX;
		iy += incY;
	}
//JMD 	GPTLstop("ddot");

	return r;
}

/* 11/2010
 * dnorm2 for PLSQR3 only
 * 8/26/2012 add gpu support
 * */
MY_FLOAT_TYPE cblas_dnrm2_parallel_plsqr3_forVecY(/*PerProcDataV3*/void * pp,
		const int incY, int itn) {
#ifdef PLSQR3
#ifdef ONEDIAGONAL
	PerProcDataV3 * p = (PerProcDataV3 * ) pp;
	int len = p->vec_y_end_gloIdx_d - p->vec_y_begin_gloIdx_d + 1;
#endif
#ifdef TRIDIAGONAL
	PPDataV3_TRI * p = (PPDataV3_TRI *) pp;
	int len = p->vec_y_d_end - p->vec_y_d_begin + 1;
#endif

//JMD	GPTLstart("dnrm2_v3");
	//kernel part
	MY_FLOAT_TYPE part_norm_k,part_norm_d ;
	if(itn==0){
		part_norm_k = cblas_dnrm2(p->matCSC_k->numRow, p->vec_y_k, incY);
		part_norm_d = cblas_dnrm2_parallel(len, p->vec_y_d, incY, itn);
	}
	else {
	#if GPUALL==1
		cublasSnrm2_w(p->gpu->matCSC_k_d->numRow, p->gpu->vec_y_k_d, incY, &part_norm_k);
		part_norm_d = cblas_dnrm2_parallel(len, p->gpu->vec_y_d_d, incY, itn);
	#else
		part_norm_k = cblas_dnrm2(p->matCSC_k->numRow, p->vec_y_k, incY);
		part_norm_d = cblas_dnrm2_parallel(len, p->vec_y_d, incY, itn);
	#endif
	}

	MY_FLOAT_TYPE norm = sqrt(part_norm_k * part_norm_k + part_norm_d * part_norm_d);
//JMD	GPTLstop("dnrm2_v3");

	return norm;
#endif
}

/*

 MY_FLOAT_TYPE cblas_dnrm2_parallel_plsqr_forVecX(const int procRank, const int numProc, const PerProcDataV3 * p,
 const MY_FLOAT_TYPE *vec , const int inc)
 {

 return cblas_dnrm2_parallel(vec_len, vec, inc);
 }
 */

/* 3/19/2012 This way of parallel computing nrm2 is inaccurate.
 * The error difference is noticeable when using large dataset,
 * but the error difference is not NOT noticeale if using small dataset
 *
 * To find accurate parallel version, see Parallel Basic Linear Algebra Subprograms (PBLAS)
 * http://www.netlib.org/scalapack/pblas_qref.html
 * http://www.netlib.org/scalapack/pblas_qref.html#PvNRM2
 *
 * 11/2010 combine MPI_reduce and MPI_Bcast to MPI_Allreduce
 * 04/2010
 * parallel version of cblas_dnrm2
 * every process returns the same norm
 * */
 MY_FLOAT_TYPE cblas_dnrm2_parallel(const int vec_dim,
		const MY_FLOAT_TYPE *vec, const int inc, const int itn) {
//#ifdef BARRIER
//        MPI_Barrier(MPI_COMM_WORLD);
//#endif
//JMD	GPTLstart("dnrm2_v2");
	MY_FLOAT_TYPE part_norm = 0.0, square = 0.0, sum = 0.0, whole_norm = 0.0;
	if (itn == 0)
		part_norm = cblas_dnrm2(vec_dim, vec, inc);
	else {
#if GPUALL==1 //5/25/2011
		cublasSnrm2_w(vec_dim, vec, inc, &part_norm);
#else
		part_norm = cblas_dnrm2(vec_dim, vec, inc);
#endif
	}
	square = part_norm * part_norm;
//	fprintf(stderr, "%s(%d)-%s: part_norm=%f square=%f\n", __FILE__,__LINE__,__FUNCTION__, part_norm, square);
	MPI_Allreduce(&square, &sum, 1, MY_MPI_FLOAT_TYPE, MPI_SUM, MPI_COMM_WORLD);
	whole_norm = sqrt(sum);
//JMD	GPTLstop("dnrm2_v2");
	return whole_norm;
}

MY_FLOAT_TYPE cblas_dnrm2(const int N, const MY_FLOAT_TYPE *vec, const int inc) {
//#ifdef BARRIER
//        MPI_Barrier(MPI_COMM_WORLD);
//#endif
//JMD	GPTLstart("cblas_dnrm2");

#ifdef USE_SIMPLE_NORM
	my_nrm2(N, vec);
#else



	MY_FLOAT_TYPE scale = 0.0, ssq = 1.0;
	int i, ix = 0;

	if (N <= 0 || inc <= 0)
		return 0;
	else if (N == 1)
		return fabs(vec[0]);

	for (i = 0; i < N; i++) {
		const MY_FLOAT_TYPE x = vec[ix];

		if (x != 0.0) {
			const MY_FLOAT_TYPE ax = fabs(x);

			if (scale < ax) {
				ssq = 1.0 + ssq * (scale / ax) * (scale / ax);
				scale = ax;
			} else {
				ssq += (ax / scale) * (ax / scale);
			}
		}

		ix += inc;
	}
//JMD	GPTLstop("cblas_dnrm2");
	return scale * sqrt(ssq);

#endif
}
/*use my nrm2 version to substitute cblas_dnrm2*/
MY_FLOAT_TYPE my_nrm2(const int N, const MY_FLOAT_TYPE *vec){
	if (N <= 0)
		return 0;
	else if (N == 1)
		return fabs(vec[0]);
	int i=0;
	MY_FLOAT_TYPE norm=0.0;
	for (i=0; i<N; i++){
		norm+=vec[i]*vec[i];
	}
	return sqrt(norm);
}


/*
 * 8/26/2012 add gpu support*/
void cblas_dscal_plsqr3(/*PerProcDataV3 */void * pp, const MY_FLOAT_TYPE alpha, const int incVec, int itn) {
#ifdef PLSQR3
#ifdef ONEDIAGONAL
	PerProcDataV3 * p = (PerProcDataV3 *) pp;
	int len = p->vec_y_end_gloIdx_d - p->vec_y_begin_gloIdx_d + 1;
#endif
#ifdef TRIDIAGONAL
	PPDataV3_TRI * p = (PPDataV3_TRI *) pp;
	int len = p->vec_y_d_end - p->vec_y_d_begin + 1;
#endif

//JMD	GPTLstart("dscal_v3");
	if(itn==0){
		//kernel
		cblas_dscal(p->matCSC_k->numRow, alpha, p->vec_y_k, incVec);
		//damping
		cblas_dscal(len, alpha, p->vec_y_d, incVec);
	}
	else {
#if GPUALL==1
		cublasSscal_w(p->gpu->matCSC_k_d->numRow, &alpha, p->gpu->vec_y_k_d, incVec);
		cublasSscal_w(len, &alpha, p->gpu->vec_y_d_d, incVec);
#else
		//kernel
		cblas_dscal(p->matCSC_k->numRow, alpha, p->vec_y_k, incVec);
		//damping
		cblas_dscal(len, alpha, p->vec_y_d, incVec);
#endif
	}

//JMD	GPTLstop("dscal_v3");

#endif
}



extern int group_size_omp;
/**
 * @date remove incVec in order to use OpenMP
 * @param N vec's length
 * @param alpha
 * @param vec
 * @param incVec not useful
 *
 */
void cblas_dscal(const int N, const MY_FLOAT_TYPE alpha,
		MY_FLOAT_TYPE *vec/*array*/, const int incVec) {
//#ifdef BARRIER
//        MPI_Barrier(MPI_COMM_WORLD);
//#endif
//JMD	GPTLstart("cblas_dscal");
	int i, ix;

	if (incVec <= 0)
		return;

	ix = OFFSET(N, incVec);

/*	for (i = 0; i < N; i++) {
		vec[ix] *= alpha;
		ix += incVec;
	}*/
	int chunk=( N + group_size_omp-1) / (group_size_omp);
#pragma omp parallel for private(i) schedule(dynamic, chunk)
	for (i = 0; i < N; i++) {
		vec[i] *= alpha;
	}
//JMD	GPTLstop("cblas_dscal");
}


///* bccblas.c
//   $Revision: 231 $ $Date: 2006-04-15 18:47:05 -0700 (Sat, 15 Apr 2006) $
//
//   ----------------------------------------------------------------------
//   This file is part of BCLS (Bound-Constrained Least Squares).
//
//   Copyright (C) 2006 Michael P. Friedlander, Department of Computer
//   Science, University of British Columbia, Canada. All rights
//   reserved. E-mail: <mpf@cs.ubc.ca>.
//
//   BCLS is free software; you can redistribute it and/or modify it
//   under the terms of the GNU Lesser General Public License as
//   published by the Free Software Foundation; either version 2.1 of the
//   License, or (at your option) any later version.
//
//   BCLS is distributed in the hope that it will be useful, but WITHOUT
//   ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
//   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General
//   Public License for more details.
//
//   You should have received a copy of the GNU Lesser General Public
//   License along with BCLS; if not, write to the Free Software
//   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301
//   USA
//   ----------------------------------------------------------------------
//*/
///*!
//   \file
//
//   This file contains C-wrappers to the BLAS (Basic Linear Algebra
//   Subprograms) routines that are used by BCLS.  Whenever possible,
//   they should be replaced by corresponding BLAS routines that have
//   been optimized to the machine being used.
//
//   Included BLAS routines:
//
//   - cblas_daxpy
//   - cblas_dcopy
//   - cblas_ddot
//   - cblas_dnrm2
//   - cblas_dscal
//*/
//
//#include "cblas.h"
//
///*!
//  \param[in]     N
//  \param[in]     alpha
//  \param[in]     X
//  \param[in]     incX
//  \param[in,out] Y
//  \param[in]     incY
//*/
//void
//cblas_daxpy( const int N, const MY_FLOAT_TYPE alpha, const MY_FLOAT_TYPE *X,
//             const int incX, MY_FLOAT_TYPE *Y, const int incY)
//{
//  int i;
//
//  if (N     <= 0  ) return;
//  if (alpha == 0.0) return;
//
//  if (incX == 1 && incY == 1) {
//      const int m = N % 4;
//
//      for (i = 0; i < m; i++)
//          Y[i] += alpha * X[i];
//
//      for (i = m; i + 3 < N; i += 4) {
//          Y[i    ] += alpha * X[i    ];
//          Y[i + 1] += alpha * X[i + 1];
//          Y[i + 2] += alpha * X[i + 2];
//          Y[i + 3] += alpha * X[i + 3];
//      }
//  } else {
//      int ix = OFFSET(N, incX);
//      int iy = OFFSET(N, incY);
//
//      for (i = 0; i < N; i++) {
//          Y[iy] += alpha * X[ix];
//          ix    += incX;
//          iy    += incY;
//      }
//  }
//}
//
///*!
//  \param[in]     N
//  \param[in]     X
//  \param[in]     incX
//  \param[out]    Y
//  \param[in]     incY
//*/
//void
//cblas_dcopy( const int N, const MY_FLOAT_TYPE *X,
//             const int incX, MY_FLOAT_TYPE *Y, const int incY)
//{
//  int i;
//  int ix = OFFSET(N, incX);
//  int iy = OFFSET(N, incY);
//
//  for (i = 0; i < N; i++) {
//      Y[iy]  = X[ix];
//      ix    += incX;
//      iy    += incY;
//  }
//}
//
///*!
//  \param[in]     N
//  \param[in]     X
//  \param[in]     incX
//  \param[in]     Y
//  \param[in]     incY
//
//  \return  Dot product of X and Y.
//
//*/
//MY_FLOAT_TYPE
//cblas_ddot( const int N, const MY_FLOAT_TYPE *X,
//            const int incX, const MY_FLOAT_TYPE *Y, const int incY)
//{
//  MY_FLOAT_TYPE r  = 0.0;
//  int    i;
//  int    ix = OFFSET(N, incX);
//  int    iy = OFFSET(N, incY);
//
//  for (i = 0; i < N; i++) {
//      r  += X[ix] * Y[iy];
//      ix += incX;
//      iy += incY;
//  }
//
//  return r;
//}
//
///*!
//  \param[in]     N
//  \param[in]     X
//  \param[in]     incX
//
//  \return Two-norm of X.
//*/
//MY_FLOAT_TYPE
//cblas_dnrm2( const int N, const MY_FLOAT_TYPE *X, const int incX)
//{
//  MY_FLOAT_TYPE
//      scale = 0.0,
//      ssq   = 1.0;
//  int
//      i,
//      ix    = 0;
//
//  if (N <= 0 || incX <= 0) return 0;
//  else if (N == 1)         return fabs(X[0]);
//
//  for (i = 0; i < N; i++) {
//      const MY_FLOAT_TYPE x = X[ix];
//
//      if (x != 0.0) {
//          const MY_FLOAT_TYPE ax = fabs(x);
//
//          if (scale < ax) {
//              ssq   = 1.0 + ssq * (scale / ax) * (scale / ax);
//              scale = ax;
//          } else {
//              ssq += (ax / scale) * (ax / scale);
//          }
//      }
//
//      ix += incX;
//  }
//
//  return scale * sqrt(ssq);
//}
//
///*!
//  \param[in]     N
//  \param[in]     alpha
//  \param[in,out] X
//  \param[in]     incX
//*/
//void
//cblas_dscal(const int N, const MY_FLOAT_TYPE alpha, MY_FLOAT_TYPE *X, const int incX)
//{
//    int i, ix;
//
//    if (incX <= 0) return;
//
//    ix = OFFSET(N, incX);
//
//    for (i = 0; i < N; i++) {
//        X[ix] *= alpha;
//        ix    += incX;
//    }
//}
