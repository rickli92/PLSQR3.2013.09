#ifndef LSQR_H_
#define LSQR_H_

#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>     /* C99 feature         */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
// includes, lsqr
#include "cblas.h"
#include "main.h"
#include "lsqrchk.h"
#include "aprod.h"
#include "FileIO.h"
#include "./tool/memory_parallel.h"
//#include "./tool/gptl.h"
#include "./clsqr2/lsqr_s.h"

#define ZERO 0.0
#define ONE  1.0


int call_lsqr(VERSION version, int procRank, int numProc, int rows, int cols,
		void *userWork, MY_FLOAT_TYPE* perProcVectorB, MY_FLOAT_TYPE* v, MY_FLOAT_TYPE* w, MY_FLOAT_TYPE* x,
		int lsqr_iteration, int colPerm_all[]);

void lsqr(
		  VERSION version,
		  int procRank,
		  int numProc,
		  int m,//row
		  int n,//column
          void (*aprod)(int version, int procRank, int numProc,
        		        int mode, int m, int n, MY_FLOAT_TYPE x[], MY_FLOAT_TYPE y[],
                        void *UsrWrk, int isFinalIteration, int iteration	),
		  MY_FLOAT_TYPE damp,
		  void   *UsrWrk,
		  MY_FLOAT_TYPE u[],     // len = m, vectorB
		  MY_FLOAT_TYPE v[],     // len = n
		  MY_FLOAT_TYPE w[],     // len = n
		  MY_FLOAT_TYPE x[],     // len = n
		  MY_FLOAT_TYPE se[],    // len at least n.  May be NULL.
		  MY_FLOAT_TYPE atol,
		  MY_FLOAT_TYPE btol,
		  MY_FLOAT_TYPE conlim,
		  int    itnlim,
		  FILE   *nout,
		  // The remaining variables are output only.
		  int    *istop_out,
		  int    *itn_out,
		  MY_FLOAT_TYPE *anorm_out,
		  MY_FLOAT_TYPE *acond_out,
		  MY_FLOAT_TYPE *rnorm_out,
		  MY_FLOAT_TYPE *arnorm_out,
		  MY_FLOAT_TYPE *xnorm_out,
		  int lsqr_iteration
		  );


#endif  /* LSQR_H_ */


