/**
 * @file lsqr.c
 * @brief Major procedure of LSQR algorithm
 */

#include "lsqr.h"
#include "cblas.h"

/** use mpiio to parallel write result to a binary file, avoid MPI reduce, much faster than ascii;
 * disable it if want ascii result file, used for check result*/
#define MPIIOWRITE


#define GEN_RESULT
//#define DEBUG

extern int group_size_omp;
/**
 *
 * @param version
 * @param procRank
 * @param numProc
 * @param numRow
 * @param numCol
 * @param userWork
 * @param perProcVectorB MY_FLOAT_TYPE* perProcVectorB only significant for PLSQR2, not for PLSQR3, since perProcData already contain vector b at initial
 * @param v
 * @param w
 * @param x
 * @param lsqr_iteration
 * @param colPerm_all only significant for PLSQRV3
 * @return
 */
int call_lsqr(VERSION version, int procRank, int numProc, int numRow,
		int numCol, void *userWork, MY_FLOAT_TYPE* perProcVectorB,
		MY_FLOAT_TYPE* v, MY_FLOAT_TYPE* w, MY_FLOAT_TYPE* x,
		int lsqr_iteration, int colPerm_all[])
{
	//initialize some variables for lsqr algorithm
	int istop_out;
	int itn_out;
	MY_FLOAT_TYPE anorm_out;
	MY_FLOAT_TYPE acond_out;
	MY_FLOAT_TYPE rnorm_out;
	MY_FLOAT_TYPE arnorm_out;
	MY_FLOAT_TYPE xnorm_out;
	MY_FLOAT_TYPE atol = 1e-10;
	MY_FLOAT_TYPE btol = 1e-10;
	MY_FLOAT_TYPE conlim = 100;
	int itnlim = 1000;
	MY_FLOAT_TYPE damp = 0.0;

	FILE *fp;


	int i, j;
	//test memory usage
//	if (procRank == 0)
//		printf("check memory usage before LSQR execution\n");
//	check_memory(procRank, numProc);
	if (procRank == 0) {
		//open the output file for lsqr algorithm
		fp = fopen("result.txt", "w");
		if (fp == NULL) {
			printf("result.txt open failed!\n");
			return -1;
		}

	}

	printf("\nStarting plsqr algorithm...procRank=%d\n", procRank);
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("lsqr");
#endif

	lsqr(version, procRank, numProc, numRow, numCol, aprod, damp, userWork,
			perProcVectorB, v, w, x, (MY_FLOAT_TYPE *) NULL, atol, btol, conlim,
			itnlim, fp, &istop_out, &itn_out, &anorm_out, &acond_out,
			&rnorm_out, &arnorm_out, &xnorm_out, lsqr_iteration);

	if (procRank == 0) {
		fclose(fp);
	}
#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstop("lsqr");
#endif

//write result to file
#ifdef MPIIOWRITE
	PPDataV3_TRI * p = (PPDataV3_TRI *) userWork;
	char fname_mpiio[BUFSIZE];
	sprintf(fname_mpiio, "p_resultx_itn=%d_NProc=%d_float%d_mpiio.bin", lsqr_iteration, numProc, 8*sizeof(MY_FLOAT_TYPE));
	mpiio_write_result(procRank, numProc, fname_mpiio, x, p->vec_x_len, p->vec_x_begin_gloIdx, numCol);
#else


	PerProcDataV3 * p = (PerProcDataV3 *) userWork;

	MY_FLOAT_TYPE * xpp = (MY_FLOAT_TYPE *) calloc(numCol,
			sizeof(MY_FLOAT_TYPE)); //solution x per process
	if (NULL == xpp) {
		fprintf(stderr, "xpp calloc fails \n");
		exit(-1);
	}
	for (i = 0; i < numCol; i++) {
		xpp[i] = 0.0;
	}
	for (i = p->vec_x_begin_gloIdx; i <= p->vec_x_end_gloIdx; i++) {
		xpp[i] = x[i - p->vec_x_begin_gloIdx];
	}

	MY_FLOAT_TYPE * complete_vecX_reduce;
	complete_vecX_reduce = (MY_FLOAT_TYPE *) calloc(numCol,
			sizeof(MY_FLOAT_TYPE));
	if (complete_vecX_reduce == NULL) {
		fprintf(stderr, "cannot allocate complete_vecX_reduce\n");
		exit(EXIT_FAILURE);
	}
	memset(complete_vecX_reduce, 0, sizeof(MY_FLOAT_TYPE) * numCol);

	MPI_Reduce(xpp, complete_vecX_reduce, numCol, MY_MPI_FLOAT_TYPE, MPI_SUM, 0,
			MPI_COMM_WORLD);
	free(xpp);

	printf("Finishing plsqr 3 algorithm =========== procRank=%d\n", procRank);

#ifdef GEN_RESULT
	if (procRank == 0) {
		int i;
		//Output solution vector in RCM order

		//reverse reorder vector x(((
		MY_FLOAT_TYPE * vecX_reverse_reorder = (MY_FLOAT_TYPE *) calloc(numCol,
				sizeof(MY_FLOAT_TYPE));
		if (vecX_reverse_reorder == NULL) {
			fprintf(stderr, "can not allocate vecX_reverse_reorder\n");
			exit(EXIT_FAILURE);
		}
		int k;
#ifdef COL_PERM //use column permutation
		for(k=0; k<numCol; k++) {
			vecX_reverse_reorder[colPerm_all[k]] = complete_vecX_reduce[k];
		}
#else
		memcpy(vecX_reverse_reorder, complete_vecX_reduce,
				sizeof(MY_FLOAT_TYPE) * numCol); //if not use column permutation
#endif
		//reverse reorder vector x)))

		//saving x solution to a file
		char filename2[1024];
		sprintf(filename2, "p_resultx_itn=%d_NProc=%d.txt", lsqr_iteration,
				numProc);
		FILE *fp2 = fopen(filename2, "w"); //result including zeros
		char filename3[1024];
		sprintf(filename3, "p_resultx_itn=%d_NProc=%d_nonZero.txt",
				lsqr_iteration, numProc);
		FILE *fp3 = fopen(filename3, "w"); // result without zeros

		for (i = 0; i < numCol; i++) { //write final solution to file
			fprintf(fp2, "x(%d)\t=\t%e\n", i + 1, vecX_reverse_reorder[i]);
			if (vecX_reverse_reorder[i] != 0.0) { //nonzero only
				fprintf(fp3, "x(%d)\t=\t%e\n", i + 1, vecX_reverse_reorder[i]);
			}
		}
		fclose(fp2);
		fclose(fp3);
		printf("\nPlease open result.txt and p_resultx.txt to check the solution!\n\n");
		free(vecX_reverse_reorder);
	}
	free(complete_vecX_reduce);
#endif//GEN_RESULT
#endif //MPIIOWRITE

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	return 0;
}


// ---------------------------------------------------------------------
// LSQR
// ---------------------------------------------------------------------

//6/3/2010
/*
 * comment all fprintf(nout,..., because IBM compiler mpixlc is not compatible
 * */
void lsqr(
		VERSION version,
		int procRank,
		int numProc,
		int numRow, //row
		int numCol, //column
		void(*aprod)(int version, int procRank, int numProc, int mode,
				int numRow, int numCol, MY_FLOAT_TYPE x[], MY_FLOAT_TYPE y[],
				void *UsrWrk, int isFinalIteration, int iteration),
		MY_FLOAT_TYPE damp,
		void *UsrWrk,
		MY_FLOAT_TYPE perProc_u_yb_v2[], // len = m, perPorcVectorB, only significant for PLSQR2, not for PLSQR3
		MY_FLOAT_TYPE v[], // len = n
		MY_FLOAT_TYPE w[], // len = n
		MY_FLOAT_TYPE x[], // len = n
		MY_FLOAT_TYPE se[], // len at least n.  May be NULL.
		MY_FLOAT_TYPE atol, MY_FLOAT_TYPE btol, MY_FLOAT_TYPE conlim,
		int itnlim,
		FILE *nout,
		// The remaining variables are output only.
		int *istop_out, int *itn_out, MY_FLOAT_TYPE *anorm_out,
		MY_FLOAT_TYPE *acond_out, MY_FLOAT_TYPE *rnorm_out,
		MY_FLOAT_TYPE *arnorm_out, MY_FLOAT_TYPE *xnorm_out, int lsqr_iteration)
{
	PPDataV3_TRI * perProcDataV3 = (PPDataV3_TRI *) UsrWrk;

	int istop = 0, itn = 0;
	MY_FLOAT_TYPE anorm = ZERO, acond = ZERO, rnorm = ZERO, arnorm = ZERO,
			xnorm = ZERO;

	//  Local variables

	const bool extra = false; // true for extra printing below.
	const bool damped = damp > ZERO;
	const bool wantse = se != NULL; //wantse is FALSE if se is NULL (in normal case, se is NULL)

	int i, maxdx, nconv, nstop;
	MY_FLOAT_TYPE alfopt, alpha, arnorm0, beta, bnorm, cs, cs1, cs2, ctol,
			delta, dknorm, dnorm, dxk, dxmax, gamma, gambar, phi, phibar, psi,
			res2, rho, rhobar, rhbar1, rhs, rtol, sn, sn1, sn2, t, tau, temp,
			test1, test2, test3, theta, t1, t2, t3, xnorm1, z, zbar;
	char enter[] = "Enter LSQR.  ", exit[] = "Exit  LSQR.  ", msg[6][100] = { {
			"The exact solution is  x = 0" }, {
			"A solution to Ax = b was found, given atol, btol" }, {
			"A least-squares solution was found, given atol" }, {
			"A damped least-squares solution was found, given atol" }, {
			"Cond(Abar) seems to be too large, given conlim" }, {
			"The iteration limit was reached" } };
	//-----------------------------------------------------------------------

	//  Format strings.
	char fmt_1000[] = " %s        Least-squares solution of  Ax = b\n"
			" The matrix  A  has %7d rows  and %7d columns\n"
			" damp   = %-22.2e    wantse = %10i\n"
			" atol   = %-22.2e    conlim = %10.2e\n"
			" btol   = %-22.2e    itnlim = %10d\n\n";
	char fmt_1200[] = "    Itn       x(1)           Function"
			"     Compatible    LS      Norm A   Cond A\n";
	char fmt_1300[] = "    Itn       x(1)           Function"
			"     Compatible    LS      Norm Abar   Cond Abar\n";
	char fmt_1400[] = "     phi    dknorm  dxk  alfa_opt\n";
	char fmt_1500_extra[] =
			" %6d %16.9e %16.9e %9.2e %9.2e %8.1e %8.1e %8.1e %7.1e %7.1e %7.1e\n";
	char fmt_1500[] = " %6d %16.9e %16.9e %9.2e %9.2e %8.1e %8.1e\n";
	char fmt_1550[] = " %6d %16.9e %16.9e %9.2e %9.2e\n";
	char fmt_1600[] = "\n";
	char fmt_2000[] = "\n"
			" %s       istop  = %-10d      itn    = %-10d\n"
			" %s       anorm  = %11.5e     acond  = %11.5e\n"
			" %s       vnorm  = %11.5e     xnorm  = %11.5e\n"
			" %s       rnorm  = %11.5e     arnorm = %11.5e\n";
	char fmt_2100[] = " %s       max dx = %7.1e occured at itn %-9d\n"
			" %s              = %7.1e*xnorm\n";
	char fmt_3000[] = " %s       %s\n";

	//  Initialize.
	if (procRank == 0) {
		if (nout != NULL) {
			fprintf(nout, fmt_1000, enter, numRow, numCol, damp, wantse, atol,
					conlim, btol, itnlim);
		}
	}

	itn = 0;
	istop = 0;
	nstop = 0;
	maxdx = 0;
	ctol = ZERO;
	if (conlim > ZERO)
		ctol = ONE / conlim;
	anorm = ZERO;
	acond = ZERO;
	dnorm = ZERO;
	dxmax = ZERO;
	res2 = ZERO;
	psi = ZERO;
	xnorm = ZERO;
	xnorm1 = ZERO;
	cs2 = -ONE;
	sn2 = ZERO;
	z = ZERO;

	//  ------------------------------------------------------------------
	//  Set up the first vectors u and v for the bidiagonalization.
	//  These satisfy  beta*u = b,  alpha*v = A(transpose)*u.
	//  ------------------------------------------------------------------


#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("dload");
#endif

	int vec_x_len_local = perProcDataV3->vec_x_len;
	dload(perProcDataV3->vec_x_len, 0.0, v);
	dload(perProcDataV3->vec_x_len, 0.0, x);
	if (wantse)
		dload(perProcDataV3->vec_x_len, 0.0, se);

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstop("dload");
#endif
	/*	if (wantse)
	 dload(numCol, 0.0, se);*/

	alpha = ZERO;
	//		beta   =   cblas_dnrm2 ( m, perProc_u, 1 );

//Initialize vector
	initGatherX(procRank, numProc, perProcDataV3);
	initGatherY(procRank, numProc, perProcDataV3);
	if(numProc>1) PrintVecInfo(procRank, numProc, perProcDataV3);

	if (procRank == 0)
		printf("------------------------iteration=%d----------------------------------\n", itn);

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("norm0");
#endif

	beta = cblas_dnrm2_parallel_plsqr3_forVecY(perProcDataV3, 1, itn);

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstop("norm0");
#endif

	if(procRank==0)	printf("%d: beta=%e from vector y\n", procRank, beta);

	if (beta > ZERO) {
#ifdef BARRIER
		MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
		GPTLstart("scale1");
#endif
		cblas_dscal_plsqr3(perProcDataV3, (ONE / beta), 1, itn); //operate on vector y
#ifdef GPTL
		GPTLstop("scale1");
#endif

		aprod(version, procRank, numProc, 2, numRow, numCol, v, NULL, UsrWrk, 0, 0);

#ifdef BARRIER
		MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
		GPTLstart("norm1");
#endif

		alpha = cblas_dnrm2_parallel(perProcDataV3->vec_x_len, v, 1, itn);

#ifdef BARRIER
		MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
		GPTLstop("norm1");
#endif
		if(procRank==0)	printf("%d: alpha=%e from vector x\n", procRank, alpha);
	}

#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("scale3");
#endif
	if (alpha > ZERO) {
		cblas_dscal(perProcDataV3->vec_x_len, (ONE / alpha), v, 1);
		cblas_dcopy(perProcDataV3->vec_x_len, v, 1, w, 1);
	}
	arnorm = arnorm0 = alpha * beta;
#ifdef GPTL
	GPTLstop("scale3");
#endif
	if (arnorm == ZERO) {
		goto goto_800;
		//			printf("%d:------------------------arnorm == ZERO----------------------------------\n",procRank);
		if (damped && istop == 2)
			istop = 3;

		if (procRank == 0)
			if (nout != NULL) {
				fprintf(nout, fmt_2000, exit, istop, itn, exit, anorm, acond,
						exit, bnorm, xnorm, exit, rnorm, arnorm);
				fprintf(nout, fmt_2100, exit, dxmax, maxdx, exit,
						dxmax / (xnorm + 1.0e-20));
				fprintf(nout, fmt_3000, exit, msg[istop]);
			}
		//  Assign output variables from local copies.
		*istop_out = istop;
		*itn_out = itn;
		*anorm_out = anorm;
		*acond_out = acond;
		*rnorm_out = rnorm;
		*arnorm_out = test2;
		*xnorm_out = xnorm;
		return;
	}

	rhobar = alpha;
	phibar = beta;
	bnorm = beta;
	rnorm = beta;

	if (procRank == 0)
		if (nout != NULL) {
			if (damped) {
				fprintf(nout, fmt_1300);
			} else {
				fprintf(nout, fmt_1200);
			}
			test1 = ONE;
			test2 = alpha / beta;

			if (extra) {
				fprintf(nout, fmt_1400);
			}
			fprintf(nout, fmt_1550, itn, x[0], rnorm, test1, test2);
			fprintf(nout, fmt_1600);
		}

	//  ==================================================================
	//  Main iteration loop.
	//  ==================================================================

#if GPUALL==1
/*GPU memory initialization for PLSQR3*/
	PPDataV3_TRI_GPU p_d;
	if(procRank==0) printf("Initialing InitPPDataV3_TRI_GPU \n");
	InitPPDataV3_TRI_GPU(procRank, &p_d, perProcDataV3, x, w/*, se, t*/);
	MY_FLOAT_TYPE * v_d =  perProcDataV3->gpu->vec_x_d;
#endif //GPUALL


#ifdef USEPAPI
	float real_time=0.0, proc_time=0.0, mflops=0.0;	long long flpops=0;	int retval=0;
	/* Setup PAPI library and begin collecting data from the counters http://icl.cs.utk.edu/projects/papi/wiki/PAPIC:PAPI_flops.3	*/
	if((retval=PAPI_flops( &real_time, &proc_time, &flpops, &mflops))<PAPI_OK)
		test_fail(__FILE__, __LINE__, "PAPI_flops", retval);
#endif



#ifdef BARRIER
	MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
	GPTLstart("LSQR_MAIN_ITERATION");
#endif

	struct timeval start, end;
	gettimeofday(&start, NULL);

	if (lsqr_iteration > 0)
		while (1) {
			//      ------------------------------------------------------------------
			//      Perform the next step of the bidiagonalization to obtain the
			//      next  beta, u, alpha, v.  These satisfy the relations
			//                 beta*u  =  A*v  -  alpha*u,
			//                alpha*v  =  A(transpose)*u  -  beta*v.
			//      ------------------------------------------------------------------
			itn = itn + 1;
			if (procRank == 0) {
				printf("------------------------iteration=%d----------------------------------\n", itn);
			}
			int lastIteration = 0; //0 means false; 1 means true;
			if (itn == lsqr_iteration) {
				lastIteration = 1;
			}

#ifdef BARRIER
			MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
			GPTLstart("scale1");
#endif
			cblas_dscal_plsqr3(perProcDataV3, (-alpha), 1, itn);
#ifdef GPTL
			GPTLstop("scale1");
#endif

			aprod(version, procRank, numProc, 1, numRow, numCol, NULL, NULL, UsrWrk, lastIteration, itn);

#ifdef BARRIER
			MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
			GPTLstart("norm1");
#endif

			beta = cblas_dnrm2_parallel_plsqr3_forVecY(perProcDataV3, 1, itn);

#ifdef GPTL
			GPTLstop("norm1");
#endif
			if(procRank==0)	printf("%d: beta=%e from vector y\n", procRank, beta);

			//      Accumulate  anorm = || Bk ||
			//                        =  sqrt( sum of  alpha**2 + beta**2 + damp**2 ).

			temp = d2norm(alpha, beta);
			temp = d2norm(temp, damp);
			anorm = d2norm(anorm, temp);

			if (beta > ZERO) {
#ifdef BARRIER
				MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
				GPTLstart("scale2");
#endif

				cblas_dscal_plsqr3(perProcDataV3, (ONE / beta), 1, itn);

#ifdef BARRIER
				MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
				GPTLstop("scale2");
//				cblas_dscal(numCol, (-beta), v, 1);
#endif

#ifdef BARRIER
				MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
				GPTLstart("scale1");
#endif
#if GPUALL ==1
				MY_FLOAT_TYPE beta1  = (-beta);
				cublasSscal_w(vec_x_len_local, &beta1,v_d, 1);
#else
				cblas_dscal(vec_x_len_local, (-beta), v, 1);
#endif
#ifdef GPTL
				GPTLstop("scale1");
#endif

				aprod(version, procRank, numProc, 2, numRow, numCol, NULL, NULL, UsrWrk, lastIteration, itn);

#ifdef BARRIER
				MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
				GPTLstart("norm1");
#endif


#if GPUALL==1
				alpha = cblas_dnrm2_parallel(perProcDataV3->vec_x_len, v_d, 1, itn);
#else
				alpha = cblas_dnrm2_parallel(perProcDataV3->vec_x_len, v, 1, itn);
#endif
				if(procRank==0)	printf("%d: alpha=%e from vector x\n", procRank, alpha);

#ifdef GPTL
				GPTLstop("norm1");
#endif
#ifdef BARRIER
				MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
				GPTLstart("scale3");
#endif
				if (alpha > ZERO) {
#if GPUALL==1
					MY_FLOAT_TYPE alpha2 = ONE / alpha;
					cublasSscal_w(vec_x_len_local, &alpha2, v_d, 1);
#else
					cblas_dscal(vec_x_len_local, (ONE / alpha), v, 1);
#endif
				}
#ifdef GPTL
				GPTLstop("scale3");
#endif
			}

			//copy vector v/x from device to host

			//      ------------------------------------------------------------------
			//      Use a plane rotation to eliminate the damping parameter.
			//      This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
			//      ------------------------------------------------------------------
			rhbar1 = rhobar;
			if (damped) {
				rhbar1 = d2norm(rhobar, damp);
				cs1 = rhobar / rhbar1;
				sn1 = damp / rhbar1;
				psi = sn1 * phibar;
				phibar = cs1 * phibar;
			}

			//      ------------------------------------------------------------------
			//      Use a plane rotation to eliminate the subdiagonal element (beta)
			//      of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
			//      ------------------------------------------------------------------
			rho = d2norm(rhbar1, beta);
			cs = rhbar1 / rho;
			sn = beta / rho;
			theta = sn * alpha;
			rhobar = -cs * alpha;
			phi = cs * phibar;
			phibar = sn * phibar;
			tau = sn * phi;

			//      ------------------------------------------------------------------
			//      Update  x, w  and (perhaps) the standard error estimates.
			//      ------------------------------------------------------------------
			t1 = phi / rho;
			t2 = -theta / rho;
			t3 = ONE / rho;
			dknorm = ZERO;
#ifdef BARRIER
			MPI_Barrier(MPI_COMM_WORLD);
#endif
#ifdef GPTL
			GPTLstart("krylov_p1");
#endif
			if (wantse) {
#if GPUALL==1 //copy out  result from device to host
				PPDataV3_TRI_GPU * gpu = perProcDataV3->gpu;
				//t=w[i]
				cudaMemcpy_w(gpu->lsqr_t_d, gpu->lsqr_w_d, perProcDataV3->vec_x_len*sizeof(MY_FLOAT_TYPE), mycudaMemcpyDeviceToDevice );
				//x[i] = t1 * t + x[i];
				cublasSaxpy_w(perProcDataV3->vec_x_len, &t1, gpu->lsqr_t_d, 1, gpu->lsqr_x_d, 1);
				//w[i] = t2 * t + v[i]; 1.copy v to w, and w=t2*t+w
				cudaMemcpy_w(gpu->lsqr_w_d, gpu->vec_x_d, perProcDataV3->vec_x_len*sizeof(MY_FLOAT_TYPE), mycudaMemcpyDeviceToDevice );
				cublasSaxpy_w(perProcDataV3->vec_x_len, &t2, gpu->lsqr_t_d, 1, gpu->lsqr_w_d, 1);
				//t = (t3 * t) * (t3 * t);
				vvmulti_w(gpu->lsqr_t_d, t3, perProcDataV3->vec_x_len);
				//se[i] = t + se[i];
				MY_FLOAT_TYPE alpha1 = 1.0f;
				cublasSaxpy_w(perProcDataV3->vec_x_len, &alpha1, gpu->lsqr_t_d, 1, gpu->lsqr_se_d, 1);
				//dknorm = t + dknorm;
				cublasSasum_w(perProcDataV3->vec_x_len, gpu->lsqr_t_d, 1, &dknorm);
#else
				for (i = 0; i < perProcDataV3->vec_x_len; i++) {
					t = w[i];
					x[i] = t1 * t + x[i];
					w[i] = t2 * t + v[i];
					t = (t3 * t) * (t3 * t);
					se[i] = t + se[i];
					dknorm = t + dknorm;
				}
#endif//GPUALL==1
			} else {
#if GPUALL==1
				PPDataV3_TRI_GPU * gpu = perProcDataV3->gpu;
//					t = w[i];
				cudaMemcpy_w(gpu->lsqr_t_d, gpu->lsqr_w_d, perProcDataV3->vec_x_len*sizeof(MY_FLOAT_TYPE), mycudaMemcpyDeviceToDevice );
//					x[i] = t1 * t + x[i];
				cublasSaxpy_w(perProcDataV3->vec_x_len, &t1, gpu->lsqr_t_d, 1, gpu->lsqr_x_d, 1);
//					w[i] = t2 * t + v[i];
				cudaMemcpy_w(gpu->lsqr_w_d, gpu->vec_x_d, perProcDataV3->vec_x_len*sizeof(MY_FLOAT_TYPE), mycudaMemcpyDeviceToDevice );
				cublasSaxpy_w(perProcDataV3->vec_x_len, &t2, gpu->lsqr_t_d, 1, gpu->lsqr_w_d, 1);
//					dknorm = (t3 * t) * (t3 * t) + dknorm;
				vvmulti_w(gpu->lsqr_t_d, t3, perProcDataV3->vec_x_len);
				cublasSasum_w(perProcDataV3->vec_x_len, gpu->lsqr_t_d, 1, &dknorm);
#else
				for (i = 0; i < perProcDataV3->vec_x_len; i++) {
					t = w[i];
					x[i] = t1 * t + x[i];
					w[i] = t2 * t + v[i];
					dknorm = (t3 * t) * (t3 * t) + dknorm;
				}
#endif // GPUALL==1
			}

#ifdef GPTL
			GPTLstop("krylov_p1");
#endif
			//      ------------------------------------------------------------------
			//      Monitor the norm of d_k, the update to x.
			//      dknorm = norm( d_k )
			//      dnorm  = norm( D_k ),        where   D_k = (d_1, d_2, ..., d_k )
			//      dxk    = norm( phi_k d_k ),  where new x = x_k + phi_k d_k.
			//      ------------------------------------------------------------------
			dknorm = sqrt(dknorm);
			dnorm = d2norm(dnorm, dknorm);
			dxk = fabs(phi * dknorm);
			if (dxmax < dxk) {
				dxmax = dxk;
				maxdx = itn;
			}

			//      ------------------------------------------------------------------
			//      Use a plane rotation on the right to eliminate the
			//      super-diagonal element (theta) of the upper-bidiagonal matrix.
			//      Then use the result to estimate  norm(x).
			//      ------------------------------------------------------------------
			delta = sn2 * rho;
			gambar = -cs2 * rho;
			rhs = phi - delta * z;
			zbar = rhs / gambar;
			xnorm = d2norm(xnorm1, zbar);
			gamma = d2norm(gambar, theta);
			cs2 = gambar / gamma;
			sn2 = theta / gamma;
			z = rhs / gamma;
			xnorm1 = d2norm(xnorm1, z);

			//      ------------------------------------------------------------------
			//      Test for convergence.
			//      First, estimate the norm and condition of the matrix  Abar,
			//      and the norms of  rbar  and  Abar(transpose)*rbar.
			//      ------------------------------------------------------------------
			acond = anorm * dnorm;
			res2 = d2norm(res2, psi);
			rnorm = d2norm(res2, phibar);
			arnorm = alpha * fabs(tau);

			//      Now use these norms to estimate certain other quantities,
			//      some of which will be small near a solution.

			alfopt = sqrt(rnorm / (dnorm * xnorm));
			test1 = rnorm / bnorm;
			test2 = ZERO;
			if (rnorm > ZERO)
				test2 = arnorm / (anorm * rnorm);
			//      if (arnorm0 > ZERO) test2 = arnorm / arnorm0;  //(Michael Friedlander's modification)
			test3 = ONE / acond;
			t1 = test1 / (ONE + anorm * xnorm / bnorm);
			rtol = btol + atol * anorm * xnorm / bnorm;

			//      The following tests guard against extremely small values of
			//      atol, btol  or  ctol.  (The user may have set any or all of
			//      the parameters  atol, btol, conlim  to zero.)
			//      The effect is equivalent to the normal tests using
			//      atol = relpr,  btol = relpr,  conlim = 1/relpr.

			t3 = ONE + test3;
			t2 = ONE + test2;
			t1 = ONE + t1;
			if (itn >= itnlim)
				istop = 5;
			if (t3 <= ONE)
				istop = 4;
			if (t2 <= ONE)
				istop = 2;
			if (t1 <= ONE)
				istop = 1;

			//      Allow for tolerances set by the user.

			if (test3 <= ctol)
				istop = 4;
			if (test2 <= atol)
				istop = 2;
			if (test1 <= rtol)
				istop = 1; //(Michael Friedlander had this commented out)

			//      ------------------------------------------------------------------
			//      See if it is time to print something.
			//      ------------------------------------------------------------------
			if (nout == NULL)
				goto goto_600;
			if (numCol <= 40)
				goto goto_400;
			if (itn <= 10)
				goto goto_400;
			if (itn >= itnlim - 10)
				goto goto_400;
			if (itn % 10 == 0)
				goto goto_400;
			if (test3 <= 2.0 * ctol)
				goto goto_400;
			if (test2 <= 10.0 * atol)
				goto goto_400;
			if (test1 <= 10.0 * rtol)
				goto goto_400;
			if (istop != 0)
				goto goto_400;
			goto goto_600;

			//      Print a line for this iteration.
			//      "extra" is for experimental purposes.

			goto_400: if (extra) {
				if (procRank == 0)
					fprintf(nout, fmt_1500_extra, itn, x[0], rnorm, test1,
							test2, anorm, acond, phi, dknorm, dxk, alfopt);
			} else {
				if (procRank == 0)
					fprintf(nout, fmt_1500, itn, x[0], rnorm, test1, test2,
							anorm, acond);
			}
			//			if (itn % 10 == 0) fprintf(nout, fmt_1600);

			//      ------------------------------------------------------------------
			//      Stop if appropriate.
			//      The convergence criteria are required to be met on  nconv
			//      consecutive iterations, where  nconv  is set below.
			//      Suggested value:  nconv = 1, 2  or  3.
			//      ------------------------------------------------------------------
			goto_600: if (istop == 0) {
				nstop = 0;
			} else {
				nconv = 1;
				nstop = nstop + 1;
				if (nstop < nconv && itn < itnlim)
					istop = 0;
			}

			//		MPI_Bcast(&istop, 1, MPI_INT, 0, MPI_COMM_WORLD);

			//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
			//		if (istop!=0) break;
			if (itn == lsqr_iteration)
				break;
			//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

		}
	//set timer: end
	gettimeofday(&end, NULL);
	if (procRank==0)
		printf("Parallel LSQR time=%lf s\n",((end.tv_sec * 1000000 + end.tv_usec)
					- (start.tv_sec * 1000000 + start.tv_usec))*1.0 / 1000000);

	//  ==================================================================
	//  End of iteration loop.
	//  ==================================================================
#ifdef USEPAPI
	/* Collect the data into the variables passed in */
	if((retval=PAPI_flops( &real_time, &proc_time, &flpops, &mflops))<PAPI_OK)
		test_fail(__FILE__, __LINE__, "PAPI_flops", retval);
//	printf("%d: Real_time:\t%f\tProc_time:\t%f\tTotal flpops:\t%lld\tMFLOPS:\t\t%f\t %s\tPASSED\n",
//			procRank, real_time, proc_time, flpops, mflops, __FILE__);
	long long flop_allproc = 0;
	MPI_Reduce(&flpops, &flop_allproc, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if(procRank==0) {
		printf("%d: \"LSQR main iteration\" total flop = %lld   flops=%le for %d processes\n", procRank, flop_allproc, 1.0*flop_allproc/proc_time, numProc);
	}
#endif

#ifdef GPTL
	GPTLstop("LSQR_MAIN_ITERATION");
#endif

#if GPUALL==1
	/*copy result from device to host*/
	cudaMemcpy_w(perProcDataV3->vec_x, perProcDataV3->gpu->vec_x_d, sizeof(MY_FLOAT_TYPE) * perProcDataV3->vec_x_len, mycudaMemcpyDeviceToHost);
	cudaMemcpy_w(perProcDataV3->vec_y_k, perProcDataV3->gpu->vec_y_k_d, sizeof(MY_FLOAT_TYPE) * perProcDataV3->matCSC_k->numRow, mycudaMemcpyDeviceToHost);
	cudaMemcpy_w(perProcDataV3->vec_y_d, perProcDataV3->gpu->vec_y_d_d, sizeof(MY_FLOAT_TYPE) * perProcDataV3->vec_y_d_len, mycudaMemcpyDeviceToHost);
	cudaMemcpy_w(x, perProcDataV3->gpu->lsqr_x_d, sizeof(MY_FLOAT_TYPE)*perProcDataV3->vec_x_len, mycudaMemcpyDeviceToHost);
	cudaMemcpy_w(w, perProcDataV3->gpu->lsqr_w_d, sizeof(MY_FLOAT_TYPE)*perProcDataV3->vec_x_len, mycudaMemcpyDeviceToHost);
	if (wantse) cudaMemcpy_w(se, perProcDataV3->gpu->lsqr_se_d, sizeof(MY_FLOAT_TYPE)*perProcDataV3->vec_x_len, mycudaMemcpyDeviceToHost);
/*free GPU memory  for PLSQR3*/
	FinalPPDataV3_TRI_GPU(perProcDataV3->gpu);
#endif //GPUALL


	//  Finish off the standard error estimates.
	if (wantse) {
		t = ONE;
		if (numRow > numCol)
			t = numRow - numCol;
		if (damped)
			t = numRow;
		t = rnorm / sqrt(t);


		int chunk=(vec_x_len_local+group_size_omp-1) / (group_size_omp);
#pragma omp parallel for private(i) schedule(dynamic, chunk)
		for (i = 0; i < vec_x_len_local; i++)
			se[i] = t * sqrt(se[i]);
	}

	//  Decide if istop = 2 or 3.
	//  Print the stopping condition.
	goto_800: if (damped && istop == 2)
		istop = 3;
	if (procRank == 0)
		if (nout != NULL) {
			fprintf(nout, fmt_2000, exit, istop, itn, exit, anorm, acond, exit,
					bnorm, xnorm, exit, rnorm, arnorm);

			fprintf(nout, fmt_2100, exit, dxmax, maxdx, exit,
					dxmax / (xnorm + 1.0e-20));
			fprintf(nout, fmt_3000, exit, msg[istop]);
		}

	//  Assign output variables from local copies.
	*istop_out = istop;
	*itn_out = itn;
	*anorm_out = anorm;
	*acond_out = acond;
	*rnorm_out = rnorm;
	*arnorm_out = test2;
	*xnorm_out = xnorm;

	return;
}
