/* lsqr.c
   This C version of LSQR was first created by
      Michael P Friedlander <mpf@cs.ubc.ca>
   as part of his BCLS package:
      http://www.cs.ubc.ca/~mpf/bcls/index.html.
   The present file is maintained by
      Michael Saunders <saunders@stanford.edu>

   31 Aug 2007: First version of this file lsqr.c obtained from
                Michael Friedlander's BCLS package, svn version number
                $Revision: 273 $ $Date: 2006-09-04 15:59:04 -0700 (Mon, 04 Sep 2006) $

                The stopping rules were slightly altered in that version.
                They have been restored to the original rules used in the f77 LSQR.
*/

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>

#ifdef __APPLE__
  #include <vecLib/vecLib.h>
#else
  #include "../cblas.h"
#endif

#define ZERO   0.0
#define ONE    1.0

// ---------------------------------------------------------------------
// d2norm  returns  sqrt( a**2 + b**2 )  with precautions
// to avoid overflow.
//
// 21 Mar 1990: First version.
// ---------------------------------------------------------------------
static float
d2norm( const float a, const float b )
{
    float scale;
    const float zero = 0.0;

    scale  = fabs( a ) + fabs( b );
    if (scale == zero)
        return zero;
    else
        return scale * sqrt( (a/scale)*(a/scale) + (b/scale)*(b/scale) );
}

static void
dload( const int n, const float alpha, float x[] )
{    
    int i;
    for (i = 0; i < n; i++) x[i] = alpha;
    return;
}

// ---------------------------------------------------------------------
// LSQR
// ---------------------------------------------------------------------
void lsqr_s(
          int m,
          int n,
          void (*aprod_s_c)(int mode, int m, int n, float x[], float y[],
                        void *UsrWrk),
          float damp,
          void   *UsrWrk,
          float u[],     // len = m
          float v[],     // len = n
          float w[],     // len = n
          float x[],     // len = n
          float se[],    // len at least n.  May be NULL.
          float atol,
          float btol,
          float conlim,
          int    itnlim,
          FILE   *nout,
          // The remaining variables are output only.
          int    *istop_out,
          int    *itn_out,
          float *anorm_out,
          float *acond_out,
          float *rnorm_out,
          float *arnorm_out,
          float *xnorm_out
         )
{
//  Local copies of output variables.  Output vars are assigned at exit.
    int
        istop  = 0,
        itn    = 0;
    float
        anorm  = ZERO,
        acond  = ZERO,
        rnorm  = ZERO,
        arnorm = ZERO,
        xnorm  = ZERO;

//  Local variables

    const bool
        extra  = false,       // true for extra printing below.
        damped = damp > ZERO,
        wantse = se != NULL;
    int
        i, maxdx, nconv, nstop;
    float
        alfopt, alpha, arnorm0, beta, bnorm,
        cs, cs1, cs2, ctol,
        delta, dknorm, dnorm, dxk, dxmax,
        gamma, gambar, phi, phibar, psi,
        res2, rho, rhobar, rhbar1,
        rhs, rtol, sn, sn1, sn2,
        t, tau, temp, test1, test2, test3,
        theta, t1, t2, t3, xnorm1, z, zbar;
    char
        enter[] = "Enter LSQR.  ",
        exit[]  = "Exit  LSQR.  ",
        msg[6][100] =
        {
            {"The exact solution is  x = 0"},
            {"A solution to Ax = b was found, given atol, btol"},
            {"A least-squares solution was found, given atol"},
            {"A damped least-squares solution was found, given atol"},
            {"Cond(Abar) seems to be too large, given conlim"},
            {"The iteration limit was reached"}
        };
//-----------------------------------------------------------------------

//  Format strings.
    char fmt_1000[] = 
        " %s        Least-squares solution of  Ax = b\n"
        " The matrix  A  has %7d rows  and %7d columns\n"
        " damp   = %-22.2e    wantse = %10i\n"
        " atol   = %-22.2e    conlim = %10.2e\n"
        " btol   = %-22.2e    itnlim = %10d\n\n";
    char fmt_1200[] =
        "    Itn       x(1)           Function"
        "     Compatible    LS      Norm A   Cond A\n";
    char fmt_1300[] =
        "    Itn       x(1)           Function"
        "     Compatible    LS      Norm Abar   Cond Abar\n";
    char fmt_1400[] =
        "     phi    dknorm  dxk  alfa_opt\n";
    char fmt_1500_extra[] =
        " %6d %16.9e %16.9e %9.2e %9.2e %8.1e %8.1e %8.1e %7.1e %7.1e %7.1e\n";
    char fmt_1500[] =
        " %6d %16.9e %16.9e %9.2e %9.2e %8.1e %8.1e\n";
    char fmt_1550[] =
        " %6d %16.9e %16.9e %9.2e %9.2e\n";
    char fmt_1600[] = 
        "\n";
    char fmt_2000[] =
        "\n"
        " %s       istop  = %-10d      itn    = %-10d\n"
        " %s       anorm  = %11.5e     acond  = %11.5e\n"
        " %s       vnorm  = %11.5e     xnorm  = %11.5e\n"
        " %s       rnorm  = %11.5e     arnorm = %11.5e\n";
    char fmt_2100[] =
        " %s       max dx = %7.1e occured at itn %-9d\n"
        " %s              = %7.1e*xnorm\n";
    char fmt_3000[] =
        " %s       %s\n";

//  Initialize.

    if (nout != NULL)
        fprintf(nout, fmt_1000,
                enter, m, n, damp, wantse,
                atol, conlim, btol, itnlim);

    itn    =   0;
    istop  =   0;
    nstop  =   0;
    maxdx  =   0;
    ctol   =   ZERO;
    if (conlim > ZERO) ctol = ONE / conlim;
    anorm  =   ZERO;
    acond  =   ZERO;
    dnorm  =   ZERO;
    dxmax  =   ZERO;
    res2   =   ZERO;
    psi    =   ZERO;
    xnorm  =   ZERO;
    xnorm1 =   ZERO;
    cs2    = - ONE;
    sn2    =   ZERO;
    z      =   ZERO;

//  ------------------------------------------------------------------
//  Set up the first vectors u and v for the bidiagonalization.
//  These satisfy  beta*u = b,  alpha*v = A(transpose)*u.
//  ------------------------------------------------------------------
    dload( n, 0.0, v );
    dload( n, 0.0, x );

    if ( wantse )
        dload( n, 0.0, se );
    
    alpha  =   ZERO;
    beta   =   cblas_dnrm2 ( m, u, 1 );

    if (beta > ZERO) {
        cblas_dscal ( m, (ONE / beta), u, 1 );
        aprod_s ( 2, m, n, v, u, UsrWrk );
        alpha  =   cblas_dnrm2 ( n, v, 1 );
    }

    if (alpha > ZERO) {
        cblas_dscal ( n, (ONE / alpha), v, 1 );
        cblas_dcopy ( n, v, 1, w, 1 );
    }

    arnorm = arnorm0 = alpha * beta;
    if (arnorm == ZERO) goto goto_800;
    
    rhobar =   alpha;
    phibar =   beta;
    bnorm  =   beta;
    rnorm  =   beta;

    if (nout != NULL) {
        if ( damped ) 
            fprintf(nout, fmt_1300);
        else
            fprintf(nout, fmt_1200);

        test1  = ONE;
        test2  = alpha / beta;
        
        if ( extra ) 
            fprintf(nout, fmt_1400);

        fprintf(nout, fmt_1550, itn, x[0], rnorm, test1, test2);
        fprintf(nout, fmt_1600);
    }


//  ==================================================================
//  Main iteration loop.
//  ==================================================================
    while (1) {
        itn    = itn + 1;
		printf("------------------------iteration=%d----------------------------------\n",	itn);

//      ------------------------------------------------------------------
//      Perform the next step of the bidiagonalization to obtain the
//      next  beta, u, alpha, v.  These satisfy the relations
//                 beta*u  =  A*v  -  alpha*u,
//                alpha*v  =  A(transpose)*u  -  beta*v.
//      ------------------------------------------------------------------
        cblas_dscal ( m, (- alpha), u, 1 );
        aprod_s ( 1, m, n, v, u, UsrWrk );
        beta   =   cblas_dnrm2 ( m, u, 1 );

//      Accumulate  anorm = || Bk ||
//                        =  sqrt( sum of  alpha**2 + beta**2 + damp**2 ).

        temp   =   d2norm( alpha, beta );
        temp   =   d2norm( temp , damp );
        anorm  =   d2norm( anorm, temp );

        if (beta > ZERO) {
            cblas_dscal ( m, (ONE / beta), u, 1 );
            cblas_dscal ( n, (- beta), v, 1 );
            aprod_s ( 2, m, n, v, u, UsrWrk );
            alpha  =   cblas_dnrm2 ( n, v, 1 );
            if (alpha > ZERO) {
                cblas_dscal ( n, (ONE / alpha), v, 1 );
            }
        }

//      ------------------------------------------------------------------
//      Use a plane rotation to eliminate the damping parameter.
//      This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
//      ------------------------------------------------------------------
        rhbar1 = rhobar;
        if ( damped ) {
            rhbar1 = d2norm( rhobar, damp );
            cs1    = rhobar / rhbar1;
            sn1    = damp   / rhbar1;
            psi    = sn1 * phibar;
            phibar = cs1 * phibar;
        }

//      ------------------------------------------------------------------
//      Use a plane rotation to eliminate the subdiagonal element (beta)
//      of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
//      ------------------------------------------------------------------
        rho    =   d2norm( rhbar1, beta );
        cs     =   rhbar1 / rho;
        sn     =   beta   / rho;
        theta  =   sn * alpha;
        rhobar = - cs * alpha;
        phi    =   cs * phibar;
        phibar =   sn * phibar;
        tau    =   sn * phi;

//      ------------------------------------------------------------------
//      Update  x, w  and (perhaps) the standard error estimates.
//      ------------------------------------------------------------------
        t1     =   phi   / rho;
        t2     = - theta / rho;
        t3     =   ONE   / rho;
        dknorm =   ZERO;

        if ( wantse ) {
            for (i = 0; i < n; i++) {
                t      =  w[i];
                x[i]   =  t1*t  +  x[i];
                w[i]   =  t2*t  +  v[i];
                t      = (t3*t)*(t3*t);
                se[i]  =  t     +  se[i];
                dknorm =  t     +  dknorm;
            }
        }
        else {
            for (i = 0; i < n; i++) {
                t      =  w[i];
                x[i]   =  t1*t  +  x[i];
                w[i]   =  t2*t  +  v[i];
                dknorm = (t3*t)*(t3*t)  +  dknorm;
            }
        }

//      ------------------------------------------------------------------
//      Monitor the norm of d_k, the update to x.
//      dknorm = norm( d_k )
//      dnorm  = norm( D_k ),        where   D_k = (d_1, d_2, ..., d_k )
//      dxk    = norm( phi_k d_k ),  where new x = x_k + phi_k d_k.
//      ------------------------------------------------------------------
        dknorm = sqrt( dknorm );
        dnorm  = d2norm( dnorm, dknorm );
        dxk    = fabs( phi * dknorm );
        if (dxmax < dxk ) {
            dxmax   =  dxk;
            maxdx   =  itn;
        }

//      ------------------------------------------------------------------
//      Use a plane rotation on the right to eliminate the
//      super-diagonal element (theta) of the upper-bidiagonal matrix.
//      Then use the result to estimate  norm(x).
//      ------------------------------------------------------------------
        delta  =   sn2 * rho;
        gambar = - cs2 * rho;
        rhs    =   phi    - delta * z;
        zbar   =   rhs    / gambar;
        xnorm  =   d2norm( xnorm1, zbar  );
        gamma  =   d2norm( gambar, theta );
        cs2    =   gambar / gamma;
        sn2    =   theta  / gamma;
        z      =   rhs    / gamma;
        xnorm1 =   d2norm( xnorm1, z     );

//      ------------------------------------------------------------------
//      Test for convergence.
//      First, estimate the norm and condition of the matrix  Abar,
//      and the norms of  rbar  and  Abar(transpose)*rbar.
//      ------------------------------------------------------------------
        acond  =   anorm * dnorm;
        res2   =   d2norm( res2 , psi    );
        rnorm  =   d2norm( res2 , phibar );
        arnorm =   alpha * fabs( tau );

//      Now use these norms to estimate certain other quantities,
//      some of which will be small near a solution.

        alfopt =   sqrt( rnorm / (dnorm * xnorm) );
        test1  =   rnorm /  bnorm;
        test2  =   ZERO;
        if (rnorm   > ZERO) test2 = arnorm / (anorm * rnorm);
//      if (arnorm0 > ZERO) test2 = arnorm / arnorm0;  //(Michael Friedlander's modification)
        test3  =   ONE   /  acond;
        t1     =   test1 / (ONE  +  anorm * xnorm / bnorm);
        rtol   =   btol  +  atol *  anorm * xnorm / bnorm;

//      The following tests guard against extremely small values of
//      atol, btol  or  ctol.  (The user may have set any or all of
//      the parameters  atol, btol, conlim  to zero.)
//      The effect is equivalent to the normal tests using
//      atol = relpr,  btol = relpr,  conlim = 1/relpr.

        t3     =   ONE + test3;
        t2     =   ONE + test2;
        t1     =   ONE + t1;
        if (itn >= itnlim) istop = 5;
        if (t3  <= ONE   ) istop = 4;
        if (t2  <= ONE   ) istop = 2;
        if (t1  <= ONE   ) istop = 1;

//      Allow for tolerances set by the user.

        if (test3 <= ctol) istop = 4;
        if (test2 <= atol) istop = 2;
        if (test1 <= rtol) istop = 1;   //(Michael Friedlander had this commented out)

//      ------------------------------------------------------------------
//      See if it is time to print something.
//      ------------------------------------------------------------------
        if (nout  == NULL     ) goto goto_600;
        if (n     <= 40       ) goto goto_400;
        if (itn   <= 10       ) goto goto_400;
        if (itn   >= itnlim-10) goto goto_400;
        if (itn % 10 == 0     ) goto goto_400;
        if (test3 <=  2.0*ctol) goto goto_400;
        if (test2 <= 10.0*atol) goto goto_400;
        if (test1 <= 10.0*rtol) goto goto_400;
        if (istop != 0        ) goto goto_400;
        goto goto_600;

//      Print a line for this iteration.
//      "extra" is for experimental purposes.

    goto_400:
        if ( extra ) {
            fprintf(nout, fmt_1500_extra,
                    itn, x[0], rnorm, test1, test2, anorm,
                    acond, phi, dknorm, dxk, alfopt);
        }
        else {
            fprintf(nout, fmt_1500,
                    itn, x[0], rnorm, test1, test2, anorm, acond);
        }
        if (itn % 10 == 0) fprintf(nout, fmt_1600);

//      ------------------------------------------------------------------
//      Stop if appropriate.
//      The convergence criteria are required to be met on  nconv
//      consecutive iterations, where  nconv  is set below.
//      Suggested value:  nconv = 1, 2  or  3.
//      ------------------------------------------------------------------
    goto_600:
        if (istop == 0) {
            nstop  = 0;
        }
        else {
            nconv  = 1;
            nstop  = nstop + 1;
            if (nstop < nconv  &&  itn < itnlim) istop = 0;
        }

        if (istop != 0) break;
        
    }
//  ==================================================================
//  End of iteration loop.
//  ==================================================================

//  Finish off the standard error estimates.

    if ( wantse ) {
        t    =   ONE;
        if (m > n)     t = m - n;
        if ( damped )  t = m;
        t    =   rnorm / sqrt( t );
      
        for (i = 0; i < n; i++)
            se[i]  = t * sqrt( se[i] );
        
    }

//  Decide if istop = 2 or 3.
//  Print the stopping condition.
 goto_800:
    if (damped  &&  istop == 2) istop = 3;
    if (nout != NULL) {
        fprintf(nout, fmt_2000,
                exit, istop, itn,
                exit, anorm, acond,
                exit, bnorm, xnorm,
                exit, rnorm, arnorm);
        fprintf(nout, fmt_2100,
                exit, dxmax, maxdx,
                exit, dxmax/(xnorm + 1.0e-20));
        fprintf(nout, fmt_3000,
                exit, msg[istop]);
    }

//  Assign output variables from local copies.
    *istop_out  = istop;
    *itn_out    = itn;
    *anorm_out  = anorm;
    *acond_out  = acond;
    *rnorm_out  = rnorm;
    *arnorm_out = test2;
    *xnorm_out  = xnorm;

    return;
}


