#ifndef APROD_H_
#define APROD_H_

//#include <stdio.h>
//#include <stdlib.h>
//#include "main.h"

//#include "GPUComputeSpMV.cuh"

void aprod(int version, int procRank, int numProc, int mode, int m /*row*/,
		int n /*column*/, MY_FLOAT_TYPE x[]/*v len=n*/,
		MY_FLOAT_TYPE y[] /*u=perPorcVectorB len=m, assign NULL if PLSQRV3*/, void *UsrWork,
		int isFinalIteration, int iteration);

#endif // APROD_H_
