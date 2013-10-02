/*
 * Ker2PLSQR3.c
 *  Created on: Feb 28, 2011
 *  Revised on: 9/2/2011
 *      Author: hh
 *
 * run: mpiexec -np 2 ./Ker2P3_format  ../../matlab_test20110219/ kernel_statistic 10
 * 		mpiexec -np 2 ./Ker2P3_format /home/hh/Documents/matlab_test20110219 kernel_statistic 10
 *  argv[1]: kernel file directory path
 *  argv[2]: kernel file list (1st col: row binary file name; 2nd col nnz per row)
 *  argv[3]: number of column
 *  argv[4]: column info: full path
 *
 *  ./Ker2P3_format /seismic10/hhuang1/elee8_20130204/test_seismicmst ker_test3_v2.bin.list_60_s 38093022 /seismic10/hhuang1/elee8_20130204/test_seismicmst/col_info.txt
 *
 *  output: matrix_bycol.mm2  (ASCII): 1-based indexing
 *  matrix_bycol.mm2.bin  (BINARY): 1-based indexing
 *  matrix_bycol.mm2.info	(MATRIX INFORMATION): 1-based indexing
 *  						columnIdx nnzThisColumn StartingIndex
 *  NOTICE: the floating point is DOUBLE for both data input and output
 */
//#define DEBUG

//#define ASCII //ascii output of data file


#include "../../source/FileIO.h"
#include "../../source/main.h"
#include <time.h>
#include <string.h>
#include <mpi.h>

struct tuple{
  int iX;
  int col_idx;
  double val;
};


int main (int argc, char **argv){

  int numProc = 0; // Number of available processes, rank=0 does not participate calculation!
  int procRank = 0; // Rank of current process
  
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numProc);
  MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

  char hostname[MAXREADNUM];
  gethostname(hostname, sizeof(hostname));
  printf("rank=%d, PID=%d on %s \n", procRank, getpid(), hostname);
  fflush(stdout);
  
  char dirpath[MAXREADNUM];
  char kernelFileList[MAXREADNUM];
  char kerFileListFullName[MAXREADNUM];
  char colInfoFullName[MAXREADNUM];
  int numRow=0, numCol=0, i=0, j=0, k=0;

	if(argc==5){
		strcpy(dirpath, argv[1]);
		strcpy(kernelFileList,argv[2]);
		numCol=atoi(argv[3]);
		strcpy(colInfoFullName , argv[4]);
	}
	sprintf(kerFileListFullName, "%s/%s",dirpath,kernelFileList);


	if (procRank == 0) {

		printf("%d: sizeof (long long) is %d bytes\n", procRank, sizeof(long long));

		numRow = countData(kerFileListFullName);
		printf("%d: The number of kernel file is %d\n", procRank, numRow);
	}
	MPI_Bcast(&numRow, 1, MPI_INT, 0, MPI_COMM_WORLD);

	char **kernelFileName;
	if (!(kernelFileName = (char **) calloc(numRow, sizeof(char *)))){
		printf("error: calloc fails\n");fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
	}
	for (i = 0; i < numRow; i++)
		if (!(kernelFileName[i] = (char *) calloc(MAXREADNUM, sizeof(char)))){// the lenglth of every data file name should not be bigger than 256 char
			printf("error: malloc fails\n");fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
		}
	//indicates the number of column of each data file.
	int * NumNZPerrow_k = (int *) calloc(numRow,sizeof(int));
	if(NULL==NumNZPerrow_k){
		fprintf(stderr, "NumNZPerrow_k allocate fails\n");exit(-1);
	}
	for (i = 0; i < numRow; i++) {
		NumNZPerrow_k[i] = 0;
	}
	getDataFileName(kernelFileName, NumNZPerrow_k, kerFileListFullName,numRow);


	int * nnzPerCol = (int *) calloc(numCol, sizeof(int));
	if(NULL==nnzPerCol){
		fprintf(stderr, " calloc fails \n");exit(-1);
	}
	FILE * fp_col_info=fopen(colInfoFullName, "r");
	for(i=0; i<numCol; i++){
		nnzPerCol[i]=0;
		fscanf(fp_col_info, "%d\n", &nnzPerCol[i]);
	}
	fclose(fp_col_info);
//	if(procRank==0){
//		for (i = 0; i < numRow; i++) {//1st iterate every kernel file
//			char kernelFileFullName[MAXREADNUM];
//			sprintf(kernelFileFullName, "%s/%s",dirpath, kernelFileName[i]);
//			if(procRank==0){
//				printf("%d: 1st loading dataFileName[%d] = %s, number of nonzero per row=%d\n",
//					procRank, i, kernelFileFullName, NumNZPerrow_k[i]);
//			}
//
//			//FILE * fip_k=fopen(kernelFileFullName, "r");
//			//add read binary files Feb 1, 2013
//			FILE * fip_k=fopen(kernelFileFullName, "rb");
//			time_t start,end;
//			time (&start);
//			if(fip_k==NULL){
//				fprintf(stderr, "Cannot open the file: %s  \n", kernelFileFullName);
//				exit(-1);
//			}
//			int colIdx, iX, iY, iZ;
//			double val;
//			struct tuple{
//			  int iX;
//			  int col_idx;
//			  double val;
//			};
//			struct tuple * read_array = (struct tuple * )malloc(sizeof(struct tuple)*NumNZPerrow_k[i]);
//			if(!read_array) fprintf(stderr, "cannot allocat read_array\n" );
//			memset(read_array, 0, sizeof(struct tuple)*NumNZPerrow_k[i]);
//			fread(read_array, sizeof(struct tuple), NumNZPerrow_k[i], fip_k);
//			printf("%d %d %le\n", read_array[0].iX,read_array[0].col_idx, read_array[0].val);
//			for(j=0; j<NumNZPerrow_k[i]; j++){
//				int c1=5;
//				//int c1=fscanf(fip_k, "%d %d %d %d %le", &colIdx, &(iX), &(iY), &(iZ), &val);
//				colIdx=read_array[j].col_idx;
//				if(c1==5){
//					colIdx--;//adjust to 0-based indexing
//					nnzPerCol[colIdx]++;
//				}
//			}
//			time (&end);
//			double dif = difftime (end,start);
//			free(read_array);
//			printf ("It takes %.2f seconds to read the file.\n", dif );
//			fclose(fip_k);
//		}
//	}

	//read column information


	MPI_Bcast(nnzPerCol, numCol, MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
#ifdef DEBUG
/*	if(procRank==numProc-1){
		for(i=0; i< numCol;i++){
			printf("%d:NNZ in column[%d]=%d\n", procRank,i, nnzPerCol[i]);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);*/
#endif
	/*
//load balance<< Example
	int colStartIdx=0, numColPerProc=0, aveNumColPerProc=0;
	if(numProc==1){
		numColPerProc=aveNumColPerProc=numCol;
	}
	else if (numProc>1){
		aveNumColPerProc=numCol/(numProc-1);
		if(procRank!=numProc-1)
			numColPerProc=numCol/(numProc-1);
		else {
			numColPerProc=numCol-(numCol/(numProc-1))*(numProc-1);
		}
	}
	colStartIdx=aveNumColPerProc*procRank;


//	int * colStartIdxSet=(int *) malloc(numProc * sizeof(*colStartIdxSet));
//	if(colStartIdxSet)
//		memset(colStartIdxSet, 0, numProc * sizeof(*colStartIdxSet));
//	MPI_Gather(&colStartIdx, 1, MPI_INT, colStartIdxSet, 1, MPI_INT, 0, MPI_COMM_WORLD);
////#ifdef DEBUG
//	if(procRank==0)
//		for (i=0; i<numProc; i++){
//			printf("colStartIdxSet[%d]=%d\n", i, colStartIdxSet[i]);
//		}
////#endif
//load balance>>
*/

//  new load balance by En-Jui Lee
        //int colStartIdx=0, numColPerProc=0, aveNumColPerProc=0;
	
	long total_nz=0, accum_nz=0;
	double average_nzPerProc=0;
	// for storing start index & number col per core
	int *colStartIdx, *numColPerProc;
	colStartIdx = (int*)malloc(sizeof(int) * numProc);
	numColPerProc = (int*)malloc(sizeof(int) * numProc);
	
	// get total nz
	for(i=0; i<numCol; i++){
	  total_nz=total_nz+nnzPerCol[i];
        }
	average_nzPerProc=(double)total_nz/numProc;  
	printf(" total nz: %ld, average nzPerProc=%lf\n",total_nz,average_nzPerProc);

	int Proc_count=0;
	
	//assign col to cores
        if(numProc==1){
	  colStartIdx[0]=0;
	  numColPerProc[0]=numCol;
        }
        else if (numProc>1){
	  colStartIdx[0]=0;
	  // when accum_nz is closest average_nzPerProc * i, assign to proc
	  for(i=0; i<numCol; i++){
	    accum_nz=accum_nz+nnzPerCol[i];
	    if(accum_nz>(average_nzPerProc*(Proc_count+1))){
	      Proc_count++;
	      colStartIdx[Proc_count]=i;
	      numColPerProc[Proc_count-1]=colStartIdx[Proc_count]-colStartIdx[Proc_count-1];
#ifdef DEBUG
	      printf("core %d : colStartIdx=%d, numColPerProc=%d\n", Proc_count-1, colStartIdx[Proc_count-1], numColPerProc[Proc_count-1]);
	      printf("core %d : colStartIdx=%d\n", Proc_count, colStartIdx[Proc_count]);
#endif
	    }
	    //the least strat
	    if(Proc_count==(numProc-1)){
	      //numColPerProc[Proc_count]=numCol-colStartIdx[Proc_count]-1;
	      numColPerProc[Proc_count]=numCol-colStartIdx[Proc_count];
	      break;
	    }
	  }
#ifdef DEBUG
	  printf("core %d : colStartIdx=%d, numColPerProc=%d\n", Proc_count, colStartIdx[Proc_count], numColPerProc[Proc_count]);
#endif
	}	  
	//new load balance

	long nnzPerProc = 0;//number of nonzero of every process including all columns in local process
	for(i=colStartIdx[procRank]; i<colStartIdx[procRank]+numColPerProc[procRank]; i++){
		nnzPerProc+=nnzPerCol[i];
	}

#ifdef DEBUG
	printf("%d: colStartIdx=%d, numColPerProc=%d, nnzPerProc=%ld\n", procRank, colStartIdx[procRank], numColPerProc[procRank], nnzPerProc);
	MPI_Barrier(MPI_COMM_WORLD);
#endif

/* nonzero in every process, 1st dimension is COLUMN index,
 * 2nd dimension is displacement in the column */
	int ** rowRecv = (int **)calloc( numColPerProc[procRank], sizeof(int *));
	if(NULL==rowRecv){
		fprintf(stderr, "rowRecv calloc fails \n");fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
	}
	double ** valueRecv = (double **)calloc(numColPerProc[procRank], sizeof(double *));
	if(NULL==valueRecv){
		fprintf(stderr, "valueRecv calloc fails \n");fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
	}
	for(i=0; i<numColPerProc[procRank]; i++){
		rowRecv[i]=(int *)calloc(nnzPerCol[colStartIdx[procRank]+i], sizeof(int));
		if(NULL==rowRecv[i]){
			fprintf(stderr, "rowRecv[i] calloc fails \n");fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
		}
		valueRecv[i]=(double *)calloc(nnzPerCol[colStartIdx[procRank]+i], sizeof(double));
		if(NULL==valueRecv[i]){
			fprintf(stderr, "valueRecv[i] calloc fails \n");fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
		}
	}
	/*counter for every column in local process*/
	int * counter = (int *)calloc(numColPerProc[procRank], sizeof(int *));
	if(NULL==counter){
		fprintf(stderr, "counter calloc fails \n"); fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
	}
	for(i=0; i<numColPerProc[procRank]; i++){
		counter[i]=0;
	}


	/* distribute row-wise matrix to column-wise matrix, 2nd iterate every kernel file */
	MPI_Status sta1, sta2, sta3;
	MPI_Request reqs1, reqs2, reqs3, reqr1, reqr2, reqr3;


	/*Scatter variable */
	int * sendcounts;//scatter send count for every process
	int * displs;//scatter displacement for every process
	/*row, col, val array, sent from root, significant only at root*/
	int * rowIdxArray;
	int * colIdxArray;
	double * valArray;

	/*number of data received from a row file (scatter)*/
	int localRecvCount=0;
	/*local data received from a row file (scatter)*/
	int * rowIdxRecv, *colIdxRecv; double * valRecv;

	if(procRank==0){
		for (i = 0; i < numRow; i++) {//2nd iterate every kernel file
			char kernelFileFullName[MAXREADNUM];
			int nnzThisRow = NumNZPerrow_k[i];
			sprintf(kernelFileFullName, "%s/%s",dirpath, kernelFileName[i]);
			printf("%d: 2nd loading dataFileName[%d] = %s, number of nonzero per row=%d\n",
					procRank, i, kernelFileFullName, nnzThisRow);
			//FILE * fip_k=fopen(kernelFileFullName, "r");
			//add read binary files Feb 1, 2013
			FILE * fip_k=fopen(kernelFileFullName, "rb");
			time_t start,end;
			//time			time (&start);
			if(fip_k==NULL){
				fprintf(stderr, "Cannot open the file: %s  \n", kernelFileFullName);fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
			}

			sendcounts = (int *)malloc(numProc * sizeof(int));//send count for every process
			memset(sendcounts, 0, numProc * sizeof(int));
			displs = (int *)malloc(numProc * sizeof(int));// displacement in for the rowIdxArray, colIdxArray, valArray for every process
			memset(displs, 0, numProc * sizeof(int));

			rowIdxArray = (int * )malloc(nnzThisRow*sizeof(int) );
			memset(rowIdxArray, 0, nnzThisRow*sizeof(int) );
			colIdxArray = (int * )malloc(nnzThisRow*sizeof(int) );
			memset(colIdxArray, 0, nnzThisRow*sizeof(int));
			valArray = (double *)malloc(nnzThisRow*sizeof(double));
			memset(valArray, 0, nnzThisRow*sizeof(int));

//read a row of kernel file

			struct tuple * read_array = (struct tuple * )malloc(sizeof(struct tuple)*nnzThisRow);
			if(!read_array) fprintf(stderr, "cannot allocat read_array\n" );
                        memset(read_array, 0, sizeof(struct tuple)*nnzThisRow);
			fread(read_array, sizeof(struct tuple), nnzThisRow, fip_k);
			int k_record=0; //saving for faster search
			for(j=0; j<nnzThisRow; j++){
				int colIdx, iX, iY, iZ;
				double val;
				int c1=5;
				//int c1=fscanf(fip_k, "%d %d %d %d %le", &colIdx, &(iX), &(iY), &(iZ), &val);
				colIdx=read_array[j].col_idx;
				val=read_array[j].val;
				if(c1==5){
					colIdx--;//adjust to 0-based indexing

					//old load balance					int targetRank=colIdx/aveNumColPerProc;
					//old load balance					if(targetRank>=numProc)//to be delete                                                                  
					//old load balance  targetRank=numProc-1;//to be delete                                                                                                                               
					//new load balance
					int targetRank=0;
					int k;
					if(colIdx>=colStartIdx[numProc-1]){
					  targetRank=numProc-1;
					}
					else{
					  for(k=k_record; k<(numProc-1); k++){
					    if((colStartIdx[k]<=colIdx) && (colIdx<colStartIdx[k+1])){
					      targetRank=k;
					      break;
					    }
					  }
					  k_record=targetRank;
					}
					
//					int targetRank=0;
//					int k;
//					for(k=0; k<numProc-1; k++){
//						if(colStartIdxSet[k]<=colIdx && colIdx<colStartIdxSet[k+1])
//							targetRank=k;
//					}
//					if(colIdx>=colStartIdxSet[numProc-1]){
//						targetRank=numProc-1;
#ifdef DEBUG
					printf("colIdx=%d targetRank=%d k_record=%d\n",colIdx, targetRank,k_record);
#endif
//					}
//#ifdef DEBUG
//					if(colIdx==38092815)
//						printf("38092815 targetRank=%d\n", targetRank);
//#endif

					sendcounts[targetRank]++;
					rowIdxArray[j]=i;
					colIdxArray[j]=colIdx;
					valArray[j]=val;
				}
			}
			free(read_array);
			//set displs for every process
			int k;
			displs[0]=0;
			for(k=1; k<numProc; k++){
				displs[k]=displs[k-1]+sendcounts[k-1];
			}
#ifdef DEBUG
			for (k=0; k<numProc; k++){
				printf("displs[%d]=%d sendcounts[%d]=%d\n", k, displs[k], k, sendcounts[k]);
			}
#endif
			//scatter sendcounts
			MPI_Scatter(sendcounts, 1, MPI_INT, &localRecvCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
			//			printf("%d: scatter sendcounts done \n", procRank);
			rowIdxRecv = (int *)malloc(localRecvCount*sizeof(int));
			colIdxRecv = (int *)malloc(localRecvCount*sizeof(int));
			valRecv = (double *)malloc(localRecvCount*sizeof(double));
			//scatter a row of kernel
			MPI_Scatterv(rowIdxArray, sendcounts, displs, MPI_INT, rowIdxRecv, localRecvCount, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Scatterv(colIdxArray, sendcounts, displs, MPI_INT, colIdxRecv, localRecvCount, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Scatterv(valArray, sendcounts, displs, MPI_DOUBLE, valRecv, localRecvCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			//			printf("%d: scatter a row done \n", procRank);

// fill received data to local process's data
			for(k=0; k<localRecvCount; k++){
				int r=0,c=0;
				double v=0.0;
				r=rowIdxRecv[k];
				c=colIdxRecv[k];
				v=valRecv[k];
				int localIdx=c-colStartIdx[procRank];
				int * c1=&counter[localIdx];
				rowRecv[localIdx][*c1]=r;
				valueRecv[localIdx][*c1]=v;
				(*c1)++;
			}

			free(sendcounts);
			free(displs);
			free(rowIdxArray);
			free(colIdxArray);
			free(valArray);
			//time			time (&end);
			//time			double dif = difftime (end,start);
			//time			printf ("It takes %.2f seconds to read and send/recv the file.\n", dif );
			fclose(fip_k);

			free(rowIdxRecv);
			free(colIdxRecv);
			free(valRecv);
		}//end of row loop
	}

	//receive data
	else if(procRank!=0){
		for (i = 0; i < numRow; i++) {
			//scatter sendcounts
			MPI_Scatter(sendcounts, 1, MPI_INT, &localRecvCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
			rowIdxRecv = (int *)malloc(localRecvCount*sizeof(int));
			if(rowIdxRecv) memset(rowIdxRecv, 0, sizeof(*rowIdxRecv)*localRecvCount);
			colIdxRecv = (int *)malloc(localRecvCount*sizeof(int));
			if(colIdxRecv) memset(colIdxRecv, 0, sizeof(*colIdxRecv)*localRecvCount);
			valRecv = (double *)malloc(localRecvCount*sizeof(double));
			if(valRecv) memset(valRecv, 0, sizeof(*valRecv)*localRecvCount);
			//scatter a row of kernel
			MPI_Scatterv(rowIdxArray, sendcounts, displs, MPI_INT, rowIdxRecv, localRecvCount, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Scatterv(colIdxArray, sendcounts, displs, MPI_INT, colIdxRecv, localRecvCount, MPI_INT, 0, MPI_COMM_WORLD);
			MPI_Scatterv(valArray, sendcounts, displs, MPI_DOUBLE, valRecv, localRecvCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

// fill received data to local process's data
			for(k=0; k<localRecvCount; k++){
				int r=0,c=0;
				double v=0.0;
				r=rowIdxRecv[k];
				c=colIdxRecv[k];
				v=valRecv[k];
				int localIdx=c-colStartIdx[procRank];
				int * c1=&counter[localIdx];
#ifdef DEBUG
				printf("localRecvCount=%d, [%d, %d]=%lf c1=%d < nnzPerCol[%d+%d]=%d \n",localRecvCount ,r,c,v, *c1, colStartIdx[procRank], localIdx, nnzPerCol[colStartIdx[procRank] + localIdx]);
#endif
				rowRecv[localIdx][*c1]=r;
				valueRecv[localIdx][*c1]=v;
				(*c1)++;
			}
			free(rowIdxRecv);
			free(colIdxRecv);
			free(valRecv);
		}
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if(procRank==0)
	{
		printf("%d: Row file distribution completed!\n", procRank);
	}
	char fon20[MAXREADNUM], fon21[MAXREADNUM], fon3[MAXREADNUM];
	FILE * fop20, * fop21;
#ifdef ASCII
	sprintf(fon20, "%s/matrix_bycol.mm2", dirpath);//matrix ascii
#endif
	sprintf(fon21, "%s/%s.bin", dirpath, "matrix_bycol.mm2");//matrix binary
	sprintf(fon3, "%s/%s.info", dirpath, "matrix_bycol.mm2");//matrix column information

	if(procRank==0){
		FILE * fop3;
		fop3 = fopen(fon3, "w");
		if(fop3==NULL){
			fprintf(stderr, "cannot open %s\n", fon3);fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
		}
		long long colStartIndex1=1;//this should be long long
		for(i=0; i<numCol; i++){
			if(0==nnzPerCol[i])
				fprintf(fop3, "%d %d %lld\n",i+1, 0, 0);//adjust to 1-based indexing
			else
				fprintf(fop3, "%d %d %lld\n",i+1, nnzPerCol[i], colStartIndex1);//adjust to 1-based indexing
			colStartIndex1+=(long long)nnzPerCol[i];
		}
		fclose(fop3);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	/*outut transpose matrix in PLSQR3 format*/
//	for(i=0; i<numProc; i++){
		/*if(procRank==i)*/

	MPI_Status sta;
	int msg;
	if(procRank-1>=0){//receive message from rank-1 and begin to work
		MPI_Recv(&msg, 1, MPI_INT, procRank-1, 00, MPI_COMM_WORLD, &sta);
		if(msg==procRank-1){
			printf("%d: receives message from %d, and outputs results \n", procRank, msg);
		}
	}

	if(procRank==0){
	  #ifdef ASCII
		fop20=fopen(fon20, "w");
		if(fop20==NULL){
			fprintf(stderr, "cannot open %s\n", fon20);fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
		}
		#endif
		fop21=fopen(fon21, "wb");
		if(fop21==NULL){
			fprintf(stderr, "cannot open %s\n", fon21);fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
		}
	}
	else{
	  #ifdef ASCII
		fop20=fopen(fon20, "a");
		if(fop20==NULL){
			fprintf(stderr, "cannot open %s\n", fon20);fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
		}
		#endif
		fop21=fopen(fon21, "ab");
		if(fop21==NULL){
			fprintf(stderr, "cannot open %s\n", fon21);fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
		}
	}
//////////////////////////////////////////////////////////
	time_t start,end;
        time (&start);
	/*
	for(j=0; j<numColPerProc; j++){
		int gloColIdx=j+colStartIdx+1;//adjust to 1-based indexing
		int k;
		for(k=0; k<nnzPerCol[gloColIdx-1]; k++){
			int gloRowIdx=rowRecv[j][k]+1;//adjust to 1-based indexing
			double val=valueRecv[j][k];
			if(0<gloRowIdx<=numRow && 0<gloColIdx<=numCol){
			  #ifdef ASCII
//					fprintf(fop20, "%d %d %e\n", gloRowIdx, gloColIdx, val);
				fprintf(fop20, "%d: %d %d %le\n", procRank, gloRowIdx, gloColIdx, val);
				#endif

				NONZERO nz;
				nz.r=gloRowIdx; nz.c=gloColIdx; nz.val=val;
				//				fwrite(&gloRowIdx, sizeof(int), 1, fop21);
				//fwrite(&gloColIdx, sizeof(int), 1, fop21);
				//fwrite(&val, sizeof(double), 1, fop21);
				fwrite(&nz, sizeof(NONZERO),1, fop21);
			}
			else{
				printf("%s(%d)-%s: \t rank=%d: gloRowIdx=%d numRow=%d gloColIdx=%d numCol=%d \n",
						__FILE__,__LINE__,__FUNCTION__, procRank,
						gloRowIdx, numRow, gloColIdx, numCol); exit(-1);
			}
		}
	}
	*/

	//add by lee
        for(j=0; j<numColPerProc[procRank]; j++){
          int gloColIdx=j+colStartIdx[procRank]+1;//adjust to 1-based indexing                                                                                                      
          int k;
          struct tuple * write_array = (struct tuple * )malloc(sizeof(struct tuple)*nnzPerCol[gloColIdx-1]);
          for(k=0; k<nnzPerCol[gloColIdx-1]; k++){
            int gloRowIdx=rowRecv[j][k]+1;//adjust to 1-based indexing                                                                                        
            double val=valueRecv[j][k];
            if(0<gloRowIdx<=numRow && 0<gloColIdx<=numCol){
              write_array[k].iX=gloRowIdx;
              write_array[k].col_idx=gloColIdx;
              write_array[k].val=val;
            }
            else{
              printf("%s(%d)-%s: \t rank=%d: gloRowIdx=%d numRow=%d gloColIdx=%d numCol=%d \n",
                     __FILE__,__LINE__,__FUNCTION__, procRank,
                     gloRowIdx, numRow, gloColIdx, numCol); exit(-1);
            }
          }
          // write to to file                                                                                                                                                      
          fwrite(write_array, sizeof(struct tuple), nnzPerCol[gloColIdx-1], fop21);
          free(write_array);
        }
	// add by lee
	time (&end);
        double dif2 = difftime (end,start);
        printf ("It takes %.2f seconds to write binary output.\n", dif2 );

	#ifdef ASCII
	fclose(fop20);
	#endif
	fclose(fop21);

	if(procRank+1<=numProc-1){//complete job, send msg to rank+1
		MPI_Send(&procRank, 1, MPI_INT, procRank+1, 00, MPI_COMM_WORLD);
	}


	MPI_Barrier(MPI_COMM_WORLD);
	if(procRank==0)
	{
		printf("%d: All done!\n", procRank);
	}

//free memory
//	free(colStartIdxSet);
	free(nnzPerCol);
	for(i=0;i<numRow; i++){
		free(kernelFileName[i]);
	}
	free(kernelFileName);

	for(i=0; i<numColPerProc[procRank]; i++){
		free(rowRecv[i]);
		free(valueRecv[i]);
	}
	free(valueRecv);
	free(rowRecv);
	free(counter);
	free(NumNZPerrow_k);
	free(colStartIdx);
	free(numColPerProc);
	MPI_Finalize();
	return 0;
}
