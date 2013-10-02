/* (Hwang-Ho) He Huang: hhuang1@uwyo.edu
 * Department of Computer Science, University of Wyoming */


//////////////////////////////////////////////


//#define DEBUG		//also has in plsqr3.c lsqr.c, aprod.c, main.c, CPUSpMV.c, GPUSpMv.cu



/////////////////////////////////////////////

//header: implement
#include "main.h"

//header: C or system library
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <mpi.h>

//header: other project

//header: this project
#include "cblas.h"
#include "lsqr.h"
#include "FileIO.h"
#include "loadbalance.h"
#include "plsqr3.h"
#include "CPUSpMV.h"





//const int MATRIX_COLUMN=10  /* 38092800*/ ;  /*444346;*/
//const bool IOASCII = true;
//const bool IOBINARY = false;

int PLSQR2_main( int argc, char * argv[], int procRank, int numProc);
int MyGPTLpr (const int procRank, const int numProc);
int main(int argc, char* argv[]);


int main(int argc, char* argv[]) {
	int numProc = 0; // Number of available processes, rank=0 does not participate calculation!
	int procRank = 0; // Rank of current process

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numProc);
	MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

	//print host name for every process
	char hostname[1024];
	gethostname(hostname, sizeof(hostname));
	printf("rank=%d, PID=%d on %s \n", procRank, getpid(), hostname);
	fflush(stdout);
	MPI_Barrier(MPI_COMM_WORLD);

#ifdef GPUALL //setaffinity_for_nvidia
//	int gpudevice= procRank % PPN;
//	int setaffinity_for_nvidia(int ,  int , int );
//	setaffinity_for_nvidia(procRank, PPN, gpudevice);
#endif

#ifdef GPTL
	GPTLsetoption(GPTLwall, 1);
	GPTLinitialize();
	GPTLstart("total");
#endif

	if(procRank==0){
#if SDVERSION==1
		printf("------------------------Single Precision, %d MPI tasks\n", numProc);
#elif SDVERSION == 2
		printf("------------------------Double Precision, %d MPI tasks\n", numProc);
#endif
	}

	PLSQR3_main(argc, argv, procRank, numProc);

#ifdef GPTL
	GPTLstop("total");
//	GPTLpr(procRank + numProc * 1000000);
	MyGPTLpr(procRank, numProc);
#endif
	MPI_Finalize();
	return 0;
}




#ifdef GPTL
int MyGPTLpr (const int id, const int numProc)   /* output file will be named "timing.<id>" */
{
  char outfile[128];         /* name of output file: timing.xxxxxx */

  if (id < 0 || id > 999999999999)
    return GPTLerror ("GPTLpr: bad id=%d for output file. Must be >= 0 and < 1000000\n", id);

//  sprintf (outfile, "timing.%d", id);
// modified by HH<<
  char dir[256];
#ifdef PLSQR2
  sprintf(dir, "v2NP%d", numProc);
#else
  sprintf(dir, "v3NP%d", numProc);
#endif
  char cmd[256];
  sprintf(cmd, "mkdir %s", dir);

  if(id==0)//only proc==0 make direcotry
	system(cmd);
  sprintf(outfile, "%s/timing.%d", dir, id);
//>>
  MPI_Barrier(MPI_COMM_WORLD);

  if (GPTLpr_file (outfile) != 0)
    return GPTLerror ("GPTLpr: Error in GPTLpr_file\n");

  return 0;
}
#endif

int PLSQR2_main( int argc, char * argv[], int procRank, int numProc)
{
	if(procRank==0){


#if GPUALL ==1
		printf("------------------------PLSQR2 GPU version---------------------------\n");
	#if TRANSPOSE_AVOID== 1
			printf("\t TRANSPOSE_AVOID enabled\n");
	#endif
#else
		printf("------------------------PLSQR2 CPU version---------------------------\n");
#endif



#ifdef USINGDAMPING
		printf("Using damping---------------------------\n\n\n");
#else
		printf("Without Using damping---------------------------\n\n\n");
#endif
	}

	//print host name for every process
	char hostname[1024];
	gethostname(hostname, sizeof(hostname));
#ifdef DEBUG
	printf("rank=%d, PID=%d on %s \n", procRank, getpid(), hostname);
	fflush(stdout);
#endif

#ifdef DEBUG
	MPI_Barrier(MPI_COMM_WORLD);
#endif

#if GPUALL==1
	initCUDADevice(procRank);
#endif



	char configFilename[MAXREADNUM], dirpath_k[MAXREADNUM],
			asciiDataStatistic[MAXREADNUM], binaryDataStatistic[MAXREADNUM],
			vectorBInfo[MAXREADNUM];
	char damping_path[MAXREADNUM], damping_info[MAXREADNUM];
	int lsqr_iteration, numColumn;
	IOTYPE inputType, isParallel, isLoadBalance;
	if (argc == 2) {
		strcpy(configFilename, argv[1]);
	}
	
	readConfig(configFilename, dirpath_k, &inputType, &lsqr_iteration,
			&numColumn, asciiDataStatistic, binaryDataStatistic, vectorBInfo,
			isParallel, isLoadBalance, damping_path, damping_info);
	if (procRank == 0) {
		printConfigParameter(procRank, configFilename, dirpath_k, inputType,
				&lsqr_iteration, asciiDataStatistic, binaryDataStatistic,
				vectorBInfo, isParallel, isLoadBalance);
		/*	printf("%d: dirpath=%s, inputType=%s, lsqr_iteration=%d, asciiDataStatistic=%s, binaryDataStatistic=%s, vectorBInfo=%s \n",
		 procRank, dirpath,(inputType) ? "true" : "false", *lsqr_iteration, asciiDataStatistic, binaryDataStatistic, vectorBInfo);
		 */
		printf("%d: damping_path=%s, damping_info=%s \n", procRank,
				damping_path, damping_info);
	}

	//	bool inputType=IOBINARY;// true if input files are ascii format, false if input files are binary format.
	char statisticFileName[1024];//="DataStatistics.txt";
	if (inputType == IOASCII)
		strcpy(statisticFileName, asciiDataStatistic);
	else if (inputType == IOBINARY)
		strcpy(statisticFileName, binaryDataStatistic);

	int numKernelFile;

	char filename[1024];
	sprintf(filename, "%s/%s", dirpath_k, statisticFileName);
	if (procRank == 0)
		printf("filename=%s\n", filename);

	if (procRank == 0) {
		numKernelFile = countData(filename);
		if (procRank == 0)
			if (numProc /*- 1*/ > numKernelFile) {
				printf("warning: the number of processes is greater than the number of data file\n");
				exit(EXIT_FAILURE);
			}
	}
	MPI_Bcast(&numKernelFile, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (procRank == numProc - 1)
		printf("%d: The number of data file is %d\n", procRank, numKernelFile);
	// read data file names
	//allocate memeory<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	//indicates every data file's name
	char **kernelDataFileName;
	if (!(kernelDataFileName = (char **) calloc(numKernelFile, sizeof(char *))))
		printf("kernelDataFileName calloc fails\n");
	int i;
	for (i = 0; i < numKernelFile; i++)
		if (!(kernelDataFileName[i] = (char *) calloc(MAXREADNUM, sizeof(char))))// the lenglth of every data file name should not be bigger than 256 char
			printf("kernelDataFileName malloc fails\n");
	//indicates the number of column of each data file.
	int * dataFileNumNonzeroPerrow_k = (int *) calloc(numKernelFile,sizeof(int));
	if(dataFileNumNonzeroPerrow_k==NULL){
		fprintf(stderr, "dataFileNumNonzeroPerrow_k malloc\n");exit(-1);
	}
	for (i = 0; i < numKernelFile; i++) {
		dataFileNumNonzeroPerrow_k[i] = 0;
	}
	//allocate memeory>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	//	::MPI_Barrier(MPI_COMM_WORLD);
	//	if(procRank==0)
	getDataFileName(kernelDataFileName, dataFileNumNonzeroPerrow_k, filename,
			numKernelFile);
	//	MPI_Bcast(dataFileName, numOfDataFile*MAXREADNUM, MPI_CHAR, 0, MPI_COMM_WORLD);
	//	MPI_Bcast(dataFileNumOfColumn, numOfDataFile, MPI_INT, 0, MPI_COMM_WORLD);
	//	MPI_Barrier(MPI_COMM_WORLD);

	//	display kernel data file information
#ifdef DEBUG
	if (procRank == 0)
		for (i = 0; i < numKernelFile; i++) {
			printf("%d: dataFileName[%d]=%s, number of nonzero per row=%d\n",
					procRank, i, kernelDataFileName[i],
					dataFileNumNonzeroPerrow_k[i]);
		}
#endif

	//3/10	process damping matrix(((((((((((((((((((((((
#ifdef USINGDAMPING
	char damping_info_fullname[1024];
	sprintf(damping_info_fullname, "%s/%s", damping_path, damping_info);
	if (0 == procRank) {
		printf("%d processing damping matrix...\ndamping_info_fullname=%s\n",
				procRank, damping_info_fullname);
	}

	int numDamping = numProc ;

	char ** dampingFileName = (char **) calloc(numDamping, sizeof(char *));
	if(dampingFileName==NULL){
		fprintf(stderr, "dampingFileName calloc fails\n");exit(-1);
	}

	for (i = 0; i < numDamping; i++) {
		dampingFileName[i] = (char *) calloc(1024, sizeof(char));
		if(dampingFileName[i]==NULL){
			fprintf(stderr, "dampingFileName[] calloc fails\n");exit(-1);
		}
	}
	int * numRowPerDampingFile = (int *) calloc(numDamping, sizeof(int));
	if(numRowPerDampingFile==NULL){
		fprintf(stderr, "numRowPerDampingFile calloc fails\n");exit(-1);
	}
	int * numNonzeroPerDampingFile = (int *) calloc(numDamping, sizeof(int));
	if(numNonzeroPerDampingFile==NULL){
		fprintf(stderr, "numNonzeroPerDampingFile calloc fails\n");exit(-1);
	}
	readDampingInfo(damping_info_fullname, numDamping, dampingFileName,
			numRowPerDampingFile, numNonzeroPerDampingFile);

	int damping_totalNumRow = 0;
	for (i = 0; i < numDamping; i++) {
		damping_totalNumRow += numRowPerDampingFile[i];
	}
	//	MPI_Barrier(MPI_COMM_WORLD);

	if (0 == procRank) {
		for (i = 0; i < numDamping; i++) {
			printf(	"%d, dampingFileName=%s, dampingFileNumRow=%d, dampingFileNumNonzero=%d\n",
					i, dampingFileName[i], numRowPerDampingFile[i],
					numNonzeroPerDampingFile[i]);
		}
		printf("damping_numRow=%d\n end of processing damping matrix...\n\n\n",
				damping_totalNumRow);
	}
#endif
	//	process damping matrix)))))))))))))))))))))))

	///////////////////////////////////////////////

#ifdef USINGDAMPING
	int whole_matrix_NumRow = numKernelFile+damping_totalNumRow;//^-^!!! using damping
	if(procRank==0){
		printf("whole matrix(including damping) number of row is %d \n", whole_matrix_NumRow);
	}
#else
	int whole_matrix_NumRow = numKernelFile;
#endif
	int whole_matrix_numCol = numColumn;

	//deal with the vector((((
	int vectorB_kernel_dim = numKernelFile;
	//this is the original vectorB read from the input
	MY_FLOAT_TYPE * oriVectorB = (MY_FLOAT_TYPE*) calloc(vectorB_kernel_dim, sizeof(MY_FLOAT_TYPE));
	if(oriVectorB==NULL){
		fprintf(stderr, "oriVectorB calloc fails\n");exit(-1);
	}
	//this vecotrB's order is reorganized according to the order of input data file scheduled by load balancing algorithm.
	MY_FLOAT_TYPE* reordered_vectorB = (MY_FLOAT_TYPE*) calloc(vectorB_kernel_dim,	sizeof(MY_FLOAT_TYPE));
	if(NULL==reordered_vectorB){
		fprintf(stderr, "reordered_vectorB calloc fails\n");exit(-1);
	}

	//3/28/10 only load measurement vector(corresponding to kernel)
	if (procRank == 0) {
		//the dimension of vector B should equal to the number of rows of matrix
		if (strcmp(vectorBInfo, "NULL") == 0) { //use randomly generated vectors
			printf("\n\n\nNo vecotr is specified, generate random vector ......\n\n\n");
			generateRandomVector(oriVectorB, vectorB_kernel_dim);
		} else { //read vector from ascii file
			readInputVector(vectorBInfo, oriVectorB, vectorB_kernel_dim);
		}
	}
	//deal with the vector))))

	//	12/2009 Load Balancing	<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
	int * numRowPerProc_k = (int *) calloc(numProc,	sizeof(int));
	if(NULL==numRowPerProc_k){
		fprintf(stderr, "numRowPerProc_k calloc fails\n");exit(-1);
	}
	int * numNZ_KerFilesetPerPorc_k = (int *) calloc(numProc,sizeof(int));
	if(NULL==numNZ_KerFilesetPerPorc_k){
		fprintf(stderr, "numNZ_KerFilesetPerPorc_k calloc fails\n");exit(-1);
	}
	int ** KerFilesetPerProc_k;
	if (procRank == 0) {
		//		procDistributedDataFileSet:  every process stores its distributed data files
		//		by the index of dataFileName
		KerFilesetPerProc_k = (int **) calloc(numProc, sizeof(int*));
		if(NULL==KerFilesetPerProc_k){
			fprintf(stderr, "KerFilesetPerProc_k calloc fails\n");exit(-1);
		}
		int k = 0;
		for (k = 0; k < numProc; k++) {
			KerFilesetPerProc_k[k] = (int *) calloc(numKernelFile,sizeof(int));
			if(NULL==KerFilesetPerProc_k[k]){
				fprintf(stderr, "KerFilesetPerProc_k[k] calloc fails\n");exit(-1);
			}
		}

		//	count how many nonzero entity that the corresponding process has

		for (k = 0; k < numProc; k++) {
			numRowPerProc_k[k] = 0;
			numNZ_KerFilesetPerPorc_k[k] = 0;
		}
		// initialize procDistributedDataFileSet to 0
		loadBalanceByNonzeroEntity(numProc, numKernelFile, kernelDataFileName,
				dataFileNumNonzeroPerrow_k, KerFilesetPerProc_k,
				numRowPerProc_k,
				numNZ_KerFilesetPerPorc_k);

		//convert the original vectorB to the re-ordered vectorB
		reorderVectorB(numProc, reordered_vectorB, oriVectorB,
				vectorB_kernel_dim, numRowPerProc_k,
				KerFilesetPerProc_k);


		//FREE MEMEORY
	}

	free(oriVectorB);

	MPI_Bcast(numRowPerProc_k, numProc, MPI_INT, 0,
			MPI_COMM_WORLD);
	MPI_Bcast(numNZ_KerFilesetPerPorc_k, numProc, MPI_INT, 0,
			MPI_COMM_WORLD);

	int * perProcDisDataFileSet_k = (int *) calloc(numKernelFile, sizeof(int));
	if(NULL==perProcDisDataFileSet_k){
		fprintf(stderr, "perProcDisDataFileSet_k calloc fails\n");exit(-1);
	}
	int perProcSizeOfDisDataFileSet_k = 0;
	int numNonzero_perProcDisDataFileSet_k = 0;
	MPI_Status status;

	if (procRank == 0) {
		int k;
		for (k = /*0*/1; k < numProc; k++)//rank =0 does not send to itself
		{
			MPI_Send(KerFilesetPerProc_k[k], numKernelFile, MPI_INT,
					k, 0, MPI_COMM_WORLD);//by procDistributedDataFileSet[i]
			//			MPI_Send(&sizeOfprocDistributedDataFileSet_k[k], 1, MPI_INT, k, 1, MPI_COMM_WORLD); //num
			//			MPI_Send(&numNonzero_perProcDistributedDataFileSet_k[k], 1, MPI_INT, k, 2, MPI_COMM_WORLD);
		}
		for (k = 0; k < numKernelFile; k++)
			perProcDisDataFileSet_k[k] = KerFilesetPerProc_k[0][k];

		for (k = 0; k < numProc; k++) {
			free(KerFilesetPerProc_k[k]);
		}
		free(KerFilesetPerProc_k);
	} else if (procRank != 0)
		MPI_Recv(perProcDisDataFileSet_k, numKernelFile, MPI_INT, 0, 0,
				MPI_COMM_WORLD, &status);

	perProcSizeOfDisDataFileSet_k= numRowPerProc_k[procRank];
	numNonzero_perProcDisDataFileSet_k= numNZ_KerFilesetPerPorc_k[procRank];
	////print result<<<
#ifdef DEBUG
	printf("%d: perProcSizeOfDisDataFileSet_k=%d, numNonzero_perProcDisDataFileSet_k=%d \n",
			procRank, perProcSizeOfDisDataFileSet_k,numNonzero_perProcDisDataFileSet_k);
	printf("%d: kernel file set:", procRank);
	int ii;
	for (ii = 0; ii < perProcSizeOfDisDataFileSet_k; ii++) {
		printf("%d\t", perProcDisDataFileSet_k[ii]);
	}
	printf("\n");
	////print result>>>
#endif


	free(numNZ_KerFilesetPerPorc_k);

	MPI_Barrier(MPI_COMM_WORLD);
	//	12/2009 END Load Balancing	>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

	if (procRank == 0)
		printf("\n------------------finish load balancing,----------------------------------\n");

	//broadcast the vector
	MPI_Bcast(reordered_vectorB, vectorB_kernel_dim, MY_MPI_FLOAT_TYPE, 0,
			MPI_COMM_WORLD);
	//END: deal with vector

	/* 3/28/10 partition the vector */
	int perProcVectorB_dim;//vector dimension
#ifdef USINGDAMPING
	perProcVectorB_dim = numRowPerProc_k[procRank]+ numRowPerDampingFile[procRank];
#else
	perProcVectorB_dim = numRowPerProc_k[procRank];
#endif

	MY_FLOAT_TYPE * perProcVectorB;

#if CUDA==1 && CUDAHOSTALLOC==1
	pinnedMemAlloc((void **)&perProcVectorB, perProcVectorB_dim, Y_D);
#else
//	size_t nbyte_B = ((perProcVectorB_dim*sizeof(MY_FLOAT_TYPE)+PAGESIZE-1)/PAGESIZE)*PAGESIZE;
	perProcVectorB =(MY_FLOAT_TYPE *) valloc(perProcVectorB_dim*sizeof(MY_FLOAT_TYPE)/*nbyte_B*/);
	if(NULL==perProcVectorB){
		fprintf(stderr, "perProcVectorB valloc fails\n");exit(-1);
	}
#endif

#ifdef USINGDAMPING
	kernelVectorPartition_perProcVectorConstruct(procRank, reordered_vectorB,
			perProcVectorB, numRowPerProc_k,numRowPerDampingFile[procRank]);
#else
	kernelVectorPartition_perProcVectorConstruct(procRank, reordered_vectorB,
				perProcVectorB, numRowPerProc_k,0);
#endif

	free(reordered_vectorB);

	/* 3/25/10: load kernel and damping data directly into CSR */
	SpMatCSR per_proc_sparseMatrixCSR_whole;
	int numOfRows_k = perProcSizeOfDisDataFileSet_k;
	int numOfColumns_k = whole_matrix_numCol;
	int numOfNonzero_k = numNonzero_perProcDisDataFileSet_k;
	int dampingRowBeginIdx = perProcSizeOfDisDataFileSet_k;


#ifdef USINGDAMPING
	int dampingFileID = procRank;
	int numOfRows_d = numRowPerDampingFile[dampingFileID],
			numOfColumns_d = whole_matrix_numCol;
	int numOfNonZeroEntries_d = numNonzeroPerDampingFile[dampingFileID];

	int matrix_perProc_numRow = numOfRows_k + numOfRows_d; /*^-^!!!damping*/
#else
	int matrix_perProc_numRow = numOfRows_k ;
#endif

	int	matrix_perProc_numCol = whole_matrix_numCol;

#ifdef USINGDAMPING
	int matrix_perProc_numNonzero = numOfNonzero_k + numOfNonZeroEntries_d; /*^-^!!!damping*/
#else
	int matrix_perProc_numNonzero = numOfNonzero_k ;
#endif

	initSpMatCSR(&per_proc_sparseMatrixCSR_whole, matrix_perProc_numRow,
			matrix_perProc_numCol, matrix_perProc_numNonzero);

	if (procRank == numProc - 1) {
		printf("%d:--------------------------Data loading (kernel and damping) in all processes starts...-------------------------------\n",procRank);
	}
#ifdef GPTL
	GPTLstart("data loading");
#endif
	if (inputType == IOASCII)
	{
#ifdef USINGDAMPING//adjust whether damping matrix's index is 0-based or 1-based
		loadMatrixCSR_ascii(procRank, &per_proc_sparseMatrixCSR_whole,
		/*kernel*/
		dirpath_k,//char dirpath_k[],
		kernelDataFileName, //char * kernelDataFileName[],
		dataFileNumNonzeroPerrow_k,//int  dataFileNumNonzeroPerrow_k[],
		perProcDisDataFileSet_k,//int perProcDisDataFileSet_k[],
		perProcSizeOfDisDataFileSet_k,//int perProcSizeOfDisDataFileSet_k,
		/*damping*/
		damping_path,//char damping_path[],
		dampingFileName,//char * dampingFileName[],
		dampingFileID//int dampingFileID)
		);
#else
		loadMatrixCSR_ascii_kernelonly(&per_proc_sparseMatrixCSR_whole,
				/*kernel*/
				dirpath_k,//char dirpath_k[],
				kernelDataFileName, //char * kernelDataFileName[],
				dataFileNumNonzeroPerrow_k,//int  dataFileNumNonzeroPerrow_k[],
				perProcDisDataFileSet_k,//int perProcDisDataFileSet_k[],
				perProcSizeOfDisDataFileSet_k);
#endif
	}
/*	else if (inputType == IOBINARY) {
		loadMatrixCSR_bin(&per_proc_sparseMatrixCSR_whole,
		kernel
		dirpath_k,//char dirpath_k[],
				kernelDataFileName, //char * kernelDataFileName[],
				dataFileNumNonzeroPerrow_k,//int  dataFileNumNonzeroPerrow_k[],
				perProcDisDataFileSet_k,//int perProcDisDataFileSet_k[],
				perProcSizeOfDisDataFileSet_k,//int perProcSizeOfDisDataFileSet_k,
				damping
				damping_path,//char damping_path[],
				dampingFileName,//char * dampingFileName[],
				dampingFileID//int dampingFileID)
		);
	}*/

	//DO NOT DELETE
#ifdef DEBUG
	printSparseMatrixCSRtoFile(procRank, &per_proc_sparseMatrixCSR_whole);
/*do not forget to check vector's value*/
#endif


	MPI_Barrier(MPI_COMM_WORLD);
#ifdef GPTL
	GPTLstop("data loading");
#endif
	if (procRank == 0)
		printf("%d:--------------------Data loading in all processes completed!------------------whole_matrix_numCol=%d\n",
				procRank, whole_matrix_numCol);


	MY_FLOAT_TYPE * v;
#if CUDA==1 && CUDAHOSTALLOC==1//pinned memory
	pinnedMemAlloc((void**)&v, whole_matrix_numCol, X_D);
#else
	size_t nbyte_v = (( whole_matrix_numCol*sizeof(MY_FLOAT_TYPE)+PAGESIZE-1)/PAGESIZE)*PAGESIZE;
	v = (MY_FLOAT_TYPE*) valloc(/*whole_matrix_numCol*sizeof(MY_FLOAT_TYPE)*/nbyte_v);
	if(NULL==v){
		fprintf(stderr, "v valloc fails\n");exit(-1);
	}
#endif
	MY_FLOAT_TYPE* w = (MY_FLOAT_TYPE*) valloc(whole_matrix_numCol * sizeof(MY_FLOAT_TYPE));
	MY_FLOAT_TYPE* x = (MY_FLOAT_TYPE*) valloc(whole_matrix_numCol * sizeof(MY_FLOAT_TYPE));

	if(NULL==w){
		fprintf(stderr, "w calloc fails\n");exit(-1);
	}
	if(NULL==x){
		fprintf(stderr, "x calloc fails\n");exit(-1);
	}
	for (i = 0; i < whole_matrix_numCol; i++) {
		v[i] = 0.0;
		w[i] = 0.0;
		x[i] = 0.0;
	}

	Per_Proc_Data_v2 perProcData;
#ifdef USINGDAMPING
	init_Per_Proc_Data_v2(numProc, procRank, &perProcData, whole_matrix_NumRow,
			whole_matrix_numCol, &per_proc_sparseMatrixCSR_whole,
			numRowPerProc_k, numRowPerDampingFile);
#else
	init_Per_Proc_Data_v2(numProc, procRank, &perProcData, whole_matrix_NumRow,
				whole_matrix_numCol, &per_proc_sparseMatrixCSR_whole,
				numRowPerProc_k, NULL);
#endif


//	 printSparseMatrixCSRtoFile(procRank, perProcData.matrix_block);

	//	print_Per_Proc_Data_BeforeCaculationtoFile( procRank,  numProc, &perProcData);
	//	print_vector(perProcVectorB, perProcVectorB_dim,"vector_rank_0");
	//perProcData must be ready/////////////////////////////////////////////////////////////////////////////////////
	MPI_Barrier(MPI_COMM_WORLD);
	//to this point, A and TA is distributed among processors and
	//ready to call plsqr algorithm
	call_lsqr(perProcData.version, procRank, numProc, whole_matrix_NumRow, whole_matrix_numCol,
			&perProcData, perProcVectorB, v, w, x, lsqr_iteration, NULL);
	//	print_Per_Proc_Data_AfterCaculationtoFile( procRank,  numProc, &perProcData);
	//	MPI_Barrier(MPI_COMM_WORLD);


	//-----------------------------free memory--------------------
	//damping related
#ifdef USINGDAMPING
	for (i = 0; i < numDamping; i++) {
		free(dampingFileName[i]);
	}
	free(dampingFileName);
	free(numRowPerDampingFile);
	free(numNonzeroPerDampingFile);
#endif
	//-------------------
	free(numRowPerProc_k);
	free(perProcDisDataFileSet_k);

	finalizeSpMatCSR(&per_proc_sparseMatrixCSR_whole);
	finialize_Per_Proc_Data_v2(procRank, numProc, &perProcData);
#if CUDA==1 && CUDAHOSTALLOC==1 //pinned memory
	pinnedMemFree(v);
	pinnedMemFree(perProcVectorB);
#else
	free(v);
	free(perProcVectorB);
#endif
	free(w);
	free(x);
	//	free(oriVectorB);
	//	free(reordered_vectorB);

	////////////////
	for (i = 0; i < numKernelFile; i++) {
		if (kernelDataFileName[i] != NULL)
			free(kernelDataFileName[i]);
	}
	if (kernelDataFileName != NULL)
		free(kernelDataFileName);
	free(dataFileNumNonzeroPerrow_k);
	///////////////

	return 0;
}



#ifdef USEPAPI
/*static*/ void test_fail(char *file, int line, char *call, int retval)
{

    printf("%s\tFAILED\nLine # %d\n", file, line);
    if ( retval == PAPI_ESYS ) {
        char buf[128];
        memset( buf, '\0', sizeof(buf) );
        sprintf(buf, "System error in %s:", call );
        perror(buf);
    }
    else if ( retval > 0 ) {
        printf("Error calculating: %s\n", call );
    }
    else {
        char errstring[PAPI_MAX_STR_LEN];
        //PAPI_perror(retval, errstring, PAPI_MAX_STR_LEN );//for papi 4.x
	PAPI_perror("");//for papi 5.x
        printf("Error in %s: %s\n", call, errstring );
    }
    printf("\n");
    exit(1);
}
#endif

//3/25/2010
//!!!DO NOT DELETE!!!
//------process kernle and damping seperately in basic sparse matrix, then merge to whole one, finally convert to CSR-----------------------------


//if (procRank != 0){//rank=0 does not participate computation, so there are totally has numOfProc-1 processes participate computation
//
//	//END OF caculate rowIndex
//
///*    	int first=0, last=0,//first is the first row in this process, last is the last row of this process
//			i=0;
//
//	for(i=1; i<procRank; i++){
//		first+=sizeOfprocDistributedDataFileSet[i];
//	}
//	last=first+perProcSizeOfDisDataFileSet-1;*/
//
////		loading kernel data from files into basic matrix, and then convert basic matrix to CSR matrix
//	int numOfRows_k=perProcSizeOfDisDataFileSet_k;/*last-first+1;*/
//	int numOfColumns_k=matrix_cols;
//	int numOfNonzero_k=numNonzero_perProcDisDataFileSet_k;
//
//	printf("%d: \tnumOfNonZeroEntries=%d\n",procRank,  /*first, last, */numOfNonzero_k);
//
//	BasicSparseMatrix per_proc_basicSparseMatrix_kernel;
//	init_BasicSparseMatrix(&per_proc_basicSparseMatrix_kernel,numOfRows_k, numOfColumns_k, numOfNonzero_k/*, first, last*/);
//
////load row file into matrix
////    	int dampingRowBeginIdx=0;
//	for(i=0; i<perProcSizeOfDisDataFileSet_k; i++){
//		int filenameIdx=perProcDisDataFileSet_k[i];
//
//		char filename[1024];//../data/
//		sprintf(filename, "%s/%s", dirpath_k,kernelDataFileName[filenameIdx]);
//		//load_csr_matrix(perProcData->matrix_block, filename, i);
//
//		if(inputType==IOASCII)
//			loadBasicSparseMatrixInGrid(&per_proc_basicSparseMatrix_kernel,filename, dataFileNumNonzeroPerrow_k[filenameIdx], /*first+i*/i);
//		else if(inputType==IOBINARY)
//			loadBasicSparseMatrixInGridBin(&per_proc_basicSparseMatrix_kernel,filename, dataFileNumNonzeroPerrow_k[filenameIdx], /*first+i*/i);
//
//		// printf("%d:dataFile[%d]=%s\n",procRank,i,filename);
//
//	}
//
//
//	int dampingRowBeginIdx=perProcSizeOfDisDataFileSet_k;
//
//	//3/11/1010 load damping matrix((((((((
//	BasicSparseMatrix per_proc_basicSparseMatrix_damping;
//	int dampingFileID=procRank-1;
//  	int numOfRows_d=numRowPerDampingFile[dampingFileID],
//  			numOfColumns_d=matrix_cols;
////      			first_d=0, last_d=numOfRows_d;
//  	int numOfNonZeroEntries_d=numNonzeroPerDampingFile[dampingFileID];
//	init_BasicSparseMatrix(&per_proc_basicSparseMatrix_damping,
//			 numOfRows_d,  numOfColumns_d,  numOfNonZeroEntries_d/*,  first_d,  last_d*/);
//	char dampingFileName1[1024];
//	sprintf(dampingFileName1,"%s/%s", damping_path, dampingFileName[dampingFileID]);
//	//3/11
//	printf("%d:------------------------------------dampingRowBeginIdx=%d---------------------------\n",procRank, dampingRowBeginIdx);
//	loadDamping(dampingFileName1, &per_proc_basicSparseMatrix_damping, dampingRowBeginIdx);
//
//	BasicSparseMatrix  per_proc_basicSparseMatrix_whole;
//
//	init_merge_BasicSparseMatrixWithDamping(procRank, &per_proc_basicSparseMatrix_kernel, &per_proc_basicSparseMatrix_damping,
//			&per_proc_basicSparseMatrix_whole);
//	// print BasicSparseMatrixInGrid into files
////		printBasicSparseMatrixtoFile(procRank, &per_proc_basicSparseMatrix_whole/*&per_proc_basicSparseMatrix_kernel*//*&per_proc_basicSparseMatrix_damping*/);
//
//	finalizeBasicSparseMatrix(&per_proc_basicSparseMatrix_kernel);
//	finalizeBasicSparseMatrix(&per_proc_basicSparseMatrix_damping);
//	//3/11/1010 load damping matrix))))))))
//
//	init_SparseMatrixCSR_withBasicSparseMatrix(&per_proc_sparseMatrixCSR_whole, &per_proc_basicSparseMatrix_whole);
//	convertToSparseMatrixCSR(&per_proc_sparseMatrixCSR_whole, &per_proc_basicSparseMatrix_whole);
//	finalizeBasicSparseMatrix(&per_proc_basicSparseMatrix_whole);
//}

