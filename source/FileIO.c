#include "FileIO.h"


//#define DEBUG
/*
 * IO and utility
 * */
//#define HALF_WARP 16

/**
 * char filename[], IN: dammping info file path
 * int numDampingFile, IN: num. of damping files
 * char * OUT: dampingFileName[],
 * int * OUT: dampingFileNumRow,
 * int * OUT: dampingFileNumNonzero
 */
void readDampingInfo(char filename[], int numDampingFile,
		char * dampingFileName[], int dampingFileNumRow[],
		int dampingFileNumNonzero[]) {
	FILE * fp=fopen(filename, "r");
	if (fp==NULL) {
		fprintf(stderr, "Cannot open the file: %s  \n", filename);
		exit(-1);
	}
	int i;
	for (i=0; i< numDampingFile; i++) {
		if (feof(fp)==0) {
			fscanf(fp, "%s %d %d", dampingFileName[i], &dampingFileNumRow[i],
					&dampingFileNumNonzero[i]);


		} else {
			fprintf(stderr, "numDampingFile does not match the file!\n");
			break;
		}
	}
	fclose(fp);
}
/*

 void testreadDampingInfo()
 {
 int i;
 //////////////////////////////////////////////
 int numDamping=12;

 char ** dampingFileName=(char **)calloc(numDamping, sizeof(char *));

 for(i=0;i<numDamping; i++){
 dampingFileName[i]=(char *)calloc(1024, sizeof(char));
 }
 int * numRowPerDampingFile=(int *)calloc(numDamping, sizeof(int));
 int * numNonzeroPerDampingFile=(int *)calloc(numDamping, sizeof(int));
 readDampingInfo("damp_CN_part.Dampinfo", numDamping, dampingFileName,
 numRowPerDampingFile, numNonzeroPerDampingFile);



 int damping_numRow=0;
 for(i=0; i<numDamping; i++){
 damping_numRow+=numRowPerDampingFile[i];
 }

 //print
 for(i=0; i<numDamping; i++){
 printf("%d, dampingFileName=%s, dampingFileNumRow=%d, dampingFileNumNonzero=%d\n",
 i, dampingFileName[i], numRowPerDampingFile[i], numNonzeroPerDampingFile[i]);
 }
 printf("damping_numRow=%d\n", damping_numRow);
 //print

 for(i=0; i<numDamping; i++){
 free(dampingFileName[i]);
 }
 free(dampingFileName);
 free(numRowPerDampingFile);
 free(numNonzeroPerDampingFile);

 }
 */

/* 3/8/2010
 * merge kernel matrix with the damping matrix in BasicSpMat format,
 * the merged matrix is stored in basicSparseMatrix_whole
 * do not need to initialize the BasicSpMat * basicSparseMatrix_whole, this function already do this,
 * but need to finalize basicSparseMatrix_whole explicitly
 * */
void init_merge_BasicSpMatWithDamping(int procRank,
		BasicSpMat * basicSparseMatrix_kernel,
		BasicSpMat * basicSparseMatrix_damping,
		BasicSpMat * basicSparseMatrix_whole) {
	//Initialize
	basicSparseMatrix_whole->numRow=basicSparseMatrix_kernel->numRow
			+ basicSparseMatrix_damping->numRow;
	basicSparseMatrix_whole->numCol
			=basicSparseMatrix_kernel->numCol;

	/*
	 printf("%d: basicSparseMatrix_kernel->numOfRows=%d, basicSparseMatrix_kernel->numOfColumns=%d, basicSparseMatrix_kernel->numOfNonZeroEntries=%d \n",
	 procRank, basicSparseMatrix_kernel->numOfRows, basicSparseMatrix_kernel->numOfColumns, basicSparseMatrix_kernel->numOfNonZeroEntries);

	 printf("%d: basicSparseMatrix_damping->numOfRows=%d, basicSparseMatrix_damping->numOfColumns=%d, basicSparseMatrix_damping->numOfNonZeroEntries=%d \n",
	 procRank, basicSparseMatrix_damping->numOfRows, basicSparseMatrix_damping->numOfColumns, basicSparseMatrix_damping->numOfNonZeroEntries);
	 */
	basicSparseMatrix_whole->numNonzero
			=basicSparseMatrix_kernel->numNonzero
					+ basicSparseMatrix_damping->numNonzero;

	basicSparseMatrix_whole->nonzeroEntry=(MatNonzeroEntry *) calloc(
			basicSparseMatrix_whole->numNonzero,
			sizeof(MatNonzeroEntry));
	if(NULL==basicSparseMatrix_whole->nonzeroEntry){
		fprintf(stderr, "basicSparseMatrix_whole->nonzeroEntry calloc fails \n");exit(-1);
	}
	basicSparseMatrix_whole->count=0;
	/*
	 printf("%d: basicSparseMatrix_whole->numOfRows=%d, basicSparseMatrix_whole->numOfColumns=%d, basicSparseMatrix_whole->numOfNonZeroEntries=%d \n",
	 procRank, basicSparseMatrix_whole->numOfRows, basicSparseMatrix_whole->numOfColumns, basicSparseMatrix_whole->numOfNonZeroEntries);
	 */

	//copy data from kernel matrix
	int i;

	for (i=0; i<basicSparseMatrix_kernel->numNonzero; i++) {
		MatNonzeroEntry * whole =
				&(basicSparseMatrix_whole->nonzeroEntry[i]);
		MatNonzeroEntry * kernel=
				&(basicSparseMatrix_kernel->nonzeroEntry[i]);
		whole->rowIndex=kernel->rowIndex;
		whole->columnIndex=kernel->columnIndex;
		whole->value=kernel->value;
	}
	//	copy data from damping matrix
	for (i=basicSparseMatrix_kernel->numNonzero; (i
			-basicSparseMatrix_kernel->numNonzero)
			<basicSparseMatrix_damping->numNonzero; i++) {
		MatNonzeroEntry * whole =
				&(basicSparseMatrix_whole->nonzeroEntry[i]);
		MatNonzeroEntry
				* damping=
						&(basicSparseMatrix_damping->nonzeroEntry[i-basicSparseMatrix_kernel->numNonzero]);
		whole->rowIndex=damping->rowIndex;
		whole->columnIndex=damping->columnIndex;
		whole->value=damping->value;
	}
}

/* 3/7/2010
 * load damping matrix into basic sparse matrix format
 * */
void loadDamping(const char dampingFileName[],
		BasicSpMat * basicSparseMatrix_damping, int rowBeginIdx) {
	FILE* fp1=fopen(dampingFileName, "r");
	if (fp1==NULL) {
		fprintf(stderr, "Can not open the file: %s \n", dampingFileName);
		exit(EXIT_FAILURE);
	}

	int * counter=&(basicSparseMatrix_damping->count);

	int no_nz,//number of nonzeros
			rowIdx,//row index
			colIdx;
	MY_FLOAT_TYPE val;
	int jj=rowBeginIdx;
	while (feof(fp1)==0) {
		int c1= fscanf(fp1, "%d %d", &no_nz, &rowIdx);

		if (c1!=2) {
			printf("c1!=2\t %d\n", jj-rowBeginIdx);
			break;
		}

		int ii;
		for (ii=0; ii<no_nz; ii++) {
			int c2= fscanf(fp1, "%d %e", &colIdx, &val);
			/*			if(c2!=2)
			 printf("c2!=2\t %d\n", counter);*/

			MatNonzeroEntry * matrixNonZeroEntry =
					&(basicSparseMatrix_damping->nonzeroEntry[*counter]);
			matrixNonZeroEntry->rowIndex=jj;
			matrixNonZeroEntry->columnIndex=colIdx;
			matrixNonZeroEntry->value=val;
			(*counter)++;
		}
		jj++;
	}

	fclose(fp1);
}


/*void loadDamping(const char dampingFileName[], BasicSpMat * basicSparseMatrix_damping)
 {
 FILE* fp=fopen(dampingFileName, "r");
 if(fp==NULL){
 fprintf(stderr, "Can not open the file: %s \n", dampingFileName);
 exit(EXIT_FAILURE);
 }

 int * counter=&(basicSparseMatrix_damping->count);

 int no_nz,//number of nonzeros
 rowIdx,//row index
 colIdx;
 MY_FLOAT_TYPE	val;
 while(feof(fp)==0 ){
 fscanf(fp, "%d %d", &no_nz, &rowIdx);
 int i;
 for(i=0; i<no_nz; i++){
 fscanf(fp, "%d %e", &colIdx, &val);
 MatrixNonZeroEntry * matrixNonZeroEntry
 =&(basicSparseMatrix_damping->nonZeroEntries[*counter]);
 matrixNonZeroEntry->rowIndex=rowIdx;
 matrixNonZeroEntry->columnIndex=colIdx;
 matrixNonZeroEntry->value=val;
 (*counter)++;
 }
 }

 fclose(fp);
 }*/

/*void testLoadDamping()
 {
 BasicSpMat basicSparseMatrix_damping;

 int numOfRows=54+54,  numOfColumns=38092800, first=0, last=numOfRows;
 int numOfNonZeroEntries=54+162;
 init_BasicSpMat(&basicSparseMatrix_damping,
 numOfRows,  numOfColumns,  numOfNonZeroEntries,  first,  last);
 loadDamping("../damping_mytest1", &basicSparseMatrix_damping);
 loadDamping("../damping_mytest2", &basicSparseMatrix_damping);

 printBasicSpMat(&basicSparseMatrix_damping);
 }*/

//read confiuration file
//revised on 1/25/10, 3/10/10
void readConfig(const char configFilename[], char dirpath[], IOTYPE * inputType,
		int * lsqr_iteration, int * numColumn, char asciiDataStatistic[],
		char binaryDataStatistic[], char vectorBInfo[], bool isParallel,
		bool isLoadBalance, char damping_path[], char damping_info[])
{
	FILE * fp =fopen(configFilename, "r");
	if (fp==NULL) {
		fprintf(stderr, "Can not open the file: %s \n", configFilename);
		exit(EXIT_FAILURE);
	}
	char var[MAXREADNUM], val[MAXREADNUM], comment[MAXREADNUM];
	while (feof(fp)==0) {
		memset(var, 0, MAXREADNUM);
		memset(val, 0, MAXREADNUM);
		memset(comment, 0, MAXREADNUM);
		//		val="";
		//		comment="";
		fscanf(fp, "%s\t%s\t%s\n", var, val, comment);
		//		fscanf(fp, "%s %s \n", var, val);
		if (0==strcmp(var, "kernelpath")) {
			strcpy(dirpath, val);
		} else if (0==strcmp(var, "kernelInputType")) {
			if (0==strcmp(val, "IOASCII")) {
				*inputType=IOASCII;
			} else if (0==strcmp(val, "IOBINARY")) {
				*inputType=IOBINARY;
			}
		}else if (0==strcmp(var, "kernel_asciiList")) {
			strcpy(asciiDataStatistic, val);
		} else if (0==strcmp(var, "kernel_binaryList")) {
			strcpy(binaryDataStatistic, val);
		}else if (0==strcmp(var, "damping_path")) {
			strcpy(damping_path, val);
		} else if (0==strcmp(var, "damping_list")) {
			strcpy(damping_info, val);
		} else if (0==strcmp(var, "vectorpath")) {
			strcpy(vectorBInfo, val);
		}else if (0==strcmp(var, "lsqr_iteration")) {
			*lsqr_iteration=atoi(val);
		} else if (0==strcmp(var, "numColumn")){
			*numColumn=atoi(val);
		}   else if (0==strcmp(var, "isParallel")) {
			if (0==strcmp(val, "TRUE")) {
				isParallel=true;
			} else
				isParallel=false;
		} else if (0==strcmp(var, "isLoadBalance")) {
			if (0==strcmp(val, "TRUE")) {
				isLoadBalance=true;
			} else
				isLoadBalance=false;
		}
	}
	fclose(fp);
}

/* void readConfig(const char  configFilename[], char  dirpath[], bool inputType, int * lsqr_iteration,
 char  asciiDataStatistic[], char binaryDataStatistic[], char vectorBInfo[],
 bool isParallel, bool isLoadBalance)
 {
 FILE * fp =fopen(configFilename, "r");
 if(fp==NULL){
 fprintf(stderr, "Can not open the file: %s \n", configFilename);
 exit(EXIT_FAILURE);
 }
 char var[MAXREADNUM], val[MAXREADNUM], comment[MAXREADNUM];
 while(feof(fp)==0 ){
 fscanf(fp, "%s\t%s\t%s\n", var, val, comment);
 if(0==strcmp(var, "dirpath")){
 strcpy(dirpath, val);
 }else if(0==strcmp(var, "inputType")){
 if(0==strcmp(val, "IOASCII")){
 inputType=IOASCII;
 }
 else if(0==strcmp(val, "IOBINARY")){
 inputType=IOBINARY;
 }
 }else if(0==strcmp(var, "lsqr_iteration")){
 *lsqr_iteration=atoi(val);
 }else if(0==strcmp(var, "asciiDataStatistic")){
 strcpy(asciiDataStatistic, val);
 }else if(0==strcmp(var, "binaryDataStatistic")){
 strcpy(binaryDataStatistic, val);
 }else if(0==strcmp(var, "vectorBInfo")){
 strcpy(vectorBInfo, val);
 }else if(0==strcmp(var, "isParallel")){
 if(0==strcmp(val, "TRUE")){
 isParallel=true;
 }
 else isParallel=false;
 }else if(0==strcmp(var, "isLoadBalance")){
 if(0==strcmp(val, "TRUE")){
 isLoadBalance=true;
 }else
 isLoadBalance=false;
 }
 }
 fclose(fp);
 } */

void printConfigParameter(const int procRank, const char configFilename[],
		const char dirpath[], const bool inputType, const int * lsqr_iteration,
		const char asciiDataStatistic[], const char binaryDataStatistic[],
		const char vectorBInfo[], const bool isParallel,
		const bool isLoadBalance) {
	printf(
			"%d: dirpath=%s, inputType=%s, lsqr_iteration=%d, asciiDataStatistic=%s, binaryDataStatistic=%s, vectorBInfo=%s \n",
			procRank, dirpath, (inputType) ? "true" : "false", *lsqr_iteration,
			asciiDataStatistic, binaryDataStatistic, vectorBInfo);
}

//read data statistics files, and get the number of all data file
int countData(const char * filename) {
	FILE * fp=fopen(filename, "r");
	// printf("countData filename=%s\n",filename);
	if (fp==NULL) {
		fprintf(stderr, "countData cannot open the file: %s  \n", filename);
		exit(-1);
	}
	int counter=0;
	while (feof(fp)==0) {//if the file does not reach its end of file
		char str1[MAXREADNUM];
		int num;
		//fgets(str, sizeof(str), fp);
		int c11=fscanf(fp, "%s %d", str1, &num);
		if (c11!=2) {
			printf("For file: (%s) in function of countData fscanf c11!=2\n",filename);
			break;
		}
		counter++;
	}

	fclose(fp);
	return counter;
}
/*
 IN: const char * filename, int numOfDataFile
 OUT: char * dataFileName[], int dataFileNumOfColumn[]
 */
void getDataFileName(char * dataFileName[], int dataFileNumOfColumn[],
		const char * filename, int numOfDataFile) {
	FILE * fp=fopen(filename, "r");
	// printf("getDataFileName filename=%s\n",filename);
	if (fp==NULL) {
		fprintf(stderr, "Cannot open the file: %s  \n", filename);
		exit(-1);
	}
	//		int counter=0;
	//		while( feof(fp)==0&& counter<=numOfDataFile){//if the file does not reach its end of file
	//
	//			fgets(dataFileName[counter], sizeof(str), fp);
	//			counter++;
	//		}
	int i;
	for (i=0; i< numOfDataFile; i++) {
		if (feof(fp)==0) {
			fscanf(fp, "%s %d", dataFileName[i], &dataFileNumOfColumn[i]);
			//				printf("dataFileName[%d]=%s\n", i,dataFileName[i]);
		}
	}
	fclose(fp);
}

/*void dataDistribut( char ** dataFileName, const int dataFileNumOfColumn[], const int numOfDataFile,
 const int procRank, const int numProc,  int * first,  int * last, int * numOfNonZeroEntriesPerProc)
 {
 //	compute the total number of nonzero elements of the entire data files
 int sumOfNonzeroElement=0;
 int i;
 for(i=0; i<numOfDataFile; i++){
 sumOfNonzeroElement+=dataFileNumOfColumn[i];
 }
 //	compute the average number of nonzero elements per process
 int average=sumOfNonzeroElement/(numProc-1);
 //	computer the first and last file distributed to every process


 //test<<
 if(procRank==(numProc-1))
 printf("%d: Sum of Non-zero Element is %d\t average=%d\n", procRank, sumOfNonzeroElement, average);
 //test>>

 int * numOfNonzeroPerProc=(int *) calloc(numProc+1, sizeof(int));
 int * rowEndIndexPerProc=(int *)calloc(numProc+1, sizeof(int));//indicates the last row data file of every process
 int j=0;
 numOfNonzeroPerProc[0]=0;
 rowEndIndexPerProc[0]=-1;
 for(i=1; i<=numProc; i++){
 numOfNonzeroPerProc[i]=rowEndIndexPerProc[i]=0;
 while(numOfNonzeroPerProc[i]<average && j<=numOfDataFile){
 numOfNonzeroPerProc[i]+=dataFileNumOfColumn[j];
 j++;
 }
 //go back to the status such that numOfNonzeroPerProc[i]<average and numOfNonzeroPerProc[i]+dataFileNumOfColumn[j]>average
 numOfNonzeroPerProc[i]-=dataFileNumOfColumn[j];
 j--;
 rowEndIndexPerProc[i]=j-1;
 }
 //test<<
 if(procRank==(numProc-1))
 {	int k;
 for(k=0; k<numProc; k++){
 printf("%d: NumOfNonzeroPerProc[%d]=%d, rowEndIndexPerProc[%d]=%d\n", procRank,
 k, numOfNonzeroPerProc[k], k, rowEndIndexPerProc[k]);
 }
 }
 //test>>
 *first=rowEndIndexPerProc[procRank-1]+1;
 *last=rowEndIndexPerProc[procRank];
 if(*first>*last)
 *first=*last;

 *numOfNonZeroEntriesPerProc=numOfNonzeroPerProc[procRank];
 //	printf("%d: num of non-zero element= %d\n", procRank, numOfNonzeroPerProc[procRank]);

 free (numOfNonzeroPerProc);
 free (rowEndIndexPerProc);
 }*/
/*
 void init_Per_Proc_Data(int numProc, int procRank, Per_Proc_Data *perProcData,int matrix_rows, int matrix_cols, SparseMatrixCSR * sparseMatrixCSR, int sizeOfprocDistributedDataFileSet[])
 {

 if(procRank!=0){
 perProcData->matrix_block = sparseMatrixCSR;//(SparseMatrixCSR*)malloc(sizeof(SparseMatrixCSR));

 perProcData->vec_y_Ax_temp = (MY_FLOAT_TYPE *) calloc(perProcData->matrix_block->numOfRows, sizeof(MY_FLOAT_TYPE));
 int i ;
 for (i = 0; i < (perProcData->matrix_block->numOfRows); i++)
 perProcData->vec_y_Ax_temp[i] = 0.0;
 }
 // the vec_x_ATy_temp on root is used for summary.
 perProcData->vec_x_ATy_temp = (MY_FLOAT_TYPE*) calloc(matrix_cols,	sizeof(MY_FLOAT_TYPE));
 int i;
 for (i = 0; i < matrix_cols; i++)
 perProcData->vec_x_ATy_temp[i] = 0.0;


 perProcData->totalNumOfRows = matrix_rows;
 perProcData->totalNumOfColumns = matrix_cols;
 perProcData->perProcRows = (int *)calloc(numProc, sizeof(int));//new int[numProc];
 //perProcData->vec_x_ATy_temp = (MY_FLOAT_TYPE*)calloc(matrix_cols, sizeof(MY_FLOAT_TYPE));//new MY_FLOAT_TYPE[matrix_cols]; // the vec_x_ATy_temp on root is used for summary.
 perProcData->perProcRows[0] = 0;

 12/2009: These part is not suitable for load balancing version
 * for (int i=1;i<numProc;i++){
 if(i!=numProc-1)
 perProcData->perProcRows[i] = matrix_rows/(numProc-1);//set correspond value based on how many rows for each proc.
 else
 perProcData->perProcRows[i]=matrix_rows-(numProc-2)*(matrix_rows/(numProc-1));//(matrix_rows-1)-(matrix_rows/(numProc-1)+1)*(numProc-2);
 }
 //12/2009
 for(i=0; i<numProc; i++){
 perProcData->perProcRows[i]=sizeOfprocDistributedDataFileSet[i];
 }
 //12/2009

 //hh:caculating current process's row number
 int rowStartPosition = 0;
 for ( i=1;i<procRank;i++){
 rowStartPosition += perProcData-> perProcRows[i];
 }
 perProcData->rowStartPositionInEntireMatrix = rowStartPosition;

 if (procRank == 0){
 //hh: perProcData in rank=0, is only used for record the final result, not for computation
 perProcData->wholeResult = (MY_FLOAT_TYPE*)calloc(matrix_rows, sizeof(MY_FLOAT_TYPE)); //new MY_FLOAT_TYPE[matrix_rows];
 perProcData->rowStartPositionInEntireMatrix = 0;
 }
 perProcData->procReceiveInd =(int *)calloc(numProc, sizeof(int)); //new int [numProc];
 perProcData->procReceiveInd[0] = 0;
 for (i=1; i<numProc; i++){
 perProcData->procReceiveInd[i] = perProcData->procReceiveInd[i-1]+ perProcData->perProcRows[i-1];
 }
 }
 */

/*3/11/2010
 *
 *
 * int numRowPerDampingFile[]: NULL if without damping matrix
 * */
void init_Per_Proc_Data_v2(int numProc, int procRank, Per_Proc_Data_v2 *perProcData,
		int matrix_rows, int matrix_cols, SpMatCSR * sparseMatrixCSR,
		int sizeOfprocDistributedDataFileSet_k[], int numRowPerDampingFile[])
{	int i;
	perProcData->version = PLSQRV2;

	perProcData->matrix_block = sparseMatrixCSR;//(SparseMatrixCSR*)malloc(sizeof(SparseMatrixCSR));

	// the vec_x_ATy_temp on root is used for summary.
#if CUDA==1 && CUDAHOSTALLOC==1 //pinned memory
	pinnedMemAlloc((void**)&(perProcData->vec_y_Ax_temp), perProcData->matrix_block->numRow, AX_D);
	pinnedMemAlloc((void**)&(perProcData->vec_x_tAy_temp), perProcData->matrix_block->numCol, TAY_D);
#else
	size_t nbyte_Ax = (perProcData->matrix_block->numRow * sizeof(MY_FLOAT_TYPE) + PAGESIZE-1)/PAGESIZE*PAGESIZE;
	perProcData->vec_y_Ax_temp = (MY_FLOAT_TYPE *) malloc(nbyte_Ax);
	if(NULL==perProcData->vec_y_Ax_temp ){
		fprintf(stderr, "%s(%d)-%s: Ax valloc fails\n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
	}
	size_t nbyte_tAy = (matrix_cols * sizeof(MY_FLOAT_TYPE) + PAGESIZE-1)/PAGESIZE*PAGESIZE;
	perProcData->vec_x_tAy_temp = (MY_FLOAT_TYPE*) malloc(nbyte_tAy);
	if(NULL==perProcData->vec_x_tAy_temp ){
		fprintf(stderr, "%s(%d)-%s: tAy valloc fails\n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
	}
#endif

	for (i = 0; i < (perProcData->matrix_block->numRow); i++)
		perProcData->vec_y_Ax_temp[i] = 0.0;

	for (i = 0; i < matrix_cols; i++)
		perProcData->vec_x_tAy_temp[i] = 0.0f;

	perProcData->totalNumOfRows = matrix_rows;
	perProcData->totalNumOfColumns = matrix_cols;

	{
		perProcData->perProcNumRow = (int *) calloc(numProc, sizeof(int));//new int[numProc];
		if(NULL==perProcData->perProcNumRow ){
			fprintf(stderr, "perProcData->perProcRows calloc fails\n");exit(-1);
		}
		perProcData->perProcNumRow[0] = 0;
		//	perProcData->perProcRows[0]=sizeOfprocDistributedDataFileSet[0];
		for (i = 0; i < numProc; i++) {
			if(numRowPerDampingFile!=NULL)
				perProcData->perProcNumRow[i] = sizeOfprocDistributedDataFileSet_k[i]	+ numRowPerDampingFile[i];
			else perProcData->perProcNumRow[i] = sizeOfprocDistributedDataFileSet_k[i];
		}
	}

#ifdef DEBUG
	printf("%d/%d: perProcData->totalNumOfRows=%d, perProcData->totalNumOfColumns=%d \n",procRank, numProc, perProcData->totalNumOfRows,  perProcData->totalNumOfColumns );
#endif

	//perProcData->vec_x_ATy_temp = (MY_FLOAT_TYPE*)calloc(matrix_cols, sizeof(MY_FLOAT_TYPE));//new MY_FLOAT_TYPE[matrix_cols]; // the vec_x_ATy_temp on root is used for summary.


	//hh:caculating current process's row number
//	int rowStartPosition = 0;

	/*if(procRank!=0)*/
/*	for (i=01; i<procRank; i++) {
		rowStartPosition += perProcData-> perProcRows[i];
	}*/
	/*perProcData->rowStartPositionInEntireMatrix = rowStartPosition;*/

/*	if (procRank == 0) {
		//hh: perProcData in rank=0, is only used for record the final result, not for computation
		//3/29
		//		perProcData->wholeResult = (MY_FLOAT_TYPE*)calloc(matrix_rows, sizeof(MY_FLOAT_TYPE));

		perProcData->rowStartPositionInEntireMatrix = 0;
	}*/


	/*if(procRank!=0)*/{
		perProcData->procReceiveInd =(int *)calloc(numProc, sizeof(int));
		if(NULL==perProcData->procReceiveInd ){
			fprintf(stderr, "perProcData->procReceiveInd calloc fails\n");exit(-1);
		}
		perProcData->procReceiveInd[0] = 0;

		for (i=1; i<numProc; i++) {
			perProcData->procReceiveInd[i] = perProcData->procReceiveInd[i-1]+ perProcData->perProcNumRow[i-1];
		}
	}

}

void finialize_Per_Proc_Data_v2(int procRank, int numProc,Per_Proc_Data_v2 *perProcData)
{

	perProcData->matrix_block = NULL;
	free(perProcData->perProcNumRow);
	free(perProcData->procReceiveInd);

#if CUDA==1 && CUDAHOSTALLOC==1 //pinned memory
	pinnedMemFree(perProcData->vec_y_Ax_temp);
	pinnedMemFree(perProcData->vec_x_tAy_temp);
#else
	free(perProcData->vec_y_Ax_temp);
	free(perProcData->vec_x_tAy_temp);
#endif


}

/*	struct Per_Proc_Data{
 SparseMatrixCSR* matrix_block;
 MY_FLOAT_TYPE* vec_y_Ax_temp; //temporary result for each process
 MY_FLOAT_TYPE* vec_x_ATy_temp;
 MY_FLOAT_TYPE* wholeResult; // the whole vector result, only on processor 0.
 //MY_FLOAT_TYPE* vectorResultTM; // the whole vector result, only on processor 0.
 int* procReceiveInd; // index when gathing data from all processors. The data on each processor has its own index.
 int* perProcRows; // array of num of rows on each processors
 int totalNumOfRows;
 int totalNumOfColumns;
 int rowStartPositionInEntireMatrix;
 };*/
//print Per_Proc_Data's every field except perProcData->matrix_block to corresponding file
void print_Per_Proc_Data_BeforeCaculationtoFile(int procRank, int numProc,
		Per_Proc_Data_v2 *perProcData) {
	FILE *fp;
	char filename[MAXREADNUM];
	sprintf(filename, "rank=%d_Per_Proc_Data_beforeCal", procRank);
	fp = fopen(filename, "w");
	if (!fp) {
		fprintf(stderr, "Cannot open the file: %s  \n", filename);
		exit(-1);
	}
	fprintf(fp, "process rank=%d, before calculation \n", procRank);
	int i;
	for (i=1; i<numProc; i++) {
		fprintf(fp, "perProcData->perProcRows[%d] %d\n", i,
				perProcData->perProcNumRow[i]);
		fprintf(fp, "perProcData->procReceiveInd[%d] %d\n", i,
				perProcData->procReceiveInd[i]);
	}
	fprintf(fp, "perProcData->totalNumOfRows %d\n", perProcData->totalNumOfRows) ;
	fprintf(fp, "perProcData->totalNumOfColumns %d\n",
			perProcData->totalNumOfColumns);
/*	fprintf(fp, "perProcData->rowStartPositionInEntireMatrix %d\n",
			perProcData->rowStartPositionInEntireMatrix);*/

	fclose(fp);
}

void print_Per_Proc_Data_AfterCaculationtoFile(int procRank, int numProc,
		Per_Proc_Data_v2 *perProcData) {
	FILE *fp;
	char filename[MAXREADNUM];
	sprintf(filename, "rank=%d_Per_Proc_Data_afterCal", procRank);
	fp = fopen(filename, "a");
	if (!fp) {
		fprintf(stderr, "Cannot open the file: %s  \n", filename);
		exit(-1);
	}
	fprintf(fp, "process rank=%d, after calculation \n", procRank);
	int i;
	for (i=1; i<numProc; i++) {
		fprintf(fp, "perProcData->perProcRows[%d] %d\n", i,
				perProcData->perProcNumRow[i]);
		fprintf(fp, "perProcData->procReceiveInd[%d] %d\n", i,
				perProcData->procReceiveInd[i]);
	}
	fprintf(fp, "perProcData->totalNumOfRows %d\n", perProcData->totalNumOfRows) ;
	fprintf(fp, "perProcData->totalNumOfColumns %d\n",
			perProcData->totalNumOfColumns);
/*	fprintf(fp, "perProcData->rowStartPositionInEntireMatrix %d\n",
			perProcData->rowStartPositionInEntireMatrix);*/

	if (procRank==0) {
		int i;
		for (i=0; i<perProcData->totalNumOfRows; i++)
/*			fprintf(fp, "perProcData->wholeResult[%d]=%e\n", i,
					perProcData->wholeResult[i]);*/
		for (i=0; i<perProcData->totalNumOfColumns; i++)
			fprintf(fp, "perProcData->procVectorResultTM[%d]=%e\n", i,
					perProcData->vec_x_tAy_temp[i]);
	} else if (procRank!=0) {
		int i;
		for (i=0; i<perProcData->matrix_block->numRow; i++)
			fprintf(fp, "perProcData->procVectorResult[%d]=%e\n", i,
					perProcData->vec_y_Ax_temp[i]);
		for (i=0; i<perProcData->matrix_block->numCol; i++)
			fprintf(fp, "perProcData->procVectorResultTM[%d]=%e\n", i,
					perProcData->vec_x_tAy_temp[i]);
	}

	fclose(fp);
}

//12/2009
void readInputVector(const char * filename, MY_FLOAT_TYPE vectorB[], int dimension) {
	//should check whether the inputed vector's dimension equals to the vector's dimension in main function
	FILE *fp1 =fopen(filename, "r");
	if (fp1==NULL) {
		fprintf(stderr, "Cannot open the file: %s\n", filename);
		exit(EXIT_FAILURE);
	}
	int i;
	for (i=0; i<dimension; i++) {
		if (feof(fp1)==0){
#if SDVERSION==1
			fscanf(fp1, "%e", &vectorB[i]);
#elif SDVERSION == 2
			fscanf(fp1, "%le", &vectorB[i]);
#endif
		}
		else {
			fprintf(stderr,	"the vectorB's dimension is not matched by the number of lines in this vector file\n");
			break;
		}
	}
	fclose(fp1);
	//display the vector
	/*	for (i = 0; i < dimension; i++){
	 fprintf(stdout, "%e\n", vectorB[i] );
	 }*/
	//

}
/*
MY_FLOAT_TYPE* readInputVectorInBinary(const char *filename, int numCols) {

	MY_FLOAT_TYPE* y = (MY_FLOAT_TYPE*) calloc(numCols, sizeof(MY_FLOAT_TYPE));

	// printf("\nLoading vector B from binary file %s ...   \n",filename);

	FILE *f;
	f = fopen(filename, "r");
	if (!f) {
		fprintf(stderr, "Cannot open vector file: %s  \n", filename);
		exit(-1);
	}

	int i = 0;
	while (i < numCols) {
		fscanf(f, "%e", &y[i]);
		i++;
	}

	if (i < numCols) {
		fprintf(stderr, "number of columns mismatch: %s  \n", filename);
		exit(-1);
	}

	fclose(f);

	return y;
}*/

/*print vector*/
void print_vector(const MY_FLOAT_TYPE * vector, const int dimension,
		const char filename[]) {
	FILE *fp1 =fopen(filename, "w");
	if (fp1==NULL) {
		fprintf(stderr, "%s(%d)-%s: Cannot open the file: %s\n", __FILE__, __LINE__, __FUNCTION__, filename);
		exit(EXIT_FAILURE);
	}

	fprintf(fp1, "dimension=%d\n", dimension);
	int i;
	for (i = 0; i < dimension; i++) {
//		if (vector[i]!=0.0)
			fprintf(fp1, "[%d]=%e\n", i, vector[i]);
	}
	fclose(fp1);
}

void print_vector_int(const int * vector, const int dimension,
		const char filename[]) {
	FILE *fp1 =fopen(filename, "w");
	if (fp1==NULL) {
		fprintf(stderr, "%s(%d)-%s: Cannot open the file: %s\n", __FILE__, __LINE__, __FUNCTION__, filename);
		exit(EXIT_FAILURE);
	}

	fprintf(fp1, "dimension=%d\n", dimension);
	int i;
	for (i = 0; i < dimension; i++) {
//		if (vector[i]!=0.0)
			fprintf(fp1, "[%d]=%d\n", i, vector[i]);
	}
	fclose(fp1);
}
/*Example
 RAND.C: This program seeds the random-number generator
* with the time, then displays 10 random integers.


#include <stdlib.h>
#include <stdio.h>
#include <time.h>

void main( void )
{
int i;

 Seed the random-number generator with current time so that
* the numbers will be different every time we run.

srand( (unsigned)time( NULL ) );

 Display 10 numbers.
for( i = 0; i < 10;i++ )
printf( " %6d\n", rand() );
}


Output

6929
8026
21987
30734
20587
6699
22034
25051
7988
10104*/
void generateRandomVector(MY_FLOAT_TYPE* y, int dimension) {
	printf("no input vector provided, generate random input vector ... \n");
	int i;
	for (i = 0; i < dimension; i++) {
		y[i] = rand() / (MY_FLOAT_TYPE)RAND_MAX;
	}

	char filename[]="RandomVector.txt";
	FILE * fp=fopen(filename, "w");
	if (fp==NULL) {
		fprintf(stderr, "generateRandomVector Cannot open the file: %s  \n", filename);
		exit(-1);
	}
#ifdef DEBUG
	for (i = 0; i < dimension; i++) {
		fprintf(fp, "%e\n", y[i]);
	}
#endif
	fclose(fp);
}


//revised by hh 10/05/2009
//print SparseMatrixCSR into files
void printSparseMatrixCSRtoFile(int rank, SpMatCSR * sparseMatrix) {
	FILE *fp;
	char filename[MAXREADNUM];
	sprintf(filename, "rank=%d_MatrixCSROutput", rank);
	fp = fopen(filename, "w");
	if (!fp) {
		fprintf(stderr, "printSparseMatrixCSRtoFile Cannot open the file: %s  \n", filename);
		exit(-1);
	}
	fprintf(fp, "process rank=%d, print SparseMatrixCSR in this rank \n", rank);
	//	fprintf(fp, "first row=%d last row=%d \n", sparseMatrix->firstRowNo, sparseMatrix->lastRowNo);
	///////////////////////////////////////////
	fprintf(fp, "ptrs:[ ");
	int i;
	for (i = 0; i <= sparseMatrix->numRow; i++) {
		fprintf(fp, "%i, ", sparseMatrix->row_ptr[i]);
	}

	fprintf(fp, " ] \n");

	for (i = 0; i < sparseMatrix->numRow; i++) {
		MY_FLOAT_TYPE t = 0;
		int firstEntryInRow = sparseMatrix->row_ptr[i];
		int lastEntryInRow = sparseMatrix->row_ptr[i+1];
		int entryIndex;
		for (entryIndex = firstEntryInRow; entryIndex < lastEntryInRow; entryIndex++) {

			int columnIndex = sparseMatrix->col_idx[entryIndex];
			fprintf(fp, "(%i,%i) = %e \n", /*i, columnIndex,*/
					i, columnIndex, sparseMatrix->val[entryIndex]);

		}
	}
	//////////////////////////////////////////////////////////////
	fclose(fp);
}

void printPaddedSparseMatrixCSR(PadSpMatCSR *sparseMatrix) {

	printf("ptrs:[ ");
	int i;
	for (i = 0; i <= sparseMatrix->numRow; i++) {
		printf("%i ", sparseMatrix->row_ptr[i]);
	}

	printf(" ] \n");

	for (i = 0; i < sparseMatrix->numRow; i++) {
		MY_FLOAT_TYPE t = 0;
		int firstEntryInRow = sparseMatrix->row_ptr[i];
		int lastEntryInRow = sparseMatrix->row_ptr[i+1];
		int entryIndex;
		for (entryIndex = firstEntryInRow; entryIndex < lastEntryInRow; entryIndex++) {

			int columnIndex = sparseMatrix->col_idx[entryIndex];
			printf("[%i][%i](row:%i, column:%i) = %e \n", i, columnIndex, i,
					columnIndex, sparseMatrix->val[entryIndex]);

		}
	}

	for (i = 0; i< sparseMatrix->row_ptr[sparseMatrix->numRow]; i++) {
		printf("column[%i] = %i \n", i, sparseMatrix->col_idx[i]);
	}

	for (i = 0; i< sparseMatrix->row_ptr[sparseMatrix->numRow]; i++) {
		printf("data[%i] = %e \n", i, sparseMatrix->val[i]);
	}
}

void printBasicSpMat(BasicSpMat *basicSparseMatrix) {
	int i;
	for (i = 0; i < basicSparseMatrix->numNonzero; i++) {
		MatNonzeroEntry entry = basicSparseMatrix->nonzeroEntry[i];
		printf("[%d][%d] = %e \n", entry.rowIndex, entry.columnIndex,
				entry.value);
	}
}

void printBasicSpMattoFile(int rank,
		BasicSpMat *basicSparseMatrix) {
	FILE *fp;
	char filename[MAXREADNUM];
	sprintf(filename, "rank=%d BasicMatrixOutput", rank);
	fp = fopen(filename, "w");
	if (!fp) {
		fprintf(stderr, "Cannot open the file: %s  \n", filename);
		exit(-1);
	}
	fprintf(fp, "process rank=%d, print BasicSpMat in this rank \n",
			rank);
	//	fprintf(fp, "first row=%d, last row=%d\n",basicSparseMatrix->firstRowNo, basicSparseMatrix->lastRowNo);
	BasicSpMat * p=basicSparseMatrix;
	//	fprintf(fp, "", );


	/////////////////////////
	int i;
	for (i = 0; i < basicSparseMatrix->numNonzero; i++) {
		fprintf(fp, "%d [%d][%d] = %e \n", i,
				basicSparseMatrix->nonzeroEntry[i].rowIndex,
				basicSparseMatrix->nonzeroEntry[i].columnIndex,
				basicSparseMatrix->nonzeroEntry[i].value);
	}
	fclose(fp);
}

/*
 * initialize SparseMatrixCSR using a basic sparse matrix
 * basic sparse matrix must be ready before invoking
 * */
void init_SparseMatrixCSR_withBasicSpMat(SpMatCSR* matrixCSR,
		BasicSpMat* basicSparseMatrix) {
	matrixCSR->numRow = basicSparseMatrix->numRow;
	matrixCSR->numCol = basicSparseMatrix->numCol;
	matrixCSR->numNonzero = basicSparseMatrix->numNonzero;

	matrixCSR->row_ptr
			= (int*)malloc(sizeof(int) * (matrixCSR->numRow + 1));
	if(NULL==matrixCSR->row_ptr){
		fprintf(stderr, "matrixCSR->row_ptr calloc fails \n");exit(-1);
	}
	matrixCSR->col_idx
			= (int*)malloc(sizeof(int) *(matrixCSR->numNonzero));
	if(NULL==matrixCSR->col_idx){
		fprintf(stderr, "matrixCSR->col_idx calloc fails \n");exit(-1);
	}
	matrixCSR->val = (MY_FLOAT_TYPE*)malloc(sizeof(MY_FLOAT_TYPE)
			*(matrixCSR->numNonzero));
	if(NULL==matrixCSR->val){
		fprintf(stderr, "matrixCSR->val calloc fails \n");exit(-1);
	}
}
/* 3/25/2010
 * */
void initSpMatCSR(SpMatCSR* matrixCSR, int numOfRows,
		int numOfColumns, int numOfNonZeroEntries) {
	matrixCSR->numRow = numOfRows;
	matrixCSR->numCol = numOfColumns;
	matrixCSR->numNonzero = numOfNonZeroEntries;

	matrixCSR->row_ptr = (int*)valloc(sizeof(int) *(matrixCSR->numRow + 1));
	if(NULL==matrixCSR->row_ptr){
		fprintf(stderr, "matrixCSR->row_ptr calloc fails\n");exit(-1);
	}
	memset(matrixCSR->row_ptr, 0, sizeof(int)*(matrixCSR->numRow + 1));

	matrixCSR->col_idx= (int*)valloc(sizeof(int) *(matrixCSR->numNonzero));
	if(NULL==matrixCSR->col_idx){
		fprintf(stderr, "matrixCSR->col_idx calloc fails\n");exit(-1);
	}
	memset(matrixCSR->col_idx, 0 , sizeof(int) * (matrixCSR->numNonzero));

	matrixCSR->val = (MY_FLOAT_TYPE*)valloc(sizeof(MY_FLOAT_TYPE)*(matrixCSR->numNonzero));
	if(NULL==matrixCSR->val ){
		fprintf(stderr, "matrixCSR->val calloc fails\n");exit(-1);
	}
	memset(matrixCSR->val, 0, sizeof(MY_FLOAT_TYPE)	*(matrixCSR->numNonzero));
//	int i;
/*	for(i=0; i<(matrixCSR->numRow + 1); i++){
		matrixCSR->row_ptr[i]=0;
	}*/

/*	for(i=0; i<matrixCSR->numNonzero; i++){
		matrixCSR->col_idx[i]=0;
		matrixCSR->val[i]=0.0;
	}*/

}
/*
void finalizeSpMatCSR(SpMatCSR* matrixCSR) {
	free(matrixCSR->row_ptr);
	free(matrixCSR->col_idx);
	free(matrixCSR->val);
}
*/



/* 3/25/2010
 * load kernel and damping (ASCII) directly into CSR
 * */

void loadMatrixCSR_ascii_kernelonly(SpMatCSR* matrixCSR,
		/*kernel*/
		char dirpath_k[], char * kernelDataFileName[],
		int dataFileNumNonzeroPerrow_k[], int perProcDisDataFileSet_k[],
		int perProcSizeOfDisDataFileSet_k)
{
	matrixCSR->row_ptr[0] = 0;
	//the index of column and data of the CSR matrix
	int totalDataIndexOfCSR=0;
	int i;
	/*----------------------process kernel data----------------------------------------*/
	for (i=0; i<perProcSizeOfDisDataFileSet_k; i++) {
		int filenameIdx=perProcDisDataFileSet_k[i];
		char kernelFilenameFullpath[1024];//../data/
		sprintf(kernelFilenameFullpath, "%s/%s", dirpath_k,
				kernelDataFileName[filenameIdx]);

		matrixCSR->row_ptr[i+1]=matrixCSR->row_ptr[i]
				+dataFileNumNonzeroPerrow_k[filenameIdx];

		//process  kernel file one by one
		FILE *fp;
		fp = fopen(kernelFilenameFullpath, "r");
		if (fp==NULL) {
			fprintf(stderr, "Cannot open the file: %s  \n", kernelFilenameFullpath);
			exit(-1);
		}
		int j;
		for (j = 0; j <dataFileNumNonzeroPerrow_k[filenameIdx]; j++) {
			MatNonzeroEntry nonzeroEntry;
			int iX, iY, iZ;
#if SDVERSION == 1
			int c1=fscanf(fp, "%d %d %d %d %e", &(nonzeroEntry.columnIndex),
					&(iX), &(iY), &(iZ), &(nonzeroEntry.value));
#elif SDVERSION == 2
			int c1=fscanf(fp, "%d %d %d %d %le", &(nonzeroEntry.columnIndex),
								&(iX), &(iY), &(iZ), &(nonzeroEntry.value));
#endif

			if (c1!=5) {//deal boundary condition
				printf("loadMatrixCSR_ascii, %s: line:%d  fscanf!=5\n", kernelFilenameFullpath, j+1);
				printf("loadMatrixCSR_ascii: %s: c1: %d\n", kernelFilenameFullpath, c1);
				if(feof(fp)!=0);
					printf("loadMatrixCSR_ascii: %s: EOF\n", kernelFilenameFullpath);
				break;
			}
			nonzeroEntry.rowIndex = -1;//ignored
			nonzeroEntry.columnIndex--; // current column index -1, adjust to zero-based indexing ??

			matrixCSR->col_idx[totalDataIndexOfCSR]=nonzeroEntry.columnIndex;
			matrixCSR->val[totalDataIndexOfCSR]=nonzeroEntry.value;
			totalDataIndexOfCSR++;
			if (totalDataIndexOfCSR>matrixCSR->numNonzero)//deal boundary condition
				printf("data index is greater matrixCSR->numOfNonZeroData\n");
		}
		fclose(fp);
	}
}

/*load both kernel and damping matrix
 * assume damping matrix 's  index is 1-based
 * */
void loadMatrixCSR_ascii(int procRank, SpMatCSR* matrixCSR,
		/*kernel*/
		char dirpath_k[], char * kernelDataFileName[],
		int numNonzeroPerRow_k[], int perProcDisDataFileSet_k[],
		int numRowPerProc_k,
		/*damping*/
		char damping_path[], char * dampingFileName[], int dampingFileID)
{
	matrixCSR->row_ptr[0] = 0;
	//the index of column and data of the CSR matrix
	int totalDataIndexOfCSR=0;
	int i;
	/*----------------------process kernel data----------------------------------------*/
	for (i=0; i<numRowPerProc_k; i++) {
		int filenameIdx=perProcDisDataFileSet_k[i];
		char kernelFilenameFullpath[1024];//../data/
		sprintf(kernelFilenameFullpath, "%s/%s", dirpath_k,
				kernelDataFileName[filenameIdx]);

		matrixCSR->row_ptr[i+1]=matrixCSR->row_ptr[i]
				+numNonzeroPerRow_k[filenameIdx];

		//process  kernel file one by one
		FILE *fp_k;
		fp_k = fopen(kernelFilenameFullpath, "r");
		if (fp_k==NULL) {
			fprintf(stderr, "loadMatrixCSR_ascii cannot open kernel file: %s  \n", kernelFilenameFullpath);
			exit(-1);
		}
		int j;
		for (j = 0; j <numNonzeroPerRow_k[filenameIdx]; j++) {
			MatNonzeroEntry nonzeroEntry;
			int iX, iY, iZ;
#if SDVERSION == 1
			int c1=fscanf(fp_k, "%d %d %d %d %e", &(nonzeroEntry.columnIndex),
					&(iX), &(iY), &(iZ), &(nonzeroEntry.value));
#elif	SDVERSION == 2
			int c1=fscanf(fp_k, "%d %d %d %d %le", &(nonzeroEntry.columnIndex),
								&(iX), &(iY), &(iZ), &(nonzeroEntry.value));
#endif
			if (c1!=5) {//deal boundary condition
				printf("loadMatrixCSR_ascii, %s: line:%d  fscanf!=5\n", kernelFilenameFullpath, j+1);
				printf("loadMatrixCSR_ascii: %s: c1: %d\n", kernelFilenameFullpath, c1);
				if(feof(fp_k)!=0);
					printf("loadMatrixCSR_ascii: %s: EOF\n", kernelFilenameFullpath);
				break;
			}
			nonzeroEntry.rowIndex = -1;//ignored
			nonzeroEntry.columnIndex--; // current column index -1, adjust to zero-based indexing ??

			matrixCSR->col_idx[totalDataIndexOfCSR]=nonzeroEntry.columnIndex;
			matrixCSR->val[totalDataIndexOfCSR]=nonzeroEntry.value;
			totalDataIndexOfCSR++;
			if (totalDataIndexOfCSR>matrixCSR->numNonzero)//deal boundary condition
				printf("data index is greater matrixCSR->numOfNonZeroData\n");
		}
		fclose(fp_k);
	}
	for(i=numRowPerProc_k; i<matrixCSR->numRow; i++){//fill damping row_ptr first
		matrixCSR->row_ptr[i+1]=matrixCSR->row_ptr[i];
	}


	/*------------------------^-^!!!process damping data------------------------------------*/
	char dampingFileNameFullpath[1024];
	sprintf(dampingFileNameFullpath, "%s/%s", damping_path,
			dampingFileName[dampingFileID]);
	FILE* fp_d = fopen(dampingFileNameFullpath, "r");
	if (fp_d == NULL) {
		fprintf(stderr, "%d: loadMatrixCSR_ascii can not open damping file: %s \n", procRank, dampingFileNameFullpath);
		exit(EXIT_FAILURE);
	}else{
		fprintf(stdout, "%d: loadMatrixCSR_ascii is loading damping file: %s \n", procRank, dampingFileNameFullpath);
	}

	int no_nz_perRow,//number of nonzeros
			rowIdx,//row index, ignored
			colIdx;
	MY_FLOAT_TYPE val;
	/*local row iIndex of local matrix*/
	int rowIdx_d = numRowPerProc_k;

	while (feof(fp_d) == 0) {
		int c1 = fscanf(fp_d, "%d %d", &no_nz_perRow, &rowIdx);
		if (c1 != 2)//deal boundary condition
		{
			printf("c1!=2\t lineNO:%d\n", rowIdx);
			break;
		}
		matrixCSR->row_ptr[rowIdx_d + 1] = matrixCSR->row_ptr[rowIdx_d] + no_nz_perRow;/*matrixCSR->rowIndex[k+1]=matrixCSR->rowIndex[k]+no_nz_perRow;*/
		int j;
		for (j = 0; j < no_nz_perRow; j++) {
#if SDVERSION == 1
			int c2 = fscanf(fp_d, "%d %e", &colIdx, &val);
#elif	SDVERSION == 2
			int c2 = fscanf(fp_d, "%d %le", &colIdx, &val);
#endif
/*@@@adjust to 0-based from 1-based@@@@@@@@@@@@@@@@@*/
//			colIdx--;

			MatNonzeroEntry nonzeroEntry;
			nonzeroEntry.rowIndex = rowIdx_d;
			nonzeroEntry.columnIndex = colIdx; /*nonzeroEntry.columnIndex--;*/
			nonzeroEntry.value = val;

			matrixCSR->col_idx[totalDataIndexOfCSR] = nonzeroEntry.columnIndex;/*matrixCSR->column[totalDataIndexOfCSR]=nonzeroEntry.columnIndex;*/
			matrixCSR->val[totalDataIndexOfCSR] = nonzeroEntry.value;/*matrixCSR->data[totalDataIndexOfCSR]=nonzeroEntry.value;*/
			totalDataIndexOfCSR++;
			if (totalDataIndexOfCSR > matrixCSR->numNonzero/*numOfNonZeroData*/)//deal boundary condition
				printf("data index is greater matrixCSR->numOfNonZeroData\n");
		}
		rowIdx_d++;
	}
	fclose(fp_d);
}
/*
 * load kernel directly into CSR
 *
 * The format of Seismic's bin data
 * The first data of the file is the total number of non zoer columns in this file
 *
 * [numNonzeroCol:int][rowIndex:int][x:int][y:int][z:int][value:doulbe]
 * [rowIndex:int][x:int][y:int][z:int][value:doulbe]
 * [rowIndex:int][x:int][y:int][z:int][value:doulbe]
 * [rowIndex:int][x:int][y:int][z:int][value:doulbe]
 * ...
 * */
void loadMatrixCSR_bin(SpMatCSR* matrixCSR,
		/*kernel*/
		char dirpath_k[], char * kernelDataFileName[],
		int dataFileNumNonzeroPerrow_k[], int perProcDisDataFileSet_k[],
		int perProcSizeOfDisDataFileSet_k,
		/*damping*/
		char damping_path[], char * dampingFileName[], int dampingFileID)
{
	matrixCSR->row_ptr[0] = 0;
	//the index of column and data of the CSR matrix
	int totalDataIndexOfCSR=0;
	int i;
	/*----------------------process kernel data----------------------------------------*/
	for (i=0; i<perProcSizeOfDisDataFileSet_k; i++) {
		int filenameIdx=perProcDisDataFileSet_k[i];
		char kernelFilenameFullpath[1024];//../data/
		sprintf(kernelFilenameFullpath, "%s/%s", dirpath_k,
				kernelDataFileName[filenameIdx]);

		matrixCSR->row_ptr[i+1]=matrixCSR->row_ptr[i]
				+dataFileNumNonzeroPerrow_k[filenameIdx];

		//process  kernel file one by one
		FILE *fp = fopen(kernelFilenameFullpath, "rb");
		if (fp==NULL) {
			fprintf(stderr, "Cannot open the file: %s  \n", kernelFilenameFullpath);
			exit(-1);
		}
		int numNonzeroCol=0;
		fread(&numNonzeroCol, sizeof(int), 1, fp);
		int readBuf4Int [4];

		int j;
		for (j = 0; j <dataFileNumNonzeroPerrow_k[filenameIdx]; j++) {
			MatNonzeroEntry nonzeroEntry;

			fread(readBuf4Int, sizeof(int), 4, fp);
			nonzeroEntry.columnIndex==readBuf4Int[0];
			fread(&(nonzeroEntry.value)/*value1*/, sizeof(MY_FLOAT_TYPE), 1, fp);

			nonzeroEntry.rowIndex = -1;//ignored
			nonzeroEntry.columnIndex--; // current column index -1, adjust to zero-based indexing ??

			matrixCSR->col_idx[totalDataIndexOfCSR]=nonzeroEntry.columnIndex;
			matrixCSR->val[totalDataIndexOfCSR]=nonzeroEntry.value;
			totalDataIndexOfCSR++;
			if (totalDataIndexOfCSR>matrixCSR->numNonzero)//deal boundary condition
				printf("data index is greater matrixCSR->numOfNonZeroData\n");
		}
		fclose(fp);
	}

}

/*

 SparseMatrixCSR* loadMMFile(const char* fileName){

 BasicSpMat* basicSparseMatrix =
 loadBasicSpMat(fileName);

 SparseMatrixCSR *matrixCSR =
 convertToSparseMatrixCSR(basicSparseMatrix);

 return matrixCSR;
 }

 */
/*

BasicSpMat* loadBasicSpMat(const char* fileName) {

	FILE *f;
	f = fopen(fileName, "r");

	if (!f) {
		fprintf(stderr, "Cannot open file: %s\n", fileName);
		return NULL;
	}

	char line[MAXREADNUM];
	while ( (fgets(line, MAXREADNUM, f)) != NULL) {
		if (line[0] != '%')
			break;
	}

	BasicSpMat *basicSparseMatrix =
			(BasicSpMat *)malloc(sizeof(BasicSpMat));

	if ( (sscanf(line, "%d %d %d", &(basicSparseMatrix->numRow),
			&(basicSparseMatrix->numCol),
			&(basicSparseMatrix->numNonzero))) != 3) {

		fprintf(
				stderr,
				"Cannot read the numRows, numCols, numNZElements from file: %s\n",
				fileName);
	}

	basicSparseMatrix->nonzeroEntry
			= (MatNonzeroEntry *) malloc(sizeof(MatNonzeroEntry)
					* (basicSparseMatrix->numNonzero));

	//	basicSparseMatrix->rowIndices
	//		= (int *) malloc(sizeof(int) * (basicSparseMatrix->numOfRows));
	//	basicSparseMatrix->columnIndices
	//		= (int *) malloc(sizeof(int) * (basicSparseMatrix->numOfColumns));

	MatNonzeroEntry entry;

	int i;
	for (i = 0; i < basicSparseMatrix->numNonzero; i++) {
		fscanf(f, "%d %d %e\n", &(entry.rowIndex), &(entry.columnIndex),
				&(entry.value));

		entry.rowIndex--;
		entry.columnIndex--; // adjust to 0-based
		(basicSparseMatrix->nonzeroEntry)[i]= entry;
	}

	qsort(basicSparseMatrix->nonzeroEntry,
			basicSparseMatrix->numNonzero, sizeof(MatNonzeroEntry),
			compareRow);

	//	printBasicSpMat(basicSparseMatrix);

	fclose(f);

	return basicSparseMatrix;
}
*/
/*
int getNumOfRows(const char* fileName) {
	char *wc = "wc -l ";

	char *wcFullCommand;

	wcFullCommand = (char *)calloc(strlen(wc) + strlen(fileName) + 1,
			sizeof(char));

	strcpy(wcFullCommand, wc);
	strcat(wcFullCommand, fileName);

	FILE* pipe = popen(wcFullCommand, "r");
	if (pipe == NULL) {
		fprintf(stderr, "reading %s failed: \n", fileName);
		return 1;
	}

}*/

int readTimeShift(const char* fileName) {

}

/*

 SparseMatrixCSR* load_csr_matrix(const char* fileName, int rowIndex){

 BasicSpMat* basicMatrix = loadBasicSpMatInGrid(fileName, rowIndex);

 SparseMatrixCSR* matrixCSR = convertToSparseMatrixCSR(basicMatrix);
 return matrixCSR;
 }
 */
//revised by hh
/*void load_csr_matrix(SparseMatrixCSR* matrixCSR, const char* fileName, int rowIndex){

 BasicSpMat basicMatrix;
 loadBasicSpMatInGrid(&basicMatrix, fileName, rowIndex);
 //after load the entire basic sparse matrix, than convert basic matrix to CSRMatrix
 convertToSparseMatrixCSR(matrixCSR, &basicMatrix);

 }*/

///compressed sparse row format
//revised by hh, checked OK
//BasicSpMat* loadBasicSpMatInGrid(const char* fileName, int rowIndex)
bool loadBasicSpMatInGrid(BasicSpMat* basicSparseMatrix,
		const char* fileName, int dataFileNumOfColumn, int rowIndex) {
	MatNonzeroEntry matrixNonZeroEntry;
	FILE *fp;
	fp = fopen(fileName, "r");
	int iX, iY, iZ;
	//	int maxColumn = 0;
	int i;
	int lastCount=basicSparseMatrix->count;
	for (i = lastCount; i <lastCount+dataFileNumOfColumn; i++) {
		fscanf(fp, "%d %d %d %d %e\n", &(matrixNonZeroEntry.columnIndex),
				&(iX), &(iY), &(iZ), &(matrixNonZeroEntry.value));

		//		if (matrixNonZeroEntry.columnIndex > maxColumn) {
		//			maxColumn = matrixNonZeroEntry.columnIndex;
		//		}
		matrixNonZeroEntry.rowIndex = rowIndex;
		matrixNonZeroEntry.columnIndex--; // current column index -1, adjust to zero-based indexing ??
		(basicSparseMatrix->nonzeroEntry)[i] = matrixNonZeroEntry;
		basicSparseMatrix->count++;
	}
	//	for (i = lastCount; i <lastCount+dataFileNumOfColumn; i++)
	//		printf("%d: %d %d %e\n", i,basicSparseMatrix->nonZeroEntries[i].rowIndex, basicSparseMatrix->nonZeroEntries[i].columnIndex, basicSparseMatrix->nonZeroEntries[i].value/*matrixNonZeroEntry.columnIndex, matrixNonZeroEntry.value*/);


	//	basicSparseMatrix->numOfColumns = maxColumn;
	//sort the nonZeroEntries array by row and column.
	//	qsort(basicSparseMatrix->nonZeroEntries,basicSparseMatrix->numOfNonZeroEntries, sizeof(MatrixNonZeroEntry),compareRow);
	//	printBasicSpMat(basicSparseMatrix);
	fclose(fp);
	//return basicSparseMatrix;
	return true;
}

/* read binary file			12/12/09
 * The format of Seismic's bin data
 * The first data of the file is the total number of non zoer columns in this file
 *
 * [numNonzeroCol:int][rowIndex:int][x:int][y:int][z:int][value:doulbe]
 * [rowIndex:int][x:int][y:int][z:int][value:doulbe]
 * [rowIndex:int][x:int][y:int][z:int][value:doulbe]
 * [rowIndex:int][x:int][y:int][z:int][value:doulbe]
 * ...
 *
 * */
bool loadBasicSpMatInGridBin(BasicSpMat* basicSparseMatrix,
		const char* fileName, int dataFileNumOfColumn, int rowIndex) {
	MatNonzeroEntry matrixNonZeroEntry;
	FILE *fp;
	fp = fopen(fileName, "rb");
	if (fp==NULL) {
		fprintf(stderr, "Cannot open the file: %s  \n", fileName);
		exit(-1);
	}
	int iX, iY, iZ;
	//	int maxColumn = 0;
	int i;
	int lastCount=basicSparseMatrix->count;
	int readBuf4Int [4];
	//	MY_FLOAT_TYPE value1;
	int numNonzeroCol=0;
	fread(&numNonzeroCol, sizeof(int), 1, fp);

	for (i = lastCount; i <lastCount+numNonzeroCol/*dataFileNumOfColumn*/; i++) {
		fread(readBuf4Int, sizeof(int), 4, fp);
		matrixNonZeroEntry.columnIndex=readBuf4Int[0];
		iX=readBuf4Int[1];
		iY=readBuf4Int[2];
		iZ=readBuf4Int[3];
		fread(&(matrixNonZeroEntry.value)/*value1*/, sizeof(MY_FLOAT_TYPE), 1, fp);
		//		matrixNonZeroEntry.value=value1;
		matrixNonZeroEntry.rowIndex = rowIndex;
		matrixNonZeroEntry.columnIndex--; // current column index -1, adjust to zero-based indexing ??
		(basicSparseMatrix->nonzeroEntry)[i] = matrixNonZeroEntry;
		basicSparseMatrix->count++;
	}
	fclose(fp);
	return true;
}

// order by row index, if row index is equal, order by column index
int compareRow(const void *entry1, const void *entry2) {

	int rowIndex1 = ((MatNonzeroEntry*) entry1)-> rowIndex;

	int rowIndex2 = ((MatNonzeroEntry*) entry2)-> rowIndex;

	if (rowIndex1 > rowIndex2) {
		return 1;

	} else if (rowIndex1 < rowIndex2) {
		return -1;

	} else {
		// rowIndex1 == rowIndex2

		int columnIndex1 = ((MatNonzeroEntry*) entry1)-> columnIndex;

		int columnIndex2 = ((MatNonzeroEntry*) entry2)-> columnIndex;

		if (columnIndex1 > columnIndex2) {
			return 1;
		} else if (columnIndex1 < columnIndex2) {
			return -1;
		}
	}

	return 0;
}

void saveResult(MY_FLOAT_TYPE *x, int numOfRows, const char* fileName) {

	FILE *fp2 = fopen(fileName, "w");

	if (fp2 == NULL) {
		printf("Fail to write the result:%s !\n", fileName);
	}
	int i;
	for (i = 0; i < numOfRows; i++) {
		fprintf(fp2, "x[%d]=\t%e\n", i, x[i]);
	}

	fclose(fp2);
}

/*

 void changeWorkingDirectory(const char* dirName){
 if(!dirExists("output")){
 mkdir("output", S_IRWXU | S_IRWXG);
 }

 chdir("./output");
 }
 */
/*

PadSpMatCSR* fillPadding(SpMatCSR* sparseMatrix){

        PadSpMatCSR* paddedMatrix =
                (PadSpMatCSR*)malloc(sizeof(PadSpMatCSR));


        int numOfRows = sparseMatrix->numRow;

        paddedMatrix->numRow = numOfRows;

        paddedMatrix->numCol = sparseMatrix->numCol;

        paddedMatrix->numNonzero = sparseMatrix->numNonzero;

        paddedMatrix->row_ptr = (int *)malloc(sizeof(int)*(numOfRows+1));

        int padNZ=0, nnzRow;


        paddedMatrix->row_ptr[0] = 0;
		int i ;
        for ( i = 0; i < numOfRows - 1; i++) {

                nnzRow = sparseMatrix->row_ptr[i+1] - sparseMatrix->row_ptr[i];

                if (nnzRow % HALF_WARP) {
                        // round to the closest multiple of IBM_HALF_WARP
                        padNZ += nnzRow + HALF_WARP -(nnzRow % HALF_WARP);
                        //printf("Row:%d --- OrigNZ: %d -- PadNZ: %d\n",i,nnzRow,(nnzRow + HALFWARP-(nnzRow%HALFWARP)));
                } else {
                        padNZ += nnzRow;
                        //printf("Row:%d --- OrigNZ: %d -- PadNZ: %d\n",i,nnzRow,nnzRow);
                }

                paddedMatrix->row_ptr[i+1] = padNZ;
        }


        nnzRow = sparseMatrix->numNonzero - sparseMatrix->row_ptr[numOfRows-1];


        paddedMatrix->row_ptr[numOfRows] = padNZ + nnzRow;

        if (nnzRow % HALF_WARP) {
                padNZ += nnzRow + HALF_WARP- (nnzRow % HALF_WARP);
                //printf("Row:%d --- OrigNZ: %d -- PadNZ: %d\n",numOfRows-1,nnzRow,(nnzRow + HALFWARP-(nnzRow%HALFWARP)));
        } else {
                padNZ += nnzRow;
                //printf("Row:%d --- OrigNZ: %d -- PadNZ: %d\n",numOfRows-1,nnzRow,nnzRow);
        }

//      paddedMatrix->ptr[numOfRows] = padNZ;

        paddedMatrix->val = (MY_FLOAT_TYPE *)malloc(sizeof(MY_FLOAT_TYPE)*padNZ);
        paddedMatrix->col_idx = (int *)malloc(sizeof(int)*padNZ);

    for ( i = 0; i < sparseMatrix->numRow; i++) {

        int firstEntryInRow = sparseMatrix->row_ptr[i];
        int lastEntryInRow = sparseMatrix->row_ptr[i+1];

        int newFirstEntryInRow = paddedMatrix->row_ptr[i];
        int newLastEntryInRow = paddedMatrix->row_ptr[i+1];

        int newEntryIndex = newFirstEntryInRow;

				int entryIndex;
                for ( entryIndex = firstEntryInRow;
                        entryIndex < lastEntryInRow; entryIndex++) {

                        int columnIndex = sparseMatrix->col_idx[entryIndex];

                        paddedMatrix->val[newEntryIndex] =
                                sparseMatrix->val[entryIndex];

                        paddedMatrix->col_idx[newEntryIndex] = columnIndex;

                        newEntryIndex++;

                }


                while(newEntryIndex < newLastEntryInRow){
                        paddedMatrix->val[newEntryIndex] = 0;

                        paddedMatrix->col_idx[newEntryIndex] = 0;

                        newEntryIndex++;
                }

    }

    return paddedMatrix;
}
*/


/*
PaddedSparseMatrixCSR* fillPadding(SparseMatrixCSR* sparseMatrix) {

	PaddedSparseMatrixCSR* paddedMatrix =
			(PaddedSparseMatrixCSR*)malloc(sizeof(PaddedSparseMatrixCSR));

	int numOfRows = sparseMatrix->numOfRows;

	paddedMatrix->numOfRows = numOfRows;

	paddedMatrix->numOfColumns = sparseMatrix->numOfColumns;

	paddedMatrix->numNonzero = sparseMatrix->numOfNonZeroData;

	paddedMatrix->ptr = (int *)malloc(sizeof(int)*(numOfRows+1));

	int padNZ=0, nnzRow;

	paddedMatrix->ptr[0] = 0;

	int i = 0;

	for (; i < numOfRows - 1; i++) {

		nnzRow = sparseMatrix->rowIndex[i+1] - sparseMatrix->rowIndex[i];

		if (nnzRow % HALF_WARP) {
			// round to the closest multiple of IBM_HALF_WARP
			padNZ += nnzRow + HALF_WARP - (nnzRow % HALF_WARP);
			//printf("Row:%d --- OrigNZ: %d -- PadNZ: %d\n",i,nnzRow,(nnzRow + HALFWARP-(nnzRow%HALFWARP)));
		} else {
			padNZ += nnzRow;
			//printf("Row:%d --- OrigNZ: %d -- PadNZ: %d\n",i,nnzRow,nnzRow);
		}

		paddedMatrix->ptr[i+1] = padNZ;
	}

	nnzRow = sparseMatrix->numOfNonZeroData
			- sparseMatrix->rowIndex[numOfRows-1];

	paddedMatrix->ptr[numOfRows] = padNZ + nnzRow;

	if (nnzRow % HALF_WARP) {
		padNZ += nnzRow + HALF_WARP - (nnzRow % HALF_WARP);
		//printf("Row:%d --- OrigNZ: %d -- PadNZ: %d\n",numOfRows-1,nnzRow,(nnzRow + HALFWARP-(nnzRow%HALFWARP)));
	} else {
		padNZ += nnzRow;
		//printf("Row:%d --- OrigNZ: %d -- PadNZ: %d\n",numOfRows-1,nnzRow,nnzRow);
	}

	//	paddedMatrix->ptr[numOfRows] = padNZ;

	paddedMatrix->data = (MY_FLOAT_TYPE *)malloc(sizeof(MY_FLOAT_TYPE)*padNZ);
	paddedMatrix->index = (int *)malloc(sizeof(int)*padNZ);

	i = 0;

	for (; i < sparseMatrix->numOfRows; i++) {

		int firstEntryInRow = sparseMatrix->rowIndex[i];
		int lastEntryInRow = sparseMatrix->rowIndex[i+1];

		int newFirstEntryInRow = paddedMatrix->ptr[i];
		int newLastEntryInRow = paddedMatrix->ptr[i+1];

		int newEntryIndex = newFirstEntryInRow;

		int entryIndex = firstEntryInRow;
		for (; entryIndex < lastEntryInRow; entryIndex++) {

			int columnIndex = sparseMatrix->column[entryIndex];

			paddedMatrix->data[newEntryIndex]
					= sparseMatrix->column[entryIndex];

			paddedMatrix->index[newEntryIndex] = columnIndex;

			newEntryIndex++;

		}

		while (newEntryIndex < newLastEntryInRow) {
			paddedMatrix->data[newEntryIndex] = 0;

			paddedMatrix->index[newEntryIndex] = 0;

			newEntryIndex++;
		}

	}

	return paddedMatrix;
} */

void freePaddedMatrix(PadSpMatCSR* paddedMatrix) {
	free(paddedMatrix->val);
	free(paddedMatrix->col_idx);
	free(paddedMatrix->row_ptr);
	free(paddedMatrix);
}

/*void getMatrixDimension(const char* dirName, int& matrix_rows, int& matrix_cols){
 DIR * dirEntry = opendir(dirName);

 }*/

int totalSize_GPUSparseMatrixCSR(){
	return(sizeof(GPUSparseMatrixCSR));
}
