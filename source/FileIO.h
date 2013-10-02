#ifndef FILEIO_H_

#define FILEIO_H_




#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <ctype.h>
#include <sys/stat.h>
#include <unistd.h>
#include <string.h>
#include <dirent.h>
#include <stdbool.h>     /* C99 feature         */
//#include "./tool/mmio.h"
#include "main.h"




void readDampingInfo(char filename[], int numDampingFile, char * dampingFileName[],
		int  dampingFileNumRow[], int  dampingFileNumNonzero[]);

//void testreadDampingInfo();

void init_merge_BasicSpMatWithDamping(int procRank, BasicSpMat * basicSparseMatrix_kernel, BasicSpMat * basicSparseMatrix_damping,
		BasicSpMat * basicSparseMatrix_whole);

void loadDamping(const char dampingFileName[], BasicSpMat * basicSparseMatrix_damping, int rowBeginIdx);

void testLoadDamping();

void readConfig(const char configFilename[], char dirpath[], IOTYPE * inputType,
		int * lsqr_iteration, int * numColumn, char asciiDataStatistic[],
		char binaryDataStatistic[], char vectorBInfo[], bool isParallel,
		bool isLoadBalance, char damping_path[], char damping_info[]);

void printConfigParameter(const int procRank, const char configFilename[],
		const char dirpath[], const bool inputType, const int * lsqr_iteration,
		const char asciiDataStatistic[], const char binaryDataStatistic[],
		const char vectorBInfo[], const bool isParallel,
		const bool isLoadBalance);

int countData(const char * filename);

void getDataFileName(char * dataFileName[], int * dataFileNumOfColumn, const char * filename, int numOfDataFile);

void dataDistribut( char ** dataFileName, const int dataFileNumOfColumn[], const int numOfDataFile,
		const int procRank, const int numProc,  int * first,  int * last, int * numOfNonZeroEntriesPerProc);

void init_Per_Proc_Data_v2(int numProc, int procRank, Per_Proc_Data_v2 *perProcData,
		int matrix_rows, int matrix_cols, SpMatCSR * sparseMatrixCSR,
		int sizeOfprocDistributedDataFileSet[], int numRowPerDampingFile[]);

void print_Per_Proc_Data_BeforeCaculationtoFile(int rank, int numProc, Per_Proc_Data_v2 *perProcData);

void finialize_Per_Proc_Data_v2(int rank, int numProc, Per_Proc_Data_v2 *perProcData);

void print_Per_Proc_Data_AfterCaculationtoFile(int procRank, int numProc, Per_Proc_Data_v2 *perProcData);

void readInputVector(const char * filename,MY_FLOAT_TYPE vectorB[], int vectorB_dimension);
//MY_FLOAT_TYPE* readInputVector(const char *filename, int numCols);

//MY_FLOAT_TYPE* readInputVectorInBinary(const char* fileName, int numCols);

void print_vector(const MY_FLOAT_TYPE * vector, const int dimension, const char filename[]);
void print_vector_int(const int * vector, const int dimension,const char filename[]);

void generateRandomVector(MY_FLOAT_TYPE *, int);

//void printSparseMatrixCSR(SparseMatrixCSR *sparseMatrix);

void printSparseMatrixCSRtoFile(int rank, SpMatCSR *sparseMatrix);

void printBasicSpMat(BasicSpMat *sparseMatrix);

void printBasicSpMattoFile(int rank, BasicSpMat *basicSparseMatrix);


void printPaddedSparseMatrixCSR(PadSpMatCSR *sparseMatrix);

SpMatCSR* convertAndLoadBinaryFile(const char *fileName);

void init_SparseMatrixCSR_withBasicSpMat(SpMatCSR* matrixCSR, BasicSpMat* basicSparseMatrix);

void initSpMatCSR(SpMatCSR* matrixCSR, int numOfRows, int numOfColumns, int numOfNonZeroEntries);

//void finalizeSpMatCSR(SpMatCSR* matrixCSR);
//SparseMatrixCSR* convertToSparseMatrixCSR(BasicSpMat* basicSparseMatrix);
//void basicSpMatTOSpMatCSR(SpMatCSR* matrixCSR, BasicSpMat* basicSparseMatrix);


void loadMatrixCSR_ascii_kernelonly(SpMatCSR* matrixCSR,
		/*kernel*/
		char dirpath_k[], char * kernelDataFileName[],
		int dataFileNumNonzeroPerrow_k[], int perProcDisDataFileSet_k[],
		int perProcSizeOfDisDataFileSet_k);

void loadMatrixCSR_ascii(int procRank, SpMatCSR* matrixCSR,
		/*kernel*/
		char dirpath_k[],
		char * kernelDataFileName[],
		int  dataFileNumOfColumn_k[],
		int perProcDisDataFileSet_k[],
		int perProcSizeOfDisDataFileSet_k,
		/*damping*/
		char damping_path[],
		char * dampingFileName[],
		int dampingFileID);

void loadMatrixCSR_bin(SpMatCSR* matrixCSR,
		/*kernel*/
		char dirpath_k[], char * kernelDataFileName[],
		int dataFileNumNonzeroPerrow_k[], int perProcDisDataFileSet_k[],
		int perProcSizeOfDisDataFileSet_k,
		/*damping*/
		char damping_path[], char * dampingFileName[], int dampingFileID);

SpMatCSR* loadMMFile(const char* fileName);

//void saveResult(MY_FLOAT_TYPE *x, int numOfRows, const char* fileName = "result.mtx");

int compareRow(const void *entry1, const void *entry2);

//BasicSpMat* loadBasicSpMat(const char* fileName);

//int getNumOfRows(const char* fileName);



void changeWorkingDirectory(const char* dirName);

//PadSpMatCSR* fillPadding(SpMatCSR* sparseMatrix);
void freePaddedMatrix(PadSpMatCSR* paddedMatrix);

//void getMatrixDimension(const char* dirName, int& matrix_rows, int& matrix_cols);

//SparseMatrixCSR* load_csr_matrix(const char* fileName, int rowIndex);
//void load_csr_matrix(SparseMatrixCSR* matrixCSR, const char* fileName, int rowIndex);



//BasicSpMat* loadBasicSpMatInGrid(const char* fileName, int rowIndex);
//bool loadBasicSpMatInGrid(BasicSpMat* basicSparseMatrix,const char* fileName, int rowIndex);

bool loadBasicSpMatInGrid(BasicSpMat* basicSparseMatrix,
		const char* fileName, int dataFileNumOfColumn, int rowIndex) ;
bool loadBasicSpMatInGridBin(BasicSpMat* basicSparseMatrix,

		const char* fileName, int dataFileNumOfColumn, int rowIndex);
int readTimeShift(const char* fileName);

int totalSize_GPUSparseMatrixCSR();

#endif /*FILEIO_H_*/
