/*
 * loadbalance.c
 *
 *  Created on: Dec 13, 2009
 *      Author: hhuang1@uwyo.edu
 */

#include "loadbalance.h"


/* 3/28
 * partition kernel vector into vector per process
 * construct vector per proc according to damping matrix*/
void kernelVectorPartition_perProcVectorConstruct(int procRank, MY_FLOAT_TYPE reordered_vectorB[], MY_FLOAT_TYPE perProcVector[],
		int sizeOfprocDistributedDataFileSet_k[], int numRowPerDampingFile)
{
	int perProc_beginIdx=0;
	int i;
	for(i=0;i<procRank; i++){
		perProc_beginIdx+=sizeOfprocDistributedDataFileSet_k[i];
	}
	int perProc_kernel_vector_dim=sizeOfprocDistributedDataFileSet_k[procRank];
	//copy kernel vector
	for(i=perProc_beginIdx; i<perProc_beginIdx+perProc_kernel_vector_dim; i++){
		perProcVector[i-perProc_beginIdx]=reordered_vectorB[i];
	}
	//copy damping vector
	for(i=perProc_kernel_vector_dim; i<perProc_kernel_vector_dim+numRowPerDampingFile; i++ ){
		perProcVector[i]=0.0;
	}

	//display the vector
/*	char filename[1024];
	sprintf(filename, "./vector/perProcVector_rank=%d", procRank);
	FILE *fp1 =fopen( filename, "w");
	if(fp1==NULL){
		fprintf(stderr, "Cannot open the file: %s\n", filename);
		exit(EXIT_FAILURE);
	}
	fprintf(fp1, "sizeOfprocDistributedDataFileSet_k[procRank]=%d, numRowPerDampingFile=%d\n",
			sizeOfprocDistributedDataFileSet_k[procRank], numRowPerDampingFile);
	for (i = 0; i <perProc_kernel_vector_dim+numRowPerDampingFile ; i++){
		fprintf(fp1, "%e\n", perProcVector[i] );
	}
	fclose(fp1);*/

}


 /* load balancing according to the nonzero entity of every row file.
  * The goal is make every process have even number of nonzero entity.
  * Before invoke this function, the data files are already sorted by reverse order of nonzero entity,
  * which means dataFileNumOfColumn[0] has the greatest number of nonzero entity,
  * and dataFileNumOfColumn[MAX-1] has the smallest number of nonzero entity.
  * Method: distributed data file with greatest number of columns to the process
  * with lowest total number of columns( summation of the total files' columns)
  */
void loadBalanceByNonzeroEntity(/*int procRank,*/ int numProc, int numOfDataFile, char *dataFileName[],
		int  dataFileNumOfColumn[], int* procDistributedDataFileSet[], int sizeOfprocDistributedDataFileSet[],
		int numNonzero_procDistributedDataFileSet[])
{

	int i;
/*	for(i=0; i<numProc; i++){
		numNonzero_procDistributedDataFileSet[i]=0;
	}*/

	//load balance technique
	for(i=0; i<numOfDataFile; i++)
	{
		int currentMinProcIndex=0;
		currentMinProcIndex=currentMinProc(numProc,procDistributedDataFileSet,sizeOfprocDistributedDataFileSet, numNonzero_procDistributedDataFileSet);
		procDistributedDataFileSet[currentMinProcIndex][sizeOfprocDistributedDataFileSet[currentMinProcIndex]]=i;
		numNonzero_procDistributedDataFileSet[currentMinProcIndex]+=dataFileNumOfColumn[i];
		sizeOfprocDistributedDataFileSet[currentMinProcIndex]++;
	}

//print result to a file
	FILE * fpschedule=fopen("LoadBalanceScheduleWithOriginalRowIndex.txt", "w");
	if(fpschedule==NULL){
		fprintf(stderr, "Cannot open the file LoadBalanceScheduleWithOriginalRowIndex.txt\n");
		exit(EXIT_FAILURE);
	}
	int sumOfCount=0;
	int sumOfIndex=0;
	for(i=0; i<numProc; i++)
	{
#ifdef DEBUG
		printf("\nprocRank=%d: sizeOfprocDistributedDataFileSet[%d]=%d, countOfprocDistributedDataFileSet[%d]=%d :\n",
				i, i, sizeOfprocDistributedDataFileSet[i],i, numNonzero_procDistributedDataFileSet[i]);
#endif
		sumOfCount+=numNonzero_procDistributedDataFileSet[i];
		sumOfIndex+=sizeOfprocDistributedDataFileSet[i];
		int j;
		for(j=0; j<sizeOfprocDistributedDataFileSet[i]; j++)
		{
			int filenameIdx=procDistributedDataFileSet[i][j];
			fprintf(fpschedule, "%d\t%s\t%d\n",filenameIdx, dataFileName[filenameIdx], dataFileNumOfColumn[filenameIdx]);
			printf("%d\t", filenameIdx);
		}
		printf("\n");
	}
	int sum=0;
	for(i=0; i<numOfDataFile; i++){
		sum+=dataFileNumOfColumn[i];
	}
	int average=sum/numProc/*(numProc-1)*/;
#ifdef DEBUG
	printf("sumOfIndex=%d, sumOfCount=%d, sum=%d, average=%d",sumOfIndex, sumOfCount, sum, average);
#endif

	fclose(fpschedule);
//print result


}


int currentMinProc(int numProc, int *procDistributedDataFileSet[], int sizeOfprocDistributedDataFileSet[],
		int numNonzero_procDistributedDataFileSet[])
{
	int min=numNonzero_procDistributedDataFileSet[0];
	int minIndex=0/*1*/;
	int i;
	for(i=0/*1*/; i<numProc; i++){// exclude proc==0
		if(min>numNonzero_procDistributedDataFileSet[i])
		{
			min=numNonzero_procDistributedDataFileSet[i];
			minIndex=i;
		}
	}
	return minIndex;
}


//convert the original vectorB to the re-ordered vectorB
//new vector is stored in vectorB
void reorderVectorB(int numProc,  MY_FLOAT_TYPE vectorB[],const MY_FLOAT_TYPE oriVectorB[], const int vectorB_dimension,
		const int sizeOfprocDistributedDataFileSet[],  int* procDistributedDataFileSet[])
{
	int i,j,k=0;
	for(i=0; i<numProc; i++)
	{
		for(j=0; j<sizeOfprocDistributedDataFileSet[i]; j++)
		{
			int filenameIdx=procDistributedDataFileSet[i][j];
			vectorB[k]=oriVectorB[filenameIdx];
			if(k<vectorB_dimension) k++;
			else fprintf(stderr, "larger than the vector's dimension=%d!\n", vectorB_dimension);
		}
	}
	//display the vector
/*	FILE * fp=fopen("reordered_vectorB", "w");
	if(fp==NULL){
		fprintf(stderr,"Cannot open the file: %s  \n","reordered_vectorB");
		exit(-1);
	}
	for (i = 0; i < vectorB_dimension; i++){
		fprintf(fp, "%e\n", vectorB[i] );
	}
	fclose(fp);*/
	//
}
