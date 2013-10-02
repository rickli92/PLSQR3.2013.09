/*
 * loadbalance.h
 *
 *  Created on: Dec 13, 2009
 *      Author: hh
 */

#ifndef LOADBALANCE_H_
#define LOADBALANCE_H_

#include <stdlib.h>
#include <stdio.h>
#include "main.h"
void kernelVectorPartition_perProcVectorConstruct(int procRank, MY_FLOAT_TYPE reorderedVectorB[], MY_FLOAT_TYPE perProcVector[],
		int sizeOfprocDistributedDataFileSet_k[], int numRowPerDampingFile);


void loadBalanceByNonzeroEntity(/*int procRank,*/ int numProc, int numOfDataFile, char *dataFileName[],
		int  dataFileNumOfColumn[], int* procDistributedDataFileSet[], int sizeOfprocDistributedDataFileSet[],
		int countOfprocDistributedDataFileSet[]);

int currentMinProc(int numProc, int *procDistributedDataFileSet[], int sizeOfprocDistributedDataFileSet[],int countOfprocDistributedDataFileSet[]);

void reorderVectorB(int numProc,  MY_FLOAT_TYPE vectorB[],const MY_FLOAT_TYPE oriVectorB[], const int vectorB_dimension,
		const int sizeOfprocDistributedDataFileSet[],  int* procDistributedDataFileSet[]);
#endif /* LOADBALANCE_H_ */
