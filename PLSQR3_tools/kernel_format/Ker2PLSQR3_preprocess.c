/*
 * Ker2PLSQR3_preprocess.c
 *
 *  Created on: Feb 8, 2013
 *      Author: hh
 *  argv[1]: kernel file directory path
 *  argv[2]: kernel file list (1st col: row binary file name; 2nd col nnz per row)
 *  argv[3]: number of column
 *
 *  ./Ker2PLQSR3_preprocess /seismic10/hhuang1/elee8_20130204/test_seismicmst ker_test3_v2.bin.list_60_s 38093022
 *
 *  OUTPUT: col_info.txt(ascii): nnz per col
 *  */

#include "../../source/FileIO.h"
#include "../../source/main.h"
#include <time.h>
#include <string.h>


struct tuple{
  int iX;//not useful
  int col_idx;
  double val;
};

int main(int argc, char * argv[]){
	char dirpath[MAXREADNUM];
	char kernelFileList[MAXREADNUM];
	char kerFileListFullName[MAXREADNUM];
	int numRow=0, numCol=0, i=0, j=0, k=0;

	if(argc==4){
		strcpy(dirpath, argv[1]);
		strcpy(kernelFileList,argv[2]);
		numCol=atoi(argv[3]);
	}
	sprintf(kerFileListFullName, "%s/%s",dirpath,kernelFileList);

	printf("sizeof (long long) is %d bytes\n", sizeof(long long));
	printf("sizeof tuple is %d\n", sizeof(struct tuple));

	numRow = countData(kerFileListFullName);
	printf("The number of kernel file is %d\n", numRow);


	char **kernelFileName;
	if (!(kernelFileName = (char **) calloc(numRow, sizeof(char *)))){
		printf("error: calloc fails\n");fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
	}
	for (i = 0; i < numRow; i++)
		if (!(kernelFileName[i] = (char *) calloc(MAXREADNUM, sizeof(char)))){// the length of every data file name should not be bigger than 256 char
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
	for(i=0; i<numCol; i++){
		nnzPerCol[i]=0;
	}

	for (i = 0; i < numRow; i++) {//1st iterate every kernel file
		char kernelFileFullName[MAXREADNUM];
		sprintf(kernelFileFullName, "%s/%s",dirpath, kernelFileName[i]);
		printf("1st loading dataFileName[%d] = %s, number of nonzero per row=%d\n",
				i, kernelFileFullName, NumNZPerrow_k[i]);


		//FILE * fip_k=fopen(kernelFileFullName, "r");
		//add read binary files Feb 1, 2013
		FILE * fip_k=fopen(kernelFileFullName, "rb");
		time_t start,end;
		time (&start);
		if(fip_k==NULL){
			fprintf(stderr, "Cannot open the file: %s  \n", kernelFileFullName);
			exit(-1);
		}
		int colIdx, iX, iY, iZ;
		double val;

		struct tuple * read_array = (struct tuple * )malloc(sizeof(struct tuple)*NumNZPerrow_k[i]);
		if(!read_array) fprintf(stderr, "cannot allocat read_array\n" );
		memset(read_array, 0, sizeof(struct tuple)*NumNZPerrow_k[i]);
		fread(read_array, sizeof(struct tuple), NumNZPerrow_k[i], fip_k);
		printf("%d %d %le\n", read_array[0].iX,read_array[0].col_idx, read_array[0].val);
		for(j=0; j<NumNZPerrow_k[i]; j++){
			int c1=5;
			//int c1=fscanf(fip_k, "%d %d %d %d %le", &colIdx, &(iX), &(iY), &(iZ), &val);
			colIdx=read_array[j].col_idx;
			if(c1==5){
				colIdx--;//adjust to 0-based indexing
				nnzPerCol[colIdx]++;
			}
		}
		time (&end);
		double dif = difftime (end,start);
		free(read_array);
		printf ("It takes %.2f seconds to read the file.\n", dif );
		fclose(fip_k);
	}


//write column information
	char filename[MAXREADNUM];
	sprintf(filename, "%s/col_info.txt", dirpath);
	printf("output %s\n", filename);
	FILE * fout_col_info = fopen(filename ,"w");
	if(!fout_col_info){
		fprintf(stderr, "%s(%d)-%s: \n", __FILE__,__LINE__,__FUNCTION__);exit(-1);
	}
	for(i=0; i<numCol; i++){
		fprintf(fout_col_info, "%d\n", nnzPerCol[i]);
	}
	fclose(fout_col_info);
	for(i=0;i<numRow; i++){
		free(kernelFileName[i]);
	}
	free(kernelFileName);
	free(nnzPerCol);
	return 0;
}
