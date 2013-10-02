/* Created by En-Jui,
 * Version 3
 * 3/10/2013
 *
 * argv[1] kernel info by col file, ascii
 * argv[2] damping file by col, binary
 * argv[3] number of column
 * argv[4] number of row in damping matrix
 * argv[5] number of non-zero in kernel matrix
 * argv[6] number of non-zero in damp
 * argv[7] max col/ave_col ratios
 * argv[8] max element/ave_element ratios
 * argv[9] number of core
 *
 * ./load_balance /home/hhuang1/seismic10.lnk/angf_decimate/dec5_plsqr3_data/matrix_bycol.mm2.info /home/hhuang1/seismic10.lnk/angf_decimate/dec5_plsqr3_data/plsqr3_reorder_dec5_all.damp.bycol.bin 302940 1910672 48059714 5895716 16
 *
 * ./load_balance_v2 matrix_bycol_v3.mm2.info damp_v7_D01_S01_col_data.bin 38093022 261330576 68074889274 818542016 64
 mv damp_row.index damp_row_64.index
 The output "damp_row_64.index" starts from the first row of damping matrix not whole matrix A.
 mv ker_col.index ker_col_64.index*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <ctype.h>
#include <string.h>

struct dampdata {
	int row_idx;
	int col_idx;
	double val;
};

#define MAX_dampCol 2500

long long str2long64(const char * str);

int main(int argc, char **argv) {

	char kernelInfoList[300];
	char dampColData[300];
	int core_num = 0, numCol = 0, dam_nz = 0, numDampRow = 0, i1 = 0;
	long long ker_nz = 0;
	double avg_nz = 0.0;
	double col_ratio=0.0,elem_ratio=0.0;
	double avg_col = 0.0, avg_damp_col = 0.0;
	int core_count = 0;
	//  int MAX_dampCol=2000;
	FILE* infile;
	FILE* outfile1;
	FILE* outfile2;

	//  strcpy(kernelInfoList, argv[1]); //kernel info file
	//  strcpy(dampColData,argv[2]);  // damp col file
	numCol = atoi(argv[3]); // col number of A
	printf("numCol: %d\n", numCol);
	numDampRow = atoi(argv[4]); // row number of damping
	printf("numDampRow: %d\n", numDampRow);
	ker_nz = str2long64(argv[5]); //
	printf("ker_nz: %lld\n", ker_nz);
	dam_nz = atoi(argv[6]); // number of non-zero in damp
	printf("dam_nz: %d\n", dam_nz);
	col_ratio = atof(argv[7]); // col ratio
	printf("col_ratio: %lf\n", col_ratio);
	elem_ratio = atof(argv[8]); // element ratio
	printf("elem_ratio: %lf\n", elem_ratio);
	core_num = atoi(argv[9]); //number of core
	printf("core_num: %d\n", core_num);

	// for storing start & end index
	int *ker_col_start, *ker_col_end, *damp_row_start, *damp_row_end;
	//int *damp_row_values;
	ker_col_start = (int*) malloc(sizeof(int) * core_num);
	ker_col_end = (int*) malloc(sizeof(int) * core_num);
	//ker_col_start[core_num] = {0};
	//ker_col_end[core_num] = {0};

	damp_row_start = (int*) malloc(sizeof(int) * core_num);
	damp_row_end = (int*) malloc(sizeof(int) * core_num);
	int *damp_row_values;
	damp_row_values = (int*) malloc(sizeof(int) * MAX_dampCol);
	//int damp_row_values[2500];
	// damp_row_start[core_num] = {0};
	//damp_row_end[core_num] = {0};

	avg_nz = ((double) (ker_nz + dam_nz + dam_nz + numDampRow + numCol))
			/ core_num; //average non-sero number
	printf("average nz: %lf\n", avg_nz);
	avg_col = ((double) numCol) / core_num; //average col number
	printf("average col: %lf\n", avg_col);
	avg_damp_col = ((double) (dam_nz + dam_nz + numDampRow)) / numCol; // average damp nz in each column
	printf("average damp nz in each column: %f\n", avg_damp_col);

	// partition column
	// (1) if accum_col > avg_col*col_ratio used the colest col
	// (2) if accum_nz2 > avg_nz*elem_ratio used the colest col
	int col = 0, nz_num = 0;
	int col_err=0, elem_err=0;
	long accum_nz = 0;
	long accum_nz2 = 0,accum_col =0;

	ker_col_start[0] = 1; // first start col = 1
	ker_col_end[core_num - 1] = numCol; //last end col = numCol
	core_count=0;
	infile = fopen(argv[1], "r");
	while ((fscanf(infile, "%d %d %ld", &col, &nz_num, &accum_nz)) != EOF) {
	  accum_nz2 = accum_nz2 + nz_num;
	  accum_col++;
	  //(1)accum_col > avg_col*col_ratio used the colest col
	  if(accum_col>=((double)(avg_col*col_ratio))){
	    core_count++;
	    ker_col_end[core_count - 1]=col;
	    ker_col_start[core_count] = col + 1;
	    //printf("core %d : col %d\n",core_count,col);
	    //printf("core %d : col %d - %d\n",core_count,ker_col_start[core_count - 1],ker_col_end[core_count - 1]);
	    accum_nz2=0;
	    accum_col=0;
	  }
	  // (2) if accum_nz2 > avg_nz*elem_ratio used the colest col
	  if((accum_nz2+accum_col+(avg_damp_col*accum_col))>=((double)(avg_nz*elem_ratio))){
	    core_count++;
	    ker_col_end[core_count - 1]=col;
	    ker_col_start[core_count] = col + 1;
	    //printf("core %d : col %d - %d\n",core_count,ker_col_start[core_count - 1],ker_col_end[core_count - 1]);
	    accum_nz2=0;
	    accum_col=0;
	    }
	  // break when the last one is filled
	  if(core_count == (core_num - 1)){
	    //printf("core %d : col %d - %d\n",core_count,ker_col_start[core_count],ker_col_end[core_count]);
	    //(1)check left col
	    if((numCol-col)>((double)(avg_col*col_ratio))){
	      fprintf(stderr, "There are %d col left, please increase col_ratio!\n",numCol-col);
	      col_err=1;
	    }
	    printf("%d cols in last core.\n",ker_col_end[core_count]-ker_col_start[core_count]);
	    //(2)check left elements
	    if((ker_nz+dam_nz+dam_nz+numDampRow+numCol)-accum_nz-col-(avg_damp_col*col)>((double)(avg_nz*elem_ratio))){
	      fprintf(stderr, "Too many elements left, please increase elem_ratio!\n");
	      elem_err=1;
	    }
	    if(elem_err>0||col_err>0)
	      return 1;
	    
	    break;
	  }
	  /*
	  
		//check non-zero exceed or not
		if ((accum_nz2 + col + (avg_damp_col * col))
				>= (avg_nz * (core_count + 1))) {
			// col number > avg_col
			//printf("coue_count : %d ; accum_nz : %ld\n",core_count,accum_nz);
			core_count++;
			if ((col - ker_col_start[core_count - 1]) > avg_col) {
				ker_col_end[core_count - 1] = col - 1;
				ker_col_start[core_count] = col;
			} else {
				ker_col_end[core_count - 1] = col;
				ker_col_start[core_count] = col + 1;
			}
		}

		// break when the last one is filled
		if (core_count == (core_num - 1)) {
			break;
		}
	  */
	}
	fclose(infile);
	//ckeck core
	if(core_count<(core_num - 1)){
	  fprintf(stderr, "The cores are not filled, please decrease col_ratio or elem_ratio!\n");
	  return 1;
	}

	// read damping in once
	infile = fopen(argv[2], "rb");

	struct dampdata * read_array = (struct dampdata *) malloc(
			sizeof(struct dampdata) * dam_nz);
	if (!read_array)
		fprintf(stderr, "cannot allocat read_array\n");
	fread(read_array, sizeof(struct dampdata), dam_nz, infile);
	fclose(infile);

	int damp_row_count = 0;
	int i2 = 0, sel_count = 0;
	long row_mean = 0;

	damp_row_start[0] = 1; // first start col = 1
	damp_row_end[core_num - 1] = numDampRow; //last end col = numCol
	core_count = 0;
	for (i1 = 0; i1 < dam_nz; i1++) {
		// find first nz in the column
		if (read_array[i1].col_idx == ker_col_end[core_count]) {
			damp_row_values[damp_row_count] = read_array[i1].row_idx;
			damp_row_count++;
		} else if (read_array[i1].col_idx > ker_col_end[core_count]) {
			//(1)start_row > previous end row
			//(2)close to mean row
			row_mean = 0;
			sel_count = 0;
			for (i2 = 0; i2 < damp_row_count; i2++) {
				if (damp_row_values[i2] > damp_row_start[core_count - 1]) {
				  sel_count++;
					row_mean = row_mean + damp_row_values[i2];
				}
			}
			//(2)close to mean row
			//printf("core_count %d & sel_count = %d\n",core_count,sel_count);
			row_mean = row_mean / sel_count;
			core_count++;
			for (i2 = 0; i2 < damp_row_count; i2++) {
				if (damp_row_values[i2] >= row_mean) {
					damp_row_end[core_count - 1] = damp_row_values[i2];
					damp_row_start[core_count] = damp_row_values[i2] + 1;
					//	  printf("core_count: %d\n",core_count);
					damp_row_count = 0;
					break;
				}
			}
		}      //else if

		// break when the last one is filled
		if (core_count == (core_num - 1)) {
			break;
		}
	}

	free(read_array);

	//print out results
	outfile1 = fopen("ker_col.index", "w");
	outfile2 = fopen("damp_row.index", "w");
	for (i1 = 0; i1 < core_num; i1++) {
		fprintf(outfile1, "%d  %d\n", ker_col_start[i1], ker_col_end[i1]);
		fprintf(outfile2, "%d  %d\n", damp_row_start[i1], damp_row_end[i1]);
	}
	fclose(outfile1);
	fclose(outfile2);

	free(ker_col_start);
	free(ker_col_end);
	free(damp_row_start);
	free(damp_row_end);
	free(damp_row_values);

	return 0;
}

//convert unsigned integer from string to long long
long long str2long64(const char * str) {
	long long num = 0;
	int n = strlen(str);
	int i = 0;
	for (i = 0; i < n; i++) {
		assert(isdigit(str[i]));
		num = num * 10 + (str[i] - '0');
	}
	printf("convert string %s to long long %lld\n", str, num);
	return num;
}
