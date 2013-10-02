#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>


/* Feb 9, 2013
 * a.out input_file input_dir output_dir
 *
 * input_file(list: first line: number of rows; second to last line: row file name)
 *
 *
 * mpiexec -np 8 ./ker_ascii_bin_intintdouble_MPI row_list_1.txt /seismic10/hhuang1/elee8_20130204/test_seismicmst /seismic10/hhuang1/elee8_20130204/test_seismicmst
 * */

#define NLEN 1024

int main(int argc, char *argv[]) {

	int rank, mpisize;
	FILE *ifp;
	FILE *infile;
	FILE *outfile;
	char kerout[NLEN];
	char kerfnam[NLEN];
	int ker_num, mpi_count;
	int* rank_id;
	int ik, col, ix, iy, iz;
	double ker;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpisize);

	ifp = fopen(argv[1], "r");
	if (ifp == NULL ) {
		fprintf(stderr, "Can't open input file!\n");
		exit(1);
	}


	fscanf(ifp, "%d", &ker_num);
	printf("%d cores and %d files\n", mpisize, ker_num);

	//***** assign rank ID
	rank_id = (int*) malloc(sizeof(int) * ker_num);
	mpi_count = 0;
	for (ik = 0; ik < ker_num; ik++) {
		rank_id[ik] = mpi_count;
		mpi_count++;
		if (mpi_count == mpisize)
			mpi_count = 0;
	}

	//  ****** convert ascii to bin ******
	for (ik = 0; ik < ker_num; ik++) {
		fscanf(ifp, "%s", kerfnam);
		if (rank == rank_id[ik]) {
			sprintf(kerout, "%s.bin", kerfnam);
			printf("rank %d : convert %s to %s\n", rank, kerfnam, kerout);

			//full name
			char kerfnam_fullname[NLEN];
			char kerout_fullname[NLEN];
			sprintf(kerfnam_fullname, "%s/%s", argv[2], kerfnam);
			printf("input: %s\n", kerfnam_fullname);
			sprintf(kerout_fullname, "%s/%s", argv[3], kerout);
			printf("output: %s\n", kerout_fullname);

			infile = fopen(kerfnam_fullname, "r");
			outfile = fopen(kerout_fullname, "wb");
			if(!infile || !outfile){
				fprintf(stderr, "cannot open %s or %s\n", kerfnam_fullname, kerout_fullname);
			}

			//=== read file & convert to bin
			while ((fscanf(infile, "%d %d %d %d %lf", &col, &ix, &iy, &iz, &ker))
					!= EOF) {
				//fwrite(&col,1,sizeof(int),outfile);
				//fwrite(&ker,1,sizeof(double),outfile);
				fwrite(&ix, sizeof(int), 1, outfile);
				fwrite(&col, sizeof(int), 1, outfile);
				fwrite(&ker, sizeof(double), 1, outfile);
			}

			fclose(infile);
			fclose(outfile);
		}	  //if
	}	  //for

	printf("rank %d : Finish!!\n", rank);
	free(rank_id);
	MPI_Finalize();
}	  // main
