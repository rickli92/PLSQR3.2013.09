###############################################################################
#make //for cpu
#
#Blue Waters:
#Cray PAPI: make CC=cc INCLUDE='-O3 ${PAPI_INCLUDE_OPTS}' LINK='-O3 -lm ${PAPI_POST_LINK_OPTS}'
#OpenMP make CC='cc -O3 -h omp'
#
#Yellowstone
# mpi-selector --set mvapich2-1.7
# use Intel compiler and IMPI(impi)
#PAPI
# module load papi
# export PAPI_INCLUDE_OPTS=-I/glade/apps/opt/papi/5.1.0.2/gnu/4.7.2/include
# export PAPI_POST_LINK_OPTS='-L/glade/apps/opt/papi/5.1.0.2/gnu/4.7.2/lib64 -lpapi'
# make CC=cc INCLUDE='-O3 ${PAPI_INCLUDE_OPTS}' LINK='-O3 ${PAPI_POST_LINK_OPTS}'
###############################################################################

CC = mpicc
#CC=cc # if use lynx.ucar.edu

#Do not use -g if using cray performance tools
#INCLUDE = -g 
INCLUDE = 
#INCLUDE = -O3 ${PAPI_INCLUDE_OPTS} 
LINK = -lm
#LINK = -O3 -lm ${PAPI_POST_LINK_OPTS} 


OBJECTS = main.o lsqr.o mycblas.o FileIO.o aprod.o loadbalance.o plsqr3.o  CPUSpMV.o tridiagonal.o\
gptl.o threadutil.o GPTLutil.o vecops.o aprod_impl.o vector_comm.o sparse_vec.o\

all:  ${OBJECTS}
	${CC} ${LINK}  -o PLSQR3 $+

##################################################################################33
main.o: main.c
	${CC} ${INCLUDE} -c $+ -o $@

#GPTL tools (following three *.o files)
gptl.o: ./tool/gptl.c
	$(CC) ${INCLUDE} -DHAVE_MPI -DHAVE_GETTIMEOFDAY -c $+  -o $@

threadutil.o: ./tool/threadutil.c
	$(CC) ${INCLUDE} -DHAVE_MPI -DHAVE_GETTIMEOFDAY -c $+  -o $@

GPTLutil.o: ./tool/GPTLutil.c
	$(CC) ${INCLUDE} -DHAVE_MPI -DHAVE_GETTIMEOFDAY -c $+  -o $@

lsqr.o: lsqr.c
	$(CC) ${INCLUDE} -lm  -c $+  -o $@

mycblas.o: mycblas.c
	$(CC) ${INCLUDE} -lm  -c $+  -o $@

FileIO.o: FileIO.c
	$(CC) ${INCLUDE} -c $+  -o $@

CPUSpMV.o: CPUSpMV.c
	$(CC) ${INCLUDE} -c $+  -o $@

aprod.o: aprod.c
	$(CC) ${INCLUDE} -c $+  -o $@

loadbalance.o: loadbalance.c
	$(CC) ${INCLUDE} -c $+  -o $@

plsqr3.o: plsqr3.c
	$(CC) ${INCLUDE} -c $+  -o $@

tridiagonal.o: tridiagonal.c
	$(CC) ${INCLUDE} -c $+  -o $@

vecops.o: vecops.c
	$(CC) ${INCLUDE} -c $+ -o $@

aprod_impl.o: aprod_impl.c
	$(CC) ${INCLUDE} -c $+ -o $@

vector_comm.o: vector_comm.c
	$(CC) ${INCLUDE} -c $+ -o $@

sparse_vec.o: sparse_vec.c
	$(CC) ${INCLUDE} -c $+ -o $@

clean:
	rm -f *.o 
cleanall: 
	rm -f  *.o ${EXE} ${EXE_GPU}
