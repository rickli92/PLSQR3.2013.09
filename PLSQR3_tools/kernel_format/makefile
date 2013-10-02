CC	=	mpicc 
FLAG   = 

all:	Ker2PLSQR3 Ker2PLSQR3_preprocess ker_ascii2bin

Ker2PLSQR3_preprocess: Ker2PLSQR3_preprocess.o FileIO.o
	${CC} ${FLAG} $+ -o $@

Ker2PLSQR3_preprocess.o: Ker2PLSQR3_preprocess.c
	${CC} ${FLAG} -c $+ -o $@

Ker2PLSQR3: Ker2PLSQR3.o FileIO.o
	${CC} ${FLAG} $+ -o $@

Ker2PLSQR3.o: Ker2PLSQR3.c
	${CC} ${FLAG} -c $+ -o $@

FileIO.o: ../../source/FileIO.c
	${CC} ${FLAG} -c $+ -o $@

ker_ascii2bin: ker_ascii2bin.c
	${CC} ${FLAG} $+ -o $@

clean: 
	rm -rf *.o
