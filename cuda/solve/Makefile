
mp_cuda_solve:
	$(CC) -lgomp -shared -o $(LIB_MP)/libmpsolver.so \
	$(LIB_MP)/mp_auxiliary.o $(LIB_MP)/mp_probing.o $(LIB_MP)/mp_solve_gmres.o $(LIB_MP)/mp_solve_lanczos.o $(LIB_MP)/mp_solve_cg.o $(LIB_MP)/mp_solve_outer.o $(LIB_MP)/mp_cuda_probing.o

mp_cuda_solve_gmres:
	nvcc -I$(INC_MP) -I$(INC_MKL) $(SRC_MP)/mp_cuda_solve_gmres.c -c -o $(LIB_MP)/mp_cuda_solve_gmres.o

mp_cuda_solve_lanczos:
	nvcc -I$(INC_MP) -I$(INC_MKL) $(SRC_MP)/mp_cuda_solve_lanczos.c -c -o $(LIB_MP)/mp_cuda_solve_lanczos.o

mp_cuda_solve_cg:
	nvcc -I$(INC_MP) -I$(INC_MKL) $(SRC_MP)/mp_cuda_solve_cg.c -c -o $(LIB_MP)/mp_cuda_solve_cg.o
