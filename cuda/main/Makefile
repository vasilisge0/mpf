
mp_cuda_main:
	$(CC) -I$(MP_CUDA_INC) -I$(MKL_INC) -I$(MP_INC) $(MKL_CFLAGS) $(MP_SRC)/mp_main.c -c -o $(MP_LIB)/mp_main.o
	$(CC) $(MP_LDFLAGS) $(MP_LIB)/mp_main.o -o $(MP_LIB)/mp
