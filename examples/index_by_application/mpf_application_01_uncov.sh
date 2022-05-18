#!/bin/bash

data_prefix="uncov_256_256_3_5"
mpf_out="../../output"
mpf_data="../../data/matrices"
mpf_lib="."

# mpf parameters

filename_A_array=("banded_covariance/dataset_uncov_256_256_3_5_A.mtx")
filename_out="dataset_uncov_256_256_3_5_A.txt"

blk_fA_array=(1)
output_type_array=(diag)
stride_array=(1 2 4)
probe_nlevels_array=(1 2 3 4)
solver_name=(cg)
solver_frame_array=(mpf)
solver_outer_array=(batch)
solver_batch_array=(1)
solver_outer_nthreads_array=(1)
solver_inner_nthreads_array=(1)
solver_inner_tol_array=("1e-8")
solver_inner_iters_array=("20")

data_type_A="MPF_REAL"
layout_A="MPF_COL_MAJOR"
matrix_type_A="MPF_MATRIX_SYMMETRIC"
output_type="MPF_DIAG_FA"

# bash script parameters

nsamples=3
header=""

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "echo mpf_application_01_uncov.sh"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "          nblk_fA: ${#blk_fA_array[@]}"
echo "      nblk_stride: ${#stride_array[@]}"
echo "          nlevels: ${#probe_nlevels_array[@]}"
echo "   nouter_solvers: ${#solver_batch_array[@]}"
echo "   nouter_solvers: ${#solver_outer_nthreads_array[@]}"
echo "   ninner_threads: ${#solver_inner_nthreads_array[@]}"
echo " ninner_tolerance: ${#solver_inner_tol_array[@]}"
echo "ninner_iterations: ${#solver_inner_iters_array[@]}"
echo "         nsamples: ${#solver_inner_iters_array[@]}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo ""
echo ""

# create output directories

echo "mpf_out: ${mpf_out}"
out_path="${mpf_out}/index_by_application/mpf_application_01_uncov"
out_path_approx="${mpf_out}/index_by_application/mpf_application_01_uncov/approx"
out_path_log="${mpf_out}/index_by_application/mpf_application_01_uncov/log"
out_path_plots="${mpf_out}/index_by_application/mpf_application_01_uncov/plots"

if [ ! -f ${out_path_approx} ]; then
  mkdir -p ${out_path_approx}
fi

if [ ! -f ${out_path_log} ]; then
  mkdir -p ${out_path_log}
fi

if [ ! -f ${out_path_plots} ]; then
  mkdir -p ${out_path_plots}
fi

# run tests

let sample_max=$nsamples-1
for iA in $(seq 1 ${#filename_A_array[@]})
do
  for i0 in $(seq 1 ${#blk_fA_array[@]})
  do
    for i1 in $(seq 1 ${#stride_array[@]})
    do
      for i2 in $(seq 1 ${#probe_nlevels_array[@]})
      do
        for i_fr in $(seq 1 ${#solver_frame_array[@]})
        do
          for i3 in $(seq 1 ${#solver_batch_array[@]})
          do
            for i4 in $(seq 1 ${#solver_outer_nthreads_array[@]})
            do
              for i5 in $(seq 1 ${#solver_inner_nthreads_array[@]})
              do
                for i_nm in $(seq 1 ${#solver_name[@]})
                do
                  for i6 in $(seq 1 ${#solver_inner_tol_array[@]})
                  do
                    for i7 in $(seq 1 ${#solver_inner_iters_array[@]})
                    do
                      for isample in $(seq 0 ${sample_max})
                      do

                        # main name for executable

                        file="blkprobe"
                        file+="_"
                        file+="${blk_fA_array[$i0-1]}"
                        file+="_"
                        file+="diag"
                        file+="_"
                        file+="${stride_array[$i1-1]}"
                        file+="_"
                        file+="${probe_nlevels_array[$i2-1]}"
                        file+="_"
                        file+="${solver_frame_array[$i_fr-1]}"
                        file+="_"
                        file+="${solver_batch_array[$i_3-1]}"
                        file+="_"
                        file+="${solver_outer_nthreads_array[$i4-1]}"
                        file+="_"
                        file+="${solver_inner_nthreads_array[$i5-1]}"
                        file+="_"
                        file+="jacobi_precond"
                        file+="_"
                        file+="nodefl"
                        file+="_"
                        file+="${solver_name[$i_nm-1]}"
                        file+="_"
                        file+="${solver_inner_tol_array[$i6-1]}"
                        file+="_"
                        file+="${solver_inner_iters_array[$i7-1]}"
                        file+="_"
                        file+="${isample}"

                        # subtype name

                        filename_out="${data_prefix}_$file"
                        filename_out+=".mtx"

                        filename_log="${data_prefix}_$file"
                        filename_log+="_log"

                        filename_caller="${data_prefix}_$file"
                        filename_caller+="_caller"

                        # call function

                        $mpf_lib"/mpf_application_01_uncov"\
                        ${blk_fA_array[$i0-1]}\
                        $mpf_data"/"${filename_A_array[$iA-1]}\
                        "$out_path_approx/$filename_out"\
                        "$out_path_log/$filename_log"\
                        $filename_caller\
                        ${stride_array[$i1-1]}\
                        ${probe_nlevels_array[$i2-1]}\
                        ${solver_batch_array[$i3-1]}\
                        ${solver_outer_nthreads_array[$i4-1]}\
                        ${solver_inner_nthreads_array[$i5-1]}\
                        ${solver_inner_tol_array[$i6-1]}\
                        ${solver_inner_iters_array[$i7-1]}

                        # write meta

                        configuration="$i0 $i1 $i2 $i_fr $i3 $i4 $i5 $i_nm $i6 $i7 $isample"

                        file_meta_contents=${filename_A_array[@]}"\n"
                        file_meta_contents+=${data_type_A}"\n"
                        file_meta_contents+=${layout_A}"\n"
                        file_meta_contents+=${matrix_type_A}"\n"
                        file_meta_contents+=${blk_fA_array[@]}"\n"
                        file_meta_contents+=${output_type_array[@]}"\n"
                        file_meta_contents+="blocking\n"
                        file_meta_contents+=${stride_array[@]}"\n"
                        file_meta_contents+=${probe_nlevels_array[@]}"\n"
                        file_meta_contents+=${solver_frame_array[@]}"\n"
                        file_meta_contents+=${solver_outer_array}"\n"
                        file_meta_contents+=${solver_batch_array[@]}"\n"
                        file_meta_contents+=${solver_outer_nthreads_array[@]}"\n"
                        file_meta_contents+=${solver_inner_nthreads_array[@]}"\n"
                        file_meta_contents+="jacobi_precond\n"
                        file_meta_contents+="nodefl\n"
                        file_meta_contents+="cg\n"
                        file_meta_contents+=${solver_inner_tol_array}"\n"
                        file_meta_contents+=${solver_inner_iters_array[@]}"\n"
                        file_meta_contents+=${nsamples}"\n"
                        file_meta_contents+=${configuration[@]}

                        echo -e ${file_meta_contents[@]} > "${out_path}/${data_prefix}_meta"
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done

# use for debugging

#gdb --args
#valgrind
#valgrind --leak-chek=full
