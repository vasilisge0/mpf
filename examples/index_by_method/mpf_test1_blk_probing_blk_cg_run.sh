#!/bin/bash

source mpf_config.sh

#blk_fA_array=(1 3) # for debug$
filename_A="dataset_uncov_128_128_3_5_A.mtx"
filename_out="dataset_uncov_128_3_5_A_stride.txt"
test_prefix="stride_exp3"
blk_fA_array=(1)
output_type_array=(diag)
#stride_array=(1 2)
stride_array=(2)
#probe_nlevels_array=(1 2 3 4 5)
probe_nlevels_array=(1)
solver_frame_array=(mpf)
solver_outer_array=(batch)
solver_batch_array=(2)
solver_outer_nthreads_array=(1)
solver_inner_nthreads_array=(1)
solver_inner_tol_array=(1e-8)
solver_inner_iters_array=(20)

# read last stored indices$
#typeset completed_file=($(cat $filename_bash_log))
i0_start=1
i1_start=1
i2_start=1
i3_start=1
i4_start=1
i5_start=1
i6_start=1
i7_start=1
i8_start=1
i9_start=1
i10_start=1
header=""

data_type_A="MPF_REAL"
layout_A="MPF_COL_MAJOR"
matrix_type_A="MPF_MATRIX_SYMMETRIC"
output_type="MPF_DIAG_FA"
solver_frame=(0)

file_bash_log+="blocking\n"${stride_array[@]}"\n"${probe_nlevels_array[@]}"\n"
file_bash_log+=${solver_frame_array[@]}"\n"${solver_outer_array}"\n"\
file_bash_log+=${solver_batch_array[@]}"\n"${solver_outer_nthreads_array[@]}"\n"\
file_bash_log+=${solver_inner_nthreads_array[@]}"\n""none\n""none\n""cg\n"${solver_inner_tol_array}"\n"\
file_bash_log+=${solver_inner_iters_array[@]}"\n"\
file_bash_log+=${configuration[@]}

filename_bash_log="test1_blk_probing_blk_cg_nn_bash_log"

echo "${#blk_fA_array[@]}"
echo "${#stride_array[@]}"
echo "${#probe_nlevels_array[@]}"
echo "${#solver_batch_array[@]}"
echo "${#solver_outer_nthreads_array[@]}"
echo "${#solver_inner_nthreads_array[@]}"
echo "${#solver_inner_tol_array[@]}"
echo "${#solver_inner_iters_array[@]}"

for i0 in $(seq $i0_start ${#blk_fA_array[@]})
do
  for i1 in $(seq $i1_start ${#stride_array[@]})
  do
    for i2 in $(seq $i2_start ${#probe_nlevels_array[@]})
    do
      for i3 in $(seq $i3_start ${#solver_batch_array[@]})
      do
        for i4 in $(seq $i4_start ${#solver_outer_nthreads_array[@]})
        do
          for i5 in $(seq $i5_start ${#solver_inner_nthreads_array[@]})
          do
            for i6 in $(seq $i6_start ${#solver_inner_tol_array[@]})
            do
              for i7 in $(seq $i7_start ${#solver_inner_iters_array[@]})
              do
                file="test1_blk_probing_blk_cg_nn"
                file+="_"
                file+="${blk_fA_array[$i0-1]}"
                file+="_"
                file+="${stride_array[$i1-1]}"
                file+="_"
                file+="${probe_nlevels_array[$i2-1]}"
                file+="_"
                file+="${solver_outer_nthreads_array[$i3-1]}"
                file+="_"
                file+="${solver_inner_nthreads_array[$i4-1]}"
                file+="_"
                file+="${solver_inner_tol_array[$i5-1]}"
                file+="_"
                file+="${solver_inner_iters_array[$i6-1]}"
                file+="_"
                file+="${solver_inner_iters_array[$i7-1]}"

                filename_out="$file"
                filename_out+=".mtx"

                filename_meta="$file"
                filename_meta+="_meta"

                filename_caller="$file"
                filename_caller+="_caller"

                $mpf_lib"/mpf_test1_blk_probing_blk_cg"\
                ${blk_fA_array[$i0-1]}\
                $mpf_data"/"$filename_A\
                $mpf_out"/approx/"$filename_out\
                $mpf_out"/meta/"$filename_meta\
                $filename_caller\
                ${stride_array[$i1-1]}\
                ${probe_nlevels_array[$i2-1]}\
                ${solver_batch_array[$i3-1]}\
                ${solver_outer_nthreads_array[$i4-1]}\
                ${solver_inner_nthreads_array[$i5-1]}\
                ${solver_inner_tol_array[$i6-1]}\
                ${solver_inner_iters_array[$i7-1]}

                configuration="$i0 $i1 $i2 $i3 $i4 $i5 $i6 $i7"

                echo "test"

                file_bash_log=${filename_A}"\n"${data_type_A}"\n"${layout_A}"\n"${matrix_type_A}"\n"${blk_fA_array[@]}"\n"${output_type_array[@]}"\n"
                file_bash_log+="blocking\n"${stride_array[@]}"\n"${probe_nlevels_array[@]}"\n"
                file_bash_log+=${solver_frame_array[@]}"\n"${solver_outer_array}"\n"
                file_bash_log+=${solver_batch_array[@]}"\n"${solver_outer_nthreads_array[@]}"\n"
                file_bash_log+=${solver_inner_nthreads_array[@]}"\n""none\n""none\n""cg\n"${solver_inner_tol_array}"\n"
                file_bash_log+=${solver_inner_iters_array[@]}"\n"
                file_bash_log+=${configuration[@]}
                echo -e ${file_bash_log[@]} > $filename_bash_log
              done
            done
          done
        done
      done
    done
  done
done
