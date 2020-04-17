#!/bin/bash

label="label_smoothed.nii.gz"
aseg_output="aseg.nii.gz"
subjects_dir="/data/deasy/DylanHsu/fsSubjects"
data_dir="/data/deasy/DylanHsu/SRS_N401/nifti"

source fsenv.sh
for case in `ls ${data_dir}`
do
  mri_convert --like /data/deasy/DylanHsu/SRS_N401/nifti/${case}/${label} ${subjects_dir}/${case}/mri/aseg.mgz ${data_dir}/${case}/${aseg_output}
done
