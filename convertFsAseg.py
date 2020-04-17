# make sure to source fsenv.sh
label_name="label_smoothed.nii.gz"
aseg_output="aseg.nii.gz"
subjects_dir="/data/deasy/DylanHsu/fsSubjects"
data_dir="/data/deasy/DylanHsu/SRS_N401/nifti"

import SimpleITK as sitk
import os,sys

cases = os.listdir(data_dir)
for case in cases:
  aseg_mgz_path = os.path.join(subjects_dir,case,'mri','aseg.mgz')
  aseg_256_path = os.path.join(data_dir, case, 'aseg256.nii.gz')
  os.system('mri_convert %s %s'%(aseg_mgz_path,aseg_256_path))
  
  aseg_256 = sitk.ReadImage(aseg_256_path)
  trueLabel = sitk.ReadImage(os.path.join(data_dir,case,label_name))
  aseg_resampled = sitk.Resample(aseg_256, trueLabel.GetSize(), sitk.Transform(), sitk.sitkNearestNeighbor, trueLabel.GetOrigin(), trueLabel.GetSpacing(), trueLabel.GetDirection(), 0, aseg_256.GetPixelID())
  sitk.WriteImage(aseg_resampled, os.path.join(data_dir, case, aseg_output) )
  os.remove(aseg_256_path)
