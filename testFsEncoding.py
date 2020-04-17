import SimpleITK as sitk
import os, sys
from fsOneHotEncoding import fsDicts
fsLabelDict, fsNamedGroups, fsNumberedGroups = fsDicts()
# testing code
#aseg=sitk.ReadImage('/data/deasy/DylanHsu/SRS_N401/nifti/JR24465FE/aseg.nii.gz')
aseg=sitk.ReadImage(sys.argv[1])
ltlmf=sitk.LabelImageToLabelMapFilter()
castImageFilter = sitk.CastImageFilter()
castImageFilter.SetOutputPixelType(sitk.sitkUInt16)
aseg_UInt16 = castImageFilter.Execute(aseg)
labelShapeFilter = sitk.LabelShapeStatisticsImageFilter()
labelShapeFilter.Execute(aseg_UInt16)
presentLabels = labelShapeFilter.GetLabels()
for mapping in fsLabelDict.items():
  i=mapping[0]
  name=mapping[1]
  if i in presentLabels:
    nvoxels = labelShapeFilter.GetNumberOfPixels(i)
  else:
    nvoxels = 0
  print("%d\t%d\t%s"%(i,nvoxels,name))


