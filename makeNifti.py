import os, sys, shutil
from glob import glob
import datetime
import pydicom
from pydicom.uid import (ExplicitVRLittleEndian, ImplicitVRLittleEndian)
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from shepperd import shepperd
import argparse

# some hardcoded variables, remove these
makeNiftiFiles = True
useDcm2niix = False
remakeCheckImages = True
doCropping = True
doSkullStripping = True
xySpacing=1.0
zSpacing=1.0

parser = argparse.ArgumentParser(description='')

parser.add_argument('dicom_dir'     , type=str, help="anonymized patient directory")
parser.add_argument('output_dir'    , type=str, help="nifti output area"           )
parser.add_argument("--overwrite"   , help="overwrite output_dir if it already exists", action="store_true")
args = parser.parse_args()

if os.path.isdir(args.output_dir):
  if not args.overwrite:
    print("exiting instead of overwriting, pass flag --overwrite to overwrite")
    sys.exit(1)
  shutil.rmtree(args.output_dir)
os.makedirs(args.output_dir)

# Instantiate SimpleITK objects
reader = sitk.ImageFileReader()
writer = sitk.ImageFileWriter()
writer.UseCompressionOn()
  
case = os.path.basename(args.dicom_dir)

plastimatchLog   = os.path.join(args.output_dir,"cmdLog.txt")

reg1Dir     = os.path.join(args.dicom_dir, "reg"            )
mr1Dir      = os.path.join(args.dicom_dir, "t1mr_spgr_post" )
ctDir       = os.path.join(args.dicom_dir, "ct"             )
rtstructDir = os.path.join(args.dicom_dir, "rtstruct"       )

for inputDir in [reg1Dir, mr1Dir, ctDir, rtstructDir]:
  assert os.path.isdir(inputDir), 'Missing input directory "%s"'%inputDir

# Load in the image series
ctDcms=glob(os.path.join(ctDir,"*.dcm"))
ctSOPInstanceUIDs = []
for ctDcm in ctDcms:
  ctDcmDataset = pydicom.dcmread(ctDcm)
  ctSOPInstanceUIDs.append(ctDcmDataset.SOPInstanceUID)
print("Found %d dicom files for CT"%len(ctSOPInstanceUIDs))

mr1Dcms=glob(os.path.join(mr1Dir,"*.dcm"))
mr1SOPInstanceUIDs = []
for mr1Dcm in mr1Dcms:
  mrDcmDataset = pydicom.dcmread(mr1Dcm)
  mr1SOPInstanceUIDs.append(mrDcmDataset.SOPInstanceUID)
print("Found %d dicom files for MR1"%len(mr1SOPInstanceUIDs))

reg1Dcms = glob(os.path.join(reg1Dir,"*.dcm"))
assert len(reg1Dcms)==1
reg1Dcm=reg1Dcms[0]
reg1DcmDataset = pydicom.dcmread(reg1Dcm)

# Done loading in the image series
# Now make sure we have the right registrations
reg1ImgSeries = reg1DcmDataset.RegistrationSequence[:]  # 2 entries
assert len(reg1ImgSeries)==2
reg1CtIndex = False
reg1MrIndex = False
for iRegSeq, regSeq in enumerate(reg1ImgSeries):
  ris = regSeq.ReferencedImageSequence
  if (
    '1.2.840.10008.5.1.4.1.1.2' in ris[0].ReferencedSOPClassUID
    #'1.2.840.10008.5.1.4.1.1.2' in ris[0].ReferencedSOPClassUID and
    #ris[0].ReferencedSOPInstanceUID in ctSOPInstanceUIDs # this check will fail for the edge cases where we deleted some CT slices
  ):
    reg1CtIndex = iRegSeq
    reg1CtFrameOfReferenceUID = regSeq.FrameOfReferenceUID

  elif (
    ris[0].ReferencedSOPInstanceUID in mr1SOPInstanceUIDs
  ):
    reg1MrIndex = iRegSeq
    flatM_CTtoMR1 = regSeq.MatrixRegistrationSequence[0].MatrixSequence[0].FrameOfReferenceTransformationMatrix
assert reg1CtIndex is not False
assert reg1MrIndex is not False
print("Successfully loaded registrations")
# Done checking registrations


# Check the structure file
# Currently only support a single RTstruct dicom file
rtstructDcms = glob(os.path.join(rtstructDir,"*.dcm"))
assert len(rtstructDcms)==1
rtstructDcm = rtstructDcms[0]
rtstructDcmDataset = pydicom.dcmread(rtstructDcm)
assert reg1CtFrameOfReferenceUID == rtstructDcmDataset.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID, "Registration 1 and RTstruct do not share a Frame of Reference UID"
print("Successfully loaded structure files")

# Begin rasterizing contours
# Extract geometric info from CT header file
ctHeaderFilePath = os.path.join(args.output_dir,"ctHeader.txt")
os.system('plastimatch header "%s" > "%s"'%(ctDir,ctHeaderFilePath))
ctHeaderFile = open(ctHeaderFilePath,"r")  
gotSpacing=False
gotSize=False
gotOrigin=False
gotDirection=False
for line in ctHeaderFile:
  tokens=line.split()
  if tokens[0] == 'Spacing':
    ctSpacing = tokens[2:5]
    gotSpacing = True
  if tokens[0] == 'Size':
    ctSize = tokens[2:5]
    gotSize = True
  if tokens[0] == 'Origin':
    ctOrigin = tokens[2:5]
    gotOrigin = True
  if tokens[0] == 'Direction':
    ctDirection = tokens[2:11]
    gotDirection = True        
assert gotOrigin and gotSpacing and gotSize and gotDirection

newWidth  = int(round(float(ctSize[0]) * float(ctSpacing[0]) / xySpacing))
newHeight = int(round(float(ctSize[1]) * float(ctSpacing[1]) / xySpacing))
newDepth  = int(round(float(ctSize[2]) * float(ctSpacing[2]) / zSpacing))

# Hack: make an image with the desired XY rasterization spacing to trick plastimatch
# Cannot apply any changes to Z spacing here or we will get random 0 slices
blankCtGeoFile = os.path.join(args.output_dir,"blankCtGeo.nii.gz")
ctGeo = sitk.Image(newWidth,newHeight,int(ctSize[2]),sitk.sitkUInt8)
ctGeo.SetOrigin(tuple( [float(x) for x in ctOrigin] ))
ctGeo.SetSpacing( (xySpacing,xySpacing, float(ctSpacing[2])) )
ctGeo.SetDirection(tuple( [float(x) for x in ctDirection] ))
writer.SetFileName(blankCtGeoFile)
writer.Execute(ctGeo)

# Use the black CT image as a reference for invoking plastimatch convert
contourDir=os.path.join(args.output_dir, 'contours')
transformArgs=(rtstructDcm, contourDir, blankCtGeoFile, blankCtGeoFile, plastimatchLog)
os.system('plastimatch convert --input "%s" --prefix-format nii.gz --output-prefix "%s" --referenced-ct "%s" --fixed "%s" >> "%s"'%transformArgs)

allContours = glob(os.path.join(contourDir, "*.nii.gz"))
gtvs = []
brainContours = []
for contour in allContours:
  basename = os.path.basename(os.path.normpath(contour))
  # MSKCC fun feature: our planners name contours beginning with a Z
  # They are only used for dose purposes, don't consider these
  if basename[0] == 'Z': 
    continue
  # RULE FOR SELECTING THE CONTOURS TO LEARN!!!
  lc = basename.lower()
  if 'gtv' in lc and 'brain' not in lc and 'notgtv' not in lc and 'not_gtv' not in lc:
    gtvs.append(contour)
  if 'brain' in lc and 'brainstem' not in lc and 'gtv' not in lc and 'ptv' not in lc:
    brainContours.append(contour)

assert len(brainContours)!=0, "No brain structure found"
assert len(brainContours)==1, "More than 1 brain structure found, check contour names"

labelCtNiftiFile = os.path.join(args.output_dir,"label.nii.gz")
os.system('plastimatch add "%s" --output "%s" >> "%s"'%('" "'.join(gtvs), labelCtNiftiFile, plastimatchLog))
print('Done producing contour NIFTI files')

## Resample any relevant contours here
brainMaskNiftiFile = os.path.join(args.output_dir,"brain_resampled.nii.gz")
if float(ctSpacing[2]) == zSpacing:
  shutil.copyfile(brainContours[0], brainMaskNiftiFile)
else:
  print('CT Z spacing and desired Z spacing are different. Resampling necessary contours to the desired Z spacing now')
  unresampledBrainMaskNiftiFile = os.path.join(contourDir, "brain_unresampled.nii.gz")
  shutil.copyfile(brainContours[0], unresampledBrainMaskNiftiFile)
  transformArgs=(unresampledBrainMaskNiftiFile,brainMaskNiftiFile,xySpacing,xySpacing,zSpacing,newWidth,newHeight,newDepth,plastimatchLog)
  os.system('plastimatch convert --interpolation nn --input "%s" --output-img "%s" --spacing "%.4f %.4f %.4f" --dim "%d %d %sd" >> "%s"'%transformArgs)
  os.remove(unresampledBrainMaskNiftiFile)
  
  unresampledLabelCtNiftiFile = os.path.join(args.output_dir, "label_unresampled.nii.gz")
  os.replace(labelCtNiftiFile, unresampledLabelCtNiftiFile)
  transformArgs=(unresampledLabelCtNiftiFile, labelCtNiftiFile,xySpacing,xySpacing,zSpacing,newWidth,newHeight,newDepth,plastimatchLog)
  os.system('plastimatch convert --interpolation nn --input "%s" --output-img "%s" --spacing "%.4f %.4f %.4f" --dim "%d %d %d" >> "%s"'%transformArgs)
  os.remove(unresampledLabelCtNiftiFile)
  print('Done resampling contours')
## Done resampling contours

# Time for the transformation...
# First convert the flat array of 16 strings into a 4x4 matrix of floats
flatM_CTtoMR1 = [float(i) for i in flatM_CTtoMR1]
M_CTtoMR1 = np.array([flatM_CTtoMR1[0:4],flatM_CTtoMR1[4:8],flatM_CTtoMR1[8:12],flatM_CTtoMR1[12:16]])
M_MR1toCT = np.linalg.inv(M_CTtoMR1)
# Get the Quaternion of the rotation represented by the 4x4 matrix
quaternion = shepperd(M_MR1toCT)
# Take the Versor transformation from the quaternion, and take the translations from the matrix herself
params = (quaternion[1],quaternion[2],quaternion[3],M_MR1toCT[0][3],M_MR1toCT[1][3],M_MR1toCT[2][3])
# Write this 6DOF transformation to a text file that can be parsed by ITK via Plastimatch
rigidFile = os.path.join(args.output_dir,"rigid_coefficients.txt")
f = open(rigidFile,"w+")
f.write("#Insight Transform File V1.0\r\n")
f.write("#Transform 0\r\n")
f.write("Transform: VersorRigid3DTransform_double_3_3\r\n")
f.write("Parameters: %.10f %.10f %.10f %.10f %.10f %.10f\r\n" % params)
f.write("FixedParameters: 0 0 0\r\n")
f.close()

# Begin conversion of MR to nifti in CT frame
mr1NiftiFile      = os.path.join(args.output_dir,"mr1.nii.gz")
mr1NiftiFileNative = os.path.join(args.output_dir,"mr1_native.nii.gz")
if False:#useDcm2niix and len(mrSOPInstanceUIDs)<300:
  try:
    os.remove(mr1NiftiFile)
  except OSError:
    pass

  mr1NiftiFileUnresampled = os.path.join(args.output_dir,'mr1_unresampled.nii.gz')
  os.system('dcm2niix -w 1 -9 -z y -o "%s" -f mr1_unresampled "%s" > "%s"'%(args.output_dir,mr1Dir,plastimatchLog))
  transformArgs = (mr1NiftiFileUnresampled,mr1NiftiFile,rigidFile,labelCtNiftiFile,plastimatchLog)
  os.system('plastimatch convert --input "%s" --output-img "%s" --xf "%s" --fixed "%s" >> "%s"'%transformArgs)
  try:
    os.remove(mr1NiftiFileUnresampled)
  except OSError:
    pass
else:   
  # Use plastimatch to convert the Dicom to nifti
  # dcm2niix doesn't handle well the MR's with really low Z spacing and 300+ slices
  transformArgs = (mr1Dir,mr1NiftiFile,rigidFile,labelCtNiftiFile,plastimatchLog)
  os.system('plastimatch convert --input "%s" --output-img "%s" --xf "%s" --fixed "%s" >> "%s"'%transformArgs)
print('Done converting and resampling MR1 into CT frame of reference')

image = sitk.ReadImage(mr1NiftiFile)
image = sitk.Cast( image, sitk.sitkFloat32 )
maskImage = sitk.Cast(image,sitk.sitkUInt8)
corrector = sitk.N4BiasFieldCorrectionImageFilter();
corrector.SetMaximumNumberOfIterations([2,2,2,2])
numberFittingLevels = 4
output = corrector.Execute(image, maskImage) # use all nonzero pixels as mask
sitk.WriteImage(output, mr1NiftiFile)
print('Done applying N4 standardization to MR')

ctNiftiFile = os.path.join(args.output_dir,"ct.nii.gz")
transformArgs = (ctDir, ctNiftiFile, labelCtNiftiFile, plastimatchLog)
os.system('plastimatch convert --input "%s" --output-img "%s" --fixed "%s" >> "%s"'%transformArgs)
print('Done converting CT')

if doSkullStripping is True:
  # Begin skull stripping
  # Apply the brain mask to the label
  unstrippedCtNiftiFile = os.path.join(args.output_dir,"ct_unstripped.nii.gz")
  unstrippedMr1NiftiFile = os.path.join(args.output_dir,"mr1_unstripped.nii.gz")
  unstrippedLabelCtNiftiFile = os.path.join(args.output_dir, "label_unstripped.nii.gz")
  
  shutil.move(ctNiftiFile, unstrippedCtNiftiFile)
  shutil.move(mr1NiftiFile, unstrippedMr1NiftiFile)
  shutil.move(labelCtNiftiFile, unstrippedLabelCtNiftiFile)
  
  # Apply the brain mask to the label and images
  maskArgs=(unstrippedMr1NiftiFile, brainMaskNiftiFile, mr1NiftiFile, plastimatchLog)
  os.system('plastimatch mask --input "%s" --mask "%s" --mask-value 0 --output "%s" >> "%s"' % maskArgs)
  maskArgs=(unstrippedCtNiftiFile, brainMaskNiftiFile, ctNiftiFile, plastimatchLog)
  os.system('plastimatch mask --input "%s" --mask "%s" --mask-value 0 --output "%s" >> "%s"' % maskArgs)
  maskArgs=(unstrippedLabelCtNiftiFile, brainMaskNiftiFile, labelCtNiftiFile, plastimatchLog)
  os.system('plastimatch mask --input "%s" --mask "%s" --mask-value 0 --output "%s" >> "%s"' % maskArgs)
  
  
if doCropping is True:
  # Move the images to an unstripped file location
  uncroppedLabelCtNiftiFile = os.path.join(args.output_dir, "label_uncropped.nii.gz")
  uncroppedCtNiftiFile = os.path.join(args.output_dir,"ct_uncropped.nii.gz")
  uncroppedMr1NiftiFile = os.path.join(args.output_dir,"mr1_uncropped.nii.gz")
  shutil.move(labelCtNiftiFile, uncroppedLabelCtNiftiFile)
  shutil.move(ctNiftiFile, uncroppedCtNiftiFile)
  shutil.move(mr1NiftiFile, uncroppedMr1NiftiFile)
  
  reader.SetFileName(uncroppedCtNiftiFile)
  ct_image = reader.Execute()
  reader.SetFileName(uncroppedMr1NiftiFile)
  mr1_image = reader.Execute()
  reader.SetFileName(uncroppedLabelCtNiftiFile)
  label = reader.Execute()
  reader.SetFileName(brainMaskNiftiFile)
  brain_contour = reader.Execute()
  label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
  label_shape_filter.Execute( brain_contour )
  bounding_box = label_shape_filter.GetBoundingBox(1)
  cropped_ct_image = sitk.RegionOfInterest(ct_image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
  cropped_mr1_image = sitk.RegionOfInterest(mr1_image, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])
  cropped_label = sitk.RegionOfInterest(label, bounding_box[int(len(bounding_box)/2):], bounding_box[0:int(len(bounding_box)/2)])      
  writer.SetFileName(ctNiftiFile)
  writer.Execute(cropped_ct_image)
  writer.SetFileName(mr1NiftiFile)
  writer.Execute(cropped_mr1_image)
  writer.SetFileName(labelCtNiftiFile)
  writer.Execute(cropped_label)
  
  for i in [uncroppedCtNiftiFile, uncroppedMr1NiftiFile, uncroppedLabelCtNiftiFile]:
    try:
      os.remove(i)
    except:
      pass    
  
  filteredLabelNiftiFile = os.path.join(args.output_dir,"label_gauss1mm.nii.gz")
  filterArgs=(filteredLabelNiftiFile,labelCtNiftiFile, plastimatchLog)
  os.system('plastimatch filter --pattern gauss --gauss-width 1.0 --output "%s" "%s" >> "%s"'%filterArgs)
  thresholdedLabelNiftiFile = os.path.join(args.output_dir,"label_smoothed.nii.gz")
  thresholdArgs=(filteredLabelNiftiFile,thresholdedLabelNiftiFile,plastimatchLog)
  os.system('plastimatch threshold --above 0.4 --input "%s" --output "%s" >> "%s"'%thresholdArgs)
  try:
    os.remove(filteredLabelNiftiFile)
  except:
    pass
  
  if doSkullStripping is True:
    # Crop the unstripped mr1 to something reasonable.
    # For now, use the Brain atlas contour, but in the future we might want to use Otsu's method instead
    # to figure out where the head is.
    margin=50 #voxels
    print('Cropping unstripped CT and MR images now...')
    unstrippedUncroppedCtNiftiFile = os.path.join(args.output_dir,"ct_unstripped_uncropped.nii.gz")
    unstrippedUncroppedMr1NiftiFile = os.path.join(args.output_dir,"mr1_unstripped_uncropped.nii.gz")
    shutil.move(unstrippedCtNiftiFile , unstrippedUncroppedCtNiftiFile)
    shutil.move(unstrippedMr1NiftiFile, unstrippedUncroppedMr1NiftiFile)
    
    reader.SetFileName(unstrippedUncroppedMr1NiftiFile)
    mr1_image = reader.Execute()
    bounding_box2 = list(bounding_box)
    bounding_box2[0] = max( bounding_box[0]-margin, 0)
    bounding_box2[1] = max( bounding_box[1]-margin, 0)
    bounding_box2[2] = max( bounding_box[2]-margin, 0)
    bounding_box2[3] = min( bounding_box[3]+2*margin, mr1_image.GetSize()[0]-bounding_box2[0])
    bounding_box2[4] = min( bounding_box[4]+2*margin, mr1_image.GetSize()[1]-bounding_box2[1])
    bounding_box2[5] = min( bounding_box[5]+2*margin, mr1_image.GetSize()[2]-bounding_box2[2])
    bounding_box2 = tuple(bounding_box2)
    
    print('Attempting to apply a RegionOfInterest starting at ', bounding_box2[0:int(len(bounding_box2)/2)], ', of size ', bounding_box2[int(len(bounding_box2)/2):], ', to image of size ', mr1_image.GetSize())
    cropped_mr1_image = sitk.RegionOfInterest(mr1_image, bounding_box2[int(len(bounding_box2)/2):], bounding_box2[0:int(len(bounding_box2)/2)])
    writer.SetFileName(unstrippedMr1NiftiFile)
    writer.Execute(cropped_mr1_image)
    
    reader.SetFileName(unstrippedUncroppedCtNiftiFile)
    ct_image = reader.Execute()
    bounding_box2 = list(bounding_box)
    bounding_box2[0] = max( bounding_box[0]-margin, 0)
    bounding_box2[1] = max( bounding_box[1]-margin, 0)
    bounding_box2[2] = max( bounding_box[2]-margin, 0)
    bounding_box2[3] = min( bounding_box[3]+2*margin, ct_image.GetSize()[0]-bounding_box2[0])
    bounding_box2[4] = min( bounding_box[4]+2*margin, ct_image.GetSize()[1]-bounding_box2[1])
    bounding_box2[5] = min( bounding_box[5]+2*margin, ct_image.GetSize()[2]-bounding_box2[2])
    bounding_box2 = tuple(bounding_box2)
    cropped_ct_image = sitk.RegionOfInterest(ct_image, bounding_box2[int(len(bounding_box2)/2):], bounding_box2[0:int(len(bounding_box2)/2)])
    print('Attempting to apply a RegionOfInterest starting at ', bounding_box2[0:int(len(bounding_box2)/2)], ', of size ', bounding_box2[int(len(bounding_box2)/2):], ', to image of size ', ct_image.GetSize())
    writer.SetFileName(unstrippedCtNiftiFile)
    writer.Execute(cropped_ct_image)
    for i in [unstrippedUncroppedCtNiftiFile, unstrippedUncroppedMr1NiftiFile]:
      try:
        os.remove(i)
      except:
        pass    
 
for contour in allContours:
  lc = basename.lower()
  if True:#'brain' not in lc:
    os.remove(contour)
os.rmdir(contourDir)
# end if not already processed


