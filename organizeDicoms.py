import os, sys, shutil
from glob import glob
import datetime

import pydicom
from pydicom.uid import (ExplicitVRLittleEndian, ImplicitVRLittleEndian)

# Place to find the cases e.g. dicom listener space
workArea = "X:\MIM_Manual\DylanCases"
# Place to write the dicom files
outputArea = "Y:\hsud3"
gracePeriod = datetime.timedelta(days=2)
ext = ".*" #ext = ".img" # Dicom files exported via listener have .img extension

def containsStrings(subject, listOfStrings):
  lc=subject.lower()
  for testString in listOfStrings:
    if testString.lower() in lc:
      return True
  return False

if len(sys.argv) < 2:
  print("Pass a text file with <patient ID> <MR date> <CT sim date>")
  exit()

patientFileName = sys.argv[1]
theCase=""
if len(sys.argv) >= 3:
  theCase = sys.argv[2]

  
patientFile = open(patientFileName,"r")
nCase=0

try:
  os.mkdir(os.path.join(outputArea,'phi_dicom'))
except FileExistsError:
  pass

for line in patientFile:
  nCase+=1
  #if theCase!="" and nCase!=theCase:
  #  continue
  # Get patient information from the text file and assign it to patientFileTokens
  patientFileTokens = line.split()
  assert len(patientFileTokens) == 5, "Rows must be: MRN Hash, txtStartDate mrDate ctSimDate"
  [patientId, patientHash, txStartDate, mrDate, ctSimDate] = patientFileTokens
  if theCase!="" and patientHash!=theCase:
    continue
  
  #patientNiftiDir = os.path.join(outputArea,'nifti','case%04d'%nCase)
  patientNiftiDir = os.path.join(outputArea,'phi_dicom',patientHash)

  os.makedirs(os.path.join(patientNiftiDir,"ct"             ),exist_ok=True)
  os.makedirs(os.path.join(patientNiftiDir,"t1mr_spgr_post" ),exist_ok=True)
  os.makedirs(os.path.join(patientNiftiDir,"reg"            ),exist_ok=True)
  os.makedirs(os.path.join(patientNiftiDir,"rtstruct"       ),exist_ok=True)
  
  # Get the patient RTstruct, REG, and MR dicom directories exported from MIM
  # There may be extras directories or files, we have to handle this
  rtstructDirs = glob(os.path.join(workArea, "*$"+patientId, "RTSTRUCT*"))
  #print("DEBUG: RTstruct dirs",rtstructDirs)
  regDirs = glob(os.path.join(workArea, "*$"+patientId, "REG*"))
  mrDirs = glob(os.path.join(workArea, "*$"+patientId, "MR*"))
  ctDirs = glob(os.path.join(workArea, "*$"+patientId, "CT*"))
  print('\nPatient ID %s has %d RTst dirs, %d REG dirs, %d MR dirs, %d CT dirs'%(patientId, len(rtstructDirs), len(regDirs), len(mrDirs), len(ctDirs)))
  if len(rtstructDirs)<1 or len(regDirs)<1 or len(mrDirs)<1 or len(ctDirs)<1:
    print('  Skipping Patient ID %s'%patientId)
  
  # Create a time window with a grace period of 3 days
  mrDateTokens = [int(i) for i in mrDate.split('/')]
  ctSimDateTokens = [int(i) for i in ctSimDate.split('/')]
  txStartDateTokens = [int(i) for i in txStartDate.split('/')]
  mrDateObj = datetime.date(mrDateTokens[2], mrDateTokens[0], mrDateTokens[1])
  ctSimDateObj = datetime.date(ctSimDateTokens[2], ctSimDateTokens[0], ctSimDateTokens[1])
  txStartDateObj = datetime.date(txStartDateTokens[2], txStartDateTokens[0], txStartDateTokens[1])
  
  if mrDateObj > ctSimDateObj:
    windowStart = ctSimDateObj - gracePeriod  
  else:
    windowStart = mrDateObj - gracePeriod
  windowEnd   = txStartDateObj + gracePeriod
  
  # Choose the correct CT dir
  theCtDir = False
  ctSOPInstanceUIDs = []
  for ctDir in ctDirs:
    ctDcms = glob(os.path.join(ctDir,"*"+ext)) 
    if len(ctDcms) is 0:
      continue
    ctDcmDataset = pydicom.dcmread(ctDcms[0])
    if hasattr(ctDcmDataset, 'PerformedProcedureStepStartDate'):
      ctDcmDateStr = ctDcmDataset.PerformedProcedureStepStartDate
    elif hasattr(ctDcmDataset, 'AcquisitionDate'):
      ctDcmDateStr = ctDcmDataset.AcquisitionDate
    elif hasattr(ctDcmDataset, 'StudyDate'):
      ctDcmDateStr = ctDcmDataset.StudyDate
    else:
      print("  ERROR: Couldn't get MR acquisition date using any method")     
    ctDcmDateObj = datetime.date(int(ctDcmDateStr[0:4]),int(ctDcmDateStr[4:6]),int(ctDcmDateStr[6:8]))
    if ctDcmDateObj > windowEnd or ctDcmDateObj < windowStart:
      continue
    print("  CT DCM date", ctDcmDateObj, "in the window of (", windowStart, ",", windowEnd, ")")
    theCtDir = ctDir
    # We found our CT dir, now get a list of SOPInstanceUIDs
    print("  Found study CT:")
    print('    %s'%ctDcms[0])
    print("  Loading its SOP Instance UIDs...")
    for ctDcm in ctDcms:
      ctDcmDataset = pydicom.dcmread(ctDcm)
      ctSOPInstanceUIDs.append(ctDcmDataset.SOPInstanceUID)
    print("  Got %d SOP Instance UIDs"%len(ctSOPInstanceUIDs))
    break
  
  if theCtDir is False:
    print('  Could not find a CT dir, skipping this patient')
    continue
  
  # Choose the correct MR dir
  theMrDir = False
  mrSOPInstanceUIDs = []
  for mrDir in mrDirs:
    mrDcms = glob(os.path.join(mrDir,"*"+ext)) 
    if len(mrDcms) is 0:
      continue
    mrDcmDataset = pydicom.dcmread(mrDcms[0])
    if hasattr(mrDcmDataset, 'PerformedProcedureStepStartDate'):
      mrDcmDateStr = mrDcmDataset.PerformedProcedureStepStartDate
    elif hasattr(mrDcmDataset, 'AcquisitionDate'):
      mrDcmDateStr = mrDcmDataset.AcquisitionDate
    elif hasattr(mrDcmDataset, 'StudyDate'):
      mrDcmDateStr = mrDcmDataset.StudyDate
    else:
      print("  ERROR: Couldn't get MR acquisition date using any method")     
    mrDcmDateObj = datetime.date(int(mrDcmDateStr[0:4]),int(mrDcmDateStr[4:6]),int(mrDcmDateStr[6:8]))
    if mrDcmDateObj > windowEnd or mrDcmDateObj < windowStart:
      continue
    print("  MR DCM date", mrDcmDateObj, "in the window of (", windowStart, ",", windowEnd, ")")
    theMrDir = mrDir
    # We found our MR dir, now get a list of SOPInstanceUIDs
    print("  Found study MR:")
    print('    %s'%mrDcms[0])
    print("  Loading its SOP Instance UIDs...")
    for mrDcm in mrDcms:
      mrDcmDataset = pydicom.dcmread(mrDcm)
      mrSOPInstanceUIDs.append(mrDcmDataset.SOPInstanceUID)
      #print("    DEBUG: SOP Instance UID %s"%mrDcmDataset.SOPInstanceUID)
    print("  Got %d SOP Instance UIDs"%len(mrSOPInstanceUIDs))
    break
  
  if theMrDir is False:
    print('  Could not find a MR dir, skipping this patient')
    continue
  
  # Now choose the correct Registration dir
  theRegDcm = False
  regSeqCTFrameOfReferenceUID = False
  regDcms = []
  for regDir in regDirs:
    regDcms += glob(os.path.join(regDir, "*"+ext))
  for regDcm in regDcms:
    regDcmDataset = pydicom.dcmread(regDcm)
    #regDcmDataset.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
    
    #print('  DEBUG: Opened reg file %s'%regDcm)
    
    # Do not perform the grace period hack for choosing the correct registration.
    # We will match the MR SOP Instance UIDs for the chosen MR scan
    # with ONE of the imaging sequences in the registration,
    # and make sure that the OTHER imaging sequence in the registration is a CT scan.
    # Then, we will save the CT Frame of Reference UID, which we will use to find the RT struct.           
    
    # Class object looks like this:
    # regDcmDataset.RegistrationSequence[0:2].ReferencedImageSequence[:].ReferencedSOPInstanceUID (or ReferencedSOPClassUID)
    regImgSeries = regDcmDataset.RegistrationSequence[:]  # 2 entries
    assert len(regImgSeries)==2
    
    # Check that this registration concerns a CT scan first
    isCTReg = False
    regSeqCTIndex = -1
    try:
      for iRegSeq, regSeq in enumerate(regImgSeries):
        # regSeq is the Registration Sequence
        # ris stands for ReferencedImageSequence
        ris = regSeq.ReferencedImageSequence
        if len(ris) is 0:
          continue
        # CT image storage Class UID can be 1.2.840.10008.5.1.4.1.1.2 or 1.2.840.10008.5.1.4.1.1.2.*
        # MR image storage Class UID can be 1.2.840.10008.5.1.4.1.1.4 or 1.2.840.10008.5.1.4.1.1.4.*
        if '1.2.840.10008.5.1.4.1.1.2' in ris[0].ReferencedSOPClassUID:
          isCTReg = True
          regSeqCTIndex = iRegSeq
    except NotImplementedError:
      print('  ERROR: Issue with REG value representation. Try deleting extra REG files, then try to download it manually.')
      #pydicom.filewriter.dcmwrite(os.path.join(os.path.join(regDir, "fixtest"+ext)),regDcmDataset,False)
      continue
    if not isCTReg:
      #print('  ERROR: No CT in registrations')
      continue
      
    # Now check that this registration concerns our chosen MR scan
    # Currently not checking the CT, need to fix
    isChosenMRReg=False
    regSeqChosenMRIndex = -1
    for iRegSeq, regSeq in enumerate(regImgSeries):
      # regSeq is the Registration Sequence
      # Don't check the one that is obviously a CT
      if iRegSeq is regSeqCTIndex:
        continue
      # ris stands for ReferencedImageSequence
      ris = regSeq.ReferencedImageSequence
      
      # Check the length first to save time
      if len(mrSOPInstanceUIDs) != len(ris):
        continue
      # Just check that one belongs
      if ris[0].ReferencedSOPInstanceUID not in mrSOPInstanceUIDs:
        continue
      isChosenMRReg = True
      regSeqChosenMRIndex = iRegSeq
      # Get the array which represents the matrix to transform from CT space to MR space
      flatM_CTtoMR = regSeq.MatrixRegistrationSequence[0].MatrixSequence[0].FrameOfReferenceTransformationMatrix
    if not isChosenMRReg:
      #print('  DEBUG: Chosen MR not in registration')
      continue
    # We have now found the correct registration. Save the frame of reference UID so we can find the structure:
    print('  Found a valid registration:')
    print('    %s'%regDcm)
    theRegDcm = regDcm
    regSeqCTFrameOfReferenceUID = regImgSeries[regSeqCTIndex].FrameOfReferenceUID
    break
  
  if theRegDcm is False:
    print('  Could not find a valid registration between CT and this MR')     
    continue
  print('  FrameOfReferenceUID is',regSeqCTFrameOfReferenceUID)
  
  theRtstructDcm = False
  rtstructDcms = []
  for rtstructDir in rtstructDirs:
    rtstructDcms += glob(os.path.join(rtstructDir, "*"+ext)) # Dicom files exported via listener have .img extension
  for rtstructDcm in rtstructDcms:
    
    # DGH note to self: Could there be more than one file in this folder using Dicom listener export?
    rtstructDcmDataset = pydicom.dcmread(rtstructDcm)
    if len(rtstructDcmDataset.ReferencedFrameOfReferenceSequence) is not 1:
      print('  Could not find a unique Frame of Reference in RT structure file')
      continue
    # Find the date and time of the rtstruct
    # Sometimes there are multiple rtstructs in a single day from the planning process
    rtstructDcmDateStr = rtstructDcmDataset.StructureSetDate
    rtstructDcmTimeStr = rtstructDcmDataset.StructureSetTime
    #rtstructDcmDateStr = rtstructDcmDataset.StudyDate
    #rtstructDcmTimeStr = rtstructDcmDataset.StudyTime
    year,month,day=[int(rtstructDcmDateStr[0:4]),int(rtstructDcmDateStr[4:6]),int(rtstructDcmDateStr[6:8])]
    hour,minute,second = [int(rtstructDcmTimeStr[0:2]),int(rtstructDcmTimeStr[2:4]),int(rtstructDcmTimeStr[4:6])]
    rtstructDcmDateObj = datetime.datetime(year,month,day,hour,minute,second)
    if regSeqCTFrameOfReferenceUID != rtstructDcmDataset.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID:
      continue
    # Cast datetime object to a date
    #if rtstructDcmDateObj.date() > windowEnd or rtstructDcmDateObj.date() < windowStart:
    #  print('  WARNING: Found RTstruct (', rtstructDcmDateObj, ') outside the time window (', windowStart, ',', windowEnd, ')')
    #  continue
    print('  Found a corresponding RTstruct:')
    print('    %s'%rtstructDcm)        
    # Take the NEWEST one
    if theRtstructDcm is not False:
      if rtstructDcmDateObj < theRtstructDcmDate:
        continue
    theRtstructDcm = rtstructDcm
    theRtstructDcmDate = rtstructDcmDateObj
  

  if theRtstructDcm is False:
    print('  ERROR: Did not find a good RTstruct')
    continue
  else: 
    print('  Chose good RTstruct:')
    print('    %s'%theRtstructDcm)
  
  shutil.copyfile(theRtstructDcm, os.path.join(patientNiftiDir,"rtstruct",os.path.basename(os.path.splitext(theRtstructDcm)[0])+'.dcm'))
  shutil.copyfile(theRegDcm     , os.path.join(patientNiftiDir,"reg"     ,os.path.basename(os.path.splitext(theRegDcm     )[0])+'.dcm'))
  for mrDcm in mrDcms:
    shutil.copyfile(mrDcm, os.path.join(patientNiftiDir,"t1mr_spgr_post",os.path.basename(os.path.splitext(mrDcm)[0])+'.dcm'))
  for ctDcm in ctDcms:
    shutil.copyfile(ctDcm, os.path.join(patientNiftiDir,"ct",os.path.basename(os.path.splitext(ctDcm)[0])+'.dcm'))
  
  print('  Done organizing DICOM files (MRN %s => %s)'%(patientId,patientHash))
  

patientFile.close()