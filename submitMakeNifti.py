import os,sys
from glob import glob
import numpy as np
# Setup job configs

jobfolder       = "./jobs/"
cpu_cores       = 8
cpu_ram         = 16
walltime        = 4
dicom_dir       = '/data/deasy/DylanHsu/anonymized_dicom_n511'
output_location = '/data/deasy/DylanHsu/SRS_N511/niftiWithThin'

try:
  os.makedirs(os.path.join(jobfolder,'logs'))
except:
  pass
for case in os.listdir(dicom_dir):
  jobname = 'makeNifti-%s'%(case)
  jobname = jobname.replace('.','p')
  jobfile = os.path.join(jobfolder,jobname+".lsf")
  # Setup job files
  f = open(jobfile,"w+")
  f.write("#!/bin/bash\n")
  f.write("#BSUB -J "+jobname+"\n")
  f.write("#BSUB -n %d\n" % cpu_cores)
  f.write("#BSUB -q cpuqueue\n")
  f.write("#BSUB -R span[hosts=1]\n")
  f.write("#BSUB -R rusage[mem=%d]\n" % (cpu_ram//cpu_cores))
  f.write("#BSUB -W %d:00\n" % walltime)
  f.write("#BSUB -o " +jobfolder+"/logs/"+jobname+"_%J.stdout\n")
  f.write("#BSUB -eo "+jobfolder+"/logs/"+jobname+"_%J.stderr\n")
  f.write("\n")
  f.write("source /home/hsud3/.bash_profile\n")
  f.write("cd /home/hsud3/srsPre \n")
  f.write("source env_plastimatch.sh \n")
  #f.write("python makeNifti.py --overwrite \"%s\" \"%s\"\n" % (os.path.join(dicom_dir,case), os.path.join(output_location,case)))
  f.write("python makeNiftiCompositeTransform.py --overwrite \"%s\" \"%s\"\n" % (os.path.join(dicom_dir,case), os.path.join(output_location,case)))
  f.close()
  # Submit jobs.
  the_command = "bsub < " + jobfile
  os.system(the_command)
