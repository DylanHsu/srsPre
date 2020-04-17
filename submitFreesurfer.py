import os
from glob import glob

# Setup job configs
prefix = 'fsReconAll'
jobfolder = "./jobs/"
cpu_cores = 4
cpu_ram = 16
#case_dir = "/data/deasy/DylanHsu/N401_unstripped/nifti"
#image_name = "img_unstripped.nii.gz"
case_dir="/data/deasy/DylanHsu/SRS_N401/nifti"
image_name = "mr1_unstripped.nii.gz"
try:
  os.makedirs(os.path.join(jobfolder,'logs'))
except:
  pass
configs = []

cases = os.listdir(case_dir)
for case in cases:
  config={}
  jobname = '%s-%s'%(prefix,case)
  config['case'] = case
  config['jobname'] = jobname 
  config['jobfile'] = jobfolder+jobname+".lsf"
  configs.append(config)
# Setup job files
for config in configs:
  image_path = os.path.join(case_dir,config['case'],image_name)
  jobname = config['jobname']
  f = open(config['jobfile'],"w+")
  f.write("#!/bin/bash\n")
  f.write("#BSUB -J "+jobname+"\n")
  f.write("#BSUB -n %d\n" % cpu_cores)
  f.write("#BSUB -q cpuqueue\n")
  f.write("#BSUB -R span[hosts=1]\n")
  f.write("#BSUB -R rusage[mem=%d]\n" % (cpu_ram//cpu_cores))
  f.write("#BSUB -W 72:00\n")
  f.write("#BSUB -o  "+jobfolder+"/logs/"+jobname+"_%J.stdout\n")
  f.write("#BSUB -eo "+jobfolder+"/logs/"+jobname+"_%J.stderr\n")
  f.write("\n")
  f.write("source /home/hsud3/.bash_profile\n")
  f.write("cd /home/hsud3/srsPre \n")
  f.write("source fsenv.sh \n")
  reconCmd  = ("recon-all-dgh -i %s -s %s -all"%(image_path,config['case'])
   +" -cw256 -openmp %d" % (cpu_cores)
   +" -nofix -notal-check"
   #+" -cw256 -parallel -openmp 8"
   #+" -no-remesh -no-autodetgwstats"
  )
  f.write(reconCmd+"\n")
  # -no-remesh      : don't "Use Martin's code to remesh ?h.orig to improve triangle quality" (something is looking for remeshed file and crashing)
  # -nofix          : don't run topology fixer
  # -notal-check    : don't check Talairach
  # -fix-diag-only  : topology fixer runs until ?h.defect_labels files
  #                      are created, then stops
  f.close()
# Submit jobs.
for config in configs:
  the_command = "bsub < " + config['jobfile']
  os.system(the_command)
