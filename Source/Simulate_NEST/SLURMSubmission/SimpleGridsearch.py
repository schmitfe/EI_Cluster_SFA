import os
import numpy as np
I_E = np.linspace(0.95, 2.0, 5)
I_I = np.linspace(0.5, 1.0, 5)

Jobfile = "jobfile.job"

for ie in I_E:
    for ii in I_I:
        with open(Jobfile, 'w') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines("#SBATCH --cpus-per-task=4\n")
            fh.writelines("#SBATCH --mem=2048mb\n")
            fh.writelines("#SBATCH --time=00:10:00\n")
            fh.writelines("#SBATCH --account=fschmi69\n")

            fh.writelines("cd /home/fschmi69/git/EI_Cluster_SFA/Source/Simulate_NEST\n")
            fh.writelines("module load miniconda/py38_4.12.0\n")
            fh.writelines("conda activate /home/fschmi69/Software/NEST3_3\n")

            #set environment variables
            fh.writelines("export I_th_E=%s\n" % ie)
            fh.writelines("export I_th_I=%s\n" % ii)
            fh.writelines("python SFA_SLURM.py\n")
        os.system("sbatch %s" % Jobfile)
        os.remove(Jobfile)





