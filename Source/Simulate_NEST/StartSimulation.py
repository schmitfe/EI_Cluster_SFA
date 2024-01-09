import subprocess
import os
import numpy as np
import pickle

DEBUG = True

# get SLURM environment variables
JobID = os.environ.get('SLURM_JOB_ID', '0')
ArrayID = os.environ.get('SLURM_ARRAY_TASK_ID', '0')

# ArrayID is the ending of the filename of the input file
# e.g. Parameters_0.pkl

# get input file
input_file = '/scratch/fschmi69/spontaneousSFA/Parameters/Parameters_' + ArrayID + '.pkl'

# open input file each tuple is one simulation
with open(input_file, 'rb') as infile:
    Parameters = pickle.load(infile)

# get parameters
fixed_parameters = Parameters['FixedParameter']
variable_parameters = Parameters['VariableParameterValues']
variable_parameter_names = Parameters['VariableParameterNames']


# create output folder if it does not exist
output_folder = '../output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# We want to run the simulations in parallel on the cluster with CPUcount subprocesses
# Each subprocess should run one simulation and write the results to its own output file, after a simulation is done
# the subprocess should start the next simulation


# We communicate the parameters as environment variables
# set fixed parameters as environment variables
for key, value in fixed_parameters.items():
    os.environ[key] = str(value)

# we want to run one subprocess for execution

for ii, VarParameter in enumerate(variable_parameters):
    # set variable parameters as environment variables
    for jj, VarParameterName in enumerate(variable_parameter_names):
        os.environ[VarParameterName] = str(VarParameter[jj])

    # run subprocess, for debugging we want to see the output
   # if ~DEBUG:
    subprocess.run(['python', 'SFA_Spontaneous_SLURM.py'])
    #else:
    #    p=subprocess.Popen(['python', 'SFA_Spontaneous_SLURM.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    # wait until subprocess is finished
    # subprocess.wait()
    #if DEBUG:
    #    (output, err) = p.communicate()
    #    subprocess
        # print output of subprocess
    #    print(p.stdout)
    #    print(p.stderr)

    # print progress
    print('Job %s/%s done' % (ii + 1, len(variable_parameters)))



