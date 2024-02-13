import subprocess
import os
import numpy as np
import pickle

DEBUG = True

# get SLURM environment variables
JobID = os.environ.get('SLURM_JOB_ID', '0')
#ArrayID = os.environ.get('SLURM_ARRAY_TASK_ID', '400')
ArrayID = str(np.random.randint(0, 59))
Timeout_Simulation = os.environ.get('Timeout_Simulation', '2400')

print(ArrayID)
#set environment variable
os.environ['SLURM_ARRAY_TASK_ID'] = ArrayID

# ArrayID is the ending of the filename of the input file
# e.g. Parameters_0.pkl, unfortunately it it can only go from 0 to 999
# we want to have a 4 digit number and add a leading number, so we can have up to 9999 simulations
leadingNumber = 1
#ArrayID = str(leadingNumber) + str(ArrayID).zfill(3)

# use common path for input and output files
CommonPath=os.environ.get('common_path', '/scratch/fschmi69/stimulationSFA/')
#CommonPath = ''

# get input file
input_file = CommonPath+'Parameters/Parameters_' + ArrayID + '.pkl'
#input_file = 'Test/Parameters_' + ArrayID + '.pkl'

# open input file each tuple is one simulation
with open(input_file, 'rb') as infile:
    Parameters = pickle.load(infile)

# get parameters
fixed_parameters = Parameters['FixedParameter']
variable_parameters = Parameters['VariableParameterValues']
variable_parameter_names = Parameters['VariableParameterNames']

# create error log folder if it does not exist
error_folder = CommonPath+'error'
if not os.path.exists(error_folder):
    os.makedirs(error_folder)
os.environ['error_path'] = error_folder+'/'

# create output folder if it does not exist
output_folder = CommonPath+'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
os.environ['output_path'] = output_folder+'/'

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
    try:
        subprocess.run(['python', 'SFA_Dynamic_SLURM.py'], timeout=int(Timeout_Simulation), check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        with open(os.environ['error_path'] + 'error_' + JobID + '_' + ArrayID + '.txt', 'a') as f:
            f.write("CalledProcessError\n")
            print("CalledProcessError")
            f.write("returncode: "+str(e.returncode)+"\n")
            f.write(e.stderr+"\n")
            f.write(';'.join(variable_parameter_names)+"\n")
            f.write(str(VarParameter))
            f.write("\n\n")
    except subprocess.TimeoutExpired as e:
        with open(os.environ['error_path'] + 'error_' + JobID + '_' + ArrayID + '.txt', 'a') as f:
            f.write("TimeoutExpired "+str(e.timeout)+"\n")
            print("TimeoutExpired")
            f.write(';'.join(variable_parameter_names)+"\n")
            f.write(str(VarParameter))
            f.write("\n\n")

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



