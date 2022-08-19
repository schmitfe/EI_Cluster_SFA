import os
import htcondor
import shutil
schedd = htcondor.Schedd()

# Check whether the specified path exists or not
isExist = os.path.exists("Logs")
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs("Logs")

# Example Two: submit by adding attributes to the Submit() object
sub = htcondor.Submit()
sub['executable']=				'/Benchmark/Gridsearch_NEST/CondorSubmission/RunSimulation3.sh'
sub['log']=                     'Logs/Log.log'
sub['output']=                  'Logs/Simulation.out'
sub['error']=                   'Logs/Error.err'
sub['request_cpus']=            '24'
sub['request_gpus']= 			'0'
sub['request_memory']=			'60GB'
sub['should_transfer_files']=   'No'


with schedd.transaction() as txn:
	sub.queue(txn)

print("Submitted 1 Job to queue!")
