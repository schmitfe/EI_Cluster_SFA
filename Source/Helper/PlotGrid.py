import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import pandas as pd

# example filename: jep_3.875_jip_3.15625_IthE_1.25_IthI_0.78_NE_1200_NI_300_tau_180.0_q_0.12777777777777777_seed_107675_TrialsPerDirection_2_ITI_500.0_PrepDur_1000.0_StimDur_300.0_PrepInt_0.5_StimInt_0.5.pkl
#Get list of all output files
output_files = glob.glob('../output/*.pkl')
#extract IthE and IthI from file name
IthE = [float(file.split('_')[5]) for file in output_files]
IthI = [float(file.split('_')[7]) for file in output_files]

#extract N_E and N_I from file name
N_E = [int(file.split('_')[9]) for file in output_files]
N_I = [int(file.split('_')[11]) for file in output_files]

#Estimate spikecounts for each file and partinioned into N_E and N_I
firing_rate = []
for index, file in enumerate(output_files):
    with open(file, 'rb') as outfile:
        spikes = pickle.load(outfile)
        params=pickle.load(outfile)
        stimulus_dict=pickle.load(outfile)

    #get spikes of excitatory neurons
    spikes_E = spikes[0][spikes[1]<N_E[index]]
    #get spikes of inhibitory neurons
    spikes_I = spikes[0][spikes[1]>=N_E[index]]
    #get spikecounts of excitatory population
    rate_E = 1000 * np.shape(spikes_E)[0] / N_E[index] / params['simtime']
    #get spikecounts of inhibitory population
    rate_I = 1000 * np.shape(spikes_I)[0] / N_I[index] / params['simtime']
    #append spikecounts to list
    firing_rate.append([rate_E, rate_I])

#convert list to numpy array
firing_rate = np.array(firing_rate)
#create pandas dataframe with IthE, IthI and spikecounts
df = pd.DataFrame({'IthE':IthE, 'IthI':IthI, 'rate_E':firing_rate[:,0], 'rate_I':firing_rate[:,1]})
df = df.sort_values(by=['IthE', 'IthI'])
print(df)

#create meshgrid for IthE and IthI
IthI_grid, IthE_grid = np.meshgrid(np.sort(np.unique(IthI)), -np.sort(-np.unique(IthE)))

#create arrays for firing rates and fill them according to IthE and IthI
firing_rate_E_grid = np.zeros(np.shape(IthE_grid))
firing_rate_I_grid = np.zeros(np.shape(IthI_grid))
for index, row in df.iterrows():
    firing_rate_E_grid[IthI_grid[0] == row['IthI'], IthE_grid[:,0] == row['IthE']] = row['rate_E']
    firing_rate_I_grid[IthI_grid[0] == row['IthI'], IthE_grid[:,0] == row['IthE']] = row['rate_I']
#reshape spikecounts to meshgrid
#spikecount_E_grid = np.reshape(df['rate_E'].values, np.shape(IthE_grid))
#spikecount_I_grid = np.reshape(df['rate_I'].values, np.shape(IthI_grid))

#plot spikecounts of excitatory population
plt.figure()
plt.pcolormesh(IthI_grid, IthE_grid, firing_rate_E_grid, vmin=0, vmax=15)
plt.xlabel('IthI')
plt.ylabel('IthE')
plt.title('Rate_E')
plt.colorbar()
plt.show()

#plot spikecounts of inhibitory population
plt.figure()
plt.pcolormesh(IthI_grid, IthE_grid, firing_rate_I_grid)
plt.xlabel('IthI')
plt.ylabel('IthE')
plt.title('Spikecount I')
plt.colorbar()
plt.show()


