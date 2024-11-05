#%%
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import csv
from torch.optim import Adam
from tqdm import tqdm
from model_fft import diffusion_pipeline
from scipy.signal import resample
import mat73
import scipy
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft
import neurokit2 as nk
def calculate_respiratory_rate(signal, sampling_rate):
    """
    Calculate the respiratory rate from a signal using FFT to find the most dominant frequency.

    :param signal: The input signal, a 1D numpy array.
    :param sampling_rate: The sampling rate of the signal in Hz.
    :return: The respiratory rate in breaths per minute (BPM).
    """

    # Length of the signal
    n = len(signal)

    # Perform FFT on the signal
    fft_values = fft(signal)
    
    # Frequency bins
    freq = np.fft.fftfreq(n, d=1/sampling_rate)
    
    # Get the magnitude of the FFT
    fft_magnitude = np.abs(fft_values)
    
    # Consider only the positive half of the frequencies
    positive_freq_idx = np.where(freq > 0)
    freq = freq[positive_freq_idx]
    fft_magnitude = fft_magnitude[positive_freq_idx]
    
    # Find the peak frequency
    peak_frequency = freq[np.argmax(fft_magnitude)]
    
    # Convert frequency to respiratory rate (breaths per minute)
    respiratory_rate_bpm = peak_frequency * 60

    return respiratory_rate_bpm


fs = 30

cutoff = 1  # Desired cutoff frequency of the filter (Hz)
order = 8  # Order of the filter

# Design Butterworth low-pass filter
nyquist = 0.5 * fs  # Nyquist Frequency
normal_cutoff = cutoff / nyquist  # Normalize the frequency
b, a = butter(order, normal_cutoff, btype='low', analog=False)

ppg_whole = []
co2_whole = []
seg_length = 5

data = scipy.io.loadmat('bidmc_data.mat')
windowed_pleth_list = []
windowed_co2_list = []
for i in range(53):
    ppg_sub = []
    co2_sub = []
    ppg = data['data'][0][i]['ppg'][0][0][0]
    co2 = data['data'][0][i]['ref']['resp_sig'][0][0][0][0][0][0][0][0]
    co2 = resample(co2, int(co2.shape[0]*0.24))
    ppg = resample(ppg, int(ppg.shape[0]*0.24))
    co2 = filtfilt(b, a, co2[:,0])
    #co2 = nk.rsp_clean(co2, sampling_rate=30)

    for seg in range(int(np.floor(co2.shape[0]/(seg_length*30)))):
        co2_sub.append(co2[None,seg_length*30*seg:seg_length*30*(seg+1),None])
        ppg_sub.append(ppg[None,seg_length*30*seg:seg_length*30*(seg+1)])

    windowed_pleth =  np.concatenate(ppg_sub, axis =0).transpose(2,0,1)
    windowed_co2 =  np.concatenate(co2_sub, axis =0).transpose(2,0,1)

    windowed_pleth_list.append(windowed_pleth)
    windowed_co2_list.append(windowed_co2)

co2  = np.concatenate(windowed_co2_list, axis = 0)
ppg  = np.concatenate(windowed_pleth_list, axis = 0)
class SimpleCSVLogger:
    def __init__(self, filepath):
        self.filepath = filepath
        # If the file doesn't exist, create one with headers.
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(['subject', 'metric1', 'value1', 'metric2', 'value2', 'metric3', 'value3'])

    def log(self, subject, metric1, value1, metric2, value2, metric3, value3):
        with open(self.filepath, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([subject, metric1, value1, metric2, value2, metric3, value3])


device = torch.device("cuda:2")
overall_breathing = 0
overall_mae = 0
#%%
# Loop over each trial and use it as the test set
for subject_id in range(53):

    test_ppg = ppg[subject_id]        # Extract the i-th trial as the test set, shape (95, 300)
    train_ppg = np.delete(ppg, subject_id, axis=0)  # Remove the i-th trial, shape (41, 95, 300)
    test_co2 = co2[subject_id]        # Extract the i-th trial as the test set, shape (95, 300)
    train_co2 = np.delete(co2, subject_id, axis=0)  # Remove the i-th trial, shape (41, 95, 300)

    train_co2 = train_co2.reshape(-1,train_co2.shape[-1])
    train_ppg = train_ppg.reshape(-1,train_ppg.shape[-1])

    
    # Apply t he filter to the signal

    for i in range(train_ppg.shape[0]):
        train_ppg[i] = -1 + 2*(train_ppg[i] - train_ppg[i].min())/(train_ppg[i].max() - train_ppg[i].min())
        train_co2[i] = (train_co2[i] - train_co2[i].min())/(train_co2[i].max() - train_co2[i].min())


    for i in range(test_ppg.shape[0]):
        test_ppg[i] = -1 + 2*(test_ppg[i] - test_ppg[i].min())/(test_ppg[i].max() - test_ppg[i].min())
        test_co2[i] = (test_co2[i] - test_co2[i].min())/(test_co2[i].max() - test_co2[i].min())


    class Diff_dataset(Dataset):
        def __init__(self,ppg, co2):
            self.ppg = ppg
            self.co2 = co2
        def __len__(self):
            return len(self.ppg)
        def __getitem__(self, index):
            return  torch.tensor(self.ppg[index,None,:], dtype = torch.float32), torch.tensor(self.co2[index,None,:], dtype = torch.float32)




    train_dataset = Diff_dataset(train_ppg, train_co2)
    val_dataset = Diff_dataset(test_ppg, test_co2)

    train_loader = DataLoader(train_dataset, batch_size=128,shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size=64,shuffle = False)





    model = diffusion_pipeline(384, 1024, 6, 128, device).to(device)
    #model = torch.nn.DataParallel(model).to(device)

    

    import time
    start = time.time()
    optimizer = Adam(model.parameters(), lr=1e-4)
    log_path = os.path.join('model_5s_double_final_corrected.csv')
    logger = SimpleCSVLogger(log_path)
    num_epochs = 400
    best_val_loss = 10000000000
    p1 = int(0.7 * num_epochs)
    p2 = int(0.99 * num_epochs)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1)

    for epoch_no in range(num_epochs):
            avg_loss = 0
            model.train()
            with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, train_batch in enumerate(it, start=1):
                    optimizer.zero_grad()
                    loss = model(train_batch[0].to(device), co2 = train_batch[1].to(device),  flag = 0)
                    loss = loss.mean(dim = 0)
                    loss.backward()
                    avg_loss += loss.item()/train_batch[0].shape[0]
                    optimizer.step()
                    it.set_postfix(
                        ordered_dict={
                            "Train: avg_epoch_loss": avg_loss / batch_no,
                            "epoch": epoch_no,
                        },
                        refresh=False,
                    )

                lr_scheduler.step()



                
    output_path = os.path.join(f'model_bi{subject_id}_final.pth')
    torch.save(model.state_dict(), output_path)
    end = time.time()
    print('overall time: ', (start - end)/60)    
    mae = 0
    num_windows = 0
    
    model.eval()

    results = []

    with tqdm(val_loader, mininterval=5.0, maxinterval=50.0) as it:
        for batch_no, val_batch in enumerate(it, start=1):
            y = model(val_batch[0].to(device), n_samples=100, flag=1)
            r = y[:,0:100,0,:].mean(dim=1).detach().cpu().numpy()
            results.append(y)
            num_windows = num_windows + val_batch[0].shape[0]

            for i in range(val_batch[0].shape[0]):
                truth = val_batch[1][i,0,:].detach().cpu().numpy()
                mae_current = np.abs(r[i] - truth).mean()
                mae = mae + mae_current

    print('mae: ', mae/num_windows)
    for i in range(len(results)):
        results[i] = results[i][:,0:100,0,:].mean(dim=1).detach().cpu().numpy()
    segment_results = np.concatenate(results, axis = 0)


    whole_trial_co2 = []
    segment_co2 = test_co2
    for i in range(0, segment_co2.shape[0]):
        whole_trial_co2.append(segment_co2[i])
    whole_trial_co2 = np.concatenate(whole_trial_co2, axis = 0)


    segment_results2 = segment_results
    whole_trial_results = []
    for i in range(0, segment_results.shape[0]):
        whole_trial_results.append(segment_results[i])
    whole_trial_results = np.concatenate(whole_trial_results, axis = 0)
    whole_trial_results = filtfilt(b, a, whole_trial_results)
    whole_trial_ppg = []
    test_ppg = test_ppg
    segment_ppg = test_ppg
    for i in range(0, segment_ppg.shape[0]):
        segment_ppg[i] = (segment_ppg[i] - segment_ppg[i].min())/(segment_ppg[i].max() - segment_ppg[i].min())
        whole_trial_ppg.append(segment_ppg[i])
    whole_trial_ppg = np.concatenate(whole_trial_ppg, axis = 0)

    E = 0
    dc_truth = []
    dc_results = []
    overlap = 600
    for i in range(int(np.floor((whole_trial_results.shape[0] - 1800)/overlap))+1):
        c_current = np.zeros((1800))
        r_current = np.zeros((1800))
        r_current = whole_trial_results[overlap*i:overlap*i + 1800].copy()
        c_current = whole_trial_co2[overlap*i:overlap*i + 1800].copy()
        RR_truth = calculate_respiratory_rate(c_current, 30)
        RR_predicted = calculate_respiratory_rate(r_current, 30)
        E = E + np.abs(RR_truth - RR_predicted)

        c_current[c_current>=0.5] = 1
        c_current[c_current<0.5] = 0
        r_current[r_current>=0.5] = 1
        r_current[r_current<0.5] = 0

        dc_truth.append(c_current.sum())
        dc_results.append(r_current.sum())
    
    print('Mean Breathing Rate Difference: ', E/int(np.floor((whole_trial_results.shape[0] - 1800)/overlap)))
    print('Whole Trial MAE: ', np.abs(whole_trial_results- whole_trial_co2).mean())
    cor = np.corrcoef(dc_truth, dc_results)
    print('duty cycle correlation:', cor)
    breathing_diff = E/int(np.floor((whole_trial_results.shape[0] - 1800)/overlap))
    mae_whole = np.abs(whole_trial_results- whole_trial_co2).mean() 
    logger.log(subject_id, 'breathing', breathing_diff, 'mae_whole', mae_whole, 'corr', cor[0,1])
    overall_breathing = overall_breathing + breathing_diff
    overall_mae = overall_mae + mae_whole
print(overall_breathing/53)
print(overall_mae/53)

