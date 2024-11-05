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
import torchmetrics

import torch.fft

class FFTLoss(nn.Module):
    def __init__(self, loss_type='magnitude', reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(self, predicted, target):
        # Compute the FFT of both predicted and target signals
        pred_fft = torch.fft.fft(predicted, dim=-1, norm = "ortho")
        target_fft = torch.fft.fft(target, dim=-1, norm = "ortho")
        
        if self.loss_type == 'magnitude':
            # Calculate the magnitude spectra
            pred_magnitude = torch.abs(pred_fft)
            target_magnitude = torch.abs(target_fft)
            loss = torch.mean((pred_magnitude - target_magnitude) ** 2, dim=-1)
        
        elif self.loss_type == 'complex':
            # Calculate the difference in complex spectra
            loss = torch.mean((pred_fft - target_fft).abs() ** 2, dim=-1)

        elif self.loss_type == 'phase':
            # Calculate the phase spectra
            pred_phase = torch.angle(pred_fft)
            target_phase = torch.angle(target_fft)
            loss = torch.mean((pred_phase - target_phase) ** 2, dim=-1)

        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class ConditionalTimeGrad(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, nonlinearity = 'tanh', batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        x = torch.nn.functional.relu(self.fc1(rnn_out))
        output = self.fc2(x)
        return output

class decoder(nn.Module):
    def __init__(self, input_dimension, hidden_dimension = 512, out_dimension = 1):
        super().__init__()
        self.fc1 = nn.Linear(input_dimension, hidden_dimension)
        self.fc2 = nn.Linear(hidden_dimension, out_dimension)
    
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x



class SignalEncoder(nn.Module):
    def __init__(self, kernel_sizes=[1, 3, 5, 7, 9, 11]):
        super().__init__()
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1, 32, kernel_size=ks, padding=ks//2)
            for ks in kernel_sizes
        ])
        
        # Initialize weights using kaiming normal initialization
        for layer in self.conv_layers:
            nn.init.kaiming_normal_(layer.weight)
    def forward(self, x):
        outputs = [conv(x) for conv in self.conv_layers]
        concatenated_output = torch.cat(outputs, dim=1)  # Shape: (B, 6*N, L)

        return concatenated_output
    
def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class dilated_conv(nn.Module):
    def __init__(self, kernel_size, in_channels = 1, out_channels = 48):
        super().__init__()
        self.layer0 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation = 1, padding=kernel_size//2*1)
        self.layer1 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation = 2, padding=kernel_size//2*2)
        self.layer2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, dilation = 4, padding=kernel_size//2*4)
        self.bn0 = nn.BatchNorm1d(out_channels, track_running_stats=False)
        self.bn1 = nn.BatchNorm1d(out_channels, track_running_stats=False)
        self.bn2 = nn.BatchNorm1d(out_channels, track_running_stats=False)
        self.relu = nn.ReLU()
        for layer in [self.layer0, self.layer1, self.layer2]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu') 
        self.bottle = Conv1d_with_init(in_channels, out_channels, 1)
    
    def forward(self, x):
        x = self.bottle(x)
        x = self.bn0(self.relu(self.layer0(x)) + x)
        x = self.bn1(self.relu(self.layer1(x)) + x)
        x = self.bn2(self.relu(self.layer2(x)) + x)
        return x
        
        

class SignalEncoder_dil(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 128, kernel_sizes=[3, 5, 7, 9, 11, 13]):
        super().__init__()
        # Store the number of output channels per convolution
        self.N = out_channels
        
        # Create multiple convolutional layers with different kernel sizes
        self.conv_layers = nn.ModuleList([
            dilated_conv(kernel_size=ks, in_channels = 1, out_channels = 32)
            for ks in kernel_sizes
        ])
        
    def forward(self, x):
        # Apply convolutions to corresponding slices
        outputs = [conv(x) for conv in self.conv_layers]
        concatenated_output = torch.cat(outputs, dim=1)  # Shape: (B, 6*N, L)    
        return concatenated_output


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class diffusion_basemodel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.diff_model = ConditionalTimeGrad(input_dim, hidden_dim, num_layers, output_dim)
        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=50,
            embedding_dim=128,
            projection_dim = 192)
        self.ppg_encoder1 = SignalEncoder_dil()
        self.ppg_encoder2 = SignalEncoder()
        self.noise_encoder = SignalEncoder()
        self.de = decoder(input_dimension=output_dim)
        self.weight = torch.nn.parameter.Parameter(torch.ones(1, 192, 1), requires_grad=True)
    def forward(self, ppg, noise, step):
        embedding = self.diffusion_embedding(step).unsqueeze(2)
        f1 = self.ppg_encoder2(ppg) + embedding + self.weight*self.ppg_encoder1(ppg)
        #f2 = self.noise_encoder(noise)
        f2 = self.noise_encoder(noise)
        f = torch.concat([f1,f2], dim = 1).permute(0,2,1)
        noise_predicted = self.de(torch.nn.functional.relu(self.diff_model(f)))
        return noise_predicted

class diffusion_pipeline(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, device):
        super().__init__()    
        self.diffusion_model = diffusion_basemodel(input_dim, hidden_dim, num_layers, output_dim)

        self.num_steps = 50
        self.beta = np.linspace(0.0001, 0.5, self.num_steps)
   
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        alpha_torch = torch.tensor(self.alpha).float().unsqueeze(1).unsqueeze(1)
        self.register_buffer('alpha_torch', alpha_torch)
        self.loss_fft = FFTLoss()
        self.device = device

    def gaussian(self, window_size, sigma):
        gauss = torch.tensor([
            -(x - window_size // 2) ** 2 / float(2 * sigma ** 2)
            for x in range(window_size)
        ])
        gauss = torch.exp(gauss)
        return gauss / gauss.sum()

    def create_window(self, window_size, sigma):
        _1D_window = self.gaussian(window_size, sigma).unsqueeze(0).unsqueeze(0)
        return _1D_window

    def ssim_1d(self, x, y, window_size=11, sigma=1.5, size_average=True):
        # Ensure the inputs have the right shape: (batch_size, 1, length)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        if len(y.shape) == 2:
            y = y.unsqueeze(1)

        window = self.create_window(window_size, sigma).to(self.device)
        mu_x = F.conv1d(x, window, padding=window_size//2, groups=1)
        mu_y = F.conv1d(y, window, padding=window_size//2, groups=1)
        
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        
        sigma_x = F.conv1d(x * x, window, padding=window_size//2, groups=1) - mu_x_sq
        sigma_y = F.conv1d(y * y, window, padding=window_size//2, groups=1) - mu_y_sq
        sigma_xy = F.conv1d(x * y, window, padding=window_size//2, groups=1) - mu_xy
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2))
        
        if size_average:
            return 1 - ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1)
        


    def train_process(self, ppg, co2):
        
        B, K, L = ppg.shape
        t = torch.randint(0, self.num_steps, [B])
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn(B, K, L).to(self.device)
        noisy_co2 = ((current_alpha ** 0.5) * co2 + (1.0 - current_alpha) ** 0.5 * noise)
        predicted = self.diffusion_model(ppg, noisy_co2, t).permute(0,2,1)
        residual = noise - predicted # (B, 1, L)
        predicted_co2 = (noisy_co2 - (1.0 - current_alpha) ** 0.5 * predicted)/(current_alpha ** 0.5)
        loss1 = (residual ** 2).sum()/L 
        loss2 = self.loss_fft(predicted_co2, co2) 
        return loss1 + loss2*1e-2
    
    def imputation(self, ppg, n_samples):
        B, K, L = ppg.shape
        imputed_samples = torch.ones(B, n_samples, K, L) # (B, N, K, L)
        for i in range(n_samples):
            sample_noise = torch.randn(B, K, L).to(self.device)
            for t in range(self.num_steps - 1, -1, -1):
                t = torch.full((B,), t)
                noise_predicted = self.diffusion_model(ppg, sample_noise, t.to(self.device)).permute(0,2,1)
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                coeff1 = torch.tensor(coeff1)
                coeff2 = torch.tensor(coeff2)
                noise = torch.randn_like(noise_predicted)
                sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                sample_noise = coeff1[0] * (sample_noise - coeff2[0] * noise_predicted)
                if t[0] > 0:
                    sample_noise += sigma[0] * noise
            
            imputed_samples[:, i] = sample_noise.detach()
        return imputed_samples
    
    def ddim_sampling(self, ppg, n_samples, n_steps=10, eta=0.0):
        B, K, L = ppg.shape
        imputed_samples = torch.ones(B, n_samples, K, L)  # (B, N, K, L)
        
        # Default to using the full number of steps if n_steps is not specified
        n_steps = n_steps or self.num_steps
        
        # Create a list of timesteps for DDIM (e.g., subsample the timesteps if needed)
        ddim_timesteps = np.linspace(self.num_steps - 1, 0, n_steps).astype(int)
        print(ddim_timesteps)
        for i in range(n_samples):
            sample_noise = torch.randn(B, K, L).to(self.device)
            for idx, t in enumerate((ddim_timesteps)):
                t_tensor = torch.full((B,), t)
                noise_predicted = self.diffusion_model(ppg, sample_noise, t_tensor.to(self.device)).permute(0, 2, 1)

                # Calculate DDIM coefficients
                alpha_t = self.alpha[t]
                #alpha_t_next = self.alpha[t - 1] if t > 0 else torch.tensor(1.0)

                if idx + 1 < len(ddim_timesteps):
                    s = ddim_timesteps[idx + 1]  # Get the next timestep in the schedule

                    alpha_t_next = self.alpha[s]  # Corresponding alpha for that timestep
                else:
                    alpha_t_next = torch.tensor(1.0).to(self.device)  # For the last step, assume it's alpha_0 = 1.0

                coeff1 = (alpha_t_next) ** 0.5
                coeff2 = (1 - alpha_t_next) ** 0.5
                
                # Predict x_0 (denoised sample)
                x0_pred = (sample_noise - (1 - alpha_t) ** 0.5 * noise_predicted) / alpha_t ** 0.5
                x0_pred = (x0_pred - x0_pred.min())/(x0_pred.max() - x0_pred.min())
                # Update sample with deterministic DDIM sampling
                if idx == len(ddim_timesteps) - 1:
                    sample_noise = x0_pred.detach()
                else:
                    # Update sample with deterministic DDIM sampling
                    sample_noise = coeff1 * x0_pred + coeff2 * noise_predicted
                # Optionally add noise if eta > 0 (stochastic DDIM)
            imputed_samples[:, i] = sample_noise.detach()

        return imputed_samples

    
    '''
    def ddim_sampling(self, ppg, n_samples, sample_steps=20):
        B, K, L = ppg.shape
        imputed_samples = torch.ones(B, n_samples, K, L)  # (B, N, K, L)
        
        # Create a linear schedule of timesteps for DDIM
        timesteps = torch.linspace(0, self.num_steps - 1, sample_steps, dtype=torch.long).to(self.device)
        print(timesteps)
        for i in range(n_samples):
            sample_noise = torch.randn(B, K, L).to(self.device)

            # DDIM deterministic reverse process
            for idx, t in enumerate(reversed(timesteps)):
                t_tensor = torch.full((B,), t, dtype=torch.long).to(self.device)
                
                # Predict noise (epsilon_theta) using the diffusion model
                noise_predicted = self.diffusion_model(ppg, sample_noise, t_tensor).permute(0, 2, 1)
                
                # DDIM deterministic update formula
                alpha_t = self.alpha[t]
                if idx + 1 < len(timesteps):
                    s = timesteps[idx + 1]  # Get the next timestep in the schedule
                    alpha_s = self.alpha[s]  # Corresponding alpha for that timestep
                else:
                    alpha_s = torch.tensor(1.0).to(self.device)  # For the last step, assume it's alpha_0 = 1.0
   
                coeff1 = (alpha_s ** 0.5) * (1 / alpha_t ** 0.5)
                coeff2 = (alpha_s ** 0.5)*((1 - alpha_s) ** 0.5)* (1 / alpha_t ** 0.5) + ((1 - alpha_s) ** 0.5)
                
                # Deterministically update the sample (without noise)
                sample_noise = coeff1 * sample_noise + coeff2 * noise_predicted
            
            # Save the imputed sample
            imputed_samples[:, i] = sample_noise.detach()

        return imputed_samples
'''
    
    def forward(self, ppg, co2 = None, n_samples = None, flag = None, n_steps = None):
        if flag == 0:
            return self.train_process(ppg, co2)
        if flag == 1:
            return self.imputation(ppg, n_samples)
        if flag ==2:
            return self.ddim_sampling(ppg, n_samples, n_steps)

