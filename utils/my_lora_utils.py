
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, resample_poly, correlate, spectrogram, stft
import lzma
import pywt
import lz4.frame
from numpy.linalg import svd
import torch
from torch.autograd import Variable

sample_rate = 1e6   # 1 MHz
bw = 125e3          # LoRa Bandwidth (125 kHz)
sf = 9       # Spreading Factor 

symbol_time = 2**sf / bw  # Symbol duration
N = 2**sf
CARRIER = 915e6
PPM36 = 36/1e6 * CARRIER 
#Another Parameter
SNR = 0
USING_AWGN = True
nfft = 256
CFO_RX = 0
# Time vector
t = np.arange(0, symbol_time, 1/sample_rate)
samplePerDfreq = sample_rate / bw
samplePerSymbol = samplePerDfreq * (2**sf)

# Generate Downchirp (Linear Frequency Modulation)
f0 = -bw/2 # Start frequency
f1 = bw/2 # End frequency

###### Discrete-time chirp (normalized version)
k = np.arange(N)
B_k = np.exp(1j * 2 * np.pi * (k**2 / (2 * N) - k / 2))
###### Discrete-time chirp (normalized version)

# Generate Upchirp (increasing frequency)
up_chirp_signal = np.exp(1j * 2 * np.pi * (f0 * t + (f1 / ( symbol_time)) * t**2))

# Generate Downchirp signal (Linear Frequency Modulation)
down_chirp_signal = np.conj(up_chirp_signal)

# Q1: Implement cosine similarity function (10 pts)


def show_multiple_spectrograms(spec_list, titles=None, ncols=3):
    n = len(spec_list)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)  # Flatten even if 1D

    for i, ax in enumerate(axes):
        if i < n:
            im = ax.imshow(spec_list[i], aspect='auto', origin='lower', cmap='jet')
            # im = ax.imshow(spec_list[i], origin='lower', cmap='jet')
            ax.set_title(titles[i] if titles else f"Spectrogram {i+1}")
            ax.set_xlabel("Time bins")
            ax.set_ylabel("Frequency bins")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

def check_simmilarity(data1,data2): #data 2 suppose to original or desired

    # Load data
    # Ensure they are the same length
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]

    # Compute magnitude of errors
    errors = np.abs(data2 - data1)
    magnitudes = np.abs(data2)

    # Compute MSPE
    mspe = np.mean((errors) ** 2)
    rmse = np.sqrt(np.mean((np.abs(data1 - data2))**2))
    print('RMSE : ',rmse)
    print("RMPE : ", mspe)
    rmse_to_max_ratio_p = (rmse / (np.max(np.abs(data1)) + 1e-9)) * 100
    rmse_to_mean_ratio_p = (rmse / (np.mean(np.abs(data1)) + 1e-9)) * 100  # or mean, or std
    # Then use a rough rule of thumb:
    rmse_a=  colortext(format(rmse_to_max_ratio_p, ".3f") + " %")
    rmse_b=  colortext(format(rmse_to_mean_ratio_p, ".3f") + " %")
    print(f'RMSE - to signal max ratio = {rmse_a}')
    print(f'RMSE - to signal mean ratio = {rmse_b}')
    # NRMSE	Interpretation
    # < 0.05	Excellent match
    # 0.05–0.1	Good
    # 0.1–0.2	Acceptable
    # > 0.2	Poor similarity
     # Ensure same length

    # Compute EVM
    error = data1 - data2
    evm_rms = np.sqrt(np.sum(np.abs(error) ** 2) / np.sum(np.abs(data2) ** 2))
    evm_rms_percent = evm_rms * 100
    evm_rms_color=  colortext(format(evm_rms, ".3f"))
    evm_rms_percent_color=  colortext(format(evm_rms_percent, ".3f") + " %")
    print(f'Error Vector Magnitude = {evm_rms_color} => {evm_rms_percent_color}')
    
    return rmse

def downsampling(data,fs,down):
    #Example for Downsampling 
    # Example input signal of length 4096
    down = down
    fs_new = fs/down
    y = resample_poly(data, up=1, down=down)
    return y,fs_new
    # Resample from 4096 → 1024 using polyphase filtering

def create_spectrogram_npy(x_ds,fs_ds,snr,symbol,no,folder=None):
    nperseg = 128
    noverlap = 64
    nfft = 512
    f, t, Sxx = spectrogram(
        x_ds, fs=fs_ds, window="hann",
        nperseg=nperseg, noverlap=noverlap,
        nfft=nfft, mode='psd', return_onesided=False
    )
    
    ###### USE DB, and normalize
    f = np.fft.fftshift(f)
    Sxx = np.fft.fftshift(Sxx, axes=0)
   
    Sxx_dB = 10 * np.log10(Sxx + 1e-12)
    # Normalize to 0–1 range
    Sxx_norm = (Sxx_dB - Sxx_dB.min()) / (Sxx_dB.max() - Sxx_dB.min())
    # ---- CROP to ±BW/2 ----
    mask = (f >= -bw/2) & (f < bw/2)
    Sxx_crop = Sxx_norm[mask, :]
    if (folder is not None):
        np.save(f'{folder}/s_sf9_bw125_{snr}_{symbol}_{no}.npy', Sxx_crop)
    return Sxx_crop,Sxx_dB.min(),Sxx_dB.max()

def create_spectrogram_npy_dual(x_ds,fs_ds,snr,symbol,no,folder_r=None,folder_i=None):
    nperseg = 128 #128
    noverlap = 64 # 64
    nfft = 512 #512
    window = "hann"
    f_new, t_new, Zxx = stft(
        x_ds, fs=fs_ds, window=window, nperseg=nperseg,
        noverlap=noverlap, nfft=nfft, padded=True, boundary="zeros",
        return_onesided=False
    )

    print(Zxx.shape)
    ###### USE DB, and normalize
    f = np.fft.fftshift(f_new)
    Zxx_shift = np.fft.fftshift(Zxx, axes=0)
    Sxx_dB = 10 * np.log10(Zxx_shift + 1e-12)
    # Normalize to 0–1 range
    Sxx_norm_0_to_1 = (Sxx_dB - Sxx_dB.min()) / (Sxx_dB.max() - Sxx_dB.min())
    Zxx_r = np.real(Sxx_norm_0_to_1)
    # Zxx_r = np.abs(Zxx_r)
    Zxx_i = np.imag(Sxx_norm_0_to_1)
    # Zxx_i = np.abs(Zxx_i)
    # Zxx_r_norm_0_to_1 = (Zxx_r - Zxx_r.min()) / (Zxx_r.max() - Zxx_r.min())
    # Zxx_i_norm_0_to_1 = (Zxx_i - Zxx_i.min()) / (Zxx_i.max() - Zxx_i.min())
    # Sxx_norm_0_to_1 = (Zxx_shift - Zxx_shift.min()) / (Zxx_shift.max() - Zxx_shift.min())
    # Zxx_r_norm = 2 * (Zxx_r - Zxx_r.min()) / (Zxx_r.max() - Zxx_r.min()) - 1
    # Zxx_i_norm = 2 * (Zxx_i - Zxx_i.min()) / (Zxx_i.max() - Zxx_i.min()) - 1
    # ---- CROP to ±BW/2 ----
    mask = (f >= -bw/2) & (f < bw/2)
    Zxx_r_crop = Zxx_r[mask, :]
    Zxx_i_crop = Zxx_i[mask, :]
    Zxx_r_crop = Zxx_r_crop
    Zxx_i_crop = Zxx_i_crop
    if (folder_r is not None) and (folder_i is not None):
        np.save(f'{folder_r}/s_sf9_bw125_{snr}_{symbol}_{no}.npy', Zxx_r_crop)
        np.save(f'{folder_i}/s_sf9_bw125_{snr}_{symbol}_{no}.npy', Zxx_i_crop)
    return Zxx_r_crop,Zxx_i_crop,Zxx_r.min(),Zxx_r.max(),Zxx_i.min(),Zxx_i.max()

def create_spectrogram_from_torch(x,sf,bw,fs,target_row,target_col,snr,symbol,no,folder_r=None):
    center = True
    input_signal_length = len(x)
    
    nfft = int(target_row * (fs/bw))#512

    if (center) :
        noverlap = (input_signal_length//(target_col -1))
    else :
        noverlap = (input_signal_length + 0 - nfft ) // (target_col - 1)
        
    nperseg = 2 * noverlap  #64 
    window = torch.hann_window(nperseg)
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    Z = torch.stft(
        x, 
        n_fft=nfft,
        hop_length= noverlap,  ##overlap
        win_length= nperseg, ##nperseg
        window=window,
        return_complex=True,       # this makes output shape (..., 2)
        center=center,
        onesided=False,
        pad_mode="constant"
    )
    
    Z_torch = Z.unsqueeze(0)  # adds batch dim
    # Crop the spectrogram
    out = spec_to_network_input(Z_torch,target_row)
    
    real_part = out[0][0]
    ima_part = out[0][1]
    magnitude = torch.abs(real_part + ima_part * 1j)

    if (folder_r is not None):
        np.save(f'{folder_r}/s_sf9_bw125_{snr}_{symbol}_{no}.npy',magnitude)
    return magnitude
   

def calculate_symbol_alliqfile_without_down_sampling(data,sf,bw,sample_rate,show=True):
    plt.figure(figsize=(20,25))
    symbol_time = 2**sf / bw  # Symbol duration
    osr = int(np.floor(sample_rate/bw))
    # Time vector
    t = np.arange(0, symbol_time, 1/sample_rate)
    # Generate Downchirp (Linear Frequency Modulation)
    f0 = -bw/2 # Start frequency
    f1 = bw/2 # End frequency
    ###### Discrete-time chirp (normalized version)
    result = []
    array_of_value = []
    # Generate Upchirp (increasing frequency)
    up_chirp_signal = np.exp(1j * 2 * np.pi * (f0 * t + (f1 / (symbol_time)) * t**2))
    # Generate Downchirp signal (Linear Frequency Modulation)
    down_chirp_signal = np.conj(up_chirp_signal)

    nsamp = int(2**sf/bw*sample_rate) # 4096 ? 512 ?
    n = len(data)
    frames = n // nsamp
    num_chunks = int(np.ceil(len(data) / nsamp))
    padded_data = np.pad(data, (0, (num_chunks * nsamp) - len(data) + (1 * nsamp)), mode='constant')
    for i in range(frames):
        start_idx = i
        end_idx = i+1
        bias = 0#2824 - 1432 # -1400 # -540
        an_data = padded_data[start_idx*nsamp + bias :end_idx*nsamp + bias]
        de_ = an_data * down_chirp_signal
        ## EXPERIMENTAL 1
        # de_fft = np.fft.fftshift(np.fft.fft(de_))
        # de_fft_abs = np.abs(de_fft)**2
        # de_fft_argmax = np.argmax(de_fft_abs)
        ## EXPERIMENTAL 2
        # spec = np.fft.fft(de_, n=(2**sf)*osr)
        # power = np.abs(spec)**2
        # power_512 = power.reshape(2**sf, osr).sum(axis=1)
        # symbol = int(np.argmax(power_512))
        # EXPERIMENTAL 3
        spectrum = np.fft.fftshift(np.fft.fft(de_))
        power = np.abs(spectrum) ** 2
        fft_len = len(power)
        center = fft_len // 2
        bins = 2 ** sf 
        upper_freq = power[center : center + bins]
        lower_freq = power[center - bins: center]  
        combine = upper_freq + lower_freq
        symbol = np.argmax(combine)
        array_of_value.append(np.max(combine))
        ## END EXPERIMENTAL
        result.append(symbol)
        if (show):
            plt.subplot(7,6,i+1)
            # plt.specgram(an_data)
            plt.plot(combine)
            plt.title(f'Peak : {symbol}')
    if (show):
        plt.show()

    ## Post Processing
    # Example array

    # Calculate average
    avg = sum(array_of_value) / len(array_of_value)
    # Set threshold: 10% below average
    threshold = avg * 0.4

    # Check each value
    tags = ["LOW" if x < threshold else "OK" for x in array_of_value]

    return result, tags

def calculate_symbol_alliqfile_with_down_sampling(data,sf,bw,sample_rate,show=True):

    plt.figure(figsize=(20,25))

    data_downsampling = resample_poly(data, up=bw, down=sample_rate)

    symbol_time = 2**sf / bw  # Symbol duration
    # Time vector
    t = np.arange(0, symbol_time, 1/bw)
    
    f0 = -bw/2 # Start frequency
    f1 = bw/2 # End frequency
    result = []
    # Generate Upchirp (increasing frequency)
    up_chirp_signal = np.exp(1j * 2 * np.pi * (f0 * t + (f1 / (symbol_time)) * t**2))
    # Generate Downchirp signal (Linear Frequency Modulation)
    down_chirp_signal = np.conj(up_chirp_signal)

    nsamp = int(2**sf) # 4096 ? 512 ?
    n = len(data_downsampling)
    frames = n // nsamp
    num_chunks = int(np.ceil(len(data_downsampling) / nsamp))
    padded_data = np.pad(data_downsampling, (0, (num_chunks * nsamp) - len(data_downsampling)), mode='constant')
    for i in range(frames):
        start_idx = i
        end_idx = i+1
        an_data = padded_data[start_idx*nsamp:end_idx*nsamp]
        de_ = an_data * down_chirp_signal
        de_fft = np.fft.fft(de_)
        de_fft_abs = np.abs(de_fft)
        de_fft_argmax = np.argmax(de_fft_abs)
        result.append(de_fft_argmax)
        if (show):
            plt.subplot(7,6,i+1)
            plt.plot(de_fft_abs)
            plt.title(f'Peak : {de_fft_argmax}')
    if (show):
        plt.show()
    return result

def quantize_iq(x, nbits=8):
    # Normalize to [-1,1] first (choose appropriate scale)
    xmax = np.max(np.abs(x))
    if xmax == 0: return x
    scale = (2**(nbits-1)-1) / xmax
    xq = np.round(x * scale) / scale
    return xq

# example: 500 Hz CFO
# x_cfo = apply_cfo(x, Fs=1e6, freq_offset_hz=500)
def apply_cfo(x, Fs, freq_offset_hz):
    n = np.arange(len(x))
    return x * np.exp(1j * 2 * np.pi * freq_offset_hz * n / Fs)

# example: linewidth 100 Hz
# x_pn = apply_phase_noise(x, Fs=1e6, linewidth_hz=100)
def apply_phase_noise(x, Fs, linewidth_hz):
    # approximate phase noise as Wiener process with PSD ~ linewidth
    n = len(x)
    dt = 1.0 / Fs
    sigma = np.sqrt(2 * np.pi * linewidth_hz * dt)  # per-sample increment std
    increments = np.random.normal(0, sigma, n)
    phase = np.cumsum(increments)  # random walk
    return x * np.exp(1j * phase)

from scipy.signal import fftconvolve

# delays = [0, int(1e-6*1e6), int(3e-6*1e6)]  # e.g., 0,1,3 microsec at Fs=1MHz
# x_mp = multipath_rayleigh(signal, delays)
def multipath_rayleigh(x, delays_samples, K_dB = 6):
    # taps: complex gains array, delays_samples: integer delays
    #### create taps
    num_taps = 3
    K = 10**(K_dB/10)
    # LOS component amplitude sqrt(K/(K+1))
    los_amp = np.sqrt(K/(K+1))
    scatter_std = np.sqrt(1/(2*(K+1)))
    taps = los_amp + (np.random.randn(num_taps) + 1j*np.random.randn(num_taps))*scatter_std

    ####
    # build impulse response
    L = max(delays_samples) + 1
    h = np.zeros(L, dtype=complex)
    for g, d in zip(taps, delays_samples):
        h[d] = g
    return fftconvolve(x, h)[:len(x)]

# # apply e.g., fd 10 Hz
# x_tv = time_varying_rayleigh(x, fd_hz=10, Fs=1e6)
def time_varying_rayleigh(x, fd_hz, Fs):
    n = len(x)
    dt = 1/Fs
    # approximate correlation time 1/fd; create complex Gaussian noise then lowpass
    white = (np.random.randn(n) + 1j*np.random.randn(n)) / np.sqrt(2)
    # simple exponential smoothing for correlation
    alpha = np.exp(-2*np.pi*fd_hz*dt)
    h = np.zeros(n, dtype=complex)
    h[0] = white[0]
    for i in range(1,n):
        h[i] = alpha*h[i-1] + np.sqrt(1-alpha**2)*white[i]
    # normalize
    h = h / np.sqrt(np.mean(np.abs(h)**2))
    return x * h

def apply_iq_imbalance(x, gain_imb_db=0.5, phase_imb_deg=2.0):
    g = 10**(gain_imb_db/20.0)
    phi = np.deg2rad(phase_imb_deg)
    i = np.real(x)
    q = np.imag(x)
    i2 = g*i*np.cos(0) - g*q*np.sin(phi)
    q2 = i*np.sin(phi) + q*np.cos(0)
    return i2 + 1j*q2

# hard clip amplitude > clip_level
def hard_clip(x, max_amp=0.8):
    mags = np.abs(x)
    phases = np.angle(x)
    mags_clipped = np.minimum(mags, max_amp)
    return mags_clipped * np.exp(1j*phases)

from scipy.signal import butter, lfilter
# # e.g., 100-200 kHz interferer band
# x_col = band_limited_noise(x, Fs=1e6, low_hz=100e3, high_hz=200e3, snr_db=10)
def band_limited_noise(x, Fs, low_hz, high_hz, snr_db):
    n = len(x)
    white = (np.random.randn(n) + 1j*np.random.randn(n)) / np.sqrt(2)
    b, a = butter(4, [low_hz/(Fs/2), high_hz/(Fs/2)], btype='band')
    colored = lfilter(b, a, white)
    # scale to desired SNR relative to signal
    Psig = np.mean(np.abs(x)**2)
    Pn = np.mean(np.abs(colored)**2)
    scale = np.sqrt(Psig / (10**(snr_db/10) * Pn))
    return x + colored * scale

def to_var(x):
    """Converts numpy to variable."""
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def spec_to_network_input(x,freq):

    """Converts numpy to variable."""
    freq_size = freq
    normalization = True
    x_image_channel = 2
    # trim
    trim_size = freq_size // 2
    # up down 拼接
    y = torch.cat((x[:, -trim_size:, :], x[:, 0:trim_size, :]), 1)

    if normalization:
        y_abs = torch.abs(y)
        y_abs_max = torch.tensor(
            list(map(lambda x: torch.max(x), y_abs)))
        y_abs_max = to_var(torch.unsqueeze(torch.unsqueeze(y_abs_max, 1), 2))
        y = torch.div(y, y_abs_max)
    
    if x_image_channel == 2:
        y = torch.view_as_real(y)  # [B,H,W,2]
        y = torch.transpose(y, 2, 3)
        y = torch.transpose(y, 1, 2)
    else:
        y = torch.angle(y)  # [B,H,W]
        y = torch.unsqueeze(y, 1)  # [B,H,W]
    return y  # [B,2,H,W]
