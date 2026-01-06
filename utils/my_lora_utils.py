
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample_poly
import torch
from torch.autograd import Variable
import os

def colortext(text,ncolor = 92):
    tetx = str(text)
    return f'\033[{ncolor}m' + tetx + "\033[0m"

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

'''
Data 1 is decompression, data 2 is original
''' 
def check_simmilarity(data1,data2): #data 2 suppose to original or desired

    # Load data
    # Ensure they are the same length
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]

    if isinstance(data1, torch.Tensor):
        data1 = data1.detach().cpu().numpy()
    if isinstance(data2, torch.Tensor):
        data2 = data2.detach().cpu().numpy()

    # Compute magnitude of errors
    errors = np.abs(data2 - data1)

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
    ##crop

    out = spec_to_network_input(Z_torch,target_row)
    real_part = out[0][0]
    ima_part = out[0][1]
    magnitude = torch.abs(real_part + ima_part * 1j)
    # Sxx_norm = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())

    if (folder_r is not None):
        np.save(f'{folder_r}/s_sf{sf}_bw125_{snr}_{symbol}_{no}.npy',magnitude)
    return magnitude
   
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

def check_and_make_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder created: {folder_path}")
    else:
        print(f"Folder already exists: {folder_path}")
