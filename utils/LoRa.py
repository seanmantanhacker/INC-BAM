import numpy as np
import numpy.matlib
from scipy.signal import chirp
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from scipy.io import savemat
from scipy import signal

import torch

class LoRa:
    def __init__(self, sf, bw):
        self.sf = sf
        self.bw = bw

    def gen_symbol(self, code_word, down=False, Fs=None):
        sf = self.sf
        bw = self.bw
        Fs = bw
        # the default sampling frequency is 1e6
        if Fs is None or Fs < 0:
            Fs = 1000000
        # bandwidth : default(125kHz)
        bw = bw
        org_Fs = Fs

        # For Nyquist Theory
        if Fs < bw:
            Fs = bw
        
        t = np.arange(0, 2**sf/bw, 1/Fs)
        num_samp = Fs * 2**sf/bw

        f0 = -bw/2
        f1 = bw/2

        chirpI = chirp(t, f0, 2**sf/bw, f1, 'linear', 0)
        chirpQ = chirp(t, f0, 2**sf/bw, f1, 'linear', -90)
        baseline = chirpI + 1j * chirpQ

        if down:
            baseline = np.conj(baseline)
        baseline = numpy.matlib.repmat(baseline,1,2)
        offset = round((2**sf - code_word) / 2**sf * num_samp)
        # print(baseline[:5])
        # print(np.shape(baseline))

#         symb = baseline[:, offset:(offset+int(num_samp))]
        symb = baseline[:, (2**sf - offset):(2**sf - offset+int(num_samp))]

        if org_Fs != Fs:
            overSamp = int(Fs / org_Fs)
            symb = symb[:, ::overSamp]

        return symb[0]

    def gen_symbol_exp(self, code_word, down=False):
        sf = self.sf
        bw = self.bw

        f_offset = bw/(2**sf) * code_word
        t_fold = (2**sf - code_word) / bw
        T = 2**sf/bw
        t1 = np.arange(0, t_fold, 1/bw)
        t2 = np.arange(t_fold, (2**sf)/bw, 1/bw)

        x1 = np.exp(1j*2*np.pi*(bw/(2*T)*(t1**2) + (f_offset - bw/2)*t1))
        x2 = np.exp(1j*2*np.pi*(bw/(2*T)*(t2**2) + (f_offset - 3*bw/2)*t2))
        result = np.concatenate((x1,x2),axis=0)
        if down:
            result = np.conj(result)
        return result
    
    def get_fft(self, signal):
        sig_fft = np.fft.fft(signal)
        return sig_fft
    
    def get_fft_abs(self, signal):
        sig_fft = self.get_fft(signal)
        sig_fft_abs = np.abs(sig_fft)
        return sig_fft_abs


    def plot_spectrogram(self, signal, noverlap, nfft):
        if noverlap is None and nfft is None:
            noverlap = 2**self.sf // 8
            nfft = 2**self.sf // 4
        plt.figure(figsize=(8,8))
        plt.specgram(signal, NFFT=nfft, noverlap=noverlap,Fs=self.bw)
        plt.show()
    
    def one_rows_two_cols(self, signal1, signal2, noverlap, nfft):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
        # 서브플롯들 사이의 간격을 조정
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.text(0.5, 0.04, 'Frequency index', ha='center')
        plt.suptitle('Spectrogram of two symbols')
        fig.text(0.04, 0.5, 'Frequency', rotation='vertical')

        formatter = FormatStrFormatter('%.3f')  # 소수점 2자리로 제한하는 포맷 설정
        ax1.xaxis.set_major_formatter(formatter)
        ax2.xaxis.set_major_formatter(formatter)

        # plt.subplot(1,2,1)
        ax1.specgram(signal1, NFFT=nfft, noverlap=noverlap, Fs=self.bw)
        # plt.subplot(1,2,2)
        ax2.specgram(signal2, NFFT=nfft, noverlap=noverlap, Fs=self.bw)
        plt.show()
    
    def plot_fft_real(self, signal):
        x = np.arange(len(signal))
        sig_fft = self.get_fft(signal)
        plt.scatter(x, sig_fft.real, c='#1e88e5',alpha=0.7)
        plt.plot(x, sig_fft.real, c='red', linestyle='dashed', alpha=0.5)
        plt.show()

    def plot_fft_imag(self, signal):
        x = np.arange(len(signal))
        sig_fft = self.get_fft(signal)
        plt.scatter(x, sig_fft.imag, c='#1e88e5',alpha=0.7)
        plt.plot(x, sig_fft.imag, c='red', linestyle='dashed', alpha=0.5)
        plt.show()

    def plot_fft_total(self, signal):
        x = np.arange(len(signal))
        sig_fft = self.get_fft(signal)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
        # 서브플롯들 사이의 간격을 조정
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        fig.text(0.5, 0.04, 'Frequency index', ha='center')
        fig.text(0.08, 0.45, 'Magnitude', rotation='vertical')
        # plt.subplot(1,2,1)
        ax1.set_title('Real Part')
        ax1.scatter(x, sig_fft.real, c='#1e88e5',alpha=0.7)
        ax1.plot(x, sig_fft.real, c='red', linestyle='dashed', alpha=0.5)

        # plt.subplot(1,2,2)
        ax2.set_title('Imaginary Part')
        ax2.scatter(x, sig_fft.imag, c='#1e88e5',alpha=0.7)
        ax2.plot(x, sig_fft.imag, c='red', linestyle='dashed', alpha=0.5)

        plt.show()

    def awgn(self, signal_, SNR_):
        sig_avg_pwr = np.mean(abs(signal_)**2)
        sig_avg_db = 10*np.log10(sig_avg_pwr)
        noise_avg_db = sig_avg_db - SNR_
        noise_avg_pwr = 10**(noise_avg_db/10)
        noise_sim = np.random.normal(0, np.sqrt(noise_avg_pwr), len(signal_))
        return signal_ + noise_sim
    
    def awgn_iq(self, signal_, SNR_):
        sig_avg_pwr = np.mean(abs(signal_)**2)      # 신호의 평균 파워
        noise_avg_pwr = sig_avg_pwr / (10**(SNR_/10))   # SNR을 고려한 노이즈 파워 계산

        # if np.isrealobj(signal_):
        #     # 평균 : 0, 표준편차 : np.sqrt(noise_avg_pwr), 데이터 수: len(signal_)
        #     noise_sim = np.random.normal(0, np.sqrt(noise_avg_pwr), len(signal_))

        # else:
        noise_sim = (np.random.normal(0, np.sqrt(noise_avg_pwr/2), len(signal_)) + 1j*np.random.normal(0, np.sqrt(noise_avg_pwr/2), len(signal_)))

        return signal_ + noise_sim
    
    # SNR에 따른 실제 가우시안 노이즈 추가 방식 및 SNR 계산
    def add_awgn_noise(self, signal, snr_db):
        """주어진 SNR(dB)에 맞게 AWGN 노이즈 추가"""
        signal_power = np.mean(np.abs(signal)**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
        return signal + noise
    
    def calculate_snr_db(self, clean_signal, noisy_signal):
        signal_power = np.mean(np.abs(clean_signal)**2)
        noise_power = np.mean(np.abs(noisy_signal - clean_signal)**2)
        snr_db = 10 * np.log10(signal_power / noise_power)
        return snr_db
    
    def generate_symbol_with_noise(self, sf, bw, generate_size, root_path, target_snr):
        lora_init = LoRa(sf, bw)
        sym_count = 0
        sym_index = 0
        for i in range(generate_size):
            val = i % int(2**sf)
            chirp = lora_init.gen_symbol_fs(val, i+7, bw, down=False, Fs=int(8*bw))
            gen_snr = target_snr
            # chirp_awgn = lora_init.add_awgn_noise(chirp, gen_snr)
            chirp_awgn = lora_init.awgn_iq(chirp, gen_snr)
            chirp_signal = chirp_awgn.reshape(1,-1)
            mat_data = {
            '__header__': b'Generating LoRa Symbol using gen_symbol()',
            '__version__': '1.0',
            '__globals__': [],
            'chirp': chirp_signal
            }
            if sym_count == (int(2**sf)):
                sym_index += 1
                sym_count = 0
            save_name = f'{sym_index}_{gen_snr}_{sf}_{bw}_0_{val}_0_0.mat'
            savemat(root_path + save_name, mat_data)
            sym_count += 1

    def generate_symbol_with_noise2(self, sf, bw, generate_size, root_path, target_snr):
        lora_init = LoRa(sf, bw)
        sym_count = 0
        sym_index = 0
        for i in range(generate_size):
            val = i % int(2**sf)
            chirp_ = lora_init.gen_symbol(val,down=False)
            chirp = signal.resample_poly(chirp_,up=8,down=1)
            gen_snr = target_snr
            # chirp_awgn = lora_init.add_awgn_noise(chirp, gen_snr)
            chirp_awgn = lora_init.awgn_iq(chirp, gen_snr)
            chirp_signal = chirp_awgn.reshape(1,-1)
            mat_data = {
            '__header__': b'Generating LoRa Symbol using gen_symbol()',
            '__version__': '1.0',
            '__globals__': [],
            'chirp': chirp_signal
            }
            if sym_count == (int(2**sf)):
                sym_index += 1
                sym_count = 0
            save_name = f'{sym_index}_{gen_snr}_{sf}_{bw}_0_{val}_0_0.mat'
            savemat(root_path + save_name, mat_data)
            sym_count += 1
    
    def fft_example(self, val):
        signal = self.gen_symbol_exp(val, sf=self.sf, down=False, Fs=self.bw)
        self.plot_fft_total(signal)
    
    def fft_example(self, val):
        signal = self.gen_symbol_exp(val, sf=self.sf, down=False, Fs=self.bw)
        self.plot_fft_total(signal)

    def gen_symbol_fs(self, code_word, sf, bw, down=False, Fs=None):
        sf = self.sf
        bw = self.bw
        # Fs = bw
        # the default sampling frequency is 1e6
        if Fs is None or Fs < 0:
            Fs = 1000000
        # bandwidth : default(125kHz)
        bw = bw
        org_Fs = Fs

        # For Nyquist Theory
        if Fs < bw:
            Fs = bw
        
        t = np.arange(0, 2**sf/bw, 1/Fs)
        # print('len t : ', len(t))
        num_samp = Fs * 2**sf/bw

        f0 = -bw/2
        f1 = bw/2

        # chirpI = chirp(t, f0, 2**sf/bw, f1, 'linear', 90)
        # chirpQ = chirp(t, f0, 2**sf/bw, f1, 'linear', 0)
        chirpI = chirp(t, f0, 2**sf/bw, f1, 'linear', 0)
        chirpQ = chirp(t, f0, 2**sf/bw, f1, 'linear', -90)
        baseline = chirpI + 1j * chirpQ

        if down:
            baseline = np.conj(baseline)
        baseline = numpy.matlib.repmat(baseline,1,2)
        offset = round((2**sf - code_word) / 2**sf * num_samp)

        symb = baseline[:, int(num_samp - offset):int(num_samp - offset+int(num_samp))]

        if org_Fs != Fs:
            overSamp = int(Fs / org_Fs)
            symb = symb[:, ::overSamp]

        return symb[0]       
"""
NOTE 
This is the first structure of BAM and Multi BAM
not support for batch training
"""
class BAM:
    def __init__(self, input_dim, output_dim, eta=1e-4):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eta = eta

        # 가중치 초기화 (입력 -> 출력)
        self.W = np.random.uniform(-0.01, 0.01, (output_dim, input_dim))

    def _output_function(self, Wx):
        return Wx  # 선형 활성화 함수

    def train(self, X):
        for i, x in enumerate(X):
            x = x.reshape(1, -1)

            # Forward pass (입력 -> 출력 -> 입력)
            y = self._output_function(self.W @ x.T)
            x_reconstructed = self._output_function(self.W.T @ y)

            # 재구성 오류 계산
            error = x - x_reconstructed.T

            # Hebbian 학습 규칙 수정: 재구성 오류를 최소화하도록 가중치 업데이트
            self.W += self.eta * np.outer(y, error)
            # self.W += self.eta * (y @ error)
            # self.W += self.eta * (y @ error)  

            if np.isnan(self.W).any():
                raise ValueError("NaN detected in weights! Check learning rate or initialization.")

    def compress(self, X):
        compressed = []
        for x in X:
            y = self._output_function(self.W @ x.T)
            compressed.append(y.T)
        return np.array(compressed)

    def decompress(self, compressed_X):
        decompressed = []
        for y in compressed_X:
            y = y.reshape(-1, 1)
            x_reconstructed = self._output_function(self.W.T @ y)
            decompressed.append(x_reconstructed.T)
        return np.array(decompressed)
"""
NOTE 
This is the first structure of BAM and Multi BAM
not support for batch training
"""
class MultiBAM:
    def __init__(self, layers_dims, eta=1e-4):
        self.bams = [
            BAM(layers_dims[i], layers_dims[i + 1], eta)
            for i in range(len(layers_dims) - 1)
        ]

    def train(self, X):
        for i, bam in enumerate(self.bams):
            bam.train(X)
            X = bam.compress(X)

    def compress(self, X):
        for bam in self.bams:
            X = bam.compress(X)
        return X

    def decompress(self, X):
        for bam in reversed(self.bams):
            X = bam.decompress(X)
        return X         
"""
NOTE Bam V2
This is the same structure of BAM and Multi BAM as V1
but support for batch training
"""
class BAMv2:
    def __init__(self, input_dim, output_dim, eta=1e-5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eta = eta

        # 가중치 초기화 (입력 -> 출력)
        self.W = np.random.uniform(-0.01, 0.01, (output_dim, input_dim))

    def _output_function(self, Wx):
        return Wx  # 선형 활성화 함수

    def train(self, X, num_epochs=1, batch_size=32, verbose=True):
        n_samples = X.shape[0]
        losses = []

        for epoch in range(num_epochs):
            perm = np.random.permutation(n_samples)
            X = X[perm]

            for i in range(0, n_samples, batch_size):
                batch = X[i:i+batch_size]
                batch_errors = []

                for x in batch:
                    x = x.reshape(1, -1)
                    y = self._output_function(self.W @ x.T)
                    x_reconstructed = self._output_function(self.W.T @ y)

                    error = x - x_reconstructed.T
                    batch_errors.append(np.mean(error**2))

                    self.W += self.eta * np.outer(y, error)

                    if np.isnan(self.W).any():
                        raise ValueError("NaN detected in weights!")

                # average error for this batch
                batch_mse = np.mean(batch_errors)
                losses.append(batch_mse)

                if verbose and i % (batch_size * 10) == 0:
                    print(f"Epoch {epoch+1}, Batch {i//batch_size+1}, MSE = {batch_mse:.6f}")

        return losses

    def compress(self, X):
        compressed = []
        for x in X:
            y = self._output_function(self.W @ x.T)
            compressed.append(y.T)
        return np.array(compressed)

    def decompress(self, compressed_X):
        decompressed = []
        for y in compressed_X:
            y = y.reshape(-1, 1)
            x_reconstructed = self._output_function(self.W.T @ y)
            decompressed.append(x_reconstructed.T)
        return np.array(decompressed)
"""
NOTE Bam V2
This is the same structure of BAM and Multi BAM as V1
but support for batch training
"""    
class MultiBAMv2:
    def __init__(self, layers_dims, eta=1e-4):
        self.bams = [
            BAMv2(layers_dims[i], layers_dims[i + 1], eta)
            for i in range(len(layers_dims) - 1)
        ]

    def train(self, X, num_epochs=1, batch_size=32):
        all_losses = []

        for i, bam in enumerate(self.bams):
            print(f"\n--- Training Layer {i+1}/{len(self.bams)} ---")
            losses = bam.train(X, num_epochs=num_epochs, batch_size=batch_size)
            all_losses.append(losses)
            X = bam.compress(X)  # feed compressed output to next layer

        return all_losses

    def compress(self, X):
        for bam in self.bams:
            X = bam.compress(X)
        return X

    def decompress(self, X):
        for bam in reversed(self.bams):
            X = bam.decompress(X)
        return X         

## NOTE Add GPU Processing wih Torch
####################################
"""
NOTE Bam V3
This is the same structure of BAM and Multi BAM V1/V2
support for batch training
And the most importantly, add torch instead of classical numpy
its increase the speed while maintain the performace
"""
class BAMv3:
    def __init__(self, input_dim, output_dim, eta=1e-5, device=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eta = eta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.W = torch.empty(output_dim, input_dim, device=self.device)
        torch.nn.init.uniform_(self.W, -0.01, 0.01)

    def _output_function(self, Wx):
        return Wx  

    def train(self, X, num_epochs=1, batch_size=32, verbose=True):
        # n_samples = X.shape[0]
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        n_samples = X.shape[0]
        losses = []

        for epoch in range(num_epochs):
            perm = torch.randperm(n_samples, device=self.device)
            X = X[perm]

            for i in range(0, n_samples, batch_size):
                batch = X[i:i+batch_size]
                batch_errors = []

                for x in batch:
                    x = x.view(1, -1)
                    y = self._output_function(self.W @ x.T)
                    x_reconstructed = self._output_function(self.W.T @ y)

                    error = x - x_reconstructed.T
                    batch_errors.append(torch.mean(error ** 2).item())

                    self.W += self.eta * (y @ error)

                    if torch.isnan(self.W).any():
                        raise ValueError("NaN detected in weights!")

                # average error for this batch
                batch_mse = sum(batch_errors) / len(batch_errors)
                losses.append(batch_mse)

                if verbose and i % (batch_size * 10) == 0:
                    print(f"Epoch {epoch+1}, Batch {i//batch_size+1}, MSE = {batch_mse:.6f}")

        return losses

    def compress(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = self._output_function(self.W @ X.T).T
        return y.detach().cpu().numpy()

    def decompress(self, compressed_X):
        Y = torch.tensor(compressed_X, dtype=torch.float32, device=self.device)
        X_reconstructed = self._output_function(self.W.T @ Y.T).T
        return X_reconstructed.detach().cpu().numpy()
    
class MultiBAMv3:
    def __init__(self, layers_dims, eta=1e-4, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bams = [
            BAMv3(layers_dims[i], layers_dims[i + 1], eta, self.device)
            for i in range(len(layers_dims) - 1)
        ]

    def train(self, X, num_epochs=1, batch_size=32):
        all_losses = []

        for i, bam in enumerate(self.bams):
            print(f"\n--- Training Layer {i+1}/{len(self.bams)} ---")
            losses = bam.train(X, num_epochs=num_epochs, batch_size=batch_size)
            all_losses.append(losses)
            X = bam.compress(X)  # feed compressed output to next layer

        return all_losses

    def compress(self, X):
        for bam in self.bams:
            X = bam.compress(X)
        return X

    def decompress(self, X):
        for bam in reversed(self.bams):
            X = bam.decompress(X)
        return X 
    
          
# # --------------------------
# # U-NET (small, configurable)
# # --------------------------
# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(inplace=True),
#         )
#     def forward(self, x):
#         return self.conv(x)

# class UNet2Ch(nn.Module):
#     def __init__(self, in_ch=2, base_filters=32):
#         super().__init__()
#         f = base_filters
#         # encoder
#         self.c1 = ConvBlock(in_ch, f)
#         self.p1 = nn.MaxPool2d(2)
#         self.c2 = ConvBlock(f, f*2)
#         self.p2 = nn.MaxPool2d(2)
#         self.c3 = ConvBlock(f*2, f*4)
#         self.p3 = nn.MaxPool2d(2)
#         # bottleneck
#         self.b = ConvBlock(f*4, f*8)
#         # decoder
#         self.u3 = nn.ConvTranspose2d(f*8, f*4, kernel_size=2, stride=2)
#         self.c4 = ConvBlock(f*8, f*4)
#         self.u2 = nn.ConvTranspose2d(f*4, f*2, kernel_size=2, stride=2)
#         self.c5 = ConvBlock(f*4, f*2)
#         self.u1 = nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2)
#         self.c6 = ConvBlock(f*2, f)
#         # final
#         self.out_conv = nn.Conv2d(f, 2, kernel_size=1)  # 2 channels: Re_hat, Im_hat

#     def forward(self, x):
#         # x: (B, 2, F, T)
#         c1 = self.c1(x)
#         p1 = self.p1(c1)
#         c2 = self.c2(p1)
#         p2 = self.p2(c2)
#         c3 = self.c3(p2)
#         p3 = self.p3(c3)

#         b = self.b(p3)

#         u3 = self.u3(b)
#         u3 = torch.cat([u3, c3], dim=1)
#         c4 = self.c4(u3)

#         u2 = self.u2(c4)
#         u2 = torch.cat([u2, c2], dim=1)
#         c5 = self.c5(u2)

#         u1 = self.u1(c5)
#         u1 = torch.cat([u1, c1], dim=1)
#         c6 = self.c6(u1)

#         out = self.out_conv(c6)  # linear outputs (can be negative)
#         return out

# # --------------------------
# # Dataset helpers
# # --------------------------
# class SpectrogramDataset(Dataset):
#     """
#     expects inputs:
#       - spec_re: numpy array (N, F, T)
#       - spec_im: numpy array (N, F, T)
#     Returns torch tensors normalized per-sample by max magnitude.
#     """
#     def __init__(self, spec_re, spec_im, normalize=True):
#         assert spec_re.shape == spec_im.shape
#         self.re = spec_re.astype(np.float32)
#         self.im = spec_im.astype(np.float32)
#         self.normalize = normalize

#     def __len__(self):
#         return self.re.shape[0]

#     def __getitem__(self, idx):
#         Re = self.re[idx]
#         Im = self.im[idx]
#         mag = np.sqrt(Re**2 + Im**2)
#         max_val = mag.max() if self.normalize else 1.0
#         if max_val == 0:
#             max_val = 1.0
#         Re_n = Re / max_val
#         Im_n = Im / max_val
#         # return stacked (2, F, T) and the scale factor for ISTFT inversion
#         return torch.from_numpy(np.stack([Re_n, Im_n], axis=0)), float(max_val)

# # --------------------------
# # Loss helpers (mag + time-domain)
# # --------------------------
# def magnitude_mse_loss(re_true, im_true, re_hat, im_hat):
#     mag = torch.sqrt(re_true**2 + im_true**2 + 1e-12)
#     mag_hat = torch.sqrt(re_hat**2 + im_hat**2 + 1e-12)
#     return F.mse_loss(mag_hat, mag)

# def waveform_l1_loss_from_reim(re_hat, im_hat, re_true, im_true, istft_params):
#     """
#     Re/Im all tensors shaped (B, F, T)
#     istft_params: dict with n_fft, hop_length, win_length, window (torch.Tensor or None)
#     Returns L1 between original time waveform and predicted waveform
#     NOTE: This expects you have the original time waveform to compare against.
#     If you don't have it, you can compare reconstructed waveform from ground-truth spectrogram.
#     """
#     # Build complex tensors
#     complex_hat = torch.complex(re_hat, im_hat)   # shape (B, F, T)
#     complex_true = torch.complex(re_true, im_true)
#     # inverse STFT: torch.istft expects (..., freq, frames)
#     x_hat = torch.istft(complex_hat, **istft_params)
#     x_true = torch.istft(complex_true, **istft_params)
#     return F.l1_loss(x_hat, x_true)

# # --------------------------
# # Training loop
# # --------------------------
# def train_unet(
#     model, dataloader, istft_params,
#     device='cuda' if torch.cuda.is_available() else 'cpu',
#     lr=1e-3, n_epochs=50, alpha=1.0, beta=0.5
# ):
#     model = model.to(device)
#     opt = torch.optim.Adam(model.parameters(), lr=lr)
#     for epoch in range(1, n_epochs+1):
#         model.train()
#         running_loss = 0.0
#         for batch_idx, (x, scales) in enumerate(dataloader):
#             # x: (B, 2, F, T)
#             x = x.to(device)      # normalized Re/Im
#             scales = torch.tensor(scales, device=device).float()

#             # Forward
#             pred = model(x)       # (B, 2, F, T)
#             re_true = x[:,0]
#             im_true = x[:,1]
#             re_hat = pred[:,0]
#             im_hat = pred[:,1]

#             # Undo normalization per-sample before waveform ISTFT if needed:
#             # shape: (B, F, T)
#             re_true_scaled = re_true * scales.view(-1,1,1)
#             im_true_scaled = im_true * scales.view(-1,1,1)
#             re_hat_scaled = re_hat * scales.view(-1,1,1)
#             im_hat_scaled = im_hat * scales.view(-1,1,1)

#             # Loss terms
#             L_mag = magnitude_mse_loss(re_true, im_true, re_hat, im_hat)  # on normalized spectrograms
#             # Wave L1 (compute with scaled re/im)
#             L_wave = waveform_l1_loss_from_reim(
#                 re_hat_scaled, im_hat_scaled, re_true_scaled, im_true_scaled, istft_params
#             )

#             loss = alpha * L_mag + beta * L_wave

#             opt.zero_grad()
#             loss.backward()
#             opt.step()

#             running_loss += loss.item()

#         avg = running_loss / len(dataloader)
#         print(f"Epoch {epoch}/{n_epochs} - Loss: {avg:.6f}")

# # --------------------------
# # Usage example (synth / adapt to your data)
# # --------------------------
# if __name__ == "__main__":
    # === Dummy / example shapes ===
    # Suppose your spectrograms have F=256 frequency bins, T=64 time frames
    # N = 200   # number of samples
    # F = 256
    # T = 64

    # # Replace these with your real spectrogram arrays (N, F, T)
    # # Here we synthesize some example complex spectrograms:
    # rng = np.random.RandomState(0)
    # spec_re = rng.normal(scale=0.1, size=(N, F, T)).astype(np.float32)
    # spec_im = rng.normal(scale=0.1, size=(N, F, T)).astype(np.float32)

    # dataset = SpectrogramDataset(spec_re, spec_im, normalize=True)
    # loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    # # ISTFT parameters - must match the STFT used to create your spectrograms
    # n_fft = (F - 1) * 2  # e.g., F = n_fft//2 + 1
    # hop_length = n_fft // 4
    # win_length = n_fft
    # window = torch.hann_window(win_length)

    # istft_params = dict(n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, center=True, normalized=False, onesided=True)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = UNet2Ch(in_ch=2, base_filters=32)
    # train_unet(model, loader, istft_params, device=device, lr=1e-3, n_epochs=30, alpha=1.0, beta=0.3)

    # # After training: inference example
    # model.eval()
    # sample, scale = dataset[0]
    # with torch.no_grad():
    #     x = sample.unsqueeze(0).to(device)  # (1,2,F,T)
    #     pred = model(x).cpu().numpy()[0]    # (2,F,T)
    # re_hat = pred[0] * scale
    # im_hat = pred[1] * scale
    # # re_hat/im_hat are your reconstructed spectrogram channels
