import torch

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
    
## NOTE Add GPU Processing wih Torch
####################################
"""
NOTE Bam V4
This is the same structure of BAM and Multi BAM V3
support for batch training, 
And the most importantly, add torch instead of classical numpy
its increase the speed while maintain the performace

The main difference is how to calculate error, this model diff prediction with clean signal
"""
class BAMv4:
    def __init__(self, input_dim, output_dim, eta=1e-5, device=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eta = eta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.W = torch.empty(output_dim, input_dim, device=self.device)
        torch.nn.init.uniform_(self.W, -0.01, 0.01)

    def _output_function(self, Wx):
        # return torch.tanh(Wx)
        return Wx  

    def train(self, X,Y,database, num_epochs=1, batch_size=32, verbose=True,type=1):
        
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
       
        if (type == 2):
            Y = torch.as_tensor(Y, dtype=torch.long, device=self.device)
            database = torch.as_tensor(database, dtype=torch.float32, device=self.device)
        n_samples = X.shape[0]
        losses = []

        for epoch in range(num_epochs):
            perm = torch.randperm(n_samples, device=self.device)
            Xp = X[perm]
            Yp = Y[perm] if type == 2 else None

            epoch_mse = 0.0
            num_batches = 0
            for i in range(0, n_samples, batch_size):
                Xb = Xp[i:i + batch_size]
                B = Xb.shape[0]

                Xb_T = Xb.T                             
                Yb = self.W @ Xb_T                
                X_hat = self.W.T @ Yb 
                 # -------- Target --------
                if type == 2:
                    clean_sym = Yp[i:i + B]
                    X_target = database[clean_sym].T   
                else:
                    X_target = Xb_T                  

                error = X_target - X_hat       
                mse = torch.mean(error ** 2)
                epoch_mse += mse.item()
                num_batches += 1
                losses.append(mse.item())
                # -------- BAM Update (batch) --------
                self.W += self.eta * (Yb @ error.T)

                if torch.isnan(self.W).any():
                    raise ValueError("NaN detected in weights!")
                
            if verbose:
                epoch_mse /= num_batches
                print(f"Epoch {epoch+1}/{num_epochs}, MSE={epoch_mse:.6f}")
                # print(f"Epoch {epoch+1}/{num_epochs}, MSE={mse.item():.6f}")

        return losses

    def compress(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = self._output_function(self.W @ X.T).T
        return y.detach().cpu().numpy()

    def decompress(self, compressed_X):
        Y = torch.tensor(compressed_X, dtype=torch.float32, device=self.device)
        X_reconstructed = self._output_function(self.W.T @ Y.T).T
        return X_reconstructed.detach().cpu().numpy()
    
class MultiBAMv4:
    def __init__(self, layers_dims, eta=1e-4, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bams = [
            BAMv4(layers_dims[i], layers_dims[i + 1], eta, self.device)
            for i in range(len(layers_dims) - 1)
        ]

    def train(self, X, Y_sym, database, num_epochs=1, batch_size=32):
        all_losses = []

        for i, bam in enumerate(self.bams):
            print(f"\n--- Training Layer {i+1}/{len(self.bams)} ---")
            if (i == 0):
                losses = bam.train(X, Y_sym, database, num_epochs=num_epochs, batch_size=batch_size,type=2)
            else :
                losses = bam.train(X, Y_sym, database, num_epochs=num_epochs, batch_size=batch_size, type=1)
            
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

from torch.utils.data import Dataset
import torch.nn as nn

class SymbolDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

        # Normalize (VERY IMPORTANT)
        self.X = self.X / (torch.norm(self.X, dim=1, keepdim=True) + 1e-8)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class SymbolClassifier(nn.Module):
    def __init__(self,layers_dims, activation=nn.ReLU):
        super().__init__()
        layers = []
        for i in range(len(layers_dims) - 1):
            layers.append(nn.Linear(layers_dims[i], layers_dims[i+1]))

            # Do NOT add activation after final layer (logits)
            if i < len(layers_dims) - 2:
                layers.append(activation())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # logits