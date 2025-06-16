import argparse
import os
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# === Argument Parsing ===
parser = argparse.ArgumentParser(description="Train VMD-based unfolding model on time-series data.")
parser.add_argument("--K", type=int, default=13, help="Number of modes")
parser.add_argument("--layers", type=int, default=1, help="Number of unfolding layers")
parser.add_argument("--input", type=str, default="./his.npz", help="Path to input .npz file")
parser.add_argument("--output", type=str,default="./unfold/his.npz", help="Path to output .npz file")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for optimizer")
parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
parser.add_argument("--dir", type=str, default="./unfold", help="Directory to save training logs and model")
args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)
log_path = os.path.join(args.dir, 'training.log')
model_path = os.path.join(args.dir, 'model.pth')


logging.basicConfig(filename=log_path,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
data2=np.load(args.input)
mean=data2['mean']
std=data2['std']
data=data2['data']
print(data.shape)
X= data[:,:,0].T
X_train, X_test = train_test_split(
   X, test_size=0.3, random_state=42, shuffle=True
)
X_val, X_test = train_test_split(
    X_test, test_size=0.5, random_state=42, shuffle=True
)
print(X_train.shape,X_val.shape ,X_test.shape)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32))


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

class VMD_innerloop(nn.Module):
    def __init__(self,device,Alpha,lambda_hat,T,f_hat_plus,freqs):
        super(VMD_innerloop,self).__init__()
        self.Alpha=Alpha
        self.lambda_hat=lambda_hat
        self.f_hat_plus=f_hat_plus
        self.freqs=freqs
       
        self.T=T
    def forward(self,n_u_hat_plus,u_hat_plus,omega_plus,sum_uk,k):
        sum_uk=n_u_hat_plus+sum_uk-u_hat_plus
        var_omega=omega_plus.clone()
        # print(self.Alpha)
        var_u_hat =((self.f_hat_plus -sum_uk-  self.lambda_hat/2)/(1+(F.softplus(self.Alpha)*(self.freqs- var_omega.clone())**2)))
        u_hat_plus=var_u_hat.clone()
        var_u=var_u_hat.clone()
 
        if k!=0:
            val=(abs(var_u[self.T //2:self.T])**2).clone()
            var_omega=torch.sum(self.freqs[self.T //2:self.T ]*(val),dim=0)/torch.sum(val,dim=0)
        return var_u_hat,var_omega,sum_uk
    
class VMD(nn.Module):
    def __init__(self,device,Alpha,lambda_hat,T,f_hat_plus,freqs,K):
        super(VMD,self).__init__()
        self.Alpha=Alpha
        self.lambda_hat=lambda_hat
        self.f_hat_plus=f_hat_plus
        self.freqs=freqs
        self.T=T
        self.K=K
        self.device=device
        self.inner_layer=self.makelayers(T=self.T,f_hat_plus=self.f_hat_plus,freqs=self.freqs)
        
    def makelayers(self,T,f_hat_plus,freqs):
         self.modules_list = nn.ModuleList([
            VMD_innerloop(device=self.device, Alpha=self.Alpha[k], lambda_hat=self.lambda_hat, T=T, f_hat_plus=f_hat_plus, freqs=freqs)
            for k in range(self.K)
        ])
    def forward(self,n_u_hat_plus,u_hat_plus,n_omega_plus,omega_plus,sum_uk): 
        k=0
        for module in self.modules_list:
                var_omega=omega_plus[k].clone()
                if k==0:
                    var_u_hat,var_omega,sum_uk = module(u_hat_plus[:,self.K-1],u_hat_plus[:,k], omega_plus[k],sum_uk,k)    
                else:
                    var_u_hat,var_omega,sum_uk = module(n_u_hat_plus[:,k-1],u_hat_plus[:,k], omega_plus[k],sum_uk,k)
                    n_omega_plus[k]=var_omega
                n_u_hat_plus=n_u_hat_plus.clone()  
                n_u_hat_plus[:,k]=var_u_hat
               
                k=k+1       
        return n_u_hat_plus,n_omega_plus,sum_uk
    
class Unfolding(nn.Module):
    def __init__(self,device,K=13,freqs_len=480,layers=1,alpha=2000):
        super(Unfolding, self).__init__()
        logging.info(f"Number of modes (K):{K}, Number of layers (N):{layers}")
        self.K=K
        self.layers=layers+1
        self.device=device
        self.freqs_len=freqs_len
        self.Alpha= nn.Parameter(torch.ones([self.K],requires_grad=True).to(device)*alpha)
       
        # self.Alpha= torch.ones([self.K]).to(device)*alpha
      
        self.lambda_hat=nn.Parameter(torch.zeros([self.layers,freqs_len],requires_grad=True,dtype=torch.cfloat).to(device))
        # self.lambda_hat=torch.ones([self.layers,freqs_len],dtype=torch.cfloat).to(device)
        self.omega_plus=torch.zeros([self.layers,self.K]).to(self.device)
        self.u_hat_plus = torch.zeros([self.layers,self.freqs_len,self.K],dtype=torch.cfloat).to(self.device)
        
        
        for i in range(self.K):
            self.omega_plus[0,i] = (0.5/K)*(i)
                
    def makelayers(self,T,f_hat_plus,freqs):
         self.modules_list = nn.ModuleList([
            VMD(device=self.device, Alpha=self.Alpha, lambda_hat=self.lambda_hat[n], T=T, f_hat_plus=f_hat_plus, freqs=freqs,K=self.K)
            for n in range(self.layers-1)
        ])
    def preprocessing (self,f):
        
        if len(f)%2:
            f = f[:-1]

        length_f=len(f)
        # Period and sampling frequency of input signal
        fs = 1./length_f
        ltemp = length_f//2 
        fMirr =  torch.cat((torch.flip(f[:ltemp],dims=[0]),f),dim=0)  
        fMirr = torch.cat((fMirr,torch.flip(f[-ltemp:],dims=[0])),dim=0)
      
        # Time Domain 0 to T (of mirrored signal)
        self.T = len(fMirr)
        t = torch.arange(1,self.T+1)/self.T   
    
        # Spectral Domain discretization
        freqs = t-0.5-(1/self.T)
        freqs=freqs.to(self.device)
        # Construct and center f_hat
        f_hat = torch.fft.fftshift((torch.fft.fft(fMirr)),dim=0)
        f_hat_plus = torch.clone(f_hat).to(self.device) #copy f_hat
        f_hat_plus[:self.T //2] = 0
        self.makelayers(self.T,f_hat_plus,freqs)
        return f_hat_plus
    
    def postprocessing (self,u_hat_plus):
        u_hat = torch.zeros([self.freqs_len,self.K],dtype=torch.cfloat).to(self.device)
        u = torch.zeros([self.K,self.freqs_len]).to(self.device)
        idxs=torch.flipud(torch.arange(1,self.T//2+1))
        u_hat[self.T//2:self.T,:]=u_hat_plus[self.T//2:self.T,:]
        
        u_hat[idxs,:] = torch.conj(u_hat_plus[self.T//2:self.T,:])
        u_hat[0,:] = torch.conj(u_hat[-1,:])   
        for k in range(self.K): 
            u[k,:]=torch.real(torch.fft.ifft(torch.fft.ifftshift(u_hat[:,k],dim=0)))


        u=u[:,self.T//4:3*self.T//4]
        return u
    
    
    def forward(self,x):
        f_hat_plus=self.preprocessing(x)
        u_hat_plus=self.u_hat_plus
        omega_plus=self.omega_plus.clone()
        sum_uk=0
        n=0
        for module in self.modules_list:
            n_u_hat_plus, omega,sum_uk = module(u_hat_plus[n+1],u_hat_plus[n],omega_plus[n+1],omega_plus[n],sum_uk)

            omega_plus[n+1]=omega.clone()
            u_hat_plus=u_hat_plus.clone()
            u_hat_plus[n+1]=n_u_hat_plus
            n=n+1
        
        loss=torch.sum(torch.abs(torch.sum(u_hat_plus[self.layers-1],axis=1)-f_hat_plus)) #+(1/frequency_differences)
        x= self.postprocessing(u_hat_plus[self.layers-1])
        return x,omega_plus,loss
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr=args.lr
early_stopping_patience =args.patience
num_epochs = args.epochs
best_val_loss = float("inf")
early_stopping_counter = 0
model=Unfolding(device,K=args.K,layers=args.layers,freqs_len=data.shape[0]*2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_params=num_params/1000000
print('model.params (M): ', num_params)
logging.info(f"Training started with lr={args.lr}, epochs={args.epochs}, patience={args.patience}")
logging.info(f"Model Parameters (M):{num_params}")
logging.info("Training started.")


def train_model(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch in loader:
        inputs = batch[0].squeeze(0).to(device)
        outputs,omega,loss = model(inputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss

def test_model(model, loader, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].squeeze(0).to(device)
            outputs,omega,loss = model(inputs)
            running_loss += loss.item()
    avg_loss = running_loss / len(loader)
    return avg_loss


train_losses = []
val_losses = []

for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, optimizer, device)
    val_loss = test_model(model, val_loader, device)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    logging.info(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, val Loss: {val_loss:.6f}")
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stopping_counter = 0
        torch.save(model.state_dict(), model_path)  # Save best model
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= early_stopping_patience:
        logging.info("Early stopping triggered.")
        print("Early stopping triggered.")
        break

test_loss = test_model(model, test_loader, device)
print(f"Testing result:{test_loss}")
logging.info(f"Testing result:{test_loss}")


torch.save(model.state_dict(), model_path)
print("Model saved.")
load_weights=torch.load(model_path)
model.load_state_dict(load_weights)

data_tensor=torch.tensor(data,dtype=torch.float32)
model_output=torch.zeros((data_tensor.shape[0],data_tensor.shape[1],args.K+3))
print(data_tensor.shape)
model.eval()
with torch.no_grad():
    for i in range(data_tensor.shape[1]):
        model_output[:,i,0]=data_tensor[:,i,0]
        input_segment=data_tensor[:,i,0]
        
        model_prediction,_,_ = model(input_segment)  # Output shape: (12)
           
        model_output[:, i, 1:(args.K+1)] = model_prediction.T # 13 outputs
        model_output[:, i, (args.K+1):] = data_tensor[:, i, 1:]  # 3 original features

numpy_array =model_output.cpu().numpy()
np.savez(args.output,data=numpy_array,mean=mean,std=std)
