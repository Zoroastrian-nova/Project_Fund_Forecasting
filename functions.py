from datetime import datetime
import torch
import os
import torch.utils.tensorboard as tensorboard
import sklearn.metrics as metrics
import torch.optim as optim
from torch import nn
import numpy as np
import torch.utils.data as data
import torch.optim.lr_scheduler as lr_scheduler
from ray import tune
from ray.train import report
from torch.utils.data import Dataset,DataLoader
from torchviz import make_dot

patience = 0
class MAELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))
#(model,train_loader,val_loader,max_epochs,optimizer,config,checkpoint_dir = "./checkpoint/",
#          criterion = nn.MSELoss(),metric = metrics.mean_absolute_error,
#          max_patience = 5,tag = "LR",device = "cuda",step_size=5, gamma=0.1)

def tuning(config):
    
    output_size = 1
    tag = config["Model"]
    input_size = config["input_size"]
    max_epochs = config["max_epochs"]
    step_size = config["step_size"]
    gamma = config["gamma"]
    lr = config["lr"]
    max_patience = config["max_patience"]
    train_loader = config["train_loader"]
    val_loader = config["val_loader"]

    criterion = nn.MSELoss()
    metric = MAELoss()
    device = "cuda"

    if tag == "LR":
        model = nn.Sequential(
                    nn.BatchNorm1d(input_size),
                    nn.Linear(input_size, 1), # input size is 10, hidden size is 20
        )
    elif tag =="LSTM":
        num_layers = config["num_layers"]
        hidden_size = config["hidden_size"]
        model = LSTMModel(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,output_size=1)
    elif tag == "Transformer":
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        num_heads = config["num_heads"]
        dropout = config["dropout"]
        model = TransformerModel(input_size, output_size, hidden_size, num_heads, num_layers, dropout)
    
    
    optimizer = optim.Adam(model.parameters(), lr=lr)


    best_val_loss = float("inf")
    time = str(datetime.now()).replace(":","_")
    writer = tensorboard.SummaryWriter(log_dir=f"logs//{tag}//{time}")
    patience = 0
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    #torch.autograd.set_detect_anomaly(True)
    # Loop over epochs
    for epoch in range(max_epochs):
        # Loop over batches in training set
        train_loss = 0
        train_acc = 0
        for i, (inputs, labels) in enumerate(train_loader):
            # Move inputs and labels to device
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            model = model.to(device)
            
            # Set model to training mode and clear gradients
            model.train()
            optimizer.zero_grad()
            
            # Forward pass and get outputs
            outputs = model(inputs)
            if len(outputs.shape) >2: 
                outputs = outputs.squeeze(-1)
            if len(labels.shape) >2: 
                labels = labels.squeeze(-1)
            
            # Compute loss and metric
            loss = criterion(outputs, labels)
            #if outputs.shape[1]>1:
            #preds = outputs.argmax(dim=1)
            #else:
            #preds = outputs
            acc = metric(outputs, labels)
            train_loss += loss.item()
            train_acc += acc

            # Backward pass and update parameters
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()
        

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        if epoch%5 ==0:
                print(f"*********\nEpoch:{epoch} Training MSE:{train_loss} Training MAE:{train_acc}\n")
        # Log batch loss and metric to TensorBoard
        writer.add_scalar("train_mse", train_loss,global_step=epoch)
        writer.add_scalar("train_mae", train_acc,global_step=epoch)
        
        # Loop over batches in validation set
        
        if (epoch%5 == 0):
            val_loss = 0
            val_acc = 0
            for i, (inputs, labels) in enumerate(val_loader):

                # Move inputs and labels to device
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()

                model = model.to(device)
                # Set model to evaluation mode and disable gradients
                model.eval()
                
                with torch.no_grad():
                    # Forward pass and get outputs
                    outputs = model(inputs)
                    if len(outputs.shape) >2: 
                        outputs = outputs.squeeze(-1)
                    if len(labels.shape) >2: 
                        labels = labels.squeeze(-1)
                    
                    # Compute loss and metric
                    loss = criterion(outputs, labels)
                    #if outputs.shape[1]>1:
                    #preds = outputs.argmax(dim=1)
                    #else:
                    #preds = outputs
                    

                    acc = metric(outputs, labels)
                    
                    
                    
                    # Accumulate validation loss and metric
                    val_loss += loss.item()
                    val_acc += acc

                    
            
            # Compute average validation loss and metric
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

            # Log batch loss and metric to TensorBoard
            writer.add_scalar(tag = "val_mse",scalar_value= val_loss,global_step=epoch)
            writer.add_scalar("val_mae", val_acc,global_step=epoch)

            print(f"*********\nEpoch:{epoch} Validation MSE:{val_loss} Validation MAE:{val_acc}\n")
            # Compare validation loss with best_val_loss and update best_val_loss, model, and patience
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, f"{tag}.pt")
                patience = 0
            else:
                patience += 1
            
            # Stop training if patience reaches threshold
            if patience >= max_patience:
                break
        
        report({
            "loss":float(train_loss),
            "train_mae":float(train_acc),
            "val_loss":float(best_val_loss),
            "val_mae":float(val_acc)
        })
        # Load best model
        #model = torch.load(f"{tag}.pt")


    # Close SummaryWriter and return best model
    #writer.close()
    #return {"loss":train_loss,"mae":train_acc}
    #return model
    #return train_loss



def run_models(config):
    
    output_size = 1
    tag = config["Model"]
    input_size = config["input_size"]
    max_epochs = config["max_epochs"]
    step_size = config["step_size"]
    gamma = config["gamma"]
    lr = config["lr"]
    max_patience = config["max_patience"]
    train_loader = config["train_loader"]
    val_loader = config["val_loader"]

    criterion = nn.MSELoss()
    metric = MAELoss()
    device = "cuda"

    if tag == "LR":
        model = nn.Sequential(
                    nn.BatchNorm1d(input_size),
                    nn.Linear(input_size, 1), # input size is 10, hidden size is 20
        )
    elif tag =="LSTM":
        num_layers = config["num_layers"]
        hidden_size = config["hidden_size"]
        model = LSTMModel(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,output_size=1)
    elif tag == "Transformer":
        hidden_size = config["hidden_size"]
        num_layers = config["num_layers"]
        num_heads = config["num_heads"]
        dropout = config["dropout"]
        model = TransformerModel(input_size, output_size, hidden_size, num_heads, num_layers, dropout)
    
    
    optimizer = optim.Adam(model.parameters(), lr=lr)


    best_val_loss = float("inf")
    time = str(datetime.now()).replace(":","_")
    writer = tensorboard.SummaryWriter(log_dir=f"logs//{tag}//{time}")
    patience = 0
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    #torch.autograd.set_detect_anomaly(True)
    # Loop over epochs
    for epoch in range(max_epochs):
        # Loop over batches in training set
        train_loss = 0
        train_acc = 0
        for i, (inputs, labels) in enumerate(train_loader):
            # Move inputs and labels to device
            inputs = inputs.to(device).float()
            labels = labels.to(device).float()
            model = model.to(device)
            
            # Set model to training mode and clear gradients
            model.train()
            optimizer.zero_grad()
            
            # Forward pass and get outputs
            outputs = model(inputs)
            if len(outputs.shape) >2: 
                outputs = outputs.squeeze(-1)
            if len(labels.shape) >2: 
                labels = labels.squeeze(-1)
            
            # Compute loss and metric
            loss = criterion(outputs, labels)
            #if outputs.shape[1]>1:
            #preds = outputs.argmax(dim=1)
            #else:
            #preds = outputs
            acc = metric(outputs, labels)
            train_loss += loss.item()
            train_acc += acc

            # Backward pass and update parameters
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()
        

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        if epoch%5 ==0:
                print(f"*********\nEpoch:{epoch} Training MSE:{train_loss} Training MAE:{train_acc}\n")
        # Log batch loss and metric to TensorBoard
        writer.add_scalar("train_mse", train_loss,global_step=epoch)
        writer.add_scalar("train_mae", train_acc,global_step=epoch)
        
        # Loop over batches in validation set
        
        if (epoch%5 == 0):
            val_loss = 0
            val_acc = 0
            for i, (inputs, labels) in enumerate(val_loader):

                # Move inputs and labels to device
                inputs = inputs.to(device).float()
                labels = labels.to(device).float()

                model = model.to(device)
                # Set model to evaluation mode and disable gradients
                model.eval()
                
                with torch.no_grad():
                    # Forward pass and get outputs
                    outputs = model(inputs)
                    if len(outputs.shape) >2: 
                        outputs = outputs.squeeze(-1)
                    if len(labels.shape) >2: 
                        labels = labels.squeeze(-1)
                    
                    # Compute loss and metric
                    loss = criterion(outputs, labels)
                    #if outputs.shape[1]>1:
                    #preds = outputs.argmax(dim=1)
                    #else:
                    #preds = outputs
                    

                    acc = metric(outputs, labels)
                    
                    
                    
                    # Accumulate validation loss and metric
                    val_loss += loss.item()
                    val_acc += acc

                    
            
            # Compute average validation loss and metric
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

            # Log batch loss and metric to TensorBoard
            writer.add_scalar(tag = "val_mse",scalar_value= val_loss,global_step=epoch)
            writer.add_scalar("val_mae", val_acc,global_step=epoch)

            print(f"*********\nEpoch:{epoch} Validation MSE:{val_loss} Validation MAE:{val_acc}\n")
            # Compare validation loss with best_val_loss and update best_val_loss, model, and patience
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model, f"{tag}.pt")
                patience = 0
            else:
                patience += 1
            
            # Stop training if patience reaches threshold
            if patience >= max_patience:
                break
        
        # Load best model
        model = torch.load(f"{tag}.pt")

    input_tensor = torch.rand_like(inputs)
    dot = make_dot(model(input_tensor), params=dict(model.named_parameters()))
    writer.add_graph(model, input_tensor)

    # Close SummaryWriter and return best model
    writer.close()
    #return {"loss":train_loss,"mae":train_acc}
    return model

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, index):
        # Return a random pair of feature and label tensors
        #rand_index = torch.randint(0, len(self.X), (1,)).item()
        #return self.X[rand_index], self.y[rand_index]
        return self.X[index], self.y[index]
    
    def __len__(self):
        # Return the size of the dataset
        return len(self.X)
    

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.normalize = nn.BatchNorm1d(input_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # 全连接层
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        x = self.normalize(x)
        x = x.view(len(x),1,-1)

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # 前向传播 LSTM
        out, (hn, cn) = self.lstm(x, (h0, c0))
        
        # 取最后一个时间步的输出
        out = self.fc(out[:, -1, :])
        return out
    

class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_heads, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.normalize = nn.LayerNorm(input_size)
        # Define the encoder layer
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
        # Define the encoder
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers,norm=self.normalize)
        self.linear = nn.Linear(input_size,output_size)
    
    def forward(self, x):
        # x: input sequence of shape (batch_size, sequence_length, input_size)
        x = self.normalize(x)
        x = x.unsqueeze(1)
        # Encode the input sequence
        x = x.permute(1, 0, 2) # reshape to (sequence_length, batch_size, input_size)
        x = self.encoder(x) # output shape: (sequence_length, batch_size, input_size)
        x = x.permute(1, 0, 2) # reshape to (batch_size, sequence_length, input_size)
        x = self.linear(x)
        return x