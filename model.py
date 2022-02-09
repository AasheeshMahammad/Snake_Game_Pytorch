import torch,os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


class Linear_QNet(nn.Module):
    
    def __init__(self,input_size,hidden_size,output_size,device) -> None:
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size,hidden_size,device=self.device)
        self.linear2 = nn.Linear(hidden_size,output_size,device=self.device)
    
    def forward(self,x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self,file_name='model.pth'):
        model_folder_path='./model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_path = os.path.join(model_folder_path,file_name)
        torch.save(self.state_dict(),file_path)

class QTrainer:

    def __init__(self,model,lr,gamma,device) -> None:
        self.model=model
        self.lr = lr
        self.gamma = gamma
        self.device=device
        self.optimizer = optim.Adam(model.parameters(),lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self,state, action, reward, next_state, alive):
        state = torch.tensor(np.array(state),dtype=torch.float,device=self.device)
        next_state = torch.tensor(np.array(next_state),dtype=torch.float,device=self.device)
        action = torch.tensor(np.array(action),dtype=torch.long,device=self.device)
        reward = torch.tensor(np.array(reward),dtype=torch.float,device=self.device)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state,0)
            next_state = torch.unsqueeze(next_state,0)
            action = torch.unsqueeze(action,0)
            reward = torch.unsqueeze(reward,0)
            alive=(alive,)
        pred = self.model(state)
        target = pred.clone()

        for ind in range(len(alive)):
            Q_new = reward[ind]
            if alive[ind]:
                Q_new = reward[ind] + self.gamma * torch.max(self.model(next_state[ind]))
            target[ind][torch.argmax(action).item()] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target,pred)
        loss.backward()

        self.optimizer.step()
                 
