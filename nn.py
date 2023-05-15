import torch as T
from torch import nn
from torch.nn import functional as F
from torch import optim


class Model(nn.Module):

    def __init__(self, lr):
        super(Model, self).__init__()

        # architecture
        self.layer1 = nn.Conv2d(1,32,3, padding=1)
        self.layer2 = nn.BatchNorm2d(32)
        self.layer3 = nn.ReLU()
        self.layer4 = nn.Conv2d(32,32,3, padding=1)
        self.layer5 = nn.BatchNorm2d(32)
        self.layer6 = nn.ReLU()
        self.layer7 = nn.Conv2d(32,1,3, padding=1)
        self.layer8 = nn.BatchNorm2d(1)
        self.layer9 = nn.ReLU()
        self.layer10 = nn.Flatten()  
        self.layer11 = nn.Linear(28*28, 10)
        self.layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7, self.layer8, self.layer9, self.layer10, self.layer11]

        # optimizer & loss
        self.optimizer   = optim.Adam(self.parameters(), lr=lr)
        self.loss  = nn.MSELoss()

        # device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.softmax(x, dim=1)

    def fit(self, X, Y, batch_size, epochs):

        n = len(X)
        losses = []
        
        for epoch in range(epochs):
            for i in range(0,n,batch_size):

                # get sample
                X_batch = X[i:i+batch_size] if i+batch_size <= n else X[i:]
                Y_batch = Y[i:i+batch_size] if i+batch_size <= n else Y[i:]
                
                # get predictions
                Z_batch = self.forward(X_batch)
                
                # train
                self.optimizer.zero_grad()
                loss = self.loss(Z_batch, Y_batch).to(self.device)
                losses.append(loss)
                loss.backward()
                self.optimizer.step()
            print(f"[{epoch}] -- loss: {losses[-1]} -- mean loss: {sum(losses)/len(losses)}")

        return T.tensor(losses).tolist()

    def save_model(self, path):
        T.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(T.load(path))