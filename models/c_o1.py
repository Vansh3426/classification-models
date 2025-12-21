import torch
from torch import nn
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.manual_seed(42)
torch.manual_seed(42)

n_samples = 1000

X , y = make_circles(n_samples ,noise = 0.03 , random_state = 42 )

circles = pd.DataFrame({"x1" : X[:,0] , "x2":X[:,1],"y" :y})

X = torch.from_numpy(X).type(torch.float).to(device)
y = torch.from_numpy(y).type(torch.float).to(device)

Xtrain , Xtest , ytrain , ytest = train_test_split(X,y,test_size= 0.2 ,random_state=42)
Xtrain , ytrain = Xtrain.to(device) , ytrain.to(device)
Xtest , ytest = Xtest.to(device) , ytest.to(device)




class CirclesModel01(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear( 2 , 5)
        self.layer2 = nn.Linear( 5 , 10)
        self.layer3 = nn.Linear( 10 , 5)
        self.layer4= nn.Linear( 5 , 1)
        self.relu =nn.ReLU()

    def forward(self , x : torch.tensor):
        return self.layer4(self.relu(self.layer3(self.relu(self.layer2(self.relu(self.layer1(x)))))))




model_0 = CirclesModel01().to(device)

# print(next(model_0.parameters()).to(device))


# #### another method to create neural network 

# model_0 = nn.Sequential(
#           nn.Linear(2,5),
#           nn.Linear(5,1)
# ).to(device)

# with torch.inference_mode():
#  pred = model_0(Xtrain)
# print(pred[:5])
# print(ytrain[:5])

loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params = model_0.parameters(),lr = 0.1)


epochs = 10000

for epoch in range(epochs):

    model_0.train()

    logits = model_0(Xtrain)
    preds = torch.round(torch.sigmoid(logits))
    
    loss = loss_fn(logits , ytrain.unsqueeze(dim =1))
    # print(f"Epoch : {epoch} , loss : {loss}")

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch : {epoch} , loss : {loss} ")



model_0.eval()

with torch.inference_mode():
    test_logits = model_0(Xtest)

    test_preds = torch.round(torch.sigmoid(test_logits)).squeeze()
   
    test_loss = loss_fn(test_logits ,  ytest.unsqueeze(dim = 1))
    
print(f"Test preds : {test_preds[:5]}")
print(f"Test Ytest : {ytest[:5]}")
print(f"Test loss : {test_loss}")
# print(torch.eq(test_preds , ytest))