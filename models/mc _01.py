import torch
from torch import nn
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



device = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.manual_seed(42)
torch.manual_seed(42)


X, y= make_blobs(n_samples =1000 ,n_features= 2 ,centers = 4 , cluster_std =2.5 ,random_state = 42 )

# circles = pd.DataFrame({"x1" : X[:,0] , "x2":X[:,1],"y" :y})

X = torch.from_numpy(X).type(torch.float).to(device)
y = torch.from_numpy(y).type(torch.long).to(device)

Xtrain , Xtest , ytrain , ytest = train_test_split(X,y,test_size= 0.2 ,random_state=42)
Xtrain , ytrain = Xtrain.to(device) , ytrain.to(device)
Xtest , ytest = Xtest.to(device) , ytest.to(device)

class BlobModel01(nn.Module):
    def __init__(self,input_features , hidden_units  , output_features ):
        super().__init__()
        self.layer=nn.Sequential( 
        nn.Linear( input_features, hidden_units), 
        nn.ReLU(),
        nn.Linear( hidden_units, hidden_units),
        nn.ReLU(),
        nn.Linear( hidden_units, hidden_units),
        nn.ReLU(),
        nn.Linear( hidden_units , hidden_units),
        nn.ReLU(),
        nn.Linear( hidden_units , output_features),
        nn.ReLU())

    def forward(self , x : torch.tensor):
        return self.layer(x)


model = BlobModel01(input_features=2,hidden_units=8,output_features=4).to(device)

# print(next(model.parameters()).to(device))

# print(Xtrain.shape, ytrain.shape)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(params = model.parameters(), lr = 0.1)


epochs = 1000

for epoch in range(epochs):
    model.train()

    logits = model(Xtrain)
    pred = torch.argmax(torch.softmax(logits, dim = 1) , dim =1)

    loss = loss_fn(logits , ytrain)
    
    optimizer.zero_grad()
    
    loss.backward()

    optimizer.step()

    if epoch % 100 == 0:
        print(f"epoch : {epoch} , loss : {loss} ")


# print(pred)
# print(ytrain)
# print(torch.eq(pred , ytrain))
# print(f"accuracy : {accuracy_score(y_true = ytrain.cpu() , y_pred = pred.cpu()) * 100}%")

model.eval()

with torch.inference_mode():

    test_logits = model(Xtest)
    test_pred = torch.argmax(torch.softmax(test_logits , dim = 1) , dim = 1 )

    test_loss = loss_fn(test_logits,ytest)
    print(f"Test loss : {test_loss}")


# print(torch.eq(test_pred , ytest))
print(f"accuracy : {accuracy_score(y_true = ytest.cpu() , y_pred = test_pred.cpu()) * 100}%")