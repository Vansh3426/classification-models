import torch
import pandas as pd
from sklearn.datasets import make_circles

n_samples = 1000

X , y = make_circles(n_samples ,noise = 0.03 , random_state = 42 )

# print(X.shape ,y.shape)

# print(X[:6] , y[:6])
circles = pd.DataFrame({"x1" : X[:,0] , "x2":X[:,1],"y" :y})

# print(df.head())

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

# print(X.type(),y.type())

