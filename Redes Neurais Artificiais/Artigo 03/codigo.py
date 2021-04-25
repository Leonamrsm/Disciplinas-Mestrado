import numpy as np
import pandas as pd

from kerneloptimizer.optimizer import KernelOptimizer

X_df = pd.DataFrame({'a':[1,2,3,4],'b':[5,6,7,8]})
X = X_df.to_numpy()
y = np.array([1,1,2,2])
d = X.shape[1]

print("initializing MLP kernel")
opt = KernelOptimizer(
    kernel='mlp',
    input_dim=d,
    hidden_dim=20,
    output_dim=50
)

print("Training MLP")
opt.fit(X,y,n_epochs=100)

lspace = opt.get_likelihood_space(X,y)

print(lspace)

print("Initializing Gaussian kernel")
opt2 = KernelOptimizer(
    kernel='gaussian'
)

print("Finding the width")
opt2.fit(X,y)

lspace2 = opt2.get_likelihood_space(X,y)

print(lspace2)


 
 



