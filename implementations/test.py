import torch
from torch.distributions import Categorical

p = torch.tensor([-1.5, -0.2])

pdparams = p
pd = Categorical(logits=pdparams)

action = pd.sample()
print(pd.log_prob(action))
