# imports for examples
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchtest import assert_vars_change

inputs = Variable(torch.randn(20, 20))
targets = Variable(torch.randint(0, 2, (20,))).long()
batch = [inputs, targets]
model = nn.Linear(20, 2)

# what are the variables?
print('Our list of parameters', [ np[0] for np in model.named_parameters() ])

# do they change after a training step?
#  let's run a train step and see
assert_vars_change(
    model=model,
    loss_fn=F.cross_entropy,
    optim=torch.optim.Adam(model.parameters()),
    batch=batch)

""" FAILURE """
# let's try to break this, so the test fails
params_to_train = [ np[1] for np in model.named_parameters() if np[0] is not 'bias' ]
# run test now
assert_vars_change(
    model=model,
    loss_fn=F.cross_entropy,
    optim=torch.optim.Adam(params_to_train),
    batch=batch)

# YES! bias did not change
