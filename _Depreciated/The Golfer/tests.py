state = torch.Tensor([envA.reset()]).to(device)
state = augment(state[0])
actionA = agentA.calc_action(state, action_noise=None).to(device)
actionB = agentB.calc_action(state, action_noise=None).to(device)

if state.dim() == 1:
    state = state.unsqueeze(0).to(device)
if actionA.dim() == 1:
    actionA = actionA.unsqueeze(0).to(device)
if actionB.dim() == 1:
    actionB = actionB.unsqueeze(0).to(device)
q_valueA = agentA.critic(state, actionA)
q_valueB = agentB.critic(state, actionB)

# print(state, action, q_value)
# print(torch.cat((state, action,q_value), 1))
print(cat_inputs(state, (actionA, q_valueA), (actionB, q_valueB)))