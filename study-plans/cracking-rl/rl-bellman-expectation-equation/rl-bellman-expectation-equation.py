import torch
def bellman_expectation_backup(P, R, policy, gamma, V):
    """
    Returns: list of length S, V_new[s] rounded to 4 decimals
    """
    V = torch.as_tensor(V, dtype=torch.float32)
    P, R, policy = torch.as_tensor(P, dtype=torch.float32), torch.as_tensor(R, dtype=torch.float32), torch.as_tensor(policy, dtype=torch.float32)
    V_new = torch.zeros_like(V)
    for s_cur in range(V.shape[-1]):
        for a in range(policy[s_cur].shape[-1]): # current state is s_cur, take action a
            for s_prime in range(P[s_cur][a].shape[-1]): # new state is s_prime
                V_new[s_cur] += policy[s_cur][a] * P[s_cur][a][s_prime] * (R[s_cur][a][s_prime] + gamma * V[s_prime])
    print(V_new)
    return V_new
