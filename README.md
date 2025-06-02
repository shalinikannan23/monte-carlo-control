# EXP-NO 05 : MONTE CARLO CONTROL ALGORITHM

## AIM
To implement Monte Carlo Control to learn an optimal policy in a grid environment and evaluate its performance in terms of 
goal-reaching probability and average return.

## PROBLEM STATEMENT
The task involves solving a Markov Decision Process (MDP) using Monte Carlo Control. 
The environment is likely a grid world where an agent must navigate through states to reach a goal while maximizing returns. 
The goal is to compute an optimal policy that achieves the highest probability of success (reaching the goal) and maximizes the average undiscounted return.

## MONTE CARLO CONTROL ALGORITHM

1. Initialize the policy randomly.

2. Generate episodes: Simulate episodes in the environment using the current policy.

3. Update action-value function Q(s,a): For each state-action pair encountered in the episode,
   update the expected return based on the actual rewards received during the episode.

5. Policy improvement: Update the policy greedily based on the updated action-value estimates.

6. Repeat the process until convergence.

## MONTE CARLO CONTROL FUNCTION

```py
DEVELOPED BY : SHALINI K
REGISTER NUMBER : 212222240095

def mc_control (env, gamma = 1.0,
                init_alpha = 0.5,min_alpha = 0.01, alpha_decay_ratio = 0.5,
                init_epsilon = 1.0, min_epsilon = 0.1, epsilon_decay_ratio = 0.9,
                n_episodes = 9000, max_steps = 400, first_visit = True):
  nS, nA = env.observation_space.n, env.action_space.n

  discounts=np.logspace(0,max_steps,num=max_steps,base=gamma,endpoint=False)
  alphas=decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
  epsilons=decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)
  pi_track=[]
  Q=np.zeros((nS,nA),dtype=np.float64)
  Q_track=np.zeros((n_episodes,nS,nA),dtype=np.float64)

  select_action=lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random() > epsilon else np.random.randint(len(Q[state]))

  for e in tqdm (range(n_episodes),leave=False):
    trajectory=generate_trajectory(select_action,Q,epsilons[e],env,max_steps)
    visited=np.zeros((nS,nA),dtype=bool)
    for t,(state,action,reward,_,_) in enumerate(trajectory):
      if visited[state] [action] and first_visit:
        continue
      visited[state][action]=True
      n_steps=len(trajectory[t:])
      G=np.sum(discounts[:n_steps]*trajectory[t:,2])
      Q[state][action]=Q[state][action]+alphas[e]*(G-Q[state][action])
      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))
  V=np.max(Q,axis=1)
  pi=lambda s: {s:a for s,a in enumerate (np.argmax(Q,axis=1))}[s]
  #return Q, V, pi, Q_track, pi_track
  return Q, V, pi
```
## OUTPUT:

![image](https://github.com/user-attachments/assets/66b55b5e-ac4a-4792-9644-e2aadf8f85c5)



## RESULT:
Thus to implement Monte Carlo Control to learn an optimal policy in a grid environment and evaluate its performance in terms of 
goal-reaching probability and average return is executed successfully.
