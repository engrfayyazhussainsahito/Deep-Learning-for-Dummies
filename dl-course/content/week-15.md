---
title: "Deep Reinforcement Learning"
phase: Advanced Topics
summary: Learning to act from reward. MDPs, returns, value functions and the Bellman equations (derived), dynamic programming, Monte Carlo and temporal-difference learning (Q-learning, SARSA), function approximation and the deadly triad, DQN with replay and target networks, Double, Dueling and prioritized variants, the policy gradient theorem (derived), REINFORCE with baselines, actor-critic, GAE (derived), A2C/A3C, TRPO and PPO, continuous control (DDPG, TD3, SAC), exploration, model-based RL and world models, offline RL, and the connection to language models (RLHF, reward models, PPO, DPO, GRPO). The lab trains DQN on CartPole in PyTorch and PPO with Stable-Baselines3.
---

:::key Learning objectives
By the end of this week you should be able to:
- Formalize a sequential decision problem as a Markov decision process and define return, policy, state-value and action-value functions.
- Derive the Bellman expectation and optimality equations and use them for policy evaluation, policy iteration and value iteration.
- Contrast Monte Carlo and temporal-difference estimates in terms of bias and variance, and explain why Q-learning is off-policy while SARSA is on-policy.
- Explain the deadly triad (function approximation, bootstrapping, off-policy data) and how DQN's replay buffer and target network mitigate it.
- Describe Double DQN, Dueling DQN and prioritized experience replay and the specific failure each one repairs.
- Derive the policy gradient theorem, REINFORCE, the baseline trick and generalized advantage estimation (GAE), and explain the bias-variance role of $\lambda$.
- Explain A2C/A3C, TRPO's trust region, and PPO's clipped objective, and predict how the clip changes the gradient.
- Compare DDPG, TD3 and SAC for continuous control and write the maximum-entropy objective.
- Summarize exploration strategies, model-based RL and world models, and offline RL with its distribution-shift problem.
- Describe the RLHF pipeline (supervised fine-tuning, reward model, PPO with a KL penalty) and how DPO and GRPO relate to it.
- Implement DQN from scratch in PyTorch for CartPole and train PPO with Stable-Baselines3.
:::

Everything in the previous weeks learned from a fixed dataset of labelled examples or from raw data. **Reinforcement learning (RL)** changes the problem. There is no teacher that says what the right output is. An agent takes actions in a world, the world responds with a new situation and a scalar **reward**, and the agent must discover, by trial and error, how to act so that the total reward over time is large. A chess program learns from a single win or loss at the end of a game that may have lasted eighty moves; a robot learns to walk from a signal that says only how far it travelled; a language model is tuned toward answers that human raters prefer. Three difficulties distinguish this from supervised learning: **credit assignment** (which of the many past actions deserves credit for the reward?), **exploration** (the agent only sees the consequences of what it tries, so it must choose what to try), and **non-stationarity** (the data the agent trains on depends on the agent's own current behaviour, so the data distribution moves as it learns).

Deep RL is RL in which the policy, the value function, or a model of the world is a neural network. It produced the best-known demonstrations of the last decade: DQN playing Atari from pixels (2013-2015), AlphaGo and AlphaZero (2016-2017), OpenAI Five and AlphaStar, dexterous robot hands, plasma control in tokamaks, and, most relevant to this course, the post-training of large language models with human feedback. It is also notoriously fragile: results depend on seeds, hyper-parameters and implementation details far more than in supervised learning. This week gives you the theory you need to know *why* the algorithms look the way they do, and the practice to run them and diagnose them.

The structure follows the logic of the field. We first fix the formalism (MDPs, value functions, Bellman equations). We then build the tabular toolbox (dynamic programming, Monte Carlo, temporal-difference learning), because every deep algorithm is one of these ideas with a network inside. Value-based deep RL (DQN and its repairs) comes next, then policy-gradient methods (REINFORCE, actor-critic, GAE, PPO), then continuous control, exploration, model-based and offline RL, and finally how all of this is used on language models. Notation follows Sutton and Barto, *Reinforcement Learning: An Introduction* (2nd ed., 2018), the standard text.

{{figure: week15-agent-env | The agent-environment loop. At each step the agent observes a state $S_t$, picks an action $A_t$ from its policy, and the environment returns a reward $R_{t+1}$ and the next state $S_{t+1}$. Everything the agent learns is derived from the stream of these tuples.}}

## Markov decision processes {#mdp}

The standard model of a sequential decision problem is the **Markov decision process (MDP)**, a tuple $(\mathcal S,\mathcal A,p,r,\gamma)$:

- $\mathcal S$ is the set of **states** and $\mathcal A$ the set of **actions** (both can be finite, as in a board game, or continuous, as in joint torques).
- $p(s'\mid s,a)$ is the **transition probability**: the chance that taking action $a$ in state $s$ leads to state $s'$.
- $r(s,a)$ (or $r(s,a,s')$) is the **reward function**; $R_{t+1}$ denotes the random reward received after the action at time $t$.
- $\gamma\in[0,1]$ is the **discount factor**, explained below.

The defining assumption is the **Markov property**: the next state and reward depend on the history only through the current state and action, $\Pr(S_{t+1},R_{t+1}\mid S_0,A_0,\dots,S_t,A_t)=\Pr(S_{t+1},R_{t+1}\mid S_t,A_t)$. The state is a sufficient statistic of the past. This is a statement about how you *define* the state: if a single Atari frame does not reveal the velocity of the ball, it is not Markov, but a stack of four frames nearly is, and that is exactly why DQN feeds four frames to its network. When the observation is not Markov, the problem is a *partially observable* MDP (POMDP), and the agent needs memory (recurrent networks or attention over history, as in [Week 7](week-07.html) and [Week 9](week-09.html)).

A **policy** $\pi(a\mid s)$ is the agent's rule for acting: a distribution over actions for each state (a deterministic policy is the special case with all mass on one action, written $a=\mu(s)$). Following $\pi$ from a start state generates a **trajectory** $\tau=(S_0,A_0,R_1,S_1,A_1,R_2,\dots)$. The goal is to find the policy that maximizes the expected sum of rewards.

### Returns and discounting

The **return** from time $t$ is the discounted sum of future rewards:

$$
G_t \;=\; R_{t+1}+\gamma R_{t+2}+\gamma^2 R_{t+3}+\cdots \;=\; \sum_{k=0}^{\infty}\gamma^{k}R_{t+k+1}.
$$

A reward $k$ steps in the future is worth $\gamma^k$ of an immediate one. Discounting does three jobs. It keeps the sum finite for infinitely long tasks when rewards are bounded ($|G_t|\le R_{\max}/(1-\gamma)$). It encodes a preference for sooner rewards, and it reduces variance, because distant rewards, which are the most uncertain, count for less. Its **effective horizon** is about $1/(1-\gamma)$ steps: with $\gamma=0.99$ the agent looks around 100 steps ahead, with $\gamma=0.9$ about ten. Choosing $\gamma$ is therefore a modelling decision about how far ahead the agent should plan, not just a numerical convenience. A useful recursion follows directly from the definition and is the seed of everything below:

$$
G_t = R_{t+1}+\gamma G_{t+1}.
$$

Many tasks are **episodic**: they end in a terminal state (a game is won, a pole falls over), after which the environment resets. Episodic tasks can use $\gamma=1$; in that case the sum is finite because the episode is. At a terminal state we define the value of the next state to be zero, which in code becomes the `(1 - done)` mask that you will see in every implementation.

### Value functions

Rewards are immediate; what the agent needs is a way to judge how *good* it is to be somewhere. The **state-value function** of a policy is the expected return when starting in $s$ and following $\pi$ thereafter, and the **action-value function** (or Q-function) is the expected return when starting in $s$, taking action $a$ first, and following $\pi$ afterwards:

$$
v_\pi(s)=\E_\pi\!\left[G_t\mid S_t=s\right],\qquad q_\pi(s,a)=\E_\pi\!\left[G_t\mid S_t=s,A_t=a\right].
$$

The two are linked by averaging over the policy: $v_\pi(s)=\sum_a\pi(a\mid s)\,q_\pi(s,a)$. The difference $A_\pi(s,a)=q_\pi(s,a)-v_\pi(s)$ is the **advantage**: how much better action $a$ is than the policy's average behaviour in $s$. Advantages will be the central quantity of policy-gradient methods. Note what a value function is: a *prediction* of future reward under a particular behaviour, learnt from experience. It turns the hard, delayed credit-assignment problem into a local one: if you have a good $v$, you can judge an action immediately by the value of where it leads.

### The Bellman equations

The recursion $G_t=R_{t+1}+\gamma G_{t+1}$ gives a self-consistency condition on value functions. Deriving it is short and worth doing carefully, since every algorithm this week is an attempt to enforce, approximate or exploit it.

:::math Derivation: Bellman expectation equation
Start from the definition and substitute the return recursion:
$$
\begin{aligned}
v_\pi(s) &= \E_\pi[G_t\mid S_t=s] \\
&= \E_\pi[R_{t+1}+\gamma G_{t+1}\mid S_t=s] \\
&= \sum_a \pi(a\mid s)\sum_{s'}p(s'\mid s,a)\Big[r(s,a,s')+\gamma\,\E_\pi[G_{t+1}\mid S_{t+1}=s']\Big] \\
&= \sum_a \pi(a\mid s)\sum_{s'}p(s'\mid s,a)\Big[r(s,a,s')+\gamma\, v_\pi(s')\Big].
\end{aligned}
$$
The first step sums over the first action (weighted by the policy) and over the next state (weighted by the dynamics). The last step uses the Markov property: given $S_{t+1}=s'$, the expected future return does not depend on how we got there, so it is $v_\pi(s')$ by definition. Doing the same for $q_\pi$:
$$
q_\pi(s,a)=\sum_{s'}p(s'\mid s,a)\Big[r(s,a,s')+\gamma\sum_{a'}\pi(a'\mid s')\,q_\pi(s',a')\Big].
$$
:::

In words: the value of a state is the expected immediate reward plus the discounted value of the successor. For a finite MDP this is a system of $|\mathcal S|$ linear equations, $\boldsymbol v_\pi=\boldsymbol r_\pi+\gamma\mathbf P_\pi\boldsymbol v_\pi$, with the unique solution $\boldsymbol v_\pi=(\mathbf I-\gamma\mathbf P_\pi)^{-1}\boldsymbol r_\pi$ (the inverse exists for $\gamma<1$ because $\mathbf P_\pi$ is a stochastic matrix, whose eigenvalues have modulus at most one). Direct solution costs $O(|\mathcal S|^3)$, so for larger problems we iterate instead.

**Optimality.** A policy $\pi$ is at least as good as $\pi'$ if $v_\pi(s)\ge v_{\pi'}(s)$ for all $s$. For finite MDPs there is always at least one optimal policy, and all optimal policies share the same **optimal value functions** $v_*(s)=\max_\pi v_\pi(s)$ and $q_*(s,a)=\max_\pi q_\pi(s,a)$. Because an optimal policy must pick the best action in every state, the expectation over the policy in the Bellman equation becomes a maximum:

$$
v_*(s)=\max_a\sum_{s'}p(s'\mid s,a)\big[r+\gamma v_*(s')\big],\qquad
q_*(s,a)=\sum_{s'}p(s'\mid s,a)\Big[r+\gamma\max_{a'}q_*(s',a')\Big].
$$

These are the **Bellman optimality equations**. They are nonlinear (because of the max) and, crucially, once you know $q_*$ the optimal policy is trivial to read off: act greedily, $\pi_*(s)=\argmax_a q_*(s,a)$. This is why so much of RL is about estimating $q_*$.

:::definition Contraction view
Define the Bellman optimality operator $(\mathcal T v)(s)=\max_a\sum_{s'}p(s'\mid s,a)[r+\gamma v(s')]$. For any two value functions, $\norm{\mathcal Tv-\mathcal Tw}_\infty\le\gamma\norm{v-w}_\infty$, i.e. $\mathcal T$ is a $\gamma$-contraction in the max norm. By the Banach fixed-point theorem, repeated application converges to the unique fixed point $v_*$, from any starting point, at a geometric rate $\gamma^k$. This single fact underpins value iteration and the convergence proofs of tabular Q-learning; its failure under function approximation is the subject of the deadly-triad section.
:::

## Dynamic programming: planning with a known model {#dp}

Suppose, for now, you know the MDP exactly: $p$ and $r$ are given. Then finding a good policy is a *planning* problem, solved by **dynamic programming (DP)**. DP matters for deep RL even though you rarely know the model, because model-free methods are best understood as sampled, approximate versions of DP updates.

**Policy evaluation.** To compute $v_\pi$ iteratively, turn the Bellman expectation equation into an update rule: start from any $v_0$ and repeatedly set
$$
v_{k+1}(s)\leftarrow\sum_a\pi(a\mid s)\sum_{s'}p(s'\mid s,a)\big[r(s,a,s')+\gamma v_k(s')\big].
$$
This is also a $\gamma$-contraction, so $v_k\to v_\pi$.

**Policy improvement.** Given $v_\pi$, form the greedy policy $\pi'(s)=\argmax_a q_\pi(s,a)$. The **policy improvement theorem** says that $\pi'$ is at least as good as $\pi$: because acting greedily for one step and then following $\pi$ is no worse than following $\pi$ throughout, and the argument can be repeated at every subsequent step, $v_{\pi'}\ge v_\pi$ everywhere. If the improvement is zero the Bellman optimality equation holds and $\pi$ is optimal.

**Policy iteration** alternates full evaluation and greedy improvement; in a finite MDP it terminates in a finite number of iterations at an optimal policy, typically in surprisingly few. **Value iteration** collapses the two steps by using the optimality equation as an update:
$$
v_{k+1}(s)\leftarrow\max_a\sum_{s'}p(s'\mid s,a)\big[r(s,a,s')+\gamma v_k(s')\big].
$$
Each sweep costs $O(|\mathcal S|^2|\mathcal A|)$. Generalized policy iteration (GPI) is Sutton and Barto's name for the unifying picture: an evaluation process pushes the value function toward $v_\pi$, an improvement process pushes the policy toward greedy with respect to the value function, and the two chase each other to the optimum. Almost every algorithm below fits this pattern, with the exact steps replaced by sampled, partial ones.

```python:value_iteration.py
import numpy as np

# A 4x4 gridworld with a goal (bottom-right, reward +1, terminal) and a pit (reward -1, terminal).
# Actions: 0=up, 1=right, 2=down, 3=left. Moves into walls leave the agent in place.
# Dynamics are slippery: with prob 0.8 the intended move, 0.1 each for the two perpendicular moves.
H = W = 4
nS, nA, gamma = H * W, 4, 0.95
goal, pit = 15, 11
moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]

def move(s, a):
    r, c = divmod(s, W)
    dr, dc = moves[a]
    r2, c2 = min(max(r + dr, 0), H - 1), min(max(c + dc, 0), W - 1)
    return r2 * W + c2

P = np.zeros((nS, nA, nS))     # P[s, a, s'] transition tensor
R = np.zeros((nS, nA, nS))     # R[s, a, s'] reward tensor
for s in range(nS):
    for a in range(nA):
        if s in (goal, pit):
            P[s, a, s] = 1.0   # absorbing terminal states
            continue
        for a2, p in [(a, 0.8), ((a + 1) % 4, 0.1), ((a - 1) % 4, 0.1)]:
            s2 = move(s, a2)
            P[s, a, s2] += p
            R[s, a, s2] = 1.0 if s2 == goal else (-1.0 if s2 == pit else -0.04)  # small step cost

def value_iteration(tol=1e-8):
    V = np.zeros(nS)
    for k in range(1000):
        # Q[s, a] = sum_s' P[s,a,s'] (R[s,a,s'] + gamma V[s'])
        Q = (P * (R + gamma * V[None, None, :])).sum(axis=2)
        V_new = Q.max(axis=1)
        V_new[[goal, pit]] = 0.0             # terminal states have value zero
        if np.abs(V_new - V).max() < tol:
            return V_new, Q, k
        V = V_new
    return V, Q, k

V, Q, iters = value_iteration()
print(f"converged in {iters} sweeps")
print(np.round(V.reshape(H, W), 3))
print("greedy policy:")
print(np.array(list("^>v<"))[Q.argmax(axis=1)].reshape(H, W))
```

Run it and you will see values that decay smoothly with distance from the goal, with the greedy policy routing around the pit (not along the cheapest straight line) because of the slippery dynamics: this is the Bellman equation planning for noise.

{{demo: week15-gridworld | Gridworld: value iteration and Q-learning side by side | 560}}

DP has two limitations that motivate the rest of the week. It needs the model $p$, and each sweep touches every state, which is impossible for Go ($\sim10^{170}$ positions) or an image-based state space. The cure for both is to learn from *sampled experience* and to *generalize* across states with function approximation.

## Monte Carlo and temporal-difference learning {#mc-td}

Without a model, the agent has only sampled transitions. Two basic ways to turn samples into value estimates differ in what they use as the learning target.

### Monte Carlo (MC) prediction

The definition $v_\pi(s)=\E[G_t\mid S_t=s]$ suggests the plain statistical approach: run complete episodes, compute the realized return $G_t$ from each visit to $s$, and average. In incremental form, with a step size $\alpha$,
$$
V(S_t)\leftarrow V(S_t)+\alpha\big[G_t-V(S_t)\big].
$$
The target $G_t$ is an **unbiased** sample of $v_\pi(S_t)$, but it has **high variance**, since it sums many random rewards, actions and transitions. MC also needs episodes to finish before any update, so it cannot learn online in continuing tasks.

### Temporal-difference (TD) learning

TD learning replaces the unknown tail of the return with the current estimate. Using $G_t=R_{t+1}+\gamma G_{t+1}$ and approximating $\E[G_{t+1}\mid S_{t+1}]\approx V(S_{t+1})$ gives **TD(0)**:
$$
V(S_t)\leftarrow V(S_t)+\alpha\,\delta_t,\qquad \delta_t=R_{t+1}+\gamma V(S_{t+1})-V(S_t).
$$
The quantity $\delta_t$ is the **TD error**: the difference between a one-step lookahead and the current prediction. The target $R_{t+1}+\gamma V(S_{t+1})$ is **biased** (it relies on a possibly wrong $V$), but it has **low variance** (only one random reward and one random transition) and it can update after every single step. This is **bootstrapping**: updating an estimate from another estimate. Bootstrapping is the source of most of RL's efficiency and, as we will see, most of its instability.

{{figure: week15-backups | Backup diagrams. Dynamic programming averages over all actions and successor states using the model. Monte Carlo follows one sampled path to the end of the episode. TD(0) takes a single sampled step and bootstraps from the current value estimate.}}

| Method | Needs model | Update after | Target | Bias | Variance |
| --- | --- | --- | --- | --- | --- |
| Dynamic programming | yes | each sweep | full expectation over one step + $V$ | bootstraps | none (exact) |
| Monte Carlo | no | end of episode | $G_t$ | unbiased | high |
| TD(0) | no | every step | $R_{t+1}+\gamma V(S_{t+1})$ | bootstraps | low |
| $n$-step TD | no | after $n$ steps | $R_{t+1}+\dots+\gamma^{n}V(S_{t+n})$ | in between | in between |

The $n$-step return interpolates, and the **$\lambda$-return** averages all of them with geometric weights $(1-\lambda)\lambda^{n-1}$. We will meet exactly this device again as GAE. In practice, TD methods usually learn faster than MC, which is the empirical result Sutton and Barto illustrate on a random-walk task and which the cliff example below also shows.

### Control: SARSA and Q-learning

To act, we want $q$ rather than $v$ (it tells us which action is best without a model). Apply the TD idea to action values. **SARSA** (named after the tuple $S_t,A_t,R_{t+1},S_{t+1},A_{t+1}$) is **on-policy**: it evaluates the policy that is generating the data, including its exploration:
$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\big[R_{t+1}+\gamma\, Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\big].
$$
**Q-learning** (Watkins, 1989) is **off-policy**: whatever action the behaviour policy takes next, it bootstraps from the best action, thereby estimating $q_*$ directly:
$$
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\big[R_{t+1}+\gamma\max_{a'}Q(S_{t+1},a')-Q(S_t,A_t)\big].
$$
The one-token difference ($Q(S_{t+1},A_{t+1})$ versus $\max_{a'}Q(S_{t+1},a')$) has real consequences. Off-policy learning means you can learn about the greedy policy while following an exploratory one, or learn from old data or from another agent's data. That freedom is what lets DQN reuse a replay buffer. With a decaying step size and every state-action pair visited infinitely often, tabular Q-learning converges to $q_*$ (Watkins and Dayan, 1992).

The classic demonstration is **cliff walking**: a 4 by 12 grid where the bottom row between the start and the goal is a cliff (reward $-100$, reset to start), and every other step costs $-1$. Following an $\varepsilon$-greedy policy, Q-learning learns the shortest path along the cliff edge, but during training the random exploratory moves occasionally push it over; SARSA, because it accounts for its own exploration in its targets, learns the longer, safer path.

```python:cliff.py
import numpy as np

H, W = 4, 12
START, GOAL = 3 * W, 3 * W + 11
moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

def step(s, a):
    r, c = divmod(s, W)
    dr, dc = moves[a]
    r, c = min(max(r + dr, 0), H - 1), min(max(c + dc, 0), W - 1)
    if r == 3 and 0 < c < 11:             # stepped into the cliff
        return START, -100.0, False
    return r * W + c, -1.0, (r * W + c == GOAL)

def train(algo, episodes=300, alpha=0.5, gamma=1.0, eps=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((H * W, 4))
    def act(s):                           # epsilon-greedy with random tie-breaking
        if rng.random() < eps:
            return int(rng.integers(4))
        return int(np.argmax(Q[s] + 1e-9 * rng.random(4)))
    returns = []
    for _ in range(episodes):
        s, a, G = START, act(START), 0.0
        for _ in range(500):
            s2, r, done = step(s, a)
            a2 = act(s2)
            # Q-learning bootstraps from the greedy action; SARSA from the action actually chosen next
            boot = Q[s2].max() if algo == "q" else Q[s2, a2]
            target = r + (0.0 if done else gamma * boot)
            Q[s, a] += alpha * (target - Q[s, a])
            s, a, G = s2, a2, G + r
            if done:
                break
        returns.append(G)
    return np.array(returns)

for algo in ("q", "sarsa"):
    R = np.mean([train(algo, seed=k) for k in range(50)], axis=0)
    print(algo, "mean online return, last 100 episodes:", R[-100:].mean().round(1))
```

```chart
{
  "type": "line",
  "caption": "Online (training-time) return per episode on cliff walking, averaged over 50 seeds and over blocks of 10 episodes, computed with the code above (epsilon = 0.1, alpha = 0.5). Q-learning has the better greedy policy but a worse online return, because its path hugs the cliff while it keeps exploring.",
  "height": 320,
  "data": { "labels": [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300],
    "datasets": [
      { "label": "Q-learning", "data": [-315,-85,-58,-53,-43,-46,-55,-49,-52,-44,-53,-52,-44,-48,-50,-47,-49,-48,-46,-46,-53,-53,-48,-49,-51,-52,-45,-51,-51,-48], "borderColor": "var(--c-red)", "pointRadius": 0, "tension": 0.2 },
      { "label": "SARSA", "data": [-315,-83,-55,-47,-41,-35,-32,-30,-30,-30,-30,-31,-27,-29,-35,-32,-31,-31,-34,-34,-34,-28,-27,-27,-25,-27,-27,-26,-27,-28], "borderColor": "var(--c-blue)", "pointRadius": 0, "tension": 0.2 }
    ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "episode"}}, "y": {"min": -120, "title": {"display": true, "text": "online return per episode"}} } }
}
```

The lesson is general. **On-policy** methods value the policy they actually execute, exploration noise included. **Off-policy** methods value a different (usually greedy) target policy, so they find the better *final* policy but can behave worse *while learning*. If you were to anneal $\varepsilon\to0$, both would converge to the optimal path.

### A bandit warm-up: the exploration-exploitation dilemma {#bandits}

Strip an MDP down to a single state and you get the **multi-armed bandit**: $k$ actions ("arms"), each with an unknown mean reward $q_*(a)$. There is no state and no long-term consequence, so what remains is the pure *exploration versus exploitation* trade-off: pull the arm that looks best now (exploit), or pull another to learn whether it is better (explore). The simplest strategy is **$\varepsilon$-greedy**: with probability $\varepsilon$ choose a random arm, otherwise choose the arm with the highest sample-mean estimate $Q(a)$. The estimate is updated incrementally, $Q_{n+1}=Q_n+\frac1n(R_n-Q_n)$, which is the same "estimate $\leftarrow$ estimate + step $\times$ error" shape as every update in this chapter.

The ten-armed testbed (Sutton and Barto, Ch. 2) draws each $q_*(a)\sim\mathcal N(0,1)$ and rewards $\mathcal N(q_*(a),1)$. The greedy learner ($\varepsilon=0$) often locks on to a mediocre arm after a lucky early sample and never recovers; a small $\varepsilon$ keeps sampling and eventually finds the best arm. The next chart averages 1000 such randomly generated problems.

```chart
{"type": "line", "caption": "Ten-armed bandit testbed, 1000 independently generated problems, sample-average estimates (computed by us with NumPy). Average reward per step, shown every 20th step. After 1000 steps, epsilon = 0.1 picks the optimal arm about 80% of the time, epsilon = 0.01 about 60%, and pure greedy about 39% (mean over the last 100 steps).", "height": 320, "data": {"labels": [0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900, 920, 940, 960, 980], "datasets": [{"label": "greedy (eps = 0)", "data": [0.021, 0.995, 1.002, 0.994, 1.063, 1.02, 1.0, 1.057, 1.034, 1.025, 1.02, 1.06, 1.032, 1.051, 1.051, 0.952, 1.063, 1.04, 1.013, 1.072, 0.959, 1.009, 1.032, 1.036, 1.034, 1.029, 1.043, 1.004, 1.013, 1.044, 0.994, 1.08, 1.081, 1.01, 1.004, 1.011, 0.995, 1.051, 0.981, 1.108, 1.069, 1.038, 1.046, 1.037, 0.976, 0.992, 1.108, 1.044, 1.069, 1.023], "borderColor": "var(--c-red)", "pointRadius": 0, "tension": 0.2}, {"label": "eps = 0.01", "data": [0.024, 0.992, 0.972, 1.041, 1.129, 1.082, 0.993, 1.131, 1.111, 1.16, 1.119, 1.116, 1.11, 1.131, 1.154, 1.129, 1.152, 1.136, 1.171, 1.156, 1.168, 1.233, 1.177, 1.194, 1.254, 1.191, 1.242, 1.203, 1.245, 1.22, 1.23, 1.21, 1.311, 1.224, 1.26, 1.257, 1.261, 1.281, 1.24, 1.243, 1.25, 1.286, 1.27, 1.271, 1.209, 1.28, 1.312, 1.298, 1.298, 1.283], "borderColor": "var(--c-green)", "pointRadius": 0, "tension": 0.2}, {"label": "eps = 0.1", "data": [0.059, 0.977, 1.04, 1.092, 1.16, 1.203, 1.207, 1.224, 1.311, 1.218, 1.254, 1.229, 1.307, 1.288, 1.339, 1.379, 1.269, 1.313, 1.24, 1.359, 1.308, 1.304, 1.339, 1.358, 1.218, 1.304, 1.35, 1.293, 1.412, 1.311, 1.365, 1.315, 1.296, 1.296, 1.374, 1.308, 1.339, 1.366, 1.327, 1.318, 1.38, 1.315, 1.352, 1.389, 1.307, 1.373, 1.377, 1.323, 1.344, 1.288], "borderColor": "var(--c-blue)", "pointRadius": 0, "tension": 0.2}]}, "options": {"scales": {"x": {"title": {"display": true, "text": "step"}}, "y": {"title": {"display": true, "text": "average reward"}}}}}
```

{{demo: week15-bandit | Bandit explorer: epsilon-greedy, UCB and Thompson sampling | 480}}

Two alternatives deserve a name. **UCB1** (Auer, Cesa-Bianchi and Fischer, 2002) adds an optimism bonus that shrinks as an arm is sampled more: choose $\argmax_a\,Q(a)+c\sqrt{\ln t/N(a)}$. It explores arms whose value is *uncertain* rather than arms chosen at random. **Thompson sampling** (Thompson, 1933) keeps a posterior over each arm's mean, samples one value per arm from the posterior, and pulls the arm with the largest sample. Both achieve regret that grows only logarithmically with time, whereas fixed-$\varepsilon$ exploration has linear regret because it never stops exploring. We return to exploration in the deep setting in [its own section](#exploration), where counting visits is no longer possible and the ideas have to be approximated.

## Function approximation and the deadly triad {#triad}

Tables do not scale. An Atari frame is $210\times160$ RGB pixels; the number of distinct states is astronomically large and the agent almost never sees the same state twice. We must **generalize**: represent $q(s,a)$ by a parametric function $\hat q(s,a;\boldsymbol\theta)$, such as a neural network, so that experience in one state informs predictions in similar ones. The natural idea is to treat the TD target as a label and do gradient descent on the squared error. For a transition $(s,a,r,s')$ the **semi-gradient Q-learning** update is
$$
\boldsymbol\theta\leftarrow\boldsymbol\theta+\alpha\Big[r+\gamma\max_{a'}\hat q(s',a';\boldsymbol\theta)-\hat q(s,a;\boldsymbol\theta)\Big]\nabla_{\boldsymbol\theta}\hat q(s,a;\boldsymbol\theta).
$$
It is called a **semi**-gradient because the target contains $\boldsymbol\theta$ too, but we *treat it as a constant* when differentiating (in PyTorch: compute the target under `torch.no_grad()` or call `.detach()`). The true gradient of the squared Bellman error (the "residual gradient") also pushes the target toward the prediction, converges more reliably in theory, and in practice learns far slower and to worse solutions, so nobody uses it for deep RL.

Unlike supervised learning, this can diverge. Sutton and Barto identify the **deadly triad**: instability and divergence become possible when three ingredients are combined.

1. **Function approximation**: generalization means that updating one state changes the values of many others, including the very states whose values we bootstrap from.
2. **Bootstrapping**: targets depend on the current estimate, so errors feed back on themselves instead of being anchored by true returns.
3. **Off-policy training**: the data distribution does not match the distribution of the policy being evaluated, so there is no guarantee that the states where errors grow are the states that get corrected.

Any two of the three are fine. Tabular Q-learning (no approximation) converges. On-policy TD with linear approximation converges. Monte Carlo (no bootstrapping) with off-policy data is well behaved. Baird's counterexample (1995), a seven-state MDP with a linear value function, makes off-policy semi-gradient TD diverge to infinity from a harmless starting point, and it takes a couple of dozen lines to reproduce. For deep Q-learning, which has all three, divergence takes the form of Q-values that explode, oscillate, or quietly become wildly overestimated. Everything in the DQN section is a practical recipe for keeping the triad under control: break the correlations in the data (replay), keep the target from moving every step (target network), and tame the scale of errors (Huber loss, gradient clipping, reward clipping).

:::intuition Why does bootstrapping with a function approximator go wrong?
Imagine a value network that has slightly overestimated $V(s')$. Because the network generalizes, the overestimate leaks onto the neighbours of $s'$, including $s$. The TD target for $s$ is $r+\gamma V(s')$, which is now too high, so we push $V(s)$ up, which leaks onto $s'$ again. With $\gamma$ near 1 and a poorly chosen feature space this feedback loop can have gain above one, and the error grows geometrically. With a table there is no leakage and the contraction property guarantees the loop dies out. Off-policy data makes it worse because nothing forces us to visit the states where the overestimate is wrong and get a corrective negative sample.
:::

## Deep Q-networks (DQN) {#dqn}

Before 2013 it was widely believed that combining Q-learning with a neural network was too unstable to be useful, despite isolated successes such as Tesauro's TD-Gammon (1992) in backgammon. **DQN** (Mnih et al., 2013 workshop paper; *Nature*, 2015, "Human-level control through deep reinforcement learning") showed that a single convolutional network with a fixed architecture and hyper-parameters could learn to play dozens of Atari 2600 games from raw pixels and the score alone, reaching human-level performance on many of them. Its network takes a stack of the last four preprocessed frames (grayscale, $84\times84$) and outputs one Q-value per discrete action, so a single forward pass scores every action; this is cheaper than feeding $(s,a)$ pairs and makes the $\max_{a'}$ trivial.

{{figure: week15-dqn | DQN architecture and training loop. The agent acts epsilon-greedily with the online network and stores transitions in a replay buffer. Each gradient step samples a random minibatch, computes targets with a periodically-updated frozen copy of the network, and regresses the online network's Q-value for the taken action toward those targets.}}

### Experience replay

A raw stream of experience is a poor training set: consecutive transitions are strongly correlated (almost identical frames), and the data distribution shifts as the policy changes, so gradient descent on the most recent transitions chases its own tail and can forget earlier behaviour. **Experience replay** (Lin, 1992) stores the last $N$ transitions $(s_t,a_t,r_{t+1},s_{t+1},d_{t+1})$ in a buffer ($N=10^6$ in DQN) and trains on minibatches sampled uniformly at random. This (i) decorrelates the minibatch, making it closer to the i.i.d. assumption of SGD, (ii) reuses each transition many times, making learning far more sample-efficient, and (iii) averages over a long history of behaviours, smoothing the data distribution. It also requires off-policy learning: the buffer holds data from old policies, which is one more reason Q-learning (not SARSA) is the right base algorithm.

### The target network

Plain semi-gradient Q-learning has the target and the prediction computed by the same, rapidly changing parameters, which is the feedback loop above. DQN keeps a second copy of the network with parameters $\boldsymbol\theta^-$ used only to compute targets, and refreshes it from the online network every $C$ steps (10,000 in the original Atari setup). The loss for a minibatch $\mathcal B$ of transitions is
$$
\mathcal L(\boldsymbol\theta)=\frac1{|\mathcal B|}\sum_{(s,a,r,s',d)\in\mathcal B}\Big(\underbrace{r+\gamma(1-d)\max_{a'}\hat q(s',a';\boldsymbol\theta^-)}_{y,\ \text{treated as constant}}-\hat q(s,a;\boldsymbol\theta)\Big)^2.
$$
For a period of $C$ steps the regression targets are stationary, turning the moving-target problem into a sequence of ordinary supervised problems. Many modern implementations use a **soft (Polyak) update**, $\boldsymbol\theta^-\leftarrow\tau\boldsymbol\theta+(1-\tau)\boldsymbol\theta^-$ with $\tau\approx0.005$, every step, which achieves the same stabilization more smoothly. The original paper also clipped rewards to $[-1,1]$ and clipped the TD error (equivalent to the Huber loss), which prevents a rare large reward from producing enormous gradients.

### The full algorithm

```text
Initialize online network Q(s,a;θ), target network Q(s,a;θ⁻) ← θ, replay buffer D
for each environment step t:
    with prob ε_t choose a random action, else a = argmax_a Q(s_t, a; θ)    # ε annealed 1.0 -> 0.05 or 0.01
    execute a, observe r, s_{t+1}, done; store (s_t, a, r, s_{t+1}, done) in D
    if t >= learning_starts and t % train_freq == 0:
        sample minibatch of B transitions from D
        y = r + γ (1 - done) max_a' Q(s', a'; θ⁻)
        minimize Huber(y - Q(s, a; θ)) with Adam / RMSProp, clip gradient norm
    every C steps (or softly each step): θ⁻ ← θ
```

Typical Atari hyper-parameters: replay 1M, minibatch 32, $\gamma=0.99$, $\varepsilon$ from 1.0 to 0.1 over the first million frames, learning rate $2.5\times10^{-4}$ (RMSProp in the original), update every 4 agent steps. The lab below uses a smaller version on CartPole with a two-hidden-layer MLP.

:::history Atari and the arcade learning environment
The Arcade Learning Environment (Bellemare et al., 2013, JAIR) exposed Atari 2600 games as an RL benchmark. DQN's headline result in the 2015 *Nature* paper was reaching more than 75% of a professional human tester's score on 29 of 49 games with a single set of hyper-parameters. Later work (Rainbow, Hessel et al., 2018, AAAI) combined six DQN extensions, and the field's benchmark moved to measuring *sample efficiency* (for example Atari 100k) since DQN needs tens of millions of frames, equivalent to many days of human play.
:::

## Improving DQN: Double, Dueling and prioritized replay {#dqn-extensions}

DQN has well-understood weaknesses. Three of the most useful repairs each address one specific failure and are cheap to add.

### Double DQN: fixing overestimation

The target $\max_{a'}\hat q(s',a')$ takes a maximum over noisy estimates. Even if each estimate is unbiased, the expected maximum of noisy values is **larger than** the maximum of the expected values: $\E[\max_a X_a]\ge\max_a\E[X_a]$ (Jensen's inequality for the convex max function). With a network full of approximation error, Q-learning is therefore systematically optimistic, and the optimism is then bootstrapped and propagated. The same network both *selects* the action that looks best and *evaluates* it, so selection noise and evaluation noise are positively correlated: an action is picked precisely because its error happens to be positive.

**Double Q-learning** (van Hasselt, 2010) decouples the two roles. **Double DQN** (van Hasselt, Guez and Silver, 2016) does it with the machinery already at hand: select the action with the *online* network, evaluate it with the *target* network:
$$
y=r+\gamma(1-d)\;\hat q\Big(s',\ \argmax_{a'}\hat q(s',a';\boldsymbol\theta);\ \boldsymbol\theta^-\Big).
$$
The change is one line of code, costs nothing, and gave large improvements on several Atari games where vanilla DQN's value estimates were wildly too high. It is not an unbiased estimator (it can underestimate slightly), but the bias is much smaller and less harmful.

```python:double_dqn_target.py
import torch

def dqn_targets(online, target, r, s_next, done, gamma, double=True):
    """Return TD targets y of shape (B,). All inputs are batched tensors:
    r (B,), s_next (B, obs_dim), done (B,) with values 0.0 or 1.0."""
    with torch.no_grad():                                   # targets are constants in the loss
        q_next_target = target(s_next)                      # (B, n_actions)
        if double:
            a_star = online(s_next).argmax(dim=1, keepdim=True)    # select with ONLINE net
            q_next = q_next_target.gather(1, a_star).squeeze(1)    # evaluate with TARGET net
        else:
            q_next = q_next_target.max(dim=1).values               # vanilla DQN: max of target
        return r + gamma * (1.0 - done) * q_next
```

### Dueling DQN: separate what the state is worth from what the action adds

In many states the choice of action barely matters (the car is not near an obstacle; the pole is nearly upright), yet a standard Q-network must learn $q(s,a)$ separately for every action. The **dueling architecture** (Wang et al., 2016) shares the convolutional or MLP trunk and then splits into two heads: a scalar **value stream** $V(s)$ and an **advantage stream** $A(s,a)$ with one output per action, recombined as
$$
Q(s,a)=V(s)+\Big(A(s,a)-\frac1{|\mathcal A|}\sum_{a'}A(s,a')\Big).
$$
The decomposition alone is not identifiable (add a constant to $V$ and subtract it from every $A$ and $Q$ is unchanged), so the mean advantage is subtracted, forcing the advantages to be centred and giving $V$ a unique meaning. The practical gain is that every update, whichever action was taken, trains the shared value stream, so $V(s)$ is learnt faster and generalizes across actions, which matters most when there are many actions with similar values. The dueling head is a change to the network only and composes with Double DQN and replay.

{{figure: week15-dueling | The dueling network. A shared trunk feeds two streams. The value stream outputs a scalar V(s); the advantage stream outputs one centred advantage per action. They are added (with the mean advantage subtracted) to give Q(s,a).}}

### Prioritized experience replay

Uniform sampling treats every stored transition as equally useful. A transition with a large TD error (a surprising outcome) has a lot to teach; one the network already predicts well has little. **Prioritized experience replay** (Schaul et al., 2016) samples transition $i$ with probability
$$
P(i)=\frac{p_i^{\alpha}}{\sum_k p_k^{\alpha}},\qquad p_i=|\delta_i|+\epsilon,
$$
where $\delta_i$ is the TD error recorded the last time $i$ was used, $\epsilon>0$ ensures every transition has some chance, and $\alpha\in[0,1]$ interpolates between uniform ($\alpha=0$) and greedy prioritization. Sampling non-uniformly changes the expected gradient, i.e. it introduces bias, so the loss for each sampled transition is multiplied by an **importance-sampling weight** $w_i=\big(\tfrac1{N P(i)}\big)^{\beta}$ (normalized by the largest weight), with $\beta$ annealed to 1 over training. Implementations use a sum-tree so that sampling and priority updates cost $O(\log N)$. New transitions are inserted with maximal priority so they are seen at least once. Prioritization usually speeds up learning on Atari substantially, but it adds hyper-parameters ($\alpha,\beta$) and can hurt on noisy-reward tasks, where TD error reflects irreducible noise rather than learnable surprise.

### Other extensions and Rainbow

Other widely used improvements: **multi-step returns** ($n$-step targets, which cut bootstrap bias at the cost of some off-policy bias), **distributional RL** (C51, Bellemare et al., 2017, learns the whole return distribution, not just its mean), **noisy networks** (Fortunato et al., 2018, learned parameter noise for exploration instead of $\varepsilon$-greedy). Hessel et al. (2018) combined these six extensions (Double, Dueling, prioritized, multi-step, distributional, noisy) in **Rainbow** and ablated them, finding that prioritized replay and multi-step returns mattered most and that the combination was better than any single one.

```chart
{
  "type": "bar",
  "caption": "Why overestimation matters: expected value of the max over k noisy estimates of k actions whose TRUE values are all exactly 0, each estimate with Gaussian noise of standard deviation 1. The single-estimator max (what vanilla DQN's target does) is positively biased and the bias grows with the number of actions; selecting with one independent estimate and evaluating with another (the Double Q-learning idea) is unbiased here. Computed analytically/with NumPy.",
  "height": 300,
  "data": { "labels": ["k=2","k=4","k=8","k=16","k=64"],
    "datasets": [
      { "label": "max of one noisy estimate (vanilla)", "data": [0.56,1.03,1.42,1.77,2.34], "backgroundColor": "var(--c-red)" },
      { "label": "Double: select with A, evaluate with B", "data": [0.0,0.0,0.0,0.0,0.0], "backgroundColor": "var(--c-blue)" }
    ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "number of actions"}}, "y": {"title": {"display": true, "text": "expected target value (true value = 0)"}} } }
}
```

## Policy-gradient methods {#policy-gradient}

Value-based methods learn $q$ and derive a policy by maximizing over actions. That has three limits. The $\max_a$ is awkward or expensive when actions are continuous. The greedy policy is deterministic, but some problems (rock-paper-scissors, partially observed environments) need a stochastic optimal policy. And a small change in $q$ can cause an abrupt change in the greedy action, which is part of what makes the triad dangerous. **Policy-gradient methods** parameterize the policy directly, $\pi_{\boldsymbol\theta}(a\mid s)$ (a softmax over action logits for discrete actions, a Gaussian with learned mean and standard deviation for continuous ones) and ascend the gradient of expected return,
$$
J(\boldsymbol\theta)=\E_{\tau\sim\pi_{\boldsymbol\theta}}\big[R(\tau)\big],\qquad R(\tau)=\sum_{t=0}^{T-1}\gamma^t r_{t+1}.
$$
The challenge: $J$ depends on $\boldsymbol\theta$ through the *distribution* of trajectories, including the unknown environment dynamics, and sampling an action is not a differentiable operation. The policy gradient theorem shows how to get a gradient anyway, with no derivative of the dynamics needed.

### The policy gradient theorem

:::math Derivation: the log-derivative trick and the policy gradient
Write the probability of a trajectory as $p_{\boldsymbol\theta}(\tau)=p(s_0)\prod_{t=0}^{T-1}\pi_{\boldsymbol\theta}(a_t\mid s_t)\,p(s_{t+1}\mid s_t,a_t)$. Then
$$
\nabla_{\boldsymbol\theta}J=\nabla_{\boldsymbol\theta}\int p_{\boldsymbol\theta}(\tau)R(\tau)\,d\tau=\int p_{\boldsymbol\theta}(\tau)\,\nabla_{\boldsymbol\theta}\log p_{\boldsymbol\theta}(\tau)\,R(\tau)\,d\tau,
$$
using the identity $\nabla p=p\,\nabla\log p$ (the **log-derivative trick**, also called the score-function or likelihood-ratio estimator). Now take the log of the trajectory probability:
$$
\log p_{\boldsymbol\theta}(\tau)=\log p(s_0)+\sum_t\log\pi_{\boldsymbol\theta}(a_t\mid s_t)+\sum_t\log p(s_{t+1}\mid s_t,a_t).
$$
Only the middle term depends on $\boldsymbol\theta$; the initial-state distribution and the dynamics drop out of the gradient. Hence
$$
\nabla_{\boldsymbol\theta}J=\E_{\tau\sim\pi_{\boldsymbol\theta}}\Big[\Big(\sum_{t=0}^{T-1}\nabla_{\boldsymbol\theta}\log\pi_{\boldsymbol\theta}(a_t\mid s_t)\Big)R(\tau)\Big].
$$
**Causality.** The action at time $t$ cannot influence rewards received before $t$. For $t'<t$, $\E[\nabla\log\pi_{\boldsymbol\theta}(a_t\mid s_t)\,r_{t'+1}]=0$, because conditioned on the history up to $s_t$ the reward $r_{t'+1}$ is fixed and $\E_{a_t\sim\pi}[\nabla\log\pi(a_t\mid s_t)]=\sum_a\nabla\pi(a\mid s_t)=\nabla 1=0$. Dropping these terms gives the lower-variance form, the **policy gradient theorem**:
$$
\boxed{\;\nabla_{\boldsymbol\theta}J=\E_{\tau\sim\pi_{\boldsymbol\theta}}\Big[\sum_{t=0}^{T-1}\nabla_{\boldsymbol\theta}\log\pi_{\boldsymbol\theta}(a_t\mid s_t)\;G_t\Big],\qquad G_t=\sum_{k\ge t}\gamma^{k-t}r_{k+1}\;}
$$
(up to a $\gamma^t$ factor on each term which practical implementations drop; the omitted factor changes the gradient to one that is not the true gradient of the discounted objective but works well in practice). The general form is $\E[\nabla\log\pi(a_t|s_t)\,\Psi_t]$ for any $\Psi_t$ whose expectation given $(s_t,a_t)$ is $q_\pi(s_t,a_t)$, up to a baseline: this is where the actor-critic family comes from.
:::

Read the formula as a recipe: *increase the log-probability of each action taken, in proportion to the return that followed it.* Good outcomes make their actions more likely; bad outcomes make them less likely. In effect we are doing supervised learning on the policy's own samples, with the return as a per-sample weight.

### REINFORCE

**REINFORCE** (Williams, 1992) is the Monte Carlo estimator of the theorem: sample trajectories by running $\pi_{\boldsymbol\theta}$, compute returns-to-go $G_t$, and take a step along $\sum_t\nabla\log\pi_{\boldsymbol\theta}(a_t\mid s_t)G_t$. In PyTorch you implement it by minimizing the **surrogate loss** $-\sum_t\log\pi_{\boldsymbol\theta}(a_t\mid s_t)\,G_t$, whose gradient (with $G_t$ treated as constant) is exactly the negative of the policy gradient estimate. REINFORCE is unbiased but its variance is large: the weight $G_t$ is a sum of many random rewards, and a trajectory-level return provides a noisy signal about any individual action. With returns that are always positive (as in CartPole, where every step gives +1), every sampled action is pushed up, only by different amounts, and learning is slow and noisy.

### Baselines

Subtracting a **baseline** $b(s_t)$ that does not depend on the action leaves the gradient unbiased but can reduce its variance sharply:
$$
\nabla_{\boldsymbol\theta}J=\E\Big[\sum_t\nabla_{\boldsymbol\theta}\log\pi_{\boldsymbol\theta}(a_t\mid s_t)\,\big(G_t-b(s_t)\big)\Big].
$$
Unbiasedness follows from the same identity as before: $\E_{a\sim\pi}[\nabla\log\pi(a\mid s)\,b(s)]=b(s)\,\nabla\sum_a\pi(a\mid s)=0$. The best simple choice is $b(s)=v_\pi(s)$, which makes the weight $G_t-v_\pi(s_t)$ an estimate of the **advantage**: actions are reinforced only when they did *better than average from that state*, and discouraged when worse. The sign of the update now carries information. In practice $v_\pi$ is learned by a second network (the **critic**) trained by regression on the returns, and a cheap extra step is to normalize advantages within a batch to zero mean and unit variance.

```python:reinforce_numpy.py
import numpy as np

# REINFORCE with a learned baseline on a 1-D "walk to the goal" task, in pure NumPy.
# States 0..N; start at N//2; actions: left/right; reaching N gives +1, reaching 0 gives 0;
# each step costs 0.01. Policy: state-independent softmax over {left,right} parameterized by logits.
rng = np.random.default_rng(0)
N, gamma = 8, 0.99
theta = np.zeros((N + 1, 2))       # tabular policy logits per state
Vb = np.zeros(N + 1)               # learned baseline V(s)

def pi(s):
    z = np.exp(theta[s] - theta[s].max())
    return z / z.sum()

def episode():
    s, traj = N // 2, []
    while 0 < s < N and len(traj) < 200:
        a = rng.choice(2, p=pi(s))
        s2 = s + (1 if a == 1 else -1)
        r = 1.0 if s2 == N else -0.01
        traj.append((s, a, r)); s = s2
    return traj

lr_pi, lr_v, history = 0.1, 0.1, []
for it in range(2000):
    traj = episode()
    G, rets = 0.0, []
    for (_, _, r) in reversed(traj):               # returns-to-go G_t
        G = r + gamma * G
        rets.append(G)
    rets.reverse()
    for (s, a, _), Gt in zip(traj, rets):
        adv = Gt - Vb[s]                           # advantage estimate with baseline
        Vb[s] += lr_v * adv                        # critic: regress V(s) toward G_t
        grad_logp = -pi(s); grad_logp[a] += 1.0    # d log pi(a|s) / d logits for a softmax
        theta[s] += lr_pi * adv * grad_logp        # ascend along grad log pi * advantage
    history.append(sum(r for _, _, r in traj))
print("mean return, first 100 episodes:", np.mean(history[:100]).round(3))
print("mean return, last 100 episodes: ", np.mean(history[-100:]).round(3))
print("P(right) per state:", np.round([pi(s)[1] for s in range(1, N)], 2))
```

Running it prints roughly the following (exact numbers depend on the seed): the average return rises from about 0.87 to about 0.97 and the learnt policy moves right with high probability in most states.

## Actor-critic methods {#actor-critic}

REINFORCE with a baseline still uses a full Monte Carlo return as the weight, so its variance grows with the episode length and it cannot update until an episode ends. **Actor-critic** methods apply the TD idea from earlier to the policy-gradient weight. They keep two function approximators: an **actor** $\pi_{\boldsymbol\theta}(a\mid s)$ and a **critic** $V_{\boldsymbol\phi}(s)$. Replace the Monte Carlo return by a one-step bootstrapped estimate, $G_t-V(s_t)\;\to\;r_{t+1}+\gamma V_{\boldsymbol\phi}(s_{t+1})-V_{\boldsymbol\phi}(s_t)=\delta_t$, the TD error. Then the actor's update is
$$
\Delta\boldsymbol\theta\propto\delta_t\,\nabla_{\boldsymbol\theta}\log\pi_{\boldsymbol\theta}(a_t\mid s_t),
$$
and the critic is trained by regression to the bootstrapped target, $\mathcal L_V=\big(r_{t+1}+\gamma(1-d)V_{\boldsymbol\phi}(s_{t+1})-V_{\boldsymbol\phi}(s_t)\big)^2$ (target held constant). The critic's TD error is a sample of the advantage: $\E[\delta_t\mid s_t,a_t]=q_\pi(s_t,a_t)-v_\pi(s_t)$ when $V=v_\pi$. We have traded the unbiased, high-variance MC weight for a low-variance, *biased* (because $V_{\boldsymbol\phi}$ is imperfect) TD weight. The two networks usually share a trunk and have separate output heads, with a combined loss
$$
\mathcal L=\underbrace{-\log\pi_{\boldsymbol\theta}(a_t\mid s_t)\,\hat A_t}_{\text{policy}}+c_v\underbrace{\big(V_{\boldsymbol\phi}(s_t)-\hat V^{\text{target}}_t\big)^2}_{\text{value}}-c_e\underbrace{\mathcal H\big[\pi_{\boldsymbol\theta}(\cdot\mid s_t)\big]}_{\text{entropy bonus}},
$$
with typical coefficients $c_v\approx0.5$ and $c_e\approx0.01$. The entropy bonus discourages the policy from collapsing to a deterministic one too early and is the simplest on-policy exploration device.

{{figure: week15-actor-critic | Actor-critic data flow. The actor chooses actions; the critic scores states. The TD error (an advantage estimate) is the learning signal for the actor, and its square is the learning signal for the critic.}}

### A2C and A3C

Training an actor-critic from a single stream of consecutive transitions gives strongly correlated updates; with no replay buffer (the method is on-policy, so old data is invalid) we need a different decorrelation device. **A3C** (Asynchronous Advantage Actor-Critic, Mnih et al., 2016) runs many CPU workers, each with its own copy of the environment and a local copy of the network. Each worker collects a short rollout (for example 5 steps), computes $n$-step advantages, and pushes its gradients asynchronously to a shared parameter server. Because workers explore different parts of the state space at different times, the gradient stream is decorrelated, and A3C matched or beat DQN on Atari in less wall-clock time on CPUs alone. **A2C** is the synchronous variant that OpenAI's baselines popularized: all workers step together, a single batch of transitions from $N$ parallel environments is assembled, and one gradient step is taken. It is simpler, deterministic, and makes better use of a GPU, and it was found to work equally well in practice, so A2C is the standard today. The $N$ parallel environments are what provides the batch diversity that replay provides in DQN.

### Generalized advantage estimation (GAE)

How many steps should the advantage look ahead? One step gives low variance and high bias (the critic is trusted completely); a full Monte Carlo return gives no critic bias and high variance. **GAE** (Schulman et al., 2016) provides a dial between the two with a single parameter $\lambda\in[0,1]$, by taking an exponentially weighted average of all $k$-step advantage estimates.

:::math Derivation: GAE
Let $\delta_t=r_{t+1}+\gamma V(s_{t+1})-V(s_t)$ be the TD error. The $k$-step advantage estimate is
$$
\hat A_t^{(k)}=\sum_{l=0}^{k-1}\gamma^l r_{t+l+1}+\gamma^kV(s_{t+k})-V(s_t).
$$
Telescoping shows that it equals a discounted sum of TD errors. Writing $\sum_{l=0}^{k-1}\gamma^l\delta_{t+l}=\sum_{l=0}^{k-1}\gamma^l\big(r_{t+l+1}+\gamma V(s_{t+l+1})-V(s_{t+l})\big)$, the value terms cancel pairwise except the endpoints, leaving
$$
\hat A_t^{(k)}=\sum_{l=0}^{k-1}\gamma^l\delta_{t+l}.
$$
So $\hat A_t^{(1)}=\delta_t$ (low variance, biased) and $\hat A_t^{(\infty)}=\sum_{l\ge0}\gamma^l\delta_{t+l}=G_t-V(s_t)$ (the Monte Carlo advantage: unbiased, high variance). GAE takes the exponentially weighted average with weights $(1-\lambda)\lambda^{k-1}$:
$$
\begin{aligned}
\hat A_t^{\text{GAE}(\gamma,\lambda)}&=(1-\lambda)\big(\hat A_t^{(1)}+\lambda\hat A_t^{(2)}+\lambda^2\hat A_t^{(3)}+\cdots\big)\\
&=(1-\lambda)\big(\delta_t+\lambda(\delta_t+\gamma\delta_{t+1})+\lambda^2(\delta_t+\gamma\delta_{t+1}+\gamma^2\delta_{t+2})+\cdots\big).
\end{aligned}
$$
Collect the coefficient of each $\delta_{t+l}$: it is $(1-\lambda)\gamma^l(\lambda^l+\lambda^{l+1}+\cdots)=(1-\lambda)\gamma^l\lambda^l/(1-\lambda)=(\gamma\lambda)^l$. Therefore
$$
\boxed{\hat A_t^{\text{GAE}(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^l\,\delta_{t+l}}
$$
This satisfies the cheap backward recursion $\hat A_t=\delta_t+\gamma\lambda\,\hat A_{t+1}$ with $\hat A_T=0$ at the end of a rollout, which is how it is computed in code. The special cases are $\lambda=0$: $\hat A_t=\delta_t$, and $\lambda=1$: $\hat A_t=G_t-V(s_t)$.
:::

If the critic were exact, every $\delta_t$ would have zero conditional mean and GAE would be unbiased for any $\lambda$; the bias comes entirely from critic error, and $\lambda<1$ damps how much of that error accumulates. The variance comes from the sum of many noisy rewards, which $\lambda<1$ truncates in effect after about $1/(1-\gamma\lambda)$ steps. Values around $\gamma=0.99$, $\lambda=0.95$ are standard defaults, but note that $\lambda$ is a bias-variance rule of thumb, not something with a theoretically optimal value. The following simulation makes the trade-off concrete.

```chart
{
  "type": "bar",
  "caption": "Bias-variance trade-off in GAE, from a NumPy simulation (3000 trials, 50-step episodes, gamma = 0.99, rewards 1 + N(0,1) noise; the critic is deliberately imperfect: it is scaled to 80% of the true value plus N(0, 0.3) noise). The true advantage at t=0 is 0, so the mean of the estimator is its bias. Small lambda: low variance but some bias; lambda = 1 (Monte Carlo) uses the imperfect critic only as a baseline at t=0, so its estimate keeps the full critic offset and the full accumulated reward noise: largest mean and largest spread.",
  "height": 320,
  "data": { "labels": ["0","0.2","0.5","0.8","0.9","0.95","0.98","1.0"],
    "datasets": [
      { "label": "mean (bias) of advantage estimate", "data": [0.18,0.29,0.38,1.06,1.87,3.16,5.26,7.74], "backgroundColor": "var(--c-red)" },
      { "label": "standard deviation", "data": [1.08,1.09,1.19,1.68,2.23,2.95,4.07,5.64], "backgroundColor": "var(--c-blue)" }
    ] },
  "options": { "scales": { "x": {"title": {"display": true, "text": "GAE lambda"}}, "y": {"title": {"display": true, "text": "value of advantage estimate at t = 0"}} } }
}
```

The chart looks at a deliberately miscalibrated critic, so the "bias" here is the effect of the baseline's error being included in the estimate. The message, though, is the usual one: very large $\lambda$ makes the estimate sensitive to the accumulated noise of the whole episode, while small $\lambda$ stays close to one-step TD errors.

```python:gae.py
import torch

def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    """rewards, values, dones: tensors of shape (T, N) from N parallel envs over T steps.
    last_value: (N,) critic value of the state after the final step (bootstrap).
    Returns advantages (T, N) and value targets (T, N)."""
    T, N = rewards.shape
    adv = torch.zeros_like(rewards)
    next_adv = torch.zeros(N)
    next_value = last_value
    for t in reversed(range(T)):
        not_done = 1.0 - dones[t]                         # cut bootstrapping across episode ends
        delta = rewards[t] + gamma * next_value * not_done - values[t]
        next_adv = delta + gamma * lam * not_done * next_adv   # A_t = delta_t + gamma*lam*A_{t+1}
        adv[t] = next_adv
        next_value = values[t]
    returns = adv + values                                 # lambda-return targets for the critic
    return adv, returns

T, N = 5, 3
r = torch.ones(T, N); v = torch.zeros(T, N); d = torch.zeros(T, N)
adv, ret = compute_gae(r, v, d, last_value=torch.zeros(N))
print(adv[:, 0])   # with V = 0, r = 1: A_t = sum_l (gamma*lam)^l over the remaining steps: 4.44, 3.66, 2.83, 1.94, 1.0
```

{{demo: week15-gae | GAE: how lambda weights the TD errors | 440}}

## TRPO and PPO: safe policy updates {#ppo}

Policy-gradient methods have a problem that supervised learning does not. A step in parameter space of fixed size can change the policy a lot or a little depending on where you are, and a **bad step ruins the data**: the next batch is collected by the new, worse policy, so recovery is hard and a performance collapse can be permanent. We want to take the largest step that is guaranteed to be *safe*, and the right notion of "size" is measured in policy space (how much the action distribution changes), not parameter space.

### The surrogate objective and the trust region

The starting point is an exact identity (Kakade and Langford, 2002) relating the performance of any two policies:
$$
J(\pi')-J(\pi)=\E_{\tau\sim\pi'}\Big[\sum_{t}\gamma^tA_\pi(s_t,a_t)\Big].
$$
The new policy is better exactly when, on states visited by the *new* policy, its actions have positive advantage under the *old* policy. The trouble is the expectation over $\pi'$'s state distribution, which we cannot sample before we have $\pi'$. If we approximate the state distribution by the old one (valid when the policies are close) and fix the action distribution with importance sampling, we get the **surrogate objective**
$$
L_{\pi_{\text{old}}}(\boldsymbol\theta)=\E_{s\sim\pi_{\text{old}},\,a\sim\pi_{\text{old}}}\Big[\frac{\pi_{\boldsymbol\theta}(a\mid s)}{\pi_{\text{old}}(a\mid s)}\,A_{\pi_{\text{old}}}(s,a)\Big]=\E\big[r_t(\boldsymbol\theta)\,\hat A_t\big],\qquad r_t(\boldsymbol\theta)=\frac{\pi_{\boldsymbol\theta}(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)}.
$$
At $\boldsymbol\theta=\boldsymbol\theta_{\text{old}}$ the ratio is 1 and the gradient of $L$ equals the ordinary policy gradient, so $L$ is a first-order-accurate local model of the true objective. It stops being accurate once the policies drift apart, which is why we need to constrain the step.

**TRPO** (Trust Region Policy Optimization, Schulman, Levine, Abbeel, Jordan and Moritz, 2015) maximizes the surrogate subject to a bound on the average KL divergence between the old and new policies:
$$
\max_{\boldsymbol\theta}\;L_{\pi_{\text{old}}}(\boldsymbol\theta)\quad\text{s.t.}\quad\bar D_{\KL}\big(\pi_{\text{old}}\,\|\,\pi_{\boldsymbol\theta}\big)\le\delta,
$$
with $\delta\approx0.01$. The paper derives a lower bound showing that, with a KL penalty, each such step guarantees monotonic improvement; the practical algorithm uses the constraint form with a larger step. Expanding $L$ to first order and the KL to second order around $\boldsymbol\theta_{\text{old}}$ gives a quadratic-constrained linear problem whose solution is the **natural gradient** step $\boldsymbol\theta\leftarrow\boldsymbol\theta+\sqrt{2\delta/(\boldsymbol g^\top\mathbf F^{-1}\boldsymbol g)}\;\mathbf F^{-1}\boldsymbol g$, where $\boldsymbol g=\nabla L$ and $\mathbf F$ is the Fisher information matrix (the Hessian of the KL). TRPO never forms $\mathbf F$: it solves $\mathbf F\boldsymbol x=\boldsymbol g$ by conjugate gradient with Fisher-vector products, then does a backtracking line search to enforce the KL constraint and an actual improvement of the surrogate. It is principled and reliable but fiddly to implement, awkward with shared policy-value networks and with dropout or other stochastic layers, and expensive per update.

### PPO: a first-order approximation of the trust region

**Proximal Policy Optimization** (Schulman, Wolski, Dhariwal, Radford and Klimov, 2017) obtains much of TRPO's stability using nothing but a clipped objective and plain minibatch SGD with Adam, so that several epochs of updates can be run on each batch of data. The **PPO-Clip** objective is
$$
L^{\text{CLIP}}(\boldsymbol\theta)=\E_t\Big[\min\Big(r_t(\boldsymbol\theta)\hat A_t,\ \operatorname{clip}\big(r_t(\boldsymbol\theta),\,1-\epsilon,\,1+\epsilon\big)\hat A_t\Big)\Big],\qquad\epsilon\approx0.1\text{ to }0.3.
$$
To see how it works, consider the two signs of the advantage.

- **$\hat A_t>0$** (the action was better than expected). We want $r_t$ to increase. The objective is $\min(r_t,\,1+\epsilon)\hat A_t$: it grows with $r_t$ until $r_t=1+\epsilon$ and is **flat** afterwards. Once the policy has made that action $1+\epsilon$ times as likely, the gradient vanishes and there is no incentive to push further.
- **$\hat A_t<0$** (the action was worse than expected). We want $r_t$ to decrease. The objective is $\max(r_t,\,1-\epsilon)\,\hat A_t$ in effect ($\min$ with a negative factor flips): it improves as $r_t$ falls until $r_t=1-\epsilon$ and is flat below that.

The $\min$ has a subtle and important role: it makes the objective a **pessimistic lower bound** of the unclipped surrogate. If the ratio has moved in the *wrong* direction (e.g. $r_t<1-\epsilon$ for a positive-advantage action: the update made a good action less likely), the unclipped term is the smaller one, so it is used and the gradient is *not* zeroed; PPO does not hide mistakes by clipping them away. The clip therefore removes only the incentive to move beyond the trust region, in the favourable direction.

{{figure: week15-ppo-clip | The PPO-Clip objective as a function of the probability ratio r. For a positive advantage (left) the objective is flat once r exceeds 1+epsilon; for a negative advantage (right) it is flat once r falls below 1-epsilon. Dashed lines show the unclipped surrogate r times A. The red segments show where the gradient is zero.}}

{{demo: week15-ppo-clip | PPO clipping explorer: objective, gradient and trust region | 520}}

:::definition Full PPO loss
In practice PPO minimizes the negative of
$$
L^{\text{PPO}}=\E_t\Big[L^{\text{CLIP}}_t-c_v\big(V_{\boldsymbol\phi}(s_t)-\hat V_t^{\text{targ}}\big)^2+c_e\,\mathcal H\big[\pi_{\boldsymbol\theta}(\cdot\mid s_t)\big]\Big],
$$
with $\hat A_t$ from GAE and $\hat V^{\text{targ}}_t=\hat A_t+V_{\text{old}}(s_t)$. One iteration: collect $T$ steps from each of $N$ parallel environments (for example $N\times T=2048$ samples), compute advantages with GAE, then run $K$ epochs (commonly 3 to 10) of minibatch SGD on that fixed batch, and discard it. Common additions are advantage normalization per minibatch, gradient-norm clipping (0.5), learning-rate annealing, and (optionally) early stopping an iteration when the sampled KL to the old policy exceeds a threshold.
:::

```python:ppo_loss.py
import torch

def ppo_loss(new_logp, old_logp, adv, values, returns, entropy,
             clip_eps=0.2, c_v=0.5, c_e=0.01):
    """All tensors have shape (B,). new_logp: log pi_theta(a|s) under the current policy;
    old_logp: log-probs recorded when data was collected (no grad); adv: advantages;
    values: current critic outputs; returns: value targets; entropy: per-sample policy entropy."""
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)           # normalize advantages per minibatch
    ratio = torch.exp(new_logp - old_logp)                  # r_t(theta) computed in log space
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    policy_loss = -torch.min(unclipped, clipped).mean()     # maximize the pessimistic bound
    value_loss = 0.5 * (values - returns).pow(2).mean()
    approx_kl = (old_logp - new_logp).mean()                # cheap diagnostic of policy drift
    clip_frac = ((ratio - 1).abs() > clip_eps).float().mean()   # fraction of clipped samples
    loss = policy_loss + c_v * value_loss - c_e * entropy.mean()
    return loss, {"kl": approx_kl.item(), "clipfrac": clip_frac.item()}

# smoke test with random data
B = 64
nl, ol = torch.randn(B) * 0.05, torch.zeros(B)
loss, info = ppo_loss(nl, ol, torch.randn(B), torch.randn(B), torch.randn(B), torch.full((B,), 0.69))
print(loss.item(), info)
```

PPO is the workhorse of deep RL because it is simple, robust to hyper-parameters (compared with other methods), and works for discrete and continuous actions, for games, robotics and language models. A large study of implementation details (Engstrom et al., 2020, "Implementation Matters in Deep Policy Gradients") found that much of PPO's advantage over TRPO is attributable to code-level optimizations (such as learning-rate annealing, value clipping, reward normalization and orthogonal initialization) rather than the clipping itself, and the clip does not strictly enforce a trust region (ratios can leave $[1-\epsilon,1+\epsilon]$ after several epochs). A good rule: monitor `approx_kl` and `clipfrac`, not just the return. Healthy values are a KL of order $10^{-2}$ or smaller and a clip fraction of 0.1 to 0.3.

## Continuous control: DDPG, TD3 and SAC {#continuous}

Robot joints, steering angles and trading sizes are continuous. DQN's $\max_a\hat q(s,a)$ would require solving an optimization problem at every step. Policy-gradient methods handle continuous actions naturally with a Gaussian policy, but PPO is on-policy and therefore sample-hungry, a real cost when each sample is a physical robot second. The methods in this section are **off-policy actor-critic** algorithms: they keep a replay buffer like DQN, learn a Q-function $q_{\boldsymbol\phi}(s,a)$ that takes the action as an *input*, and train an actor to output actions that maximize it.

### DDPG

**Deep Deterministic Policy Gradient** (Lillicrap et al., 2016) builds on the deterministic policy gradient theorem (Silver et al., 2014). The actor is a deterministic function $a=\mu_{\boldsymbol\theta}(s)$ and the critic $q_{\boldsymbol\phi}(s,a)$ is trained with a DQN-style Bellman loss, with target networks for both:
$$
y=r+\gamma(1-d)\,q_{\boldsymbol\phi^-}\big(s',\mu_{\boldsymbol\theta^-}(s')\big),\qquad\mathcal L(\boldsymbol\phi)=\E\big[(q_{\boldsymbol\phi}(s,a)-y)^2\big].
$$
The $\max_{a'}$ of Q-learning is replaced by evaluating the target actor's action. The actor ascends the critic: by the chain rule,
$$
\nabla_{\boldsymbol\theta}J\approx\E_{s\sim\mathcal D}\Big[\nabla_a q_{\boldsymbol\phi}(s,a)\big|_{a=\mu_{\boldsymbol\theta}(s)}\;\nabla_{\boldsymbol\theta}\mu_{\boldsymbol\theta}(s)\Big].
$$
In code you simply backpropagate `-q(s, mu(s)).mean()` into the actor's parameters, with the critic's parameters excluded from that optimizer step. Because the policy is deterministic, exploration is added externally (Gaussian or Ornstein-Uhlenbeck noise on the action), and targets are updated by Polyak averaging. DDPG is sample-efficient but notoriously brittle: it is sensitive to hyper-parameters, and the critic's overestimation errors are exploited by the actor, which learns to climb narrow peaks of the *approximate* Q-function that do not exist in reality.

### TD3

**Twin Delayed DDPG** (Fujimoto, van Hoof and Meger, 2018) adds three fixes that target that exact failure.

1. **Clipped double-Q learning.** Learn two critics $q_{\boldsymbol\phi_1},q_{\boldsymbol\phi_2}$ and use the *smaller* of the two target values, $y=r+\gamma(1-d)\min_{i=1,2}q_{\boldsymbol\phi_i^-}(s',a')$. Taking the minimum counters the upward bias of a single estimate, as in the Double DQN analysis.
2. **Delayed policy updates.** Update the actor (and the target networks) less often than the critics (e.g. one actor step per two critic steps), so the actor follows a critic that has had time to settle.
3. **Target policy smoothing.** Add clipped noise to the target action, $a'=\operatorname{clip}\big(\mu_{\boldsymbol\theta^-}(s')+\operatorname{clip}(\varepsilon,-c,c),\,a_{\text{low}},a_{\text{high}}\big)$ with $\varepsilon\sim\mathcal N(0,\sigma^2)$, which regularizes the critic by forcing similar actions to have similar values and prevents the actor from exploiting sharp peaks.

TD3 is a strong, simple baseline for state-based continuous control.

### SAC and the maximum-entropy objective

**Soft Actor-Critic** (Haarnoja, Zhou, Abbeel and Levine, 2018) changes the objective itself. In **maximum-entropy RL** the agent maximizes reward *and* the entropy of its policy:
$$
J(\pi)=\E_{\tau\sim\pi}\Big[\sum_{t}\gamma^t\Big(r(s_t,a_t)+\alpha\,\mathcal H\big[\pi(\cdot\mid s_t)\big]\Big)\Big],\qquad\mathcal H[\pi(\cdot\mid s)]=-\E_{a\sim\pi}\log\pi(a\mid s).
$$
The temperature $\alpha>0$ sets the exchange rate between reward and randomness. Why add entropy? It builds exploration into the objective (the agent prefers to keep options open), it prevents premature collapse to a deterministic policy, it makes the learnt behaviour more robust (the agent learns many ways to solve a task, so perturbations hurt less), and it tends to give smoother, more stable optimization. In the soft setting the Bellman backup includes an entropy term in the target:
$$
y=r+\gamma(1-d)\Big(\min_{i=1,2}q_{\boldsymbol\phi_i^-}(s',a')-\alpha\log\pi_{\boldsymbol\theta}(a'\mid s')\Big),\qquad a'\sim\pi_{\boldsymbol\theta}(\cdot\mid s').
$$
The actor is a *stochastic* Gaussian policy (squashed through $\tanh$ to respect action bounds) trained to minimize
$$
\mathcal L_\pi(\boldsymbol\theta)=\E_{s\sim\mathcal D,\ a\sim\pi_{\boldsymbol\theta}}\Big[\alpha\log\pi_{\boldsymbol\theta}(a\mid s)-\min_iq_{\boldsymbol\phi_i}(s,a)\Big],
$$
that is, to produce actions that are valuable and high-entropy. This is differentiable through the sampled action by the **reparameterization trick** ([Week 12](week-12.html)): $a=\tanh(\mu_{\boldsymbol\theta}(s)+\sigma_{\boldsymbol\theta}(s)\odot\xi)$, $\xi\sim\mathcal N(0,\mathbf I)$. In the follow-up paper (Haarnoja et al., 2018b) the temperature is learnt automatically by treating the entropy as a constraint: $\alpha$ is adjusted by gradient descent on $\alpha(-\log\pi(a|s)-\bar{\mathcal H})$ to hold the policy entropy near a target $\bar{\mathcal H}$, often $-\dim(\mathcal A)$. Together this makes SAC (off-policy, stochastic, twin critics, auto-tuned temperature) the standard sample-efficient method for continuous control, and usually the first thing to try on a new robotics-style task.

| Algorithm | Policy | On/off-policy | Action space | Key stabilizer | Typical use |
| --- | --- | --- | --- | --- | --- |
| DQN | greedy on Q | off | discrete | replay + target net | Atari, small discrete tasks |
| A2C | stochastic | on | both | parallel envs, advantage | baseline, simple |
| PPO | stochastic | on | both | clipped ratio, GAE | default robust choice; LLM RLHF |
| DDPG | deterministic | off | continuous | target nets, noise | historical; brittle |
| TD3 | deterministic | off | continuous | twin critics, delay, smoothing | state-based control |
| SAC | stochastic (tanh-Gaussian) | off | continuous | entropy bonus, twin critics | robotics, sample-efficient control |

{{figure: week15-taxonomy | A map of the algorithms in this chapter. Value-based, policy-based and actor-critic methods overlap, and each can be on-policy or off-policy; model-based methods add a learnt model of the environment.}}

```python:sac_losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, i, o, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))
    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    """Tanh-squashed Gaussian policy. Returns an action and its log-probability."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.body = MLP(obs_dim, 2 * act_dim)
    def forward(self, s):
        mu, log_std = self.body(s).chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        dist = torch.distributions.Normal(mu, log_std.exp())
        u = dist.rsample()                                   # reparameterized sample, keeps gradient
        a = torch.tanh(u)
        # change-of-variables correction for the tanh squashing
        logp = dist.log_prob(u).sum(-1) - (2 * (torch.log(torch.tensor(2.0)) - u - F.softplus(-2 * u))).sum(-1)
        return a, logp

def sac_losses(actor, q1, q2, q1_targ, q2_targ, batch, alpha=0.2, gamma=0.99):
    s, a, r, s2, d = batch                                    # shapes (B,obs) (B,act) (B,) (B,obs) (B,)
    with torch.no_grad():
        a2, logp2 = actor(s2)
        sa2 = torch.cat([s2, a2], -1)
        q_targ = torch.min(q1_targ(sa2), q2_targ(sa2)).squeeze(-1) - alpha * logp2   # soft target
        y = r + gamma * (1 - d) * q_targ
    sa = torch.cat([s, a], -1)
    critic_loss = F.mse_loss(q1(sa).squeeze(-1), y) + F.mse_loss(q2(sa).squeeze(-1), y)
    a_new, logp = actor(s)                                    # fresh actions for the actor loss
    sa_new = torch.cat([s, a_new], -1)
    q_pi = torch.min(q1(sa_new), q2(sa_new)).squeeze(-1)
    actor_loss = (alpha * logp - q_pi).mean()                 # maximize Q + entropy
    return critic_loss, actor_loss

B, od, ad = 32, 8, 2
actor, q1, q2, q1t, q2t = Actor(od, ad), MLP(od + ad, 1), MLP(od + ad, 1), MLP(od + ad, 1), MLP(od + ad, 1)
batch = (torch.randn(B, od), torch.rand(B, ad) * 2 - 1, torch.randn(B), torch.randn(B, od), torch.zeros(B))
cl, al = sac_losses(actor, q1, q2, q1t, q2t, batch)
print(cl.item(), al.item())
```

In the full algorithm, the critic and actor losses use separate optimizers (the actor loss must not update the critics' parameters, which `optimizer.zero_grad()` and stepping only the actor's optimizer ensures), and the target critics follow the online critics by Polyak averaging.

## Exploration strategies {#exploration}

Exploration is the part of RL with no analogue in supervised learning, and the part where deep RL is still weakest. Strategies form a ladder of increasing sophistication.

- **Undirected noise.** $\varepsilon$-greedy for discrete actions, Gaussian or Ornstein-Uhlenbeck noise for continuous ones, and the entropy bonuses of A2C, PPO and SAC. These work when rewards are dense and the task requires only local variations. They fail when reaching any reward requires a long, specific sequence of actions: a random policy has an exponentially small chance of producing it.
- **Optimism in the face of uncertainty.** Prefer actions or states whose value is uncertain. UCB for bandits is the template; in deep RL, ensembles of Q-networks (bootstrapped DQN, Osband et al., 2016) provide uncertainty by disagreement and enable *deep exploration*, where one sampled network's policy is followed for a whole episode (like Thompson sampling).
- **Intrinsic motivation and curiosity.** Add a bonus reward for novelty: **count-based** bonuses $\beta/\sqrt{N(s)}$ (extended to high dimensions with pseudo-counts from density models, Bellemare et al., 2016), or **prediction-error** bonuses. In the Intrinsic Curiosity Module (Pathak et al., 2017) the bonus is the error of a learnt forward model in a learnt feature space; in Random Network Distillation (Burda et al., 2019) it is the error of a predictor network trying to match a fixed random network's output on the state, which is high for states rarely visited. RND was the first method to exceed average human performance on the notoriously hard-exploration Atari game Montezuma's Revenge without demonstrations. A caveat is the **noisy TV problem**: a purely random source of observations never becomes predictable, and a prediction-error agent can be trapped watching it.
- **Goal-conditioned and structured exploration.** Learn skills or subgoals, or maintain an archive of promising states and return to them (Go-Explore, Ecoffet et al., 2019, 2021).
- **Using demonstrations or a pretrained model.** Imitation learning gives the agent a good starting distribution. This is the most reliable approach in practice, and in LLM post-training it is the first stage: supervised fine-tuning on human-written examples before any reinforcement learning (see below).

:::tip Practical exploration advice
Start with the simplest thing the algorithm provides (entropy bonus for PPO, a decaying $\varepsilon$ for DQN, auto-tuned temperature for SAC), shape the reward only if you must (and then check the agent is not exploiting the shaping), and look at *behaviour*, not just the return curve. A flat curve at zero return over many episodes is an exploration problem; a curve that rises and collapses is usually an optimization or stability problem.
:::

## Model-based RL and world models {#model-based}

Everything so far is **model-free**: the agent never builds an explicit predictor of what happens next, and it pays for that with sample complexity (DQN needs on the order of $10^7$ frames per game). **Model-based RL** learns a model $\hat p(s'\mid s,a)$ and $\hat r(s,a)$ from experience and then uses it to improve the policy with little or no further environment interaction. Having a model offers two routes.

**Planning at decision time.** Use the model to search for good actions from the current state. *Model predictive control (MPC)* samples candidate action sequences, rolls them out in the model for a short horizon, and executes the first action of the best sequence, then replans. PETS (Chua et al., 2018) pairs this with an ensemble of probabilistic neural-network dynamics models so that the planner knows where the model is uncertain. **AlphaZero** (Silver et al., 2017/2018) plans with Monte Carlo tree search using the *known* rules of Go, chess and shogi, guided by a policy-value network trained on the search's own outputs; **MuZero** (Schrittwieser et al., 2020) removes the need for known rules by learning a latent model that predicts only quantities useful for planning (reward, value and policy), and matched AlphaZero in board games while also working on Atari.

**Learning from imagined data.** Use the model as a simulator to generate extra training experience for a model-free learner. Sutton's **Dyna** (1990) does this in the tabular case: after each real step, apply the Q-learning update to a few simulated transitions from the model as well. In the deep setting, MBPO (Janner et al., 2019) trains a SAC agent on short model rollouts branched from real states, since long rollouts compound model error.

**World models.** Ha and Schmidhuber's *World Models* (2018) trained a VAE to compress frames into latent codes, an RNN to predict the next latent, and a tiny controller acting on the latents, then even trained the controller entirely *inside the learnt dream*. The **Dreamer** family (Hafner et al., 2020 onward) learns a recurrent latent dynamics model (the RSSM) from pixels and trains an actor-critic purely on imagined latent trajectories, and later versions solve a wide range of tasks with one set of hyper-parameters. Conceptually, this is the link to generative modelling in [Weeks 12](week-12.html) and [13](week-13.html): a world model is a conditional generative model of the future.

The fundamental difficulty is **model bias and compounding error**. A one-step model with 1% error leads to rollouts that, after 50 steps, drift into regions of state space the model never saw, and the policy optimizer is an adversary that will find and exploit every flaw (just like the actor exploiting an approximate critic). Common remedies are short rollouts, ensembles with uncertainty penalties, and learning the model in a latent space that keeps only decision-relevant information. Model-based methods typically win by an order of magnitude in sample efficiency on continuous control, and the gap narrows when the dynamics are complex or discontinuous (contact-rich manipulation).

## Offline reinforcement learning {#offline}

**Offline (batch) RL** learns a policy from a *fixed* dataset of transitions collected by some other behaviour policy, with no environment interaction during training. It matters when exploration is expensive or dangerous (healthcare, autonomous driving, industrial control), and it promises to bring to RL the data-scaling recipe that works in supervised learning. It is also much harder than it looks.

The core problem is **distribution shift** (Levine, Kumar, Tucker and Fu, 2020, tutorial). An off-policy algorithm such as DQN or SAC queries $q(s',a')$ at actions $a'$ chosen by the *learnt* policy, including actions that never appear in the dataset. There the Q-function has no data to constrain it, and function approximators extrapolate arbitrarily; the $\max$ or the actor then seeks out precisely the actions with the highest *erroneous* values. In online RL the agent tries the action and finds out it was bad. Offline, no correction ever arrives and errors compound through bootstrapping. Naively running DQN or SAC on a fixed dataset typically fails even when the dataset contains expert data.

Offline methods therefore keep the learnt policy **close to the data**, or keep values **pessimistic** away from it:

- **Policy constraints** (BCQ, Fujimoto et al., 2019; BEAR, Kumar et al., 2019; TD3+BC, Fujimoto and Gu, 2021): restrict or regularize the policy toward the behaviour policy. TD3+BC literally adds a behaviour-cloning term to the TD3 actor loss.
- **Conservative value learning** (CQL, Kumar et al., 2020): add a penalty that pushes down Q-values on out-of-distribution actions while keeping them up on dataset actions, so the learnt Q lower-bounds the true value.
- **In-sample methods** (IQL, Kostrikov et al., 2022): learn values using only actions that appear in the dataset, via expectile regression, and extract a policy by advantage-weighted regression, never querying unseen actions.
- **Sequence-modelling view** (Decision Transformer, Chen et al., 2021): treat trajectories as token sequences of (return-to-go, state, action) and train a causal Transformer by supervised learning to predict actions conditioned on the desired return. It avoids bootstrapping altogether, which makes it simple and stable, at the price of limited ability to *stitch* together good sub-trajectories from different episodes.

:::warning Offline evaluation is hard
You cannot measure a policy's return without running it. Off-policy evaluation with importance sampling has enormous variance over long horizons, and fitted-Q evaluation inherits the extrapolation issue. Many papers therefore report online returns in a simulator, which is not available in the settings that motivate offline RL. Treat reported offline-RL benchmark numbers with care, and when possible keep a small online evaluation budget.
:::

## Reinforcement learning and language models {#llm-rl}

The most consequential application of deep RL today is not a game or a robot but **post-training of large language models** ([Week 10](week-10.html)). A pretrained LM is trained to imitate text, which is not the same as being helpful, honest or harmless, or being correct. Reinforcement learning from human feedback (RLHF) steers the model with a learnt reward signal instead.

### Casting text generation as an MDP

Treat the LM as the **policy** $\pi_{\boldsymbol\theta}(y_t\mid x,y_{<t})$. The **state** is the prompt $x$ plus the tokens generated so far, the **action** is the next token (a discrete action space of 30k to 200k tokens), and the transition is deterministic (append the token). An **episode** is one full response $y=(y_1,\dots,y_T)$ ending at an end-of-sequence token. Reward is typically given only at the end of the response (a sparse, delayed reward, so credit assignment across hundreds of tokens is the main difficulty) from a **reward model**, though some implementations add a per-token KL penalty.

{{figure: week15-rlhf | The RLHF pipeline (as in InstructGPT, Ouyang et al., 2022). Stage 1: supervised fine-tuning (SFT) on demonstrations. Stage 2: train a reward model on human preference comparisons. Stage 3: optimize the policy against the reward model with PPO, with a KL penalty to the SFT model.}}

### Stage 1: supervised fine-tuning

Fine-tune the pretrained model by ordinary cross-entropy on a set of high-quality prompt-response demonstrations written by humans. This gives the RL stage a sensible starting policy (so the early samples are not gibberish) and is the "demonstrations for exploration" idea above. Call the result $\pi_{\text{ref}}$ (the reference or SFT policy).

### Stage 2: the reward model

Writing a numeric reward function for "a good answer" is impossible, but people can *compare* two answers reliably. Collect prompts $x$ with pairs of responses $(y_w,y_l)$, where $y_w$ was preferred. Train a **reward model** $r_{\boldsymbol\phi}(x,y)$ (usually the same LM with a scalar head replacing the vocabulary head) under the **Bradley-Terry** model of preferences, which says the probability that $y_w$ beats $y_l$ is a logistic function of the reward difference:
$$
P(y_w\succ y_l\mid x)=\sigma\big(r_{\boldsymbol\phi}(x,y_w)-r_{\boldsymbol\phi}(x,y_l)\big),\qquad
\mathcal L_{\text{RM}}(\boldsymbol\phi)=-\E_{(x,y_w,y_l)}\Big[\log\sigma\big(r_{\boldsymbol\phi}(x,y_w)-r_{\boldsymbol\phi}(x,y_l)\big)\Big].
$$
This is binary logistic regression on score differences. Only differences matter, so the reward's offset is arbitrary; implementations often normalize rewards. The reward model is a *proxy*: it is trained on finite human data and is accurate only near the distribution on which it was trained.

### Stage 3: policy optimization with a KL penalty

Now optimize the policy against the proxy reward, but anchored to the reference model:
$$
\max_{\boldsymbol\theta}\;\E_{x\sim\mathcal D,\ y\sim\pi_{\boldsymbol\theta}(\cdot\mid x)}\Big[r_{\boldsymbol\phi}(x,y)\Big]-\beta\,\E_x\Big[\KL\big(\pi_{\boldsymbol\theta}(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)\Big].
$$
The KL term does the work that the trust region does in PPO, but over the *whole training run*. It keeps the policy where the reward model is reliable, prevents mode collapse onto a single high-reward string, and preserves fluency. Without it, the policy finds inputs on which the reward model gives high scores for the wrong reasons: **reward hacking**, a form of **Goodhart's law**. Gao, Schulman and Hilton (2023) measured this "overoptimization" with a synthetic gold reward model and found that the gold reward first rises and then *falls* as the policy moves further (in KL) from the initial policy while the proxy reward keeps rising. The optimization is run with PPO: the value function is a second network (often initialized from the reward model), advantages come from GAE over tokens with the per-token KL penalty included in the reward, and the clipped objective keeps each batch update small. That makes RLHF with PPO resource-heavy: you hold in memory the policy, the reference model, the reward model and the value model.

```python:rlhf_reward_loss.py
import torch
import torch.nn.functional as F

def reward_model_loss(r_chosen, r_rejected):
    """Bradley-Terry loss. Both tensors have shape (B,): scalar rewards for preferred / dispreferred responses."""
    return -F.logsigmoid(r_chosen - r_rejected).mean()

def kl_shaped_token_rewards(logp_policy, logp_ref, final_reward, beta=0.05):
    """Per-token rewards for PPO in RLHF. logp_* have shape (B, T) (log-prob of each sampled token under the
    policy / reference); final_reward (B,) is the reward-model score given to the last token only.
    Per-token reward = -beta * (logp_policy - logp_ref), plus the sequence reward at the final position."""
    rewards = -beta * (logp_policy - logp_ref)             # sampled estimate of the KL penalty, per token
    rewards[:, -1] = rewards[:, -1] + final_reward
    return rewards

B, T = 4, 6
print(reward_model_loss(torch.tensor([2.0, 1.0, 0.5, 3.0]), torch.tensor([1.0, 1.5, -0.5, 0.0])).item())
print(kl_shaped_token_rewards(torch.randn(B, T) - 2, torch.randn(B, T) - 2, torch.randn(B)).shape)
```

### DPO: the same objective without an explicit RL loop

The KL-regularized objective above has a closed-form optimal policy, $\pi^*(y\mid x)\propto\pi_{\text{ref}}(y\mid x)\exp\big(r(x,y)/\beta\big)$. Rearranging gives the reward as $r(x,y)=\beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)}+\beta\log Z(x)$, where the partition function $Z(x)$ cancels when two responses to the same prompt are compared in the Bradley-Terry model. **Direct Preference Optimization** (Rafailov et al., 2023) substitutes this implicit reward into the preference loss and obtains a supervised objective on the policy itself:
$$
\mathcal L_{\text{DPO}}(\boldsymbol\theta)=-\E_{(x,y_w,y_l)}\Big[\log\sigma\Big(\beta\log\frac{\pi_{\boldsymbol\theta}(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_{\boldsymbol\theta}(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\Big)\Big].
$$
There is no reward model to train, no sampling from the policy during training, and no value function: just a classification-style loss on fixed preference pairs, needing the log-probabilities of both responses under the current and reference models. It optimizes the *same* KL-regularized objective, assuming the Bradley-Terry model. The trade-offs: DPO is simpler, cheaper and more stable; but it is **offline** with respect to the preference data (it never explores beyond the responses in the dataset, so it can suffer the distribution-shift issues of the offline section), and online RL with a reward model can exploit fresh samples and, when a *verifiable* reward exists, optimize it directly. Which approach wins depends on the data and task and is an active empirical question, so treat claims of one dominating the other with care.

### GRPO and verifiable rewards

For tasks where correctness can be checked automatically, such as math answers or code that passes unit tests, a learnt reward model is unnecessary: use the checker as the reward ("reinforcement learning with verifiable rewards"). **Group Relative Policy Optimization (GRPO)**, introduced in the DeepSeekMath paper (Shao et al., 2024) and used for DeepSeek-R1 (DeepSeek-AI, 2025), keeps PPO's clipped objective and KL penalty but drops the learnt value function. For each prompt it samples a *group* of $G$ responses, scores them, and uses the group's own statistics as the baseline:
$$
\hat A_i=\frac{R_i-\operatorname{mean}(R_1,\dots,R_G)}{\operatorname{std}(R_1,\dots,R_G)}.
$$
Every token in response $i$ shares the advantage $\hat A_i$. This is the baseline trick again, with the baseline estimated by Monte Carlo from sibling samples instead of a critic, which saves the memory and compute of the value network and avoids having to train a good critic over long sequences. It is closely related to the leave-one-out baselines used in REINFORCE-style RLHF (e.g. RLOO, Ahmadian et al., 2024). The price is that the group must be large enough for the baseline to be informative, and prompts for which all samples are right or all wrong provide no gradient.

:::note What is and is not settled
The following are well established: RLHF with PPO and a KL penalty works for instruction following; reward models are imperfect proxies that can be over-optimized; DPO and GRPO are simplified relatives that share PPO's ideas (preference or group-relative signals, KL anchoring to a reference). Less settled: how much of the reasoning ability gained by RL with verifiable rewards is new capability versus better elicitation of what pretraining already provided, and which of PPO, DPO, GRPO and their variants is best in a given setting. Expect this part of the field to change quickly.
:::

## Lab: DQN on CartPole, then PPO with Stable-Baselines3 {#lab}

In this lab you build a complete DQN from scratch in PyTorch (about 100 lines), train it on Gymnasium's `CartPole-v1`, then solve the same task with PPO from the Stable-Baselines3 library and compare. In CartPole the state is four numbers (cart position, cart velocity, pole angle, pole angular velocity), there are two actions (push left or right), the reward is +1 per step, and an episode ends when the pole tilts more than about 12 degrees, the cart leaves the track, or after 500 steps. Return 500 is a perfect score. Install with `pip install torch gymnasium stable-baselines3` (Colab has most of this).

### Step 1: the replay buffer and Q-network

```python:dqn_cartpole.py
import random
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size FIFO buffer of (s, a, r, s', terminated) transitions with uniform sampling."""
    def __init__(self, capacity, obs_dim):
        self.cap, self.i, self.n = capacity, 0, 0
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.d = np.zeros(capacity, dtype=np.float32)

    def add(self, s, a, r, s2, d):
        self.s[self.i], self.a[self.i], self.r[self.i], self.s2[self.i], self.d[self.i] = s, a, r, s2, d
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.n, size=batch_size)
        t = lambda x: torch.as_tensor(x[idx], device=device)
        return t(self.s), t(self.a), t(self.r), t(self.s2), t(self.d)

class QNet(nn.Module):
    """MLP that maps an observation (B, 4) to one Q-value per action (B, 2)."""
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x):
        return self.net(x)
```

### Step 2: the training loop

Note the two details that most often go wrong: the done flag used for bootstrapping must be `terminated` (a true terminal state), *not* `truncated` (time limit), and targets are computed without gradient.

```python:dqn_train.py
def train_dqn(total_steps=30_000, gamma=0.99, lr=1e-3, batch_size=64, buffer_size=50_000,
              learning_starts=1_000, train_freq=1, target_tau=0.01, eps_start=1.0, eps_end=0.05,
              eps_decay_steps=10_000, double=True, seed=0, verbose=True):
    env = gym.make("CartPole-v1")
    obs_dim, n_actions = env.observation_space.shape[0], env.action_space.n
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    online, target = QNet(obs_dim, n_actions).to(device), QNet(obs_dim, n_actions).to(device)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    buf = ReplayBuffer(buffer_size, obs_dim)

    s, _ = env.reset(seed=seed)
    ep_return, returns = 0.0, []
    for step in range(1, total_steps + 1):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay_steps)
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = int(online(torch.as_tensor(s, device=device).unsqueeze(0)).argmax(dim=1).item())
        s2, r, terminated, truncated, _ = env.step(a)
        buf.add(s, a, r, s2, float(terminated))          # bootstrap through truncation, not termination
        s, ep_return = s2, ep_return + r
        if terminated or truncated:
            returns.append(ep_return)
            s, _ = env.reset()
            ep_return = 0.0

        if step >= learning_starts and step % train_freq == 0:
            bs, ba, br, bs2, bd = buf.sample(batch_size)
            with torch.no_grad():
                if double:                                # Double DQN: online selects, target evaluates
                    a_star = online(bs2).argmax(dim=1, keepdim=True)
                    q_next = target(bs2).gather(1, a_star).squeeze(1)
                else:
                    q_next = target(bs2).max(dim=1).values
                y = br + gamma * (1.0 - bd) * q_next
            q = online(bs).gather(1, ba.unsqueeze(1)).squeeze(1)    # Q(s, a_taken), shape (B,)
            loss = F.smooth_l1_loss(q, y)                 # Huber loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 10.0)
            opt.step()
            with torch.no_grad():                         # Polyak (soft) target update
                for p_t, p in zip(target.parameters(), online.parameters()):
                    p_t.mul_(1 - target_tau).add_(target_tau * p)

        if verbose and step % 2_000 == 0 and returns:
            print(f"step {step:6d} | eps {eps:.2f} | mean return (last 10) {np.mean(returns[-10:]):6.1f} | loss {loss.item() if step >= learning_starts else float('nan'):.4f}")
    return online, returns

if __name__ == "__main__":
    model, returns = train_dqn()
```

On a typical run, the mean return climbs from about 20 (random behaviour) to several hundred within 20,000 to 30,000 steps, though DQN on CartPole is noisy and often oscillates (it may reach 500, collapse, and recover). Different seeds will differ a lot, so run at least three and plot the mean. This variability is a typical feature of deep RL, not a bug in your code.

### Step 3: evaluate the greedy policy and inspect Q-values

```python:dqn_eval.py
def evaluate(model, episodes=10, seed=100):
    env = gym.make("CartPole-v1")
    scores = []
    for k in range(episodes):
        s, _ = env.reset(seed=seed + k)
        done, total = False, 0.0
        while not done:
            with torch.no_grad():
                a = int(model(torch.as_tensor(s, device=device).unsqueeze(0)).argmax(dim=1).item())
            s, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated
        scores.append(total)
    return np.mean(scores), np.std(scores)

mean, std = evaluate(model)
print(f"greedy evaluation: {mean:.1f} +- {std:.1f}")
# Sanity check on the value scale: with r = +1 per step and gamma = 0.99, the maximum possible
# discounted return is (1 - 0.99**500)/(1 - 0.99) ~= 99.3, so Q(s, a) should never exceed about 100.
s0, _ = gym.make("CartPole-v1").reset(seed=0)
print("Q(s0, .) =", model(torch.as_tensor(s0, device=device).unsqueeze(0)).detach().cpu().numpy())
```

The last lines show a useful debugging habit: compute the *maximum possible* value analytically and check that your learnt Q-values respect it. If they greatly exceed it, you have an overestimation or target bug.

### Step 4: PPO with Stable-Baselines3

```python:ppo_sb3.py
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

# 8 parallel environments, as in A2C/PPO: the batch diversity replaces the replay buffer
vec_env = make_vec_env("CartPole-v1", n_envs=8, seed=0)

model = PPO(
    "MlpPolicy", vec_env,
    n_steps=256,            # rollout length per env -> batch of 8 * 256 = 2048 samples per update
    batch_size=64,          # minibatch size for the SGD epochs
    n_epochs=10,            # K epochs per batch
    gamma=0.99, gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0, vf_coef=0.5,
    learning_rate=3e-4,
    verbose=1, seed=0,
)
model.learn(total_timesteps=100_000)

eval_env = gym.make("CartPole-v1")
mean_r, std_r = evaluate_policy(model, eval_env, n_eval_episodes=20, deterministic=True)
print(f"PPO evaluation: {mean_r:.1f} +- {std_r:.1f}")
model.save("ppo_cartpole")
```

With `verbose=1`, SB3 prints a table at every update including `approx_kl`, `clip_fraction`, `entropy_loss`, `explained_variance` and `value_loss`. Learn to read these: `explained_variance` near 1 means the critic predicts returns well, `clip_fraction` shows what share of samples hit the clip, and `approx_kl` should stay small (about 0.01 or lower).

### Step 5: experiments to run

1. **Ablate the stabilizers.** Set `target_tau=1.0` (target network equals online network), then set `buffer_size` to 64 (almost no replay) and compare learning curves over three seeds each. Which stabilizer matters more on CartPole?
2. **Double vs vanilla.** Run `double=False` versus `double=True` and print the mean predicted Q-value at the start state. Vanilla values tend to be higher.
3. **PPO hyper-parameters.** Vary `clip_range` in {0.05, 0.2, 0.5} and `n_epochs` in {3, 10, 30}, tracking `approx_kl` and the final return.
4. **Sample efficiency.** Plot return against environment steps for DQN and PPO. On CartPole both solve the task; which needs fewer environment steps in your runs, and what changes if you count wall-clock time?
5. **Discount.** Try $\gamma\in\{0.9,0.99,0.999\}$ for DQN. A short horizon makes the task look easier but can fail to look far enough ahead.

## Common mistakes and debugging tips {#mistakes}

Deep RL bugs are especially hard because the failure symptom ("the return does not go up") fits dozens of causes, and the training signal is noisy. Check cheap, deterministic things first.

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| Q-values explode or grow without bound | Missing target network, or targets not detached; learning rate too high; no reward scaling | Use `torch.no_grad()` for targets, soft target updates, Huber loss, gradient clipping, smaller LR |
| Q-values far above the maximum possible return | Overestimation (max over noisy values), or bootstrapping through terminal states | Double DQN or twin critics; multiply bootstrap by `(1 - terminated)` |
| Agent learns nothing; return flat at the random level | Exploration too weak, reward never seen, or learning never started (`learning_starts` larger than steps) | Check the minimum number of steps to reach a reward, raise entropy bonus or epsilon, verify the buffer is filling |
| Return rises then collapses (policy gradient) | Updates too large, no advantage normalization, learning rate too high | Smaller LR, lower clip range, fewer epochs, watch `approx_kl`; early-stop an update when KL is large |
| Value loss large and critic fails to fit | Returns on very different scales | Normalize observations and rewards (running statistics), use value clipping or a Huber loss |
| Time-limit episodes treated as failures | Using `done = terminated or truncated` in the bootstrap | Bootstrap through truncation: use only `terminated` for the `(1 - done)` mask |
| Policy collapses to deterministic too early | Entropy too low, or `log_std` initialized small | Entropy bonus, larger initial std, SAC auto-temperature |
| Works for one seed, not for the next | High variance of RL | Run 5 to 10 seeds, report mean and a confidence interval, never a best-of-seeds curve |
| Off-by-one in returns or GAE | Ordering errors in backward recursions; forgetting to cut at episode ends | Unit-test GAE on a hand-computed 3-step example; test with a constant-reward environment where the answer is known |
| Training uses stale data in an on-policy method | Reusing an old batch or an old log-prob | Recompute `old_logp` at collection time; discard the batch after $K$ epochs |

:::tip A debugging order that saves days
1. Test your algorithm on a trivial environment where you know the answer (a one-step bandit, or a 2-state chain). It must solve it in seconds.
2. Check the data: print a few (s, a, r, s', done) tuples and confirm shapes, dtypes, and that rewards are what you expect.
3. Check a scale: what are the largest plausible returns and Q-values? Do the network's outputs obey it?
4. Log diagnostics every update: episode return and length, loss terms, policy entropy, approximate KL, clip fraction, explained variance, gradient norm, mean $|Q|$.
5. Only then tune hyper-parameters, one at a time, with several seeds. "Deep reinforcement learning that matters" (Henderson et al., 2018) documents how easy it is to draw wrong conclusions from too few seeds.
:::

## Exercises {#exercises}

:::exercise Exercise 1: Bellman by hand
A deterministic chain has states $s_1\to s_2\to\text{terminal}$ under the only available policy, with rewards $R=1$ for leaving $s_1$ and $R=2$ for leaving $s_2$. With $\gamma=0.5$, compute $v_\pi(s_1)$ and $v_\pi(s_2)$ (a) from the definition of the return, and (b) by solving the Bellman expectation equations. Then state what changes at $\gamma=1$ and why this is still well defined.
:::details Solution
(a) From $s_2$: $G=2$, so $v(s_2)=2$. From $s_1$: $G=1+\gamma\cdot2=1+0.5\cdot2=2$, so $v(s_1)=2$. (b) The equations are $v(s_2)=2+\gamma\cdot0$ (terminal value is zero) and $v(s_1)=1+\gamma v(s_2)$, which give the same numbers. At $\gamma=1$: $v(s_2)=2$, $v(s_1)=3$; the sum is finite because the episode has finite length, so the undiscounted return is well defined.
:::
:::

:::exercise Exercise 2: Q-learning versus SARSA update
In state $s'$ the current estimates are $Q(s',a_1)=2$ and $Q(s',a_2)=1$. You have $Q(s,a)=0.5$, observe $r=1$, move to $s'$, and the $\varepsilon$-greedy behaviour policy happens to choose the exploratory action $a_2$ there. With $\alpha=0.1$ and $\gamma=0.9$, compute the updated $Q(s,a)$ under SARSA and under Q-learning. Which one would be unchanged if the behaviour policy had picked $a_1$?
:::details Solution
Q-learning target: $1+0.9\max(2,1)=2.8$, so $Q\leftarrow0.5+0.1(2.8-0.5)=0.73$. SARSA target uses the action actually chosen, $a_2$: $1+0.9\cdot1=1.9$, so $Q\leftarrow0.5+0.1(1.9-0.5)=0.64$. If $a_1$ had been picked, both targets equal $2.8$ and the two updates agree (0.73). Q-learning's target never depends on the exploratory choice, which is exactly what "off-policy" means here.
:::
:::

:::exercise Exercise 3: Reading the PPO clip
Let $\epsilon=0.2$. For $\hat A=+2$ and for $\hat A=-2$, evaluate $\min\big(r\hat A,\operatorname{clip}(r,0.8,1.2)\hat A\big)$ and say whether the gradient with respect to $r$ is nonzero, for $r\in\{0.7,\,1.1,\,1.5\}$. Which cases correspond to "the update made the action's probability change in the wrong direction", and what does PPO do in them?
:::details Solution
$\hat A=+2$: $r=0.7$: $\min(1.4,1.6)=1.4$, gradient $2$ (unclipped branch is smaller). $r=1.1$: $\min(2.2,2.2)=2.2$, gradient $2$. $r=1.5$: $\min(3.0,2.4)=2.4$, gradient $0$ (clipped, no further incentive).
$\hat A=-2$: $r=0.7$: $\min(-1.4,-1.6)=-1.6$, gradient $0$ (clipped, already decreased enough). $r=1.1$: $\min(-2.2,-2.2)=-2.2$, gradient $-2$. $r=1.5$: $\min(-3.0,-2.4)=-3.0$, gradient $-2$ (unclipped, strongly pushes back).
Wrong-direction cases are $(\hat A>0,\,r=0.7)$ (a good action became less likely) and $(\hat A<0,\,r=1.5)$ (a bad action became more likely). In both the $\min$ selects the unclipped term, so the full gradient remains and PPO actively corrects the mistake.
:::
:::

:::exercise Exercise 4: Compute GAE
A 3-step episode ends in a terminal state. Rewards are $(1,1,1)$, the critic outputs $V=(0.5,0.5,0.5)$ for the three states, and the terminal state has value 0. With $\gamma=0.9$ and $\lambda=0.8$, compute the three TD errors and the GAE advantages $\hat A_0,\hat A_1,\hat A_2$ using the backward recursion.
:::details Solution
$\delta_2=1+0.9\cdot0-0.5=0.5$; $\delta_1=1+0.9\cdot0.5-0.5=0.95$; $\delta_0=0.95$. With $\gamma\lambda=0.72$: $\hat A_2=0.5$; $\hat A_1=0.95+0.72\cdot0.5=1.31$; $\hat A_0=0.95+0.72\cdot1.31=1.8932$. You can verify with `compute_gae` from the page after setting `values` and `rewards` accordingly (a single environment, `last_value=0`).
:::
:::

:::exercise Exercise 5: Why the baseline is unbiased
Show that $\E_{a\sim\pi_{\boldsymbol\theta}(\cdot|s)}\big[\nabla_{\boldsymbol\theta}\log\pi_{\boldsymbol\theta}(a\mid s)\,b(s)\big]=\mathbf 0$ for any function $b$ of the state. Then, for a one-dimensional Gaussian policy $\pi(a|s)=\mathcal N(\mu_{\boldsymbol\theta}(s),\sigma^2)$ with fixed $\sigma$, compute $\nabla_\mu\log\pi(a|s)$ and explain in words what the policy-gradient update does to $\mu$ when the advantage is positive.
:::details Solution
For discrete actions, $\sum_a\pi(a|s)\nabla\log\pi(a|s)\,b(s)=b(s)\sum_a\nabla\pi(a|s)=b(s)\nabla\sum_a\pi(a|s)=b(s)\nabla1=0$; for continuous actions replace the sum by an integral (interchanging gradient and integral under standard regularity). For the Gaussian, $\log\pi=-\frac{(a-\mu)^2}{2\sigma^2}+\text{const}$, so $\nabla_\mu\log\pi=(a-\mu)/\sigma^2$. With positive advantage, the update moves $\mu$ *toward the sampled action* $a$ (the step is proportional to $(a-\mu)/\sigma^2\cdot A$); with negative advantage it moves $\mu$ away from it. Note the $1/\sigma^2$ factor: smaller $\sigma$ gives larger, noisier updates to the mean.
:::
:::

:::exercise Exercise 6: A dueling head
Write a PyTorch `nn.Module` for a dueling Q-network for an input of dimension `obs_dim` and `n_actions` actions, with a shared 128-unit hidden layer and separate value and advantage heads, using the mean-advantage subtraction. Check that adding a constant to the advantage head's output leaves $Q$ unchanged.
:::details Solution
```python
import torch
import torch.nn as nn

class DuelingQNet(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)              # V(s): (B, 1)
        self.adv = nn.Linear(hidden, n_actions)        # A(s, a): (B, n_actions)

    def forward(self, x, adv_shift=0.0):
        h = self.trunk(x)
        v, a = self.value(h), self.adv(h) + adv_shift   # adv_shift only used for the invariance check
        return v + a - a.mean(dim=1, keepdim=True)      # (B, n_actions)

net = DuelingQNet(4, 2)
x = torch.randn(5, 4)
print(torch.allclose(net(x), net(x, adv_shift=3.7), atol=1e-6))   # True: shift cancels in the mean
```
The identifiability problem is resolved because only advantages relative to their mean affect $Q$.
:::
:::

:::exercise Exercise 7: Overestimation by the max
Let $X_1,X_2$ be independent $\mathcal N(0,1)$ estimates of two actions whose true values are both 0. Compute $\E[\max(X_1,X_2)]$ exactly. Then explain why selecting with $X_1,X_2$ and evaluating with an independent second pair $Y_1,Y_2$ removes the bias.
:::details Solution
$\max(X_1,X_2)=\tfrac12(X_1+X_2)+\tfrac12|X_1-X_2|$. The first term has mean 0. $X_1-X_2\sim\mathcal N(0,2)$, so $\E|X_1-X_2|=\sqrt2\cdot\sqrt{2/\pi}=2/\sqrt\pi$. Hence $\E[\max]=\tfrac12\cdot2/\sqrt\pi=1/\sqrt\pi\approx0.564$, matching the chart (0.56). In the double estimator the action $\hat a=\argmax_iX_i$ is chosen by $X$, but evaluated with $Y_{\hat a}$, which is independent of the selection event and has mean 0, so $\E[Y_{\hat a}]=0$: the evaluation noise no longer correlates with the selection noise.
:::
:::

:::exercise Exercise 8 (challenging): A divergent off-policy TD example
Consider two states with a *linear* value function $V(s_1)=w$, $V(s_2)=2w$ (a single weight, features 1 and 2). All rewards are 0. The only transition is $s_1\to s_2$, and $s_2$ transitions to itself (continuing task). Training updates only on transitions out of $s_1$ (off-policy: the data distribution never samples $s_2$) using semi-gradient TD(0) with step size $\alpha$ and discount $\gamma$. Derive the update of $w$, find the condition on $\gamma$ under which $w$ diverges from any non-zero start, and relate it to the deadly triad. What happens if we also train on transitions out of $s_2$ with the on-policy frequency?
:::details Solution
The TD error at $s_1$ is $\delta=0+\gamma V(s_2)-V(s_1)=2\gamma w-w=(2\gamma-1)w$, and $\nabla_wV(s_1)=1$. So $w\leftarrow w+\alpha(2\gamma-1)w=\big(1+\alpha(2\gamma-1)\big)w$. For $\gamma>1/2$ the factor exceeds 1 and $|w|\to\infty$ geometrically, even though the true values are all zero ($w=0$ is the fixed point). All three triad ingredients are present: linear function approximation (shared weight), bootstrapping (target uses $V(s_2)$), off-policy sampling (only $s_1$ updates). On-policy, the stationary distribution of this chain puts all its mass on the absorbing state $s_2$, so the updates come from $s_2$: the TD error there is $2\gamma w-2w=2(\gamma-1)w$ with gradient 2, giving $w\leftarrow\big(1-4\alpha(1-\gamma)\big)w$, which converges to 0 for small $\alpha$. Mixing the two kinds of updates at weights $d_1,d_2$ gives a net factor $1+\alpha\big(d_1(2\gamma-1)-4d_2(1-\gamma)\big)$, so divergence depends on how far the sampling distribution is from the on-policy one. This is the Tsitsiklis and Van Roy style example that Sutton and Barto use to motivate the triad.
:::
:::

:::exercise Exercise 9 (challenging): Derive DPO
Starting from the KL-regularized objective $\max_\pi\E_{y\sim\pi}[r(x,y)]-\beta\KL(\pi\|\pi_{\text{ref}})$ for a fixed prompt $x$, (a) show that the optimal policy is $\pi^*(y|x)=\pi_{\text{ref}}(y|x)\exp(r(x,y)/\beta)/Z(x)$, (b) express $r$ in terms of $\pi^*$, and (c) substitute into the Bradley-Terry loss to obtain the DPO loss. Why does $Z(x)$ disappear?
:::details Solution
(a) The objective is $\sum_y\pi(y)\big[r(y)-\beta\log\pi(y)+\beta\log\pi_{\text{ref}}(y)\big]=-\beta\sum_y\pi(y)\log\frac{\pi(y)}{\pi_{\text{ref}}(y)\exp(r(y)/\beta)}$. Let $Z=\sum_y\pi_{\text{ref}}(y)\exp(r(y)/\beta)$ and $q(y)=\pi_{\text{ref}}(y)\exp(r(y)/\beta)/Z$, a valid distribution. Then the objective equals $-\beta\,\KL(\pi\|q)+\beta\log Z$, which is maximized (value $\beta\log Z$) at $\pi=q$. (b) Taking logs, $r(y)=\beta\log\frac{\pi^*(y)}{\pi_{\text{ref}}(y)}+\beta\log Z(x)$. (c) The Bradley-Terry probability depends on $r(x,y_w)-r(x,y_l)$, and $\beta\log Z(x)$ appears in both, so it cancels: the difference is $\beta\log\frac{\pi^*(y_w)}{\pi_{\text{ref}}(y_w)}-\beta\log\frac{\pi^*(y_l)}{\pi_{\text{ref}}(y_l)}$. Replacing the unknown $\pi^*$ by the trainable $\pi_{\boldsymbol\theta}$ and taking the negative log-likelihood of the observed preferences gives exactly the $\mathcal L_{\text{DPO}}$ on this page.
:::
:::

:::exercise Exercise 10: What would happen if...
For each change to the DQN code in the lab, predict the symptom and the reason: (a) remove `torch.no_grad()` and `.detach()` so gradients flow into the target; (b) use `done = terminated or truncated` in the bootstrap mask; (c) set `buffer_size=64`; (d) set `gamma=0.999`; (e) in RLHF, set the KL coefficient $\beta$ to 0.
:::details Solution
(a) The loss now also moves the target toward the prediction (a residual gradient); learning is typically slower and biased toward trivially consistent but poor value functions, and it uses extra memory. (b) At time-limit truncations the agent is told the state has value zero though it does not, so values near step 500 are biased low and the agent can learn that "survive long" is worthless; use only `terminated`. (c) The buffer holds only recent, correlated transitions: the minibatch is nearly one trajectory segment, so learning is unstable and forgets earlier behaviour. (d) The effective horizon is about 1000 steps, longer than any episode; values are large and updates propagate slowly, so learning slows and targets are more sensitive to error. (e) The policy is free to move anywhere that raises the proxy reward: reward hacking, loss of fluency, mode collapse to a few high-scoring outputs, and a falling true quality while the proxy score rises.
:::
:::

:::exercise Exercise 11 (coding): Tabular Q-learning on the slippery gridworld
Using the transition tensors `P` and `R` from the value-iteration code, implement tabular Q-learning with $\varepsilon$-greedy exploration by sampling transitions from `P`. Compare the learnt greedy policy and $\max_aQ(s,a)$ with the value-iteration solution after 50,000 steps.
:::details Solution
```python
import numpy as np

def q_learning(P, R, gamma, goal, pit, steps=50_000, alpha=0.1, eps=0.2, seed=0):
    rng = np.random.default_rng(seed)
    nS, nA, _ = P.shape
    Q = np.zeros((nS, nA))
    s = 0
    for _ in range(steps):
        a = rng.integers(nA) if rng.random() < eps else int(np.argmax(Q[s] + 1e-9 * rng.random(nA)))
        s2 = rng.choice(nS, p=P[s, a])                     # sample next state from the model
        target = R[s, a, s2] + (0.0 if s2 in (goal, pit) else gamma * Q[s2].max())
        Q[s, a] += alpha * (target - Q[s, a])
        s = 0 if s2 in (goal, pit) else s2                 # restart after a terminal state
    return Q

# Q = q_learning(P, R, gamma, goal, pit)
# print(np.abs(Q.max(1) - V).max())   # should be small (a few hundredths) away from rarely visited states
```
Expect the values to agree closely on frequently visited states and to be noisier (or still close to zero) in states the exploratory policy rarely reaches, and expect that a constant step size leaves a residual noise floor proportional to $\alpha$.
:::
:::

## Key takeaways {#takeaways}

:::key Key takeaways
- An MDP, with the Markov property, a policy, a discount, and value functions $v_\pi,q_\pi$, is the common language. The Bellman equations express a value as immediate reward plus discounted successor value, and the optimality equations replace the policy average by a max.
- Dynamic programming needs a model; Monte Carlo is unbiased but high-variance; TD bootstraps for low variance at the price of bias. Q-learning is off-policy (learns $q_*$), SARSA is on-policy.
- Function approximation, bootstrapping and off-policy data together (the deadly triad) can diverge. DQN's replay buffer, target network and Huber loss are practical antidotes; Double DQN fixes max-overestimation, Dueling separates $V$ and $A$, prioritized replay samples by TD error.
- The policy gradient theorem $\nabla J=\E[\sum_t\nabla\log\pi(a_t|s_t)\,G_t]$ needs no model; baselines (best: $v_\pi$) cut variance without bias; actor-critic bootstraps; GAE$(\gamma,\lambda)$ $=\sum_l(\gamma\lambda)^l\delta_{t+l}$ trades bias for variance.
- PPO's clipped objective approximates TRPO's trust region cheaply and is the default on-policy algorithm; watch KL and clip fraction. SAC and TD3 are the standard off-policy methods for continuous control, SAC adding a maximum-entropy objective.
- Exploration is the hard part: entropy and $\varepsilon$ for easy tasks, curiosity or count bonuses for sparse ones. Model-based RL buys sample efficiency at the price of model error; offline RL must contend with distribution shift via conservatism or in-sample learning.
- RLHF = SFT, then a Bradley-Terry reward model, then PPO with a KL anchor to the reference policy. DPO optimizes the same KL-regularized objective with a supervised loss on preference pairs; GRPO drops the critic in favour of group-relative baselines.
- RL results are noisy: use several seeds, log diagnostics, and test on trivial environments first.
:::

## Further reading {#reading}

Essential papers and books are marked with a star.

- (Essential) Sutton and Barto, *Reinforcement Learning: An Introduction*, 2nd edition, MIT Press, 2018 (free online from the authors). Chapters 3, 6 and 13 correspond to this week.
- (Essential) Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, 2015 (DQN; earlier version: "Playing Atari with Deep Reinforcement Learning," arXiv:1312.5602, 2013).
- (Essential) Schulman, Wolski, Dhariwal, Radford and Klimov, "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
- (Essential) Haarnoja, Zhou, Abbeel and Levine, "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor," ICML 2018 (arXiv:1801.01290).
- Williams, "Simple statistical gradient-following algorithms for connectionist reinforcement learning," *Machine Learning*, 1992 (REINFORCE).
- van Hasselt, Guez and Silver, "Deep Reinforcement Learning with Double Q-learning," AAAI 2016; Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning," ICML 2016; Schaul et al., "Prioritized Experience Replay," ICLR 2016; Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning," AAAI 2018.
- Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning," ICML 2016 (A3C).
- Schulman, Levine, Abbeel, Jordan and Moritz, "Trust Region Policy Optimization," ICML 2015; Schulman, Moritz, Levine, Jordan and Abbeel, "High-Dimensional Continuous Control Using Generalized Advantage Estimation," ICLR 2016.
- Lillicrap et al., "Continuous control with deep reinforcement learning," ICLR 2016 (DDPG); Fujimoto, van Hoof and Meger, "Addressing Function Approximation Error in Actor-Critic Methods," ICML 2018 (TD3).
- Engstrom et al., "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO," ICLR 2020; Henderson et al., "Deep Reinforcement Learning That Matters," AAAI 2018.
- Levine, Kumar, Tucker and Fu, "Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems," arXiv:2005.01643, 2020.
- Silver et al., "Mastering the game of Go without human knowledge," *Nature*, 2017; Schrittwieser et al., "Mastering Atari, Go, chess and shogi by planning with a learned model," *Nature*, 2020 (MuZero); Hafner et al., "Dream to Control: Learning Behaviors by Latent Imagination," ICLR 2020.
- (Essential for LLMs) Ouyang et al., "Training language models to follow instructions with human feedback," NeurIPS 2022 (InstructGPT); Christiano et al., "Deep reinforcement learning from human preferences," NeurIPS 2017.
- Rafailov et al., "Direct Preference Optimization: Your Language Model Is Secretly a Reward Model," NeurIPS 2023; Shao et al., "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models," arXiv:2402.03300, 2024 (GRPO); Gao, Schulman and Hilton, "Scaling Laws for Reward Model Overoptimization," ICML 2023.
- Stable-Baselines3 documentation (stable-baselines3.readthedocs.io), Spinning Up in Deep RL by OpenAI (spinningup.openai.com), and the CleanRL single-file implementations (Huang et al., JMLR 2022).
