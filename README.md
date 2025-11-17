# CS672: Introduction to Reinforcement Learning - Homework Implementations

This repository contains my homework assignments for the KAIST (Fall 2025) graduate course **CS672: Introduction to Reinforcement Learning**, taught by Prof. Sungjin Ahn.

As a Master's student, my goal is to build a deep, practical understanding of modern reinforcement learning by implementing its core algorithms from the ground up, based on the course lectures and the foundational textbook by Sutton & Barto.

---

## 🚀 Assignments & Core Concepts

The assignments are structured to follow the course's progression from foundational MDPs and Dynamic Programming to Model-Free methods and modern Deep Policy Gradient algorithms.

### Homework 1: Dynamic Programming & MDPs

This assignment implements the foundational algorithms for solving known Markov Decision Processes (MDPs): **Policy Iteration** and **Value Iteration**.

#### 1. Markov Decision Processes (MDPs) & Bellman Equations

An MDP is the formal framework for RL, defined by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$. The goal is to find a policy $\pi$ that maximizes the expected discounted return.

* **State-Value Function ($v_\pi$):** The expected return starting from state $s$ and following policy $\pi$. It is defined by the **Bellman Expectation Equation**:
    $$v_{\pi}(s) = \sum_{a}\pi(a|s)\sum_{s',r}p(s',r|s,a)[r + \gamma v_{\pi}(s')]$$

* **Action-Value Function ($q_\pi$):** The expected return starting from state $s$, taking action $a$, and then following policy $\pi$.
    $$q_{\pi}(s,a) = \sum_{s',r}p(s',r|s,a)[r + \gamma \sum_{a'}\pi(a'|s')q_{\pi}(s',a')]$$

#### 2. Policy and Value Iteration

* **Policy Iteration:** An algorithm that finds the optimal policy $\pi_*$ by iteratively:
    1.  **Policy Evaluation:** Calculating the value function $v_\pi$ for the current policy $\pi$ (e.g., by iterating the Bellman expectation backup).
    2.  **Policy Improvement:** Creating a new, greedy policy $\pi'$ based on the action-values $q_\pi(s,a)$ derived from $v_\pi$.

* **Value Iteration:** An algorithm that directly finds the optimal value function $v_*$ by iteratively applying the **Bellman Optimality backup**:
    $$v_{k+1}(s) = \max_{a} \sum_{s',r}p(s',r|s,a)[r + \gamma v_{k}(s')]$$

---

### Homework 2: Model-Free Prediction & Control

This assignment moves to Model-Free (MF) methods, where the agent learns directly from experience samples (episodes) without knowing the MDP's transition $\mathcal{P}$ or reward $\mathcal{R}$ functions.

#### 1. Monte-Carlo (MC) & Temporal-Difference (TD)

* **Monte-Carlo Prediction:** Estimates $v_\pi(s)$ by averaging the full returns $G_t$ observed from state $s$ over many episodes.
    * **Update:** $V(S_t) \leftarrow V(S_t) + \alpha(G_t - V(S_t))$

* **Temporal-Difference (TD(0)) Prediction:** Estimates $v_\pi(s)$ by **bootstrapping**—updating the current estimate $V(S_t)$ towards an estimated target $R_{t+1} + \gamma V(S_{t+1})$.
    * **Update:** $V(S_t) \leftarrow V(S_t) + \alpha(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))$

#### 2. Sarsa (On-Policy) and Q-Learning (Off-Policy)

* **Sarsa (On-Policy):** A TD control algorithm that learns the action-value function $q_\pi$. It is "on-policy" because it uses an action $A'$ selected by its *current* policy to form the TD target.
    * **Update:** $Q(S, A) \leftarrow Q(S, A) + \alpha(R + \gamma Q(S', A') - Q(S, A))$

* **Q-Learning (Off-Policy):** A TD control algorithm that learns the optimal action-value function $q_*$ directly, regardless of the policy being followed. It is "off-policy" because its target is formed by the *greedy* (optimal) action, not the action $A'$ that was actually taken.
    * **Update:** $Q(S, A) \leftarrow Q(S, A) + \alpha(R + \gamma \max_{a'} Q(S', a') - Q(S, A))$

#### 3. Off-Policy Learning with Importance Sampling

* **Key Idea:** To evaluate a **target policy $\pi$** using data generated from a **behavior policy $\mu$**, we must correct for the mismatch in action probabilities.
* **Importance Sampling Ratio:** $\rho_t = \frac{\pi(A_t|S_t)}{\mu(A_t|S_t)}$.
* **Off-Policy TD Update:** The TD update is weighted by the importance sampling ratio.
    * **Update:** $V(S_t) \leftarrow V(S_t) + \alpha \rho_t (R_{t+1} + \gamma V(S_{t+1}) - V(S_t))$

---

### Homework 3: Policy Gradient Methods

This assignment explores **Policy Gradient (PG)** methods, which optimize a parameterized policy $\pi_\theta(a|s)$ directly by performing gradient ascent on the objective function $J(\theta)$.

#### 1. REINFORCE Algorithm

REINFORCE is a foundational Monte-Carlo policy gradient algorithm. It relies on the **Policy Gradient Theorem**:
$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) G_t]$$

* **Key Idea:** The algorithm increases the log-probability of an action $A_t$ in proportion to the total return $G_t$ that followed it. "Good" actions get reinforced.
* **Drawback:** This estimator suffers from **high variance** because the return $G_t$ can vary significantly, even for good policies.

#### 2. REINFORCE with Baseline

To reduce variance, a state-dependent **baseline $b(s)$** (commonly the state-value function $V(s)$) is subtracted from the return.

* **Advantage Function:** The term $A(s, a) = G_t - V(s)$ is known as the **Advantage**. It measures "how much better" the return was than the average expectation for that state.
* **Unbiased Gradient:** This subtraction does not introduce bias, as $\mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) b(s)] = 0$.
* **Gradient:**
    $$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) (G_t - V(s))]$$

---

## 📚 Acknowledgements & Academic Integrity

This code is for academic purposes and is based on the lectures and materials for **CS672: Introduction to Reinforcement Learning (Fall 2025)** at KAIST, taught by **Prof. Sungjin Ahn**.
