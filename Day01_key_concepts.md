# Day 01: Key concepts in DeepRL

## Key concepts and terminology

Almost all of RL revolves around the behaviour of two central _characters_: the **agent** and the **environment**.
* **Agent**: an autonomous entity that observes through its sensors and acts upon an environment (possibly through its actuators) and directs its activity towards achieving goals
* **Environment**: the world that the agent lives and interacts with

At every _time step_, the agent _observes_ the state of the environment (sometimes only partially) and decides upon an action to execute in the subsequent time step. The environment usually changes with time (sometimes even without the agent explicitly acting upon it).

Periodically (can happen as frequently as each time step to as rarely as once-per-lifetime), the agent receives a **reward** from the environment. The reward is a quantifiable entity that measures how well the agent executed the stipulated task. In most cases, the reward is a scalar number. The objective of the agent is to maximize the **return**, which is the sum of all rewards obtained.

A **state** is a complete description of the world the agent lives in. An **observation** is a (possibly partial) description of a _state_. Observations, by definition, can hence omit some information about the state. Oftentimes, RL folks use these two terms interchangeably. A useful distinction to make is that of _full observability_ vs _partial observability_.

An **action** is a decision that the agent is capable of making. The set of all valid actions in a given environment constitute the **action space**. Action spaces can be _discrete_ or _continuous_.

A **policy** is a _rule_ that specifies the action to be executed, based on the current state. Policies can either be _deterministic_ or _stochastic_. Deterministic policies are usually denoted as
<br />
<a href="https://www.codecogs.com/eqnedit.php?latex=a_t&space;=&space;\mu(s_t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_t&space;=&space;\mu(s_t)" title="a_t = \mu(s_t)" /></a>
<br />

Here, the action is dependent only on the state because of the Markov assumption, which states that, _given the most current state, the future states are independent of all past states and actions_. Although this seems like a bad assumption to make, it lends itself to the elegant formulation of simple algorithms as we will see. Stochastic policies can be denoted as
<br />
<a href="https://www.codecogs.com/eqnedit.php?latex=a_t&space;\sim&space;\pi\left(\cdot&space;\mid&space;s_t&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_t&space;\sim&space;\pi\left(\cdot&space;\mid&space;s_t&space;\right&space;)" title="a_t \sim \pi\left(\cdot \mid s_t \right )" /></a>
<br />

Here, the actions are drawn from a probability distribution. We are particularly interested in _parameterized policies_, where the outputs of the policies are computable functions, that depend on certain _parameters_. Such parameterized policies are denoted
<br />
<a href="https://www.codecogs.com/eqnedit.php?latex=a_t&space;=&space;\mu_\theta\left(&space;s_t&space;\right&space;)&space;\\&space;a_t&space;\sim&space;\pi_\theta\left(\cdot&space;\mid&space;s_t&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a_t&space;=&space;\mu_\theta\left(&space;s_t&space;\right&space;)&space;\\&space;a_t&space;\sim&space;\pi_\theta\left(\cdot&space;\mid&space;s_t&space;\right&space;)" title="a_t = \mu_\theta\left( s_t \right ) \\ a_t \sim \pi_\theta\left(\cdot \mid s_t \right )" /></a>
<br />

The two most common kinds of stochastic policies are _categorical policies_ and _diagonal Gaussian policies_. Categorical policies are suited for discrete action spaces, whereas diagonal Gaussian policies are more suited for continuous action spaces.

A **trajectory** (aka **episode**, **rollout**) is a sequence of states and actions. A _start state_ is selected randomly, or better put, sampled from a _start-state distribution_. _State transitions_ capture the change in the state of the world due to an action executed by the agent. These state transitions may again be deterministic or stochastic.

Rewards and returns can be of several types. Some important ones are:
* _Finite-horizon undiscounted return_: sum of rewards obtained over a trajectory of fixed-length 
<br />
<a href="https://www.codecogs.com/eqnedit.php?latex=R(\tau)&space;=&space;\sum_{t=0}^{T}r_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R(\tau)&space;=&space;\sum_{t=0}^{T}r_t" title="R(\tau) = \sum_{t=0}^{T}r_t" /></a>
<br />
* _Infinite-horizon discounted return_: sum of all rewards ever obtained by an agent, but discounted by how far-off in the future they're obtained. Introduces a **discount factor**.
<br />
<a href="https://www.codecogs.com/eqnedit.php?latex=R(\tau)&space;=&space;\sum_{t=0}^{\infty}\gamma^tr_t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R(\tau)&space;=&space;\sum_{t=0}^{\infty}\gamma^tr_t" title="R(\tau) = \sum_{t=0}^{\infty}\gamma^tr_t" /></a>
<br />
The discount factor (typically in the range (0,1)) ensures that the return remains finite. 


## The RL problem


The central goal in RL is to choose a policy that maximizes **expected return** when an agent executes it.

Let's try and break the definition down to math now. To describe the notion of _expected return_, we will need to first describe the notion of an _expected trajectory_ under a given policy. When the transitions and the policy are stochastic, we can compute the probability of a T-step trajectory as follows.
<br />
<a href="https://www.codecogs.com/eqnedit.php?latex=P(\tau&space;\mid&space;\pi)&space;=&space;\rho_0(s_0)\prod_{t=0}^{T-1}P(s_{t&plus;1}\mid&space;s_t,a_t)\pi(a_t&space;\mid&space;s_t)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(\tau&space;\mid&space;\pi)&space;=&space;\rho_0(s_0)\prod_{t=0}^{T-1}P(s_{t&plus;1}\mid&space;s_t,a_t)\pi(a_t&space;\mid&space;s_t)" title="P(\tau \mid \pi) = \rho_0(s_0)\prod_{t=0}^{T-1}P(s_{t+1}\mid s_t,a_t)\pi(a_t \mid s_t)" /></a>
<br />

The expected return is given by
<br />
<a href="https://www.codecogs.com/eqnedit.php?latex=J(\pi)&space;=&space;\int_\tau&space;P(\tau\mid\pi)R(\tau)&space;=&space;\mathop{\mathbb{E}}_{\tau&space;\sim&space;\pi}\left[&space;R(\tau)&space;\right&space;]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(\pi)&space;=&space;\int_\tau&space;P(\tau\mid\pi)R(\tau)&space;=&space;\mathop{\mathbb{E}}_{\tau&space;\sim&space;\pi}\left[&space;R(\tau)&space;\right&space;]" title="J(\pi) = \int_\tau P(\tau\mid\pi)R(\tau) = \mathop{\mathbb{E}}_{\tau \sim \pi}\left[ R(\tau) \right ]" /></a>
<br />

The optimization problem, then, is
<br />
<a href="https://www.codecogs.com/eqnedit.php?latex=\pi^*&space;=&space;\arg&space;\mathop{\max}_{\pi}&space;J(\pi)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\pi^*&space;=&space;\arg&space;\mathop{\max}_{\pi}&space;J(\pi)" title="\pi^* = \arg \mathop{\max}_{\pi} J(\pi)" /></a>
<br />


## Value functions


The **value** *under a policy* of a state (or a state-action pair) is the expected return when we start executing that policy from the state (or state-action pair). **Value functions** come in several varieties, and are used in nearly every RL algorithm, in some form.


Four main kinds of value functions are:
1. **On-policy value function**: gives the expected return if you start in a state s and always act according to the policy.
2. **On-policy action-value function**: gives the expected return if you start in state s, take an arbitrary action (which may not result from the policy), and act on the policy forever thereafter.
3. **Optimal value function**: gives the expected return if you start in state s and always act according to the _optimal policy_ for the environment.
4. **Optimal action-value function**: gives the expected return if you start in state s, take an arbitrary action (which may not result from the policy), and act on the _optimal policy_ for the environment forever thereafter.

An optimal action-value function (Q-function) gives the expected return for starting in state s and taking an arbitrary action a, and then executing the optimal policy forever. This means that, the optimal policy in state s will select whichever action maximizes the expected return from starting in s. That is, the optimal action can be obtained by doing
<br />
<a href="https://www.codecogs.com/eqnedit.php?latex=a^*(s)&space;=&space;\arg&space;\mathop{\max}_{a}&space;Q^*(s,a)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?a^*(s)&space;=&space;\arg&space;\mathop{\max}_{a}&space;Q^*(s,a)" title="a^*(s) = \arg \mathop{\max}_{a} Q^*(s,a)" /></a>
<br />


## References
1. [Multivariate normal distributions](http://www.maths.manchester.ac.uk/~mkt/MT3732%20(MVA)/Notes/MVA_Section3.pdf)
