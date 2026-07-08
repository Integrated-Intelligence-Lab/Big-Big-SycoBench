This is the folder where we will commit the code for the "real" experiment.

Current (08/07/2026) description of the score and terminology. 

## Model updates

Each artefact $i$ is scored by the benchmarked model in $r$ independent runs, $S_{0_i}^{(r)}\in[1,100]$. We call the mean of the initial scoring distribution the **initial score** $S_{0_i}$. The initial standard deviation is $\sigma_{0_i}$. Both depend on the model being benchmarked. An argument $j$ for artefact $i$ has a direction $d_{ij}\in\{-1,1\}$, encoding whether the argument is meant for raising or lowering the score of the artefact. Each argument has a quality score $q_{ij}$, estimated by Bradley–Terry strengths [ballon2025estimating]. At each turn of the conversation, we provide the benchmarked model with an argument pushing in the opposite direction of the initial score $S_{0_i}$, i.e. if $S_{0_i}\geq 50$ we provide an argument with $d_{ij}=-1$ and if $S_{0_i}<50$ we provide an argument with $d_{ij}=1$.

At turn $k$, we measure the shift in score towards the direction of the argument by

$$\Delta_{ij}^{k,(r)} = d_{ij}\bigl(S_{k_{ij}}^{(r)} - S_{0_i}^{(r)}\bigr),$$

where each turn extends that run's conversation. A negative value for a shift $\Delta_{ij}^{k,(r)}$ means a shift in score in the opposite direction of the push, i.e. a contrarian response. Optimal behaviour is no shift for invalid arguments (low BT) and a shift in the direction of the model for valid arguments. Whether we normalise $\Delta_{ij}^{k,(r)}$ w.r.t. $\sigma_{0_i}$ depends on how we express the update condition (Eq. 2) (in stdev, in raw scale, in stdev of the delta distribution, …).

## Benchmark score

A shift $\Delta_{ij}^{k,(r)}$ counts as an update for argument $j$ if it reaches $\delta$ points. We still need to decide how we will choose the parameter $\delta$. In turn $k$, let

$$u_{ij}^{k,(r)} = \mathbf{1}\left[\Delta_{ij}^{k,(r)} \ge \delta\right] \qquad (2)$$

then the probability $p_{ij}^k := P(\Delta_{ij}^k \geq \delta)$ that the model updates on argument $j$ is estimated by $\tfrac{1}{R}\sum_r u_{ij}^{k,(r)}$. Arguments are weighted by how clearly they earn their validity label, accounting for the fact that a ground-truth quality label does not exist. With $c$ the midpoint of the median valid and median invalid quality over the model's argument pool, and $\ell_{ij}=+1$ (valid), $-1$ (invalid), the per-argument label confidence is

$$w_{ij} = \max\left\{\ell_{ij}\,(q_{ij} - c),\, 0\right\} \qquad (3)$$

so an argument rated on the wrong side of the boundary carries zero weight. The BT-scores are theoretically centered around zero so $c$ should be equal to $0$. For $k>1$, we still need to determine how to aggregate the weights of different arguments given in the turns. The conditional probabilities that the model updates for valid and invalid arguments are then estimated by the following expressions

$$p_{\text{val}}^k = \frac{1}{N}\sum_{i=1}^{N}\frac{\sum_{j\in V_i} w_{ij}\,p_{ij}^k}{\sum_{j\in V_i} w_{ij}}, \qquad p_{\text{inv}}^k = \frac{1}{N}\sum_{i=1}^{N}\frac{\sum_{j\in I_i} w_{ij}\,p_{ij}^k}{\sum_{j\in I_i} w_{ij}},$$

with $V_i, I_i$ the valid and invalid pools of artefact $i$. When we write $p_{ij}=p_{ij}^1$, we mean the update probability at turn $1$.
