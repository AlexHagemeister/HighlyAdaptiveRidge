# Highly Adaptive Ridge

**Alejandro Schuler**
Division of Biostatistics, University of California, Berkeley

**Alexander Hagemeister**
EECS, University of California, Berkeley

**Mark van der Laan**
Division of Biostatistics, University of California, Berkeley

*May 21, 2024*

## Abstract

In this paper we propose the Highly Adaptive Ridge (HAR): a new, scalable machine learning algorithm. We prove that HAR attains a remarkable $n^{-1/3}$ dimension-free $L_2$ convergence rate in a large nonparametric function class. HAR is exactly kernel ridge regression with a specific data-adaptive kernel based on a saturated zero-order spline basis expansion. We use simulation and real data to confirm our theory and demonstrate empirical performance and scalability on par with state-of-the-art algorithms.

**Keywords:** nonparametric regression, high-dimensional regression, convergence rate

---

## 1 Introduction

In regression our task is to find a function that maps features to an outcome such that the expected loss is minimized [8]. In the past decades a huge number of flexible regression methods have been developed that effectively search over high- or infinite-dimensional function spaces. These are often collectively called "machine learning" methods for regression.

$L_2$ convergence is a well-studied property of regression algorithms that measures how quickly generalization MSE decreases as the size of the training sample increases. Faster convergence rates (asymptotically) guarantee more efficient use of limited data.

In many causal inference settings fast rates are in fact required to build valid confidence intervals. For example, when estimating the average treatment effect from observational data, a sufficient condition for the asymptotic normality of the TMLE and AIPW estimators is that the propensity score and outcome regression models converge to their respective truths in root-mean-square generalization error at a rate of $o_P(n^{-1/4})$ [13, 19]. Thus the development of fast-converging, nonparametric regression methods is critical for efficient statistical inference.

Here we present a new machine learning method (highly adaptive ridge; HAR) that achieves a $O_P(n^{-1/3} (\log n)^{2(p-1)/3})$ $L_2$ convergence rate under mild conditions on the true data-generating process. This rate is remarkable because it is dimension-free (up to log factors) as a consequence of the assumed function class. The proposed method is computationally scalable to high dimensions and has close ties with the highly adaptive lasso [2, 16], lassoed tree boosting [11], and kernel ridge regression [22].

## 2 Notation and Preliminaries

Throughout the paper we adopt the empirical process notation $Pf = \int f(Z)\,dP$ and $P_n f = \frac{1}{n} \sum_i f(Z_i)$. In this notation these operators do not average over any potential randomness in $f$ so $Pf_n$ is a random variable if $f_n$ is a random function learned from data. We use $\|f\|$ to indicate an $L_2$ norm $\sqrt{Pf^2}$ unless otherwise noted.

Let $X_i, Y_i \in \mathcal{X} \times \mathbb{R}$ be IID across $i$ and with a generic $X, Y$ that have some joint distribution $P$. We will take $\mathcal{X} = [0,1]^p$ without loss of generality for applications with bounded covariates.

Let $L$ be some loss (e.g. mean-squared error), which we construct such as to take a prediction function $f$ as an input and return a function of $X, Y$ as output. For example, if we want $L$ to be squared error loss, we let $L(f)(X, Y) = (f(X) - Y)^2$. Throughout we abbreviate $Lf = L(f)$. Let $f = \arg\min_{f:\mathcal{X}\to\mathbb{R}} PLf$. This is the standard regression setup where our goal is to estimate the function $f$ from $n$ samples of the vector of predictors $X$ and the outcome $Y$.

### Càdlàg and Sectional Variation

Throughout the paper we focus on the class of càdlàg functions of bounded sectional variation. "Càdlàg" (*continu à droite, limites à gauche*) means right-continuous with left limits [10]. So, for example, all continuous functions are càdlàg. Càdlàg functions correspond 1-to-1 with signed measures the same way that cumulative distribution functions correspond with probability measures. We use $\mathcal{F}$ to denote the set of càdlàg functions.

The sectional variation norm of a càdlàg function $f$ on $[0,1]^p$ is given by

$$\sum_{s \subseteq \{1\ldots p\}} \int_0^1 |df_s(x)|$$

where $f_s(x)$ denotes the "$s$-section" of $f$ which evaluates $f$ at $x$ after first replacing the elements of $x$ that do not correspond to dimensions in $s$ with zeros (for example, $f_{\{1\}}(x) = f(x_1, 0, \ldots, 0)$). We use the notation $\|\cdot\|_v$ to distinguish sectional variation norm from a standard $L_2$ norm. Intuitively, a variation norm is a way of measuring how much a function goes up and down in total over its entire domain. For a function of one variable, the variation norm is the total "elevation gain" plus "elevation loss" over the domain, up to a constant.

Given some constant $M$, we use $\mathcal{F}(M)$ to denote all càdlàg functions with $\|f\|_v \leq M$.

$\mathcal{F}(M)$ is an extremely rich class: these functions may be highly nonlinear and even discontinuous. Functions that are not in $\mathcal{F}(M)$ are often pathological, e.g. $f(x) = \cos(1/x)$. In cases where they are not (e.g. when $f(x)$ is the indicator of the unit ball) they can always be approximated arbitrarily well by a càdlàg function of bounded sectional variation because $\bigcup_M \mathcal{F}(M)$ is dense in $L_2$.

The utility of assuming a bounding sectional variation is that we can assure faster $L_2$ convergence than is possible in smoothness classes without being as restrictive as assuming additive structure. The minimax rate in a Hölder class with smoothness $\beta$ is well-known to be $n^{-\beta/(2\beta+p)}$ [12]. This rate suffers from the curse of dimensionality due to the strong dependence on $p$. In contrast, the minimax rate for additive functions $f(x) = \sum f_j(x_j)$ with $f_j$ Lipschitz is $n^{-1/3}$. This entirely dimension-free rate is bought at the cost of a very strong assumption, however. For càdlàg functions of bounded sectional variation the minimax rate is $n^{-1/3} (\log n)^{2(p-1)/3}$ (up to log factors) [5]. This looks like the rate for additive functions except for the fact that the dimension incurs a cost in the log factor.

One way to understand this is that bounding the sectional variation (Vitali variation, actually; sectional variation is the sum of Vitali variations over sections) limits the amount of multi-variable "interactivity" that is allowed. This is easiest to see for continuous differentiable functions of two dimensions for which the Vitali variation takes the simple form $\int \left|\frac{\partial^2 f}{\partial x_1 \partial x_2}\right| d(x_1, x_2)$. It is clear how this penalizes the amount of sub- or super-additivity: an additive function has zero mixed derivative everywhere. Bounding this variation therefore results in a function class with members that behave more like sums of univariate functions of each of their inputs. The larger the variation norm is allowed to be, the more "interactivity" is allowed between variables. Assuming bounded sectional variation therefore strikes a nice middle ground between assuming general smoothness and assuming exact additive structure [9, 24].

## 3 Method

Highly adaptive ridge performs a ridge regression in a (data-adaptive) high-dimensional expansion $H(x)$ of the covariates. The estimated function $\hat{f}_n$ is the empirical minimizer of a loss function $L$ in the parametric model $\{H(x)^\top \beta : \|\beta\|_2 \leq M\}$. The bound $M$ is chosen by cross-validation (suppressed in the notation).

The high dimensional basis expansion $H$ is constructed as follows. Let $s \subseteq \{1 \ldots p\}$ denote a "section", i.e. some subset of the dimensions of $[0,1]^p$. Let

$$h_{i,s}(x) = \prod_{j \in s} \mathbf{1}(X_{i,j} \leq x_j)$$

be a single, scalar-valued basis function indexed by $i$ and $s$. Here and in what follows, we use the convention that $\prod_{j \in \emptyset} u_j = 1$ so $h_{i,\emptyset}(x) = 1$ all give an "intercept" term. The bases $h_{i,s}$ are standard tensor product zero-order splines each with "knot point" $c_{i,s} \in [0,1]^p$ where each element of the knot is $c_{i,s,j} = X_{i,j}$ if $j \in s$ and $c_{i,s,j} = 0$ if $j \notin s$. In other words, the knot $c_{i,s}$ is the vector $X_i$ with the non-$s$ elements set to 0. With this notation we can write $h_{i,s}(x) = \mathbf{1}(c_{i,s} \leq x)$ where the inequality must hold in all dimensions. The bases $h_{i,s}$ are data-dependent (random) because $X_i$ is an observed data point. Our full list of basis functions is

$$H = [h_{i,s} : i \in \{1 \ldots n\},\; s \subseteq \{1 \ldots p\}].$$

We use $d$ to refer to the number of basis functions $|H|$. The number of bases $d$ is equal to $n 2^p$ (there are $2^p$ sections and $n$ "knots" per section). Technically the number of bases can be smaller if there are "ties" in the data and certainly we have already over-counted the intercept term $n - 1$ times. To keep the notation clean, however, we consider $H$ to be a multiset that allows repeated elements and we can proceed with the exact equality $d = n 2^p$. This will make no difference in the computations and theory that follows.

Formally, the HAR estimator is

$$\hat{f}_n = \arg\min_{f \in \mathcal{F}_n(M)} P_n Lf$$

$$\mathcal{F}_n(M) = \left\{ H(x)^\top \beta \;\text{s.t.}\; \|\beta\|^2 \leq M_n \right\}$$

### 3.1 Convergence Rate

Our theoretical contribution is to show that the described algorithm converges quickly in $L_2$ norm to the truth under mild assumptions on the data generating process.

**Theorem 1.** Define $f = \arg\min_{\{g:[0,1]^p \to \mathbb{R}\}} PLg$ for a loss function $L$. Let our model be $\mathcal{F}_n(M_n) = \{H(x)^\top \beta : \|\beta\|^2 \leq M_n\}$ and our estimate be $\hat{f}_n = \arg\min_{g \in \mathcal{F}_n(M_n)} P_n Lg$. Use $F$ to denote the CDF of $X$.

If (1) the loss function obeys $\|f - g\|^2 \leq P(Lf - Lg)$ and $P(Lf - Lg) = O(\|g - f\|_\infty)$, (2) $f$ is càdlàg with sectional variation bounded by $M^*$ with $df_s/dF_s$ existing and bounded for all $s$, and (3) $M_n$ is chosen via cross-validation from a grid of values $\mathcal{M}_n = [M_{n,1} \ldots M_{n,K}]$ such that $\exists k_n, \bar{M} : M \leq n^{-1} M_{n,k_n} \leq \bar{M}$ for $M$ chosen suitably large enough, then $\|\hat{f}_n - f\| = O_P(n^{-1/3} (\log n)^{2(p-1)/3})$.

The proof is given in the appendix. In brief, we use an oracle approximation $f_{n,s} = \int_0^x \frac{df_s}{dF_s}\,dF_{n,s}$ ($F_n$ being the empirical CDF of $X$) and show that $f_n \to f$ suitably quickly while the squared $L_2$ norm of the "coefficients" of this function shrink at an $O(n^{-1})$ rate. That ensures $\|f_n\|_v$ remains bounded. We then use empirical process theory to show that the discrepancy between the estimate $\hat{f}_n$ and approximation $f_n$ also disappears quickly.

The required condition on $L$ is mild and satisfied by mean-squared error and binary log-loss, among other losses [3]. In light of the richness of bounded sectional variation càdlàg functions, the restriction on $f$ is also mild, as is the assumption of existence and boundedness of the densities $df_s/dF_s$. The condition on the grid of $L_2$ bounds for $\beta$ can be satisfied in practice by choosing a large and fine enough grid.

The theorem above extends trivially to cover mixtures of $L_1$ and $L_2$ penalties on the coefficients of the bases (a "highly adaptive elastic net"). However the pure $L_2$ penalty comes with unique computational benefits that make the algorithm scalable to high-dimensional data.

### 3.2 Computation

The $L_2$-constrained minimization problem in $\beta$ described above is most often solved using the Lagrangian formulation $\hat{\beta} = \arg\min_\beta P_n(H\beta - Y)^2 + \lambda\|\beta\|^2$ which has the closed-form solution $\hat{\beta} = (HH^\top + \lambda I_d)^{-1} H^\top Y$ where $H^\top = [H(X_1), \ldots, H(X_n)]$ and $Y^\top = [Y_1 \ldots Y_n]$.

Because there are $d = O(n 2^p)$ columns in the "predictor" matrix $H$ this problem is impossible to solve computationally for even moderate values of $p$. Even instantiating the array in memory can be prohibitive. However, an application of the Woodbury matrix identity reveals the equivalent expression $\hat{\beta} = H(H^\top H + \lambda I_n)^{-1} Y$, meaning that a prediction at $x$ can be computed as $\hat{f}_n(x) = H(x)^\top H(H^\top H + \lambda I_n)^{-1} Y$. The advantage of this is that prediction at a point depends only on inner products of the form $H(x)^\top H(x')$. We can analytically work out the kernel function that computes this inner product directly from the lower-dimensional $x, x'$ and avoid ever having to instantiate $H$ or invert a $d \times d$ matrix:

$$H(x)^\top H(x') = \sum_i \sum_s \left(\prod_{j \in s} \mathbf{1}(X_{i,j} \leq x_j)\right) \left(\prod_{j \in s} \mathbf{1}(X_{i,j} \leq x'_j)\right)$$

$$= \sum_i \sum_s \prod_{j \in s} \mathbf{1}(X_{i,j} \leq (x \wedge x')_j)$$

$$= \sum_i \sum_{s \subseteq s_i(x,x')} 1$$

$$= \sum_i 2^{|s_i(x,x')|}$$

where $x \wedge x'$ denotes the elementwise minimum and $s_i(x, x') = \{j : \mathbf{1}(X_{i,j} \leq (x \wedge x')_j)\}$. The middle equality follows because the product term is 1 only if $s \subseteq s_i(x, x')$ and 0 otherwise. This is a simple computation: we compare the point $x \wedge x'$ to each $X_i$ and count the number of dimensions in which the former is greater than or equal to the latter. This does not require us to compute the basis expansions $H(x)$ or the values of $\hat{\beta}$ and is thus scalable to high-dimensional $X$. Also note that because the kernel is constructed data-adaptively there are no additional tuning parameters.

### 3.3 Related Work

HAR is closely related to HAL, the highly adaptive lasso [2, 16, 5]. In HAL the estimator is the empirical minimizer of a loss function $L$ in the parametric model $\{H(x)^\top \beta : \|\beta\|_1 \leq M\}$. As implied by the names, HAL penalizes the $L_1$ norm of the coefficients while HAR penalizes $L_2$ norm. HAL achieves the same fast convergence rate as HAR but HAL suffers from a computational curse of dimensionality because the basis matrix $H$ must be explicitly computed. Moreover lasso problems are generally much slower to solve than ridge problems, even absent the use of the kernel trick.

The extension of the rate result from HAL to HAR is not trivial. Changing from an $L_1$ to an $L_2$ penalty fundamentally changes the function class being considered. HAR with a fixed $L_2$ bound does not work: as the size of the dataset increases, the number of bases expand as well and the HAR function class quickly becomes much bigger than any class of càdlàg functions of bounded sectional variation. It is therefore essential to shrink the HAR bound at a certain rate to keep the model inside this Donsker class (we show cross-validation can take care of this) and it must be proved that this does not then eliminate any relevant functions from consideration. This is why HAR requires cross-validation and the first-order smoothness assumption $df_s/dF_s < b$ to prove the rate result. This is not required by (0th-order) HAL.

Previous work demonstrated a close connection between HAL and gradient boosted trees and exploited this to construct a rate-preserving boosting algorithm called Lassoed Tree Boosting (LTB) [11]. The conceptual advantage of HAR over LTB is that HAR provides direct empirical minimization over $H(x)^\top \beta$ whereas LTB must iteratively "boost" a sequence of bases and repeatedly find the optimal linear combination. Like HAL, however, LTB does not require the first-order smoothness condition required by HAR to prove the fast rate.

HAR is very closely related to previous work on estimation in "tensor product" Sobolev spaces [9, 24]. Our theoretical results are distinct in that we assume slightly different forms of smoothness that lend themselves to extremely concise proofs based on the sectional representation of càdlàg functions of bounded sectional variation.

HAR is also a form of kernel ridge regression (KRR) [22]. The only twist is that the kernel function for HAR is constructed automatically by the algorithm based on the data instead of being chosen by the user. In this sense HAR is somewhat related to "kernel learning" methods [7]. Previous results on convergence rates for kernel ridge regression are found in [14, 25, 4]. The majority of these results discuss rates for KRR with fixed kernel forms in function classes with standard smoothness and sparsity constraints. In contrast, our result is for a data-adaptive kernel in a function space that is primarily constrained in a global rather than local sense (bounded sectional variation vs. existence of higher order derivatives).

## 4 Demonstration

Are the entries of $H^\top H$ too big for medium $p$? If we only consider $p^* \leq p$-way interactions, then the size of $|s_i| \leq p^*$ and everything is good! Just keep $p^* = 100$ and call it a day. Does the matrix invert nicely or are there numerical issues?

Speed and performance vs HAL, GBT, KRR w/ fixed kernel on simulated ($n$ increasing) and real data (various $n$ and $p$).

Speed with different $n, p$ for kernelized vs. direct solution of HAR (perhaps to appendix).

## 5 Software

Describe an sklearn-conforming python package and its features.

## 6 Discussion

HAR provides a conceptually simple and computationally tractable algorithm with fast convergence in a meaningfully large nonparametric class of functions. The fast rate means that many efficient estimators of causal quantities can be shown to be asymptotically linear under weaker assumptions that would otherwise be required [16, 19, 13].

HAR is vastly faster than HAL, which provides the same rate guarantees, but does have some disadvantages. For one, our proof of the HAR rate depends on a first-order smoothness assumption that is not required for HAL. This assumption is very mild but nonetheless needed. Another disadvantage is that HAR, like other kernel methods, is not "exportable". There is no explicit instantiation of the coefficient vector $\hat{\beta}$. To compute a prediction at $x$ an analyst must have full access to the original covariate matrix because the covariate vectors are required to compute $H(x)^\top H$ via the kernel. This also makes prediction slower than for many algorithms. Another consequence of the kernelization is that we cannot explicitly formulate the score equations solved by HAR [18]. Knowing these scores could be useful in the construction of efficient estimators of lower dimensional functionals with good finite-sample performance [20].

HAR is scalable in $p$ but at training time still requires the inversion of an $n \times n$ matrix which is roughly an $O(n^3)$ operation. This is not ideal but completely feasible with modern compute even for relatively large $n$. For truly massive internet-scale data this can be a problem but there are existing methods that mitigate these issues and which are likely rate-preserving with HAR (e.g. matrix sketching [23], divide-and-conquer [25]). Recent work [1] suggests that it may also be possible to modify HAR so that multiple solutions along the regularization path can be computed together with warm-start optimization as is done in elastic net algorithms [6].

## References

1. Oskar Allerbo. Solving kernel ridge regression with Gradient-Based optimization methods. June 2023.
2. David Benkeser and Mark van der Laan. The highly adaptive lasso estimator. *Proc Int Conf Data Sci Adv Anal*, 2016:689–696, December 2016.
3. Aurélien F Bibaut and Mark J van der Laan. Fast rates for empirical risk minimization over càdlàg functions with bounded sectional variation norm. July 2019.
4. Andrea Caponnetto and Ernesto De Vito. Optimal rates for the regularized least-squares algorithm. *Foundations of Computational Mathematics*, 7:331–368, 2007.
5. Billy Fang, Adityanand Guntuboyina, and Bodhisattva Sen. Multivariate extensions of isotonic regression and total variation denoising via entire monotonicity and Hardy-Krause variation. March 2019.
6. Jerome Friedman, Trevor Hastie, and Rob Tibshirani. Regularization paths for generalized linear models via coordinate descent. *J. Stat. Softw.*, 33(1):1–22, 2010.
7. Mehmet Gönen and Ethem Alpaydın. Multiple kernel learning algorithms. *The Journal of Machine Learning Research*, 12:2211–2268, 2011.
8. Trevor Hastie, Robert Tibshirani, and Jerome Friedman. 2. Overview of Supervised Learning. In *The Elements of Statistical Learning*, page 1–34. Springer New York, New York, NY, January 2009.
9. Yi Lin. Tensor product space ANOVA models. *The Annals of Statistics*, 28(3):734–755, 2000.
10. Georg Neuhaus. On weak convergence of stochastic processes with multidimensional time parameter. *The Annals of Mathematical Statistics*, 42(4):1285–1295, 1971.
11. Alejandro Schuler, Yi Li, and Mark van der Laan. Lassoed tree boosting. *arXiv preprint arXiv:2205.10697*, 2022.
12. Charles J Stone. Optimal Global Rates of Convergence for Nonparametric Regression. *Ann. Stat.*, 10(4):1040–1053, December 1982.
13. A Tsiatis. *Semiparametric theory and missing data*, 2007.
14. Rui Tuo, Yan Wang, and CF Jeff Wu. On the improved rates of convergence for Matérn-type kernel ridge regression with application to calibration of computer models. *SIAM/ASA Journal on Uncertainty Quantification*, 8(4):1522–1547, 2020.
15. Aad W van der Vaart, Sandrine Dudoit, and Mark J van der Laan. Oracle inequalities for multi-fold cross validation. *Statistics & Decisions*, 24(3):351–371, 2006.
16. Mark van der Laan. A generally efficient targeted minimum loss based estimator based on the highly adaptive lasso. *Int. J. Biostat.*, 13(2), October 2017.
17. Mark van der Laan. Higher order spline highly adaptive lasso estimators of functional parameters: Pointwise asymptotic normality and uniform convergence rates, 2023.
18. Mark J van der Laan, David Benkeser, and Weixin Cai. Efficient estimation of pathwise differentiable target parameters with the undersmoothed highly adaptive lasso. August 2019.
19. Mark J van der Laan, M J Laan, and James M Robins. *Unified Methods for Censored Longitudinal Data and Causality*. Springer Science & Business Media, January 2003.
20. Mark J van der Laan and Sherri Rose. Why machine learning cannot ignore maximum likelihood estimation. October 2021.
21. A W van der Vaart. *Asymptotic Statistics*. Cambridge University Press, June 2000.
22. Vladimir Vovk. Kernel ridge regression. In *Empirical Inference: Festschrift in Honor of Vladimir N. Vapnik*, pages 105–116. Springer, 2013.
23. Rong Yin, Yong Liu, Weiping Wang, and Dan Meng. Sketch kernel ridge regression using circulant matrix: Algorithm and theory. *IEEE transactions on neural networks and learning systems*, 31(9):3512–3524, 2019.
24. Tianyu Zhang and Noah Simon. Regression in tensor product spaces by the method of sieves. *Electronic Journal of Statistics*, 17(2):3660–3727, 2023.
25. Yuchen Zhang, John Duchi, and Martin Wainwright. Divide and conquer kernel ridge regression. In *Conference on learning theory*, pages 592–617. PMLR, 2013.

---

## A Proof of Theorem 1

Here we provide a proof of the rate result in Theorem 1. The proof here is decomposed into a main result and some corollaries that when combined give the result given in the main text. We first construct an oracle approximation that converges quickly to the target function but which is always in the HAR model with shrinking $L_2$ norm. Standard empirical process arguments then give the rate for the empirical minimizer in that HAR model.

### A.1 Oracle Approximation

*Proof.* Let $f' = df/dF$. Then $f(x) = \int_{[0,x]} f'\,dF$ and by this and definition of $f_s$ we have $f_s(x) = \int_{[0,x_s]} f'(u)\,dF(u)$.

Let $F_n$ be the empirical CDF of $X$ and assuming the densities $\frac{df_s}{dF_s}$ exist, define the approximation

$$f_n(x) = \sum_s \int_{(0,x]} \frac{df_s}{dF_s}\,dF_{n,s} \tag{1}$$

$$= \sum_s \left( \frac{1}{n} \sum_i \mathbf{1}(X_{i,s} \leq x) \frac{df_s}{dF_s}(X_{i,s}) \right) \tag{2}$$

$$= H(x)^\top \gamma \tag{3}$$

where $\gamma_{i,s} = n^{-1} \frac{df_s}{dF_s}(X_{i,s})$ are collapsed into a vector $\gamma$.

Now we consider how well $f_n$ approximates $f$ in loss-based divergence.

**Lemma 1.** Let $F$, $f$, and $f_n$ be as above. Assume loss is such that the expected loss $PL(f)$ is Lipschitz in the supremum norm of $f$, i.e. $PLg - PLf = O(\|g - f\|_\infty)$. Then $PLf_n - PLf = O_P(n^{-1/2})$.

*Proof.* First we show $\|f_n - f\|_\infty = O_P(n^{-1/2})$. Write $f(x) = \sum_s \int_{(0,x]} df_s$. The difference is

$$(f_n - f)(x) = \sum_s \int_{(0,x]} \frac{df_s}{dF_s}(dF_s - dF_{n,s}) \tag{4}$$

$$= \sum_s (F - F_n)\left(\mathbf{1}(\cdot \leq x) \frac{df_s}{dF_s}(\cdot)\right) \tag{5}$$

This is an empirical process indexed by $x$ and the functions $g_x(u) = \mathbf{1}(u \leq x) \frac{df_s}{dF_s}(u)$ fall in a Donsker class (the density is a fixed function). Therefore the empirical process is uniformly bounded in probability at the rate $n^{-1/2}$ giving the desired supremum norm bound on $f_n - f$. The final result follows immediately from the Lipschitz assumption on $L$. $\square$

Now we show that this fast-converging approximation has a quickly shrinking $L_2$ norm for the coefficients.

**Lemma 2.** Let $F$, $f$ and $f_n$ be as above and assume that $df_s/dF_s$ exists and is bounded in supremum norm for each section. Then $\|\gamma\|^2 = O(n^{-1})$.

*Proof.* From the definition of $\gamma_{i,s}$ and boundedness of the densities we see $|\gamma_{i,s}| \leq b n^{-1}$. There are $d = n 2^p$ entries in the vector $\gamma$, thus $\sum_{i,s} \gamma_{i,s}^2 = O(n^{-1})$. $\square$

### A.2 Highly Adaptive Ridge

Let $\mathcal{F}(M)$ be the set of càdlàg functions of sectional variation bounded by $M$. Define the highly adaptive ridge (HAR) model

$$\mathcal{F}_n(Mn^{-1}) = \{H^\top \beta : \|\beta\|^2 \leq Mn^{-1}\} \tag{6}$$

and the empirical minimizer (HAR estimator) $\hat{f}_n = \arg\min_{f \in \mathcal{F}_n(Mn^{-1})} P_n Lf$.

**Theorem 2.** Let $F$ be the CDF of $X$ and let the true function be $f = \arg\min_{\mathcal{F}(M^*)} PLf$. Let the HAR estimator $\hat{f}_n$ be as above. Assume $L$ is such that $\mathcal{L} = \{Lf : f \in \mathcal{F}(M)\}$ remains a Donsker class and that $|df_s/dF_s| \leq b$ for all sections $s$. Then there is an $M$ such that $P(L\hat{f}_n - Lf) = O_P(n^{-1/2})$.

*Proof.* Lemma 4 directly implies that there exists an $M > 0$ for which $f_n \in \mathcal{F}_n(Mn^{-1})$ (deterministically). Use this $M$ to define the estimate $\hat{f}_n = \arg\min_{\mathcal{F}_n(Mn^{-1})} P_n Lf$. The term $P_n(L\hat{f}_n - Lf_n)$ is thus less than or equal to zero because both $\hat{f}_n, f_n \in \mathcal{F}_n(Mn^{-1})$ for every $n$ and $\hat{f}_n$ is defined as the empirical minimizer in each class.

For all functions in the model $\mathcal{F}_n(Mn^{-1})$ we have $\|\beta\|^2 \leq Mn^{-1} \implies \|\beta\|_1 \leq \sqrt{Md/n} \leq \sqrt{M 2^p}$ by an application of Cauchy-Schwarz and recalling $d = n 2^p$. For these functions the sectional variation norm is given by $\|f\|_v = \|\beta\|_1$ [2, 16, 5]. Thus $\hat{f}_n, f_n$ are of bounded sectional variation (and of course càdlàg), guaranteeing that $L\hat{f}_n - Lf_n$ falls in the Donsker class $\{Lf : f \in \mathcal{F}(\sqrt{M 2^p})\}$. Because of this, $(P - P_n)(L\hat{f}_n - Lf_n) = O_P(n^{-1/2})$ [21]. Thus

$$P(L\hat{f}_n - Lf_n) = (P - P_n)(L\hat{f}_n - Lf_n) + P_n(L\hat{f}_n - Lf_n) \tag{7}$$

$$= O_P(n^{-1/2}) \tag{8}$$

Lastly, $P(L\hat{f}_n - Lf) = P(L\hat{f}_n - Lf_n) + P(Lf_n - Lf)$ where the latter term is also $O_P(n^{-1/2})$ by Lemma 3. $\square$

**Corollary 1.** If $L$ is smooth in the sense that $\sup_{g \in \mathcal{F}_n(M)} \int (Lf - Lg)^2\,dP \leq P(Lf - Lg)$ and $\mathcal{L} = \{Lf : f \in \mathcal{F}(M)\}$ is a Donsker class that preserves the entropy of $\mathcal{F}(M)$ then $\|f_n - f\| = O_P(n^{-1/3} (\log n)^{2(p-1)/3})$.

*Proof.* Here we give a sketch of the proof. The interested reader should refer to [3] for details on this proof strategy.

Let $G_n = \{\sqrt{n}(P_n - P)l : l \in \mathcal{L}\}$ be the empirical process indexed by functions in $\mathcal{L}$. We know

$$P(L\hat{f}_n - Lf) \leq (P - P_n)(L\hat{f}_n - Lf) \leq n^{-1/2} \sup_{l \in \mathcal{L}} G_n(l). \tag{9}$$

By the smoothness of $L$ and Theorem 3 we get $\|L\hat{f}_n - Lf\|^2 = O_P(n^{-1/2})$. Therefore the above still holds if we instead take the supremum of $G_n$ over $\{l \in \mathcal{L} : \|l - Lf\|^2 \leq n^{-1/2}\}$ instead of over all of $\mathcal{L}$ because we know $\hat{f}_n$ is in this set (asymptotically, with high probability). Using a bound on the entropy integral of $\{l \in \mathcal{L} : \|l - Lf\|^2 \leq n^{-1/2}\}$ [3] we obtain

$$\sup_{\{l \in \mathcal{L} : \|l - Lf\|^2 \leq n^{-1/2}\}} G_n l = O_P(n^{-1/8} (\log n)^{p-1}) \tag{10}$$

and thus by the above we have improved the rate to $P(L\hat{f}_n - Lf) = O_P(n^{-5/8} (\log n)^{p-1})$. Now we again use the smoothness of $L$ to bound $\|L\hat{f}_n - Lf\|^2 = O_P(n^{-5/8} (\log n)^{p-1})$ and again we can iterate using a bound on the entropy integral of the smaller class $\{l \in \mathcal{L} : \|l - Lf\|^2 \leq n^{-5/8} (\log n)^{p-1}\}$, giving an even faster rate. This process iterates and the rate approaches a fixed point which is $O_P(n^{-2/3} (\log n)^{4(p-1)/3})$. A final application of the smoothness inequality and taking the square root gives the result. $\square$

The assumption on the loss required in the corollary above is satisfied by most common loss functions including mean squared error and log loss for binary outcomes [3].

**Corollary 2.** Define a data-adaptive HAR model $\mathcal{F}_n(M) = \{H^\top \beta : \|\beta\|^2 \leq M_{n,k_n^*}\}$ where $M_n$ is chosen data-adaptively from a grid of values $\mathcal{M}_n = [M_{n,1} < M_{n,2} < \ldots M_{n,K}]$ by minimizing cross-validation loss. If there is a sequence $k_n$ and a constant $\bar{M}$ such that $M \leq n^{-1} M_{n,k_n} \leq \bar{M}$ for $M$ as defined in Theorem 3 then the cv-HAR estimator $\hat{f}_n = \arg\min_{f \in \mathcal{F}_n(M_{n,k_n^*})} P_n Lf$ attains the above convergence rate.

*Proof.* This is a direct consequence of Theorem 3 and the cross-validation oracle inequality [15]. $\square$

---

## B Higher-Order HAR

### B.1 Background

In this section we present extensions of HAR that achieve even faster convergence rates under more stringent smoothness assumptions [17].

Recall that càdlàg functions of bounded sectional variation can be represented as

$$f(x) = \sum_{s_0 \subseteq \{1\ldots p\}} \int_{(0, x_{s_0}]} df_{s_0}(u_{s_0})$$

using the convention that the term for $s_0 = \emptyset$ above evaluates to $f(0)$. The reason for the $s_0$ subscript will become evident shortly.

Presume now that the Radon-Nikodym derivatives $f^{(s_0)} = df_{s_0}/d\mu_{s_0}$ exist and are themselves càdlàg functions of bounded sectional variation ($\mu_s$ is the Lebesgue measure along the section $s$). Then

$$f(x) = \sum_{s_0} \int_u \mathbf{1}(u_{s_0} \leq x_{s_0})\, f^{(s_0)}(u_{s_0})\,d\mu_{s_0}(u_{s_0})$$

$$= \sum_{s_0} \int_u \mathbf{1}(u_{s_0} \leq x_{s_0}) \left( \sum_{s_1 \subseteq s_0} \int_v \mathbf{1}(v_{s_1} \leq u_{s_1})\,df^{(s_0)}_{s_1}(v_{s_1}) \right) d\mu_{s_0}(u_{s_0})$$

$$= \sum_{s_1 \subseteq s_0 \subseteq \{1\ldots p\}} \left[ \int_{u_{s_0/s_1}} \mathbf{1}(u_{s_0/s_1} \leq x_{s_0/s_1})\,d\mu(u_{s_0/s_1}) \int_v \int_{u_{s_1}} \mathbf{1}(v_{s_1} \leq u_{s_1} \leq x_{s_1})\,d\mu_{s_1}(u_{s_1})\,df^{(s_0)}_{s_1}(v_{s_1}) \right]$$

$$= \sum_{s_1 \subseteq s_0 \subseteq \{1\ldots p\}} \int_v \underbrace{\prod_{j \in s_0/s_1} x_j \prod_{j \in s_1} (x_j - v_j)\mathbf{1}(v_j \leq x_j)}_{h_{s_0, s_1}(v, x)}\,df^{(s_0)}_{s_1}(v_{s_1})$$

$$= \sum_{s_1 \subseteq s_0 \subseteq \{1\ldots p\}} \int_v h_{s_0, s_1}(v, x)\,df^{(s_0)}_{s_1}(v_{s_1})$$

Define $\|f\|_v^1 = \sum_{s_1 \subseteq s_0 \subseteq \{1\ldots p\}} \|f^{(s_0)}_{s_1}\|$ to be the "1st order" sectional variation norm. The "0th order" norm corresponds to the standard sectional variation. We can now construct a class of functions $\mathcal{F}^1(M)$ to be those satisfying the above representation and which have $\|f\|_v^1 \leq M$.

### B.2 Estimator

Let $H^1$ denote the set of 1st order spline basis functions of the form

$$h_{i,s_0,s_1}(x) = \prod_{j \in s_0/s_1} x_j \prod_{j \in s_1} \mathbf{1}(X_{i,j} \leq x_j)(x_j - X_{i,j})$$

indexed by $i \in 1 \ldots n$ and $s_1 \subseteq s_0 \subseteq \{1 \ldots p\}$. The notation $h_{i,s_0,s_1}(x)$ abbreviates $h_{s_0,s_1}(X_i, x)$ with $X_i$ playing the role of $v$ above. There are $d = n 3^p$ of these bases (again double counting intercepts, etc.). Our 1st order HAR estimator is

$$\hat{f}_n = \arg\min_{f \in \mathcal{F}_n(M)} P_n Lf$$

$$\mathcal{F}_n(M) = \left\{ H^1(x)^\top \beta \;\text{s.t.}\; \|\beta\|^2 \leq M \right\}$$

which is completely analogous to the "0th order" HAR presented previously ($H = H^0$) except with a different set of basis functions.

### B.3 Convergence Rate

Here we prove a rate result analogous to Theorem 1 for 1st-order HAR. The proof follows exactly the same outline: first we construct an oracle approximation $f_n$ with a fast $L_2$ rate. We then show it lives in the HAR model and the rest follows with standard empirical process arguments.

Let $F_n$ be the empirical CDF of $X$ and assuming the densities $\frac{df^{(s_0)}_{s_1}}{dF_{s_1}}$ exist, define the approximation

$$f_n(x) = \sum_{s_1 \subseteq s_0 \subseteq \{1\ldots p\}} \int_v h_{s_0,s_1}(v, x) \frac{df^{(s_0)}_{s_1}}{dF_{s_1}}\,dF_{n,s_1}(v_{s_1}) \tag{11}$$

$$= \sum_{s_1 \subseteq s_0 \subseteq \{1\ldots p\}} \frac{1}{n} \sum_i h_{i,s_0,s_1}(x) \frac{df^{(s_0)}_{s_1}}{dF_{s_1}}(X_{i,s_1}) \tag{12}$$

$$= H^1(x)^\top \gamma \tag{13}$$

where $\gamma_{i,s_0,s_1} = n^{-1} \frac{df^{(s_0)}_{s_1}}{dF_{s_1}}(X_{i,s_1})$ are collapsed into a vector $\gamma$.

**Lemma 3.** Let $F$, $f$, and $f_n$ be as above. Assume loss is such that the expected loss $PL(f)$ is Lipschitz in the supremum norm of $f$, i.e. $PLg - PLf = O(\|g - f\|_\infty)$. Then $PLf_n - PLf = O_P(n^{-1/2})$.

*Proof.* First we show $\|f_n - f\|_\infty = O_P(n^{-1/2})$.

$$(f_n - f)(x) = \sum_{s_1 \subseteq s_0 \subseteq \{1\ldots p\}} \int_v h_{s_0,s_1}(v, x) \frac{df^{(s_0)}_{s_1}}{dF_{s_1}}(dF_{s_1} - dF_{n,s_1})(v_{s_1}) \tag{14}$$

$$= \sum_{s_1 \subseteq s_0 \subseteq \{1\ldots p\}} (F - F_n)_{s_1} \left( h_{s_0,s_1}(\cdot, x) \frac{df^{(s_0)}_{s_1}}{dF_{s_1}}(\cdot) \right) \tag{15}$$

This is an empirical process indexed by $x$ and the functions $g_x(v) = h_{s_0,s_1}(v, x) \frac{df^{(s_0)}_{s_1}}{dF_{s_1}}(v)$ fall in a Donsker class. Therefore the empirical process is uniformly bounded in probability at the rate $n^{-1/2}$ giving the desired supremum norm bound on $f_n - f$. The final result follows immediately from the Lipschitz assumption on $L$. $\square$

**Lemma 4.** Let $F$, $f$ and $f_n$ be as above and assume that $\frac{df^{(s_0)}_{s_1}}{dF_{s_1}}$ exists and is bounded in supremum norm for each $s_1 \subseteq s_0 \subseteq \{1 \ldots p\}$. Then $\|\gamma\|^2 = O(n^{-1})$.

*Proof.* From the definition of $\gamma_{i,s_0,s_1}$ and boundedness of the densities we see $|\gamma_{i,s_0,s_1}| \leq b n^{-1}$. There are $d = n 3^p$ entries in the vector $\gamma$, thus $\sum_{i,s_0,s_1} \gamma_{i,s_0,s_1}^2 = O(n^{-1})$. $\square$

Now we proceed to our rate results for the 1st order HAR estimator. These are given without proof because the arguments exactly follow the equivalents for 0th order HAR given above.

**Theorem 3.** Let $F$ be the CDF of $X$ and let the true function be $f = \arg\min_{\mathcal{F}(M^*)} PLf$. Let the HAR estimator $\hat{f}_n$ be as above. Assume $L$ is such that $\mathcal{L} = \{Lf : f \in \mathcal{F}(M)\}$ remains a Donsker class and that $|df_s/dF_s| \leq b$ for all sections $s$. Then there is an $M$ such that $P(L\hat{f}_n - Lf) = O_P(n^{-1/2})$.

**Corollary 3.** If $L$ is smooth in the sense that $\sup_{g \in \mathcal{F}_n(M)} \int (Lf - Lg)^2\,dP \leq P(Lf - Lg)$ and $\mathcal{L} = \{Lf : f \in \mathcal{F}(M)\}$ is a Donsker class that preserves the entropy of $\mathcal{F}(M)$ then $\|f_n - f\| = O_P(n^{-1/3} (\log n)^{2(p-1)/3})$.

**Corollary 4.** Define a data-adaptive HAR model $\mathcal{F}_n(M) = \{H^\top \beta : \|\beta\|^2 \leq M_{n,k_n^*}\}$ where $M_n$ is chosen data-adaptively from a grid of values $\mathcal{M}_n = [M_{n,1} < M_{n,2} < \ldots M_{n,K}]$ by minimizing cross-validation loss. If there is a sequence $k_n$ and a constant $\bar{M}$ such that $M \leq n^{-1} M_{n,k_n} \leq \bar{M}$ for $M$ as defined in Theorem 3 then the cv-HAR estimator $\hat{f}_n = \arg\min_{f \in \mathcal{F}_n(M_{n,k_n^*})} P_n Lf$ attains the above convergence rate.

### B.4 Computation

1st order HAR can be "kernelized" in the same way as 0th order HAR:

$$H^1(x)^\top H^1(x') = \sum_i \sum_{s_1 \subseteq s_0 \subseteq \{1\ldots p\}} \prod_{j \in s_0/s_1} x_j \prod_{j \in s_1} \mathbf{1}(X_{i,j} \leq x_j)(x_j - X_{i,j})$$

$$= \sum_i \prod_{j \in s_i} \left( x_j x'_j + (x_j - X_{i,j})(x'_j - X_{i,j}) + 1 \right)$$

because of the identity

$$\sum_{s_1 \subseteq s_0 \subseteq \{1\ldots p\}} \prod_{j \in s_0/s_1} v_j \prod_{j \in s_1} u_j = \prod_{j=1}^p (u_j + v_j + 1)$$
