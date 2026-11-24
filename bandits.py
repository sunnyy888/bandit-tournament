"""Bandit strategies and scenarios, vectorized across independent runs.

Every strategy sees R independent runs at once: state arrays have shape (R, K). Each time step every run
picks one arm, gets a Bernoulli reward, and updates. Regret is the expected (pseudo-)regret used in
DSO 429 HW4: sum over steps of (best true rate at that step - true rate of the arm pulled), so it measures
choices, not luck.
"""
import numpy as np

# ---------------------------------------------------------------- scenarios
def _many_arms():
    r = np.random.default_rng(7)                          # fixed draw so the scenario is the same every time
    p = np.sort(r.uniform(0.03, 0.07, 20))[::-1].copy()
    return p


def drift_rates(t, T):
    """Three arms whose rates drift: A falls 6% -> 3%, B rises 4% -> 6.5%, C oscillates around 5%."""
    s = t / T
    return np.array([0.06 - 0.03 * s, 0.04 + 0.025 * s, 0.05 + 0.01 * np.sin(2 * np.pi * 2 * s)])


SCENARIOS = {
    "close":       {"title": "Close arms", "p": np.array([0.050, 0.055, 0.060]), "T": 50_000,
                    "desc": "3 arms, 5.0% / 5.5% / 6.0%: hard to tell apart"},
    "far":         {"title": "Far arms", "p": np.array([0.02, 0.05, 0.10]), "T": 50_000,
                    "desc": "3 arms, 2% / 5% / 10%: easy to tell apart"},
    "many":        {"title": "Many arms", "p": _many_arms(), "T": 50_000,
                    "desc": "20 arms spread between 3% and 7%"},
    "drift":       {"title": "Drifting rates", "p": None, "T": 50_000,
                    "desc": "A falls 6% to 3%, B rises 4% to 6.5%, C oscillates 4-6%; the best arm changes over time"},
    "small":       {"title": "Small traffic", "p": np.array([0.050, 0.065, 0.040]), "T": 2_000,
                    "desc": "HW4 upgrade-offer rates (5% / 6.5% / 4%), only 2,000 visitors"},
    "large":       {"title": "Large traffic", "p": np.array([0.050, 0.065, 0.040]), "T": 180_000,
                    "desc": "HW4 upgrade-offer rates, 180,000 visitors (the HW4 3D season)"},
}


def rates_at(scn, t):
    return drift_rates(t, scn["T"]) if scn["p"] is None else scn["p"]


def n_arms(scn):
    return 3 if scn["p"] is None else len(scn["p"])


# ---------------------------------------------------------------- strategies
def argmax_random(x, rng):
    """Row-wise argmax with ties broken uniformly at random."""
    noise = rng.random(x.shape)
    m = x.max(axis=1, keepdims=True)
    return np.argmax(np.where(x == m, noise, -1.0), axis=1)


class Strategy:
    name = "base"

    def __init__(self, R, K, T, rng):
        self.R, self.K, self.T, self.rng = R, K, T, rng
        self.n = np.zeros((R, K))
        self.s = np.zeros((R, K))

    def means(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.n > 0, self.s / np.maximum(self.n, 1), np.inf)   # untried arms look best

    def update(self, arm, reward, t):
        idx = np.arange(self.R)
        self.n[idx, arm] += 1
        self.s[idx, arm] += reward


class EpsGreedy(Strategy):
    def __init__(self, R, K, T, rng, eps):
        super().__init__(R, K, T, rng)
        self.eps = eps
        self.name = f"ε-greedy {eps}"

    def epsilon(self, t):
        return self.eps

    def choose(self, t):
        greedy = argmax_random(self.means(), self.rng)
        explore = self.rng.random(self.R) < self.epsilon(t)
        rand = self.rng.integers(0, self.K, self.R)        # explore uniformly over all arms (leader included)
        return np.where(explore, rand, greedy)


class DecayingEps(EpsGreedy):
    def __init__(self, R, K, T, rng, c=30):
        super().__init__(R, K, T, rng, eps=None)
        self.c = c
        self.name = f"decaying ε = min(1, {c}K/t)"

    def epsilon(self, t):
        return min(1.0, self.c * self.K / (t + 1))


class UCB1(Strategy):
    name = "UCB1"

    def choose(self, t):
        with np.errstate(divide="ignore", invalid="ignore"):
            bonus = np.sqrt(2 * np.log(max(t, 1)) / self.n)
        idx = np.where(self.n > 0, self.s / np.maximum(self.n, 1) + bonus, np.inf)
        return argmax_random(idx, self.rng)


class Thompson(Strategy):
    name = "Thompson sampling"

    def choose(self, t):
        return argmax_random(self.rng.beta(1 + self.s, 1 + self.n - self.s), self.rng)


class DiscountedThompson(Strategy):
    """Thompson with exponential forgetting: old evidence decays by gamma each step (bonus strategy for drift)."""

    def __init__(self, R, K, T, rng, gamma=0.999):
        super().__init__(R, K, T, rng)
        self.gamma = gamma
        self.name = f"discounted Thompson γ={gamma}"

    def choose(self, t):
        return argmax_random(self.rng.beta(1 + self.s, 1 + self.n - self.s), self.rng)

    def update(self, arm, reward, t):
        self.n *= self.gamma
        self.s *= self.gamma
        super().update(arm, reward, t)


class ExploreThenCommit(Strategy):
    def __init__(self, R, K, T, rng, m):
        super().__init__(R, K, T, rng)
        self.m = m
        self.name = f"explore-then-commit {m}/arm"
        self.commit = None

    def choose(self, t):
        if t < self.m * self.K:
            return np.full(self.R, t % self.K)              # round-robin exploration
        if self.commit is None:
            self.commit = argmax_random(self.s / np.maximum(self.n, 1), self.rng)
        return self.commit


class Oracle(Strategy):
    """Always pulls the truly best arm at each step (regret 0 by definition; used in tests)."""
    name = "oracle"

    def __init__(self, R, K, T, rng, scn):
        super().__init__(R, K, T, rng)
        self.scn = scn

    def choose(self, t):
        return np.full(self.R, int(np.argmax(rates_at(self.scn, t))))


STRATEGIES = {
    "eps0.05": lambda R, K, T, rng: EpsGreedy(R, K, T, rng, 0.05),
    "eps0.1": lambda R, K, T, rng: EpsGreedy(R, K, T, rng, 0.1),
    "eps0.2": lambda R, K, T, rng: EpsGreedy(R, K, T, rng, 0.2),
    "decay": lambda R, K, T, rng: DecayingEps(R, K, T, rng, 30),
    "ucb1": lambda R, K, T, rng: UCB1(R, K, T, rng),
    "ts": lambda R, K, T, rng: Thompson(R, K, T, rng),
    "etc100": lambda R, K, T, rng: ExploreThenCommit(R, K, T, rng, 100),
    "etc1000": lambda R, K, T, rng: ExploreThenCommit(R, K, T, rng, 1000),
    "etc5000": lambda R, K, T, rng: ExploreThenCommit(R, K, T, rng, 5000),
    "dts": lambda R, K, T, rng: DiscountedThompson(R, K, T, rng, 0.999),
}


# ---------------------------------------------------------------- simulation
def checkpoints(T, n=100):
    return np.unique(np.linspace(1, T, n).astype(int))


def simulate(scn, make, R=200, seed=0, record=100):
    """Run R independent runs of one strategy on one scenario. Returns (times, regret[R, len(times)], pulls[R, K])."""
    rng = np.random.default_rng(seed)
    K, T = n_arms(scn), scn["T"]
    strat = make(R, K, T, rng) if callable(make) else make
    times = checkpoints(T, record)
    out = np.zeros((R, len(times)))
    cum = np.zeros(R)
    pulls = np.zeros((R, K))                               # actual pull counts (strat.n may be discounted)
    rows = np.arange(R)
    j = 0
    static = scn["p"] is not None
    p = scn["p"] if static else None
    for t in range(T):
        if not static:
            p = rates_at(scn, t)
        arm = strat.choose(t)
        pa = p[arm]
        reward = (rng.random(R) < pa).astype(float)
        cum += p.max() - pa
        pulls[rows, arm] += 1
        strat.update(arm, reward, t)
        if t + 1 == times[j]:
            out[:, j] = cum
            j += 1
            if j == len(times):
                break
    return times, out, pulls
