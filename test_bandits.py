"""Tests for bandits.py. Run: .venv/bin/python test_bandits.py   (about 40 seconds)"""
import numpy as np

import bandits as B

ok = []


def check(name, cond, detail=""):
    ok.append(bool(cond))
    print("%s  %s%s" % ("PASS" if cond else "FAIL", name, (": " + detail) if detail else ""))


HW = B.SCENARIOS["large"]                                   # HW4 3D: 5% / 6.5% / 4%, 180,000 visitors

# 1. regret accounting
for key in ("close", "drift"):
    scn = B.SCENARIOS[key]
    _, reg, _ = B.simulate(scn, lambda R, K, T, rng: B.Oracle(R, K, T, rng, scn), R=5, seed=1)
    check(f"oracle has zero regret ({key})", np.allclose(reg, 0), f"max {reg.max():.2e}")


class AlwaysC(B.Strategy):
    def choose(self, t):
        return np.full(self.R, 2)


_, reg, _ = B.simulate(HW, AlwaysC, R=3, seed=1)
check("always showing C costs (0.065 - 0.04) x 180,000 = 4,500", np.allclose(reg[:, -1], 4500), f"{reg[0, -1]:.1f}")

# 2. random tie-breaking is uniform
rng = np.random.default_rng(0)
picks = B.argmax_random(np.tile([1.0, 3.0, 3.0, 3.0, 0.0], (60000, 1)), rng)
share = np.bincount(picks, minlength=5) / len(picks)
check("argmax ties broken uniformly", share[0] == 0 and share[4] == 0 and np.allclose(share[1:4], 1 / 3, atol=0.01), str(share.round(3)))

# 3. explore-then-commit: exact exploration, then commits; with far arms it always finds the best
scn = B.SCENARIOS["far"]
_, reg, pulls = B.simulate(scn, B.STRATEGIES["etc1000"], R=50, seed=2)
check("ETC 1000/arm on far arms: regret = 1000 x (0.08 + 0.05) = 130 in every run", np.allclose(reg[:, -1], 130), f"{reg[:, -1].min():.1f}-{reg[:, -1].max():.1f}")
check("ETC pulls each losing arm exactly 1000 times", np.allclose(pulls[:, :2], 1000))

# 4. UCB1 index on a hand example: t = 100, arm with 3/20 -> 0.15 + sqrt(2 ln 100 / 20) = 0.8286
u = B.UCB1(1, 2, 100, np.random.default_rng(0))
u.n[:] = [[20, 80]]
u.s[:] = [[3, 40]]
i0, i1 = 0.15 + np.sqrt(2 * np.log(100) / 20), 0.5 + np.sqrt(2 * np.log(100) / 80)
check("UCB1 picks the arm with the larger index", u.choose(100)[0] == int(i1 > i0) and abs(i0 - 0.8286) < 1e-4, "indexes %.4f vs %.4f" % (i0, i1))

# 5. epsilon-greedy explores eps of the time, uniformly over all arms
e = B.EpsGreedy(200000, 3, 10, np.random.default_rng(3), 0.1)
e.n[:] = 10
e.s[:] = [2, 8, 5]
c = np.bincount(e.choose(5), minlength=3) / 200000
check("ε-greedy 0.1: leader gets 0.9 + 0.1/3", abs(c[1] - (0.9 + 0.1 / 3)) < 0.003, str(c.round(4)))
check("decaying ε schedule", B.DecayingEps(1, 3, 10, None, 30).epsilon(0) == 1 and abs(B.DecayingEps(1, 3, 10, None, 30).epsilon(8999) - 0.01) < 1e-12)

# 6. discounted Thompson forgets
d = B.DiscountedThompson(1, 2, 10, np.random.default_rng(0), gamma=0.5)
d.update(np.array([0]), np.array([1.0]), 0)
d.update(np.array([1]), np.array([0.0]), 1)
check("discounted Thompson decays old evidence", np.allclose(d.n, [[0.5, 1.0]]) and np.allclose(d.s, [[0.5, 0.0]]))

# 7. reproducibility
a = B.simulate(B.SCENARIOS["small"], B.STRATEGIES["ts"], R=20, seed=11)[1]
b = B.simulate(B.SCENARIOS["small"], B.STRATEGIES["ts"], R=20, seed=11)[1]
check("same seed -> identical results", np.array_equal(a, b))

# 8. cross-check with my HW4 3D (200 seasons, separate code and random streams)
#    HW4 precision_1E_3D.py: eps-greedy 0.1 mean 280.7 (sd 90.2); Thompson 54.2 (sd 23.3)
finals = {}
for key, hw_mean, hw_sd in (("eps0.1", 280.7, 90.2), ("ts", 54.2, 23.3)):
    reg = finals[key] = B.simulate(HW, B.STRATEGIES[key], R=200, seed=99)[1][:, -1]
    se = np.sqrt(reg.var(ddof=1) / 200 + hw_sd ** 2 / 200)
    z = (reg.mean() - hw_mean) / se
    check(f"HW4 3D cross-check, {key}: mean regret matches HW4 within sampling error", abs(z) < 3,
          f"{reg.mean():.1f} vs HW4 {hw_mean} (z = {z:.2f})")
# and the theory floor: eps-greedy explores 10% uniformly -> at least 0.1 x (0.015 + 0 + 0.025)/3 x 180,000 = 240 on average
floor = 0.1 * (0.015 + 0 + 0.025) / 3 * 180000
check("ε-greedy 0.1 mean regret sits above its theoretical floor of 240", finals["eps0.1"].mean() > floor, f"{finals['eps0.1'].mean():.1f} > {floor:.0f}")
print("\n%d/%d PASS" % (sum(ok), len(ok)))
raise SystemExit(0 if all(ok) else 1)
