# Leaderboards

200 runs per strategy per scenario. Final expected regret (lost conversions vs always pulling the best arm), mean with 95% CI. ★ = 95% CI overlaps the leader's.

## Close arms (T = 50,000)

3 arms, 5.0% / 5.5% / 6.0%: hard to tell apart.

| Rank | Strategy | Mean regret | 95% CI | Median | 90th pct |
|---|---|---|---|---|---|
| 1 ★ | Thompson sampling | 66.7 | [61.0, 72.3] | 56.3 | 118.7 |
| 2 | ε-greedy 0.1 | 88.5 | [77.1, 99.8] | 44.0 | 247.4 |
| 3 | ε-greedy 0.2 | 97.1 | [87.5, 106.7] | 65.9 | 232.5 |
| 4 | explore-then-commit 5000/arm | 97.7 | [88.9, 106.6] | 75.0 | 250.0 |
| 5 | ε-greedy 0.05 | 98.6 | [83.8, 113.5] | 38.9 | 253.7 |
| 6 | explore-then-commit 1000/arm | 125.4 | [104.6, 146.3] | 15.0 | 250.0 |
| 7 | decaying ε = min(1, 30K/t) | 128.7 | [105.6, 151.9] | 9.6 | 480.7 |
| 8 | discounted Thompson γ=0.999 | 196.7 | [194.7, 198.7] | 197.1 | 214.9 |
| 9 | explore-then-commit 100/arm | 199.1 | [171.8, 226.3] | 250.0 | 498.5 |
| 10 | UCB1 | 203.6 | [202.2, 205.1] | 203.2 | 217.1 |

## Far arms (T = 50,000)

3 arms, 2% / 5% / 10%: easy to tell apart.

| Rank | Strategy | Mean regret | 95% CI | Median | 90th pct |
|---|---|---|---|---|---|
| 1 ★ | Thompson sampling | 25.3 | [24.1, 26.5] | 23.8 | 35.9 |
| 2 | decaying ε = min(1, 30K/t) | 37.0 | [28.7, 45.3] | 28.9 | 33.4 |
| 3 | explore-then-commit 1000/arm | 130.0 | [130.0, 130.0] | 130.0 | 130.0 |
| 4 | ε-greedy 0.05 | 135.9 | [127.7, 144.0] | 112.9 | 180.9 |
| 5 | explore-then-commit 100/arm | 219.3 | [121.0, 317.5] | 13.0 | 13.0 |
| 6 | ε-greedy 0.1 | 230.7 | [226.0, 235.4] | 219.8 | 255.6 |
| 7 | UCB1 | 376.5 | [372.9, 380.0] | 377.2 | 411.0 |
| 8 | discounted Thompson γ=0.999 | 396.8 | [393.7, 399.8] | 394.2 | 427.3 |
| 9 | ε-greedy 0.2 | 439.8 | [437.7, 441.8] | 436.0 | 457.1 |
| 10 | explore-then-commit 5000/arm | 650.0 | [650.0, 650.0] | 650.0 | 650.0 |

## Many arms (T = 50,000)

20 arms spread between 3% and 7%.

| Rank | Strategy | Mean regret | 95% CI | Median | 90th pct |
|---|---|---|---|---|---|
| 1 ★ | decaying ε = min(1, 30K/t) | 174.9 | [157.7, 192.0] | 103.4 | 340.8 |
| 2 ★ | ε-greedy 0.1 | 202.9 | [188.5, 217.3] | 158.7 | 357.2 |
| 3 | ε-greedy 0.05 | 217.0 | [198.1, 235.9] | 189.1 | 414.1 |
| 4 | ε-greedy 0.2 | 269.6 | [258.7, 280.6] | 243.9 | 372.5 |
| 5 | Thompson sampling | 314.5 | [307.6, 321.4] | 311.1 | 375.9 |
| 6 | explore-then-commit 100/arm | 398.6 | [349.1, 448.1] | 368.1 | 976.2 |
| 7 | explore-then-commit 1000/arm | 421.1 | [406.2, 435.9] | 343.2 | 578.7 |
| 8 | UCB1 | 709.6 | [708.1, 711.2] | 710.5 | 723.4 |
| 9 | discounted Thompson γ=0.999 | 737.9 | [736.7, 739.2] | 737.8 | 749.8 |
| 10 | explore-then-commit 5000/arm | 838.3 | [838.3, 838.3] | 838.3 | 838.3 |

## Drifting rates (T = 50,000)

A falls 6% to 3%, B rises 4% to 6.5%, C oscillates 4-6%; the best arm changes over time.

| Rank | Strategy | Mean regret | 95% CI | Median | 90th pct |
|---|---|---|---|---|---|
| 1 ★ | discounted Thompson γ=0.999 | 221.4 | [219.1, 223.7] | 220.6 | 243.2 |
| 2 ★ | Thompson sampling | 227.2 | [213.7, 240.7] | 189.7 | 366.9 |
| 3 | UCB1 | 254.6 | [251.9, 257.2] | 254.6 | 278.9 |
| 4 | ε-greedy 0.2 | 290.4 | [278.0, 302.7] | 286.9 | 397.5 |
| 5 | ε-greedy 0.1 | 313.3 | [295.9, 330.6] | 317.9 | 475.3 |
| 6 | ε-greedy 0.05 | 340.6 | [320.8, 360.5] | 350.5 | 584.3 |
| 7 | explore-then-commit 100/arm | 457.5 | [436.2, 478.9] | 603.8 | 603.8 |
| 8 | decaying ε = min(1, 30K/t) | 459.0 | [440.1, 477.8] | 446.0 | 605.2 |
| 9 | explore-then-commit 1000/arm | 548.0 | [531.0, 565.0] | 624.8 | 624.8 |
| 10 | explore-then-commit 5000/arm | 548.5 | [530.6, 566.5] | 663.2 | 663.2 |

## Small traffic (T = 2,000)

HW4 upgrade-offer rates (5% / 6.5% / 4%), only 2,000 visitors.

| Rank | Strategy | Mean regret | 95% CI | Median | 90th pct |
|---|---|---|---|---|---|
| 1 ★ | decaying ε = min(1, 30K/t) | 12.7 | [11.1, 14.3] | 6.7 | 29.5 |
| 2 ★ | ε-greedy 0.1 | 14.1 | [12.4, 15.8] | 8.1 | 30.9 |
| 3 ★ | ε-greedy 0.2 | 14.2 | [12.8, 15.5] | 9.7 | 29.1 |
| 4 ★ | ε-greedy 0.05 | 14.3 | [12.3, 16.3] | 5.8 | 32.6 |
| 5 ★ | Thompson sampling | 14.5 | [13.5, 15.4] | 13.0 | 24.2 |
| 6 ★ | explore-then-commit 100/arm | 16.2 | [14.0, 18.4] | 4.0 | 46.5 |
| 7 | discounted Thompson γ=0.999 | 17.1 | [16.3, 17.8] | 16.1 | 25.3 |
| 8 | UCB1 | 23.7 | [23.5, 23.9] | 23.6 | 25.8 |
| 9 | explore-then-commit 1000/arm | 26.7 | [26.7, 26.7] | 26.7 | 26.7 |
| 10 | explore-then-commit 5000/arm | 26.7 | [26.7, 26.7] | 26.7 | 26.7 |

## Large traffic (T = 180,000)

HW4 upgrade-offer rates, 180,000 visitors (the HW4 3D season).

| Rank | Strategy | Mean regret | 95% CI | Median | 90th pct |
|---|---|---|---|---|---|
| 1 ★ | Thompson sampling | 59.2 | [48.0, 70.3] | 50.7 | 83.2 |
| 2 | ε-greedy 0.05 | 191.0 | [170.9, 211.0] | 131.0 | 308.4 |
| 3 | explore-then-commit 5000/arm | 200.0 | [200.0, 200.0] | 200.0 | 200.0 |
| 4 | explore-then-commit 1000/arm | 239.1 | [142.0, 336.3] | 40.0 | 40.0 |
| 5 | decaying ε = min(1, 30K/t) | 274.2 | [167.7, 380.6] | 12.2 | 576.9 |
| 6 | ε-greedy 0.1 | 281.6 | [269.2, 294.1] | 249.9 | 348.1 |
| 7 | ε-greedy 0.2 | 500.3 | [495.7, 504.8] | 488.3 | 538.8 |
| 8 | UCB1 | 857.2 | [849.8, 864.6] | 856.4 | 921.1 |
| 9 | discounted Thompson γ=0.999 | 1237.1 | [1230.4, 1243.8] | 1237.2 | 1295.2 |
| 10 | explore-then-commit 100/arm | 1279.9 | [1046.5, 1513.2] | 4.0 | 4496.5 |
