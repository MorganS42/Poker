import math
import random
import itertools
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# strength to name
hand_category = {
    8: "Straight Flush",
    7: "Four of a Kind",
    6: "Flush",
    5: "Full House",
    4: "Straight",
    3: "Three of a Kind",
    2: "Two Pair",
    1: "Pair",
    0: "High Card"
}

HAND_SIZE = 2
COMMUNITY_SIZE = 5
players = 6

suits = ["S", "C", "D", "H"]
values = ["6", "7", "8", "9", "T", "J", "Q", "K", "A"]

def reset():
    deck = []
    for suit in suits:
        for value in values:
            deck.append(value + suit)
    return deck

def deal():
    deck = reset()
    random.shuffle(deck)
    hands = [deck[2 * i : 2 * i + 2] for i in range(players)]
    community = deck[2 * players: 2 * players + COMMUNITY_SIZE]
    return hands, community

rank_map = {r : i for i, r in enumerate("..6789TJQKA")}

def card_rank(c):
    return rank_map[c[0]]

def straight_high_shortdeck(ranks):
    r = sorted(set(ranks))
    if len(r) < 5:
        return 0

    for i in range(len(r) - 4):
        w = r[i:i+5]
        if w == list(range(w[0], w[0] + 5)):
            return w[-1]

    # A 6 7 8 9 Straight
    wheel_set = {14, 6, 7, 8, 9}
    if wheel_set.issubset(r):
        return 9

    return 0

rank5memoized = {}

def rank5(cards):
    if cards in rank5memoized.keys():
        return rank5memoized.get(cards)

    ranks = [card_rank(c) for c in cards]
    suits_ = [c[1] for c in cards]

    flush = len(set(suits_)) == 1
    sh = straight_high_shortdeck(ranks)

    if flush and sh:
        out = (8, sh, sorted(ranks, reverse=True))
        rank5memoized[cards] = out
        return out

    counts = {}
    for v in ranks:
        counts[v] = counts.get(v, 0) + 1
    items = sorted(((cnt, val) for val, cnt in counts.items()), reverse=True)

    if items[0][0] == 4:
        quad = items[0][1]
        kicker = max(v for v in ranks if v != quad)
        out = (7, quad, kicker)
        rank5memoized[cards] = out
        return out

    if items[0][0] == 3 and items[1][0] == 2:
        fullhouse_rank = items[0][1]
        pair_rank = items[1][1]
        fullhouse_tuple = (5, fullhouse_rank, pair_rank)
        fh = fullhouse_tuple
    else:
        fh = None

    if flush:
        out = (6, sorted(ranks, reverse=True))
        rank5memoized[cards] = out
        return out

    if fh is not None:
        out = fh
        rank5memoized[cards] = out
        return out

    if sh:
        out = (4, sh, sorted(ranks, reverse=True))
        rank5memoized[cards] = out
        return out

    if items[0][0] == 3:
        trips = items[0][1]
        kickers = sorted((v for v in ranks if v != trips), reverse=True)[:2]
        out = (3, trips, kickers)
        rank5memoized[cards] = out
        return out

    if items[0][0] == 2 and items[1][0] == 2:
        ph = max(items[0][1], items[1][1])
        pl = min(items[0][1], items[1][1])
        kicker = max(v for v in ranks if v != ph and v != pl)
        out = (2, ph, pl, kicker)
        rank5memoized[cards] = out
        return out

    if items[0][0] == 2:
        pair = items[0][1]
        kickers = sorted((v for v in ranks if v != pair), reverse=True)[:3]
        out = (1, pair, kickers)
        rank5memoized[cards] = out
        return out

    out = (0, sorted(ranks, reverse=True))
    rank5memoized[cards] = out
    return out

def best5of7(cards7):
    best = None
    best_combo = None
    for combo in itertools.combinations(cards7, 5):
        r = rank5(combo)
        if best is None or r > best:
            best = r
            best_combo = combo
    return best, list(best_combo)

def evaluate(hands, community):
    out = []
    for h in hands:
        out.append(best5of7(h + community))
    return out

def winners(hands, community):
    evals = evaluate(hands, community)
    best = max(e[0] for e in evals)
    idxs = [i for i, e in enumerate(evals) if e[0] == best]
    return idxs, [evals[i][1] for i in idxs], best

def canonical_starting_hand(two_cards):
    a, b = two_cards
    ra, rb = a[0], b[0]
    sa, sb = a[1], b[1]

    if rank_map[ra] < rank_map[rb]:
        ra, rb, sa, sb = rb, ra, sb, sa

    if ra == rb:
        return ra + rb

    suited = (sa == sb)
    return ra + rb + ("s" if suited else "o")

def simulate(n_sims):
    win_counts = Counter()
    dealt_counts = Counter()

    for _ in range(n_sims):
        hands, community = deal()

        for h in hands:
            dealt_counts[canonical_starting_hand(h)] += 1

        win_idxs, _, _ = winners(hands, community)
        for i in win_idxs:
            label = canonical_starting_hand(hands[i])
            win_counts[label] += 1

    return win_counts, dealt_counts

# win_counts, dealt_counts = simulate(200000)

# strength = {
#     label: win_counts[label] / dealt_counts[label]
#     for label in dealt_counts
# }

# # # Top 10
# # relevant = sorted(strength.items(), key=lambda x: x[1], reverse=True)[:10]

# # Bottom 10
# relevant = sorted(strength.items(), key=lambda x: x[1], reverse=False)[:10]

# labels = [label for label, _ in relevant]
# vals   = [val for _, val in relevant]

# plt.figure()
# plt.bar(labels, vals)
# plt.xticks(rotation = 45, ha = "right")
# plt.title("Estimated win rate")
# plt.ylabel("Win rate when dealt")
# plt.tight_layout()
# plt.show()

# def simulate_hand_categories(n_sims):
#     category_counts = Counter()

#     for _ in range(n_sims):
#         hands, community = deal()
#         win_idxs, _, best_rank = winners(hands, community)

#         category = best_rank[0]
#         category_counts[hand_category[category]] += 1

#     return category_counts

# def simulate_all_hand_categories(n_sims):
#     category_counts = Counter()

#     for _ in range(n_sims):
#         hands, community = deal()

#         for h in hands:
#             best_rank, _ = best5of7(h + community)
#             category = best_rank[0]

#             category_counts[hand_category[category]] += 1

#     return category_counts

# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
# import matplotlib.pyplot as plt

# sims = 10_00

# player_values = list(range(2, 11))

# order = [
#     "Straight",
#     "Full House",
#     "Two Pair",
#     "Three of a Kind",
#     "Flush",
#     "Pair",
#     "Four of a Kind",
#     "Straight Flush",
#     "High Card"
# ]

# labels = order

# num_players = len(player_values)
# num_cats = len(labels)

# data = np.zeros((num_players, num_cats), dtype=float)

# for i, players in enumerate(player_values):
#     category_counts = simulate_hand_categories(sims)

#     for j, label in enumerate(labels):
#         # data[i, j] = 0 if category_counts[label] == 0 else -math.log(1 - category_counts[label] / sims)
#         data[i, j] = category_counts[label] / sims
#     print(players)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")

# x_pos = np.arange(num_cats)
# y_pos = np.arange(num_players)

# xpos, ypos = np.meshgrid(x_pos, y_pos)
# xpos = xpos.ravel()
# ypos = ypos.ravel()
# zpos = np.zeros_like(xpos)

# dx = 0.5 * np.ones_like(xpos)
# dy = 0.5 * np.ones_like(ypos)
# dz = data.ravel()

# norm = plt.Normalize(dz.min(), dz.max())  
# colors = cm.viridis(norm(dz))

# ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors, shade=True, zsort='average')

# ax.set_title("Winning Hand Categories")
# # ax.set_xlabel("Hand category")
# ax.set_ylabel("Number of players")
# ax.set_zlabel("Proportion")

# ax.set_xticks(np.arange(num_cats) + 0.25)
# ax.set_xticklabels(labels, rotation=45, ha="right")

# ax.set_yticks(np.arange(num_players) + 0.25)
# ax.set_yticklabels(player_values)

# plt.tight_layout()
# plt.show()

sims = 1_000_000
win_counts, dealt_counts = simulate(sims)

strength = {
    label: win_counts[label] / dealt_counts[label]
    for label in dealt_counts
}

rank_order = values[::-1]
n = len(rank_order)

grid = np.zeros((n, n))
labels_grid = [[None]*n for _ in range(n)]

for i, r1 in enumerate(rank_order):
    for j, r2 in enumerate(rank_order):
        # diagonal
        if r1 == r2:
            key = r1 + r2
        # above
        elif i < j:
            key = r1 + r2 + "s"
        # below
        else:
            hi, lo = (r1, r2) if rank_map[r1] > rank_map[r2] else (r2, r1)
            key = hi + lo + "o"

        val = strength.get(key, np.nan)
        grid[i, j] = val
        labels_grid[i][j] = key

flat_vals = grid[~np.isnan(grid)].ravel()
sorted_vals = np.sort(flat_vals)

def to_percentile(x):
    idx = np.searchsorted(sorted_vals, x, side="right")
    return 100 * idx / len(sorted_vals)

percentile_grid = np.vectorize(to_percentile)(grid)

fig, ax = plt.subplots(figsize=(8, 7))

display = percentile_grid

im = ax.imshow(display, cmap="coolwarm", origin="upper")

ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(rank_order)
ax.set_yticklabels(rank_order)

ax.set_xlabel("Second card rank")
ax.set_ylabel("First card rank")
ax.set_title("Starting hand strength (percentile)")

plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

for i in range(n):
    for j in range(n):
        v = display[i, j]
        if np.isnan(v):
            continue
        hand_label = labels_grid[i][j]
        ax.text(
            j, i,
            f"{hand_label}\n{v:.1f}%",
            ha="center", va="center", fontsize=7, color="black"
        )

fig.colorbar(im, ax=ax, label="Percentile")
plt.tight_layout()
plt.show()
