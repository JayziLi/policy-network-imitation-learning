"""
Generalized closed-form ReLU policy network for the CS540 A2 competition,
supporting any odd k (not just k=5 like Part 2's exact behavior-cloning net).

Design: for a car with k front sensors (index 0..k-1, mid = k//2 straight
ahead), we don't need an *exact* rule (unlike Part 2's Q6) -- we need good
racing behavior, so:
  - "danger to the left"  = max(s[mid-1], s[mid-2 if exists]) + small avg of
                             further-out left sensors (peripheral awareness)
  - "danger to the right" = mirror of the above
  - "clear ahead"          = s[mid] + speed_bias   (speed_bias tunes how much
                             the car prefers speeding up over turning when
                             it's a close call -- the main strategy knob)
The max(a,b) trick (max(a,b) = (a+b)/2 + |a-b|/2) is exact and only needs the
two ReLU hidden layers once; everything else is a plain weighted sum, so this
works for any k with h1 >= 13, h2 >= 3.
"""
import numpy as np


def relu(x):
    return np.maximum(x, 0.0)


def softmax_rows(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def forward(W1, W2, W3, X):
    n = X.shape[0]
    ones = np.ones((n, 1))
    a1 = relu(np.hstack([X, ones]) @ W1)
    a2 = relu(np.hstack([a1, ones]) @ W2)
    z3 = np.hstack([a2, ones]) @ W3
    return softmax_rows(z3)


def fmt_matrix(W, decimals=6):
    return "\n".join(",".join(f"{v:.{decimals}f}" for v in row) for row in W)


def fmt_net(W1, W2, W3, decimals=6):
    return fmt_matrix(W1, decimals) + "\n-----\n" + fmt_matrix(W2, decimals) + "\n-----\n" + fmt_matrix(W3, decimals)


def crafted_net_general(k, h1, h2, speed_bias=0.0, scale=100.0,
                         peripheral_weight=0.3, left_eps=0.0002, speed_eps=0.0003,
                         none_bias=-1000.0):
    assert k >= 3 and k % 2 == 1
    mid = k // 2
    left = list(range(0, mid))     # indices 0..mid-1, ordered outer->inner
    right = list(range(mid + 1, k))  # indices mid+1..k-1, ordered inner->outer

    # inner-most 1 or 2 sensors per side get the exact max() treatment;
    # the rest (if any) get averaged in as a smaller linear "peripheral" term.
    left_inner = left[-2:] if len(left) >= 2 else left[-1:]
    left_outer = left[:-2] if len(left) >= 2 else []
    right_inner = right[:2] if len(right) >= 2 else right[:1]
    right_outer = right[2:] if len(right) >= 2 else []

    h1_needed = 4 + 2 * 5  # differences/pass-throughs, generous
    h1 = max(h1, 13)
    h2 = max(h2, 3)

    W1 = np.zeros((k + 1, h1))
    col = 0
    # relu(s3) passthroughs for every sensor (used for outer averaging + inner max legs)
    passthrough_col = {}
    for i in range(k):
        W1[i, col] = 1.0
        passthrough_col[i] = col
        col += 1

    def add_pair_diff(a, b):
        nonlocal col
        c1, c2 = col, col + 1
        W1[a, c1] = 1; W1[b, c1] = -1   # relu(s_a - s_b)
        W1[b, c2] = 1; W1[a, c2] = -1   # relu(s_b - s_a)
        col += 2
        return c1, c2

    left_pair_cols = None
    if len(left_inner) == 2:
        left_pair_cols = add_pair_diff(left_inner[0], left_inner[1])
    right_pair_cols = None
    if len(right_inner) == 2:
        right_pair_cols = add_pair_diff(right_inner[0], right_inner[1])

    assert col <= h1, f"h1={h1} too small, need >= {col}"

    W2 = np.zeros((h1 + 1, h2))
    # h2[0] = "left danger" score, h2[1] = "right danger" score, h2[2] = s[mid]
    if left_pair_cols:
        a, b = left_inner
        c1, c2 = left_pair_cols
        W2[passthrough_col[a], 0] = 0.5; W2[passthrough_col[b], 0] = 0.5
        W2[c1, 0] = 0.5; W2[c2, 0] = 0.5
    else:
        W2[passthrough_col[left_inner[0]], 0] = 1.0
    for i in left_outer:
        W2[passthrough_col[i], 0] = peripheral_weight / max(len(left_outer), 1)

    if right_pair_cols:
        a, b = right_inner
        c1, c2 = right_pair_cols
        W2[passthrough_col[a], 1] = 0.5; W2[passthrough_col[b], 1] = 0.5
        W2[c1, 1] = 0.5; W2[c2, 1] = 0.5
    else:
        W2[passthrough_col[right_inner[0]], 1] = 1.0
    for i in right_outer:
        W2[passthrough_col[i], 1] = peripheral_weight / max(len(right_outer), 1)

    W2[passthrough_col[mid], 2] = 1.0

    W3 = np.zeros((h2 + 1, 4))
    W3[0, 0] = scale; W3[h2, 0] = left_eps         # turn left
    W3[1, 1] = scale; W3[h2, 1] = 0.0               # turn right
    W3[2, 2] = scale; W3[h2, 2] = speed_eps + speed_bias * scale  # speed up
    W3[h2, 3] = none_bias

    return W1, W2, W3


if __name__ == "__main__":
    # quick self-test: k=5 should reduce to behavior close to Part 2's rule
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(20, 5))
    W1, W2, W3 = crafted_net_general(5, 13, 3, speed_bias=0.0)
    P = forward(np.round(W1, 4), np.round(W2, 4), np.round(W3, 4), X)
    print(P.argmax(axis=1))
