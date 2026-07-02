"""Generate the final Q9 competition network (k=5, speed_bias=0.05) in the
4-decimal-rounded format matching the rest of the assignment, ready to paste
into the page's Q9 "First Network" / "Second Network" fields."""
import numpy as np
from competition_net import crafted_net_general, fmt_net, forward

W1, W2, W3 = crafted_net_general(5, 13, 4, speed_bias=0.05)
W1, W2, W3 = np.round(W1, 4), np.round(W2, 4), np.round(W3, 4)
net_text = fmt_net(W1, W2, W3, decimals=4)

with open("output/competition_net_final.txt", "w", newline="\n") as f:
    f.write(net_text + "\n")

print(net_text)
print()
print(f"Architecture: k=5, h1={W1.shape[1]}, h2={W2.shape[1]}")
