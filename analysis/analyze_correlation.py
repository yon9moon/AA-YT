import json
import numpy as np

# load data
with open("/root/AA-YT/analysis/output.json", "r") as f:
    data = json.load(f)

abs_errs = np.array(data.get("absolute_errors", []), dtype=float)
truths = np.array(data.get("truth", []), dtype=float)

if abs_errs.size == 0 or truths.size == 0:
    print("empty arrays, nothing to do")
    exit(1)

# ensure shapes match
if abs_errs.shape != truths.shape:
    print("warning: shapes of absolute_errors and truth differ", abs_errs.shape, truths.shape)
    # if one is longer than the other, truncate to min length
    n = min(len(abs_errs), len(truths))
    abs_errs = abs_errs[:n]
    truths = truths[:n]

# select fields
truth_first2 = truths[:, :2]
err_last2 = abs_errs[:, 2:]

# compute correlations
print("Correlation coefficients between individual components:")
for i in range(2):
    for j in range(2):
        coeff = np.corrcoef(truth_first2[:, i], err_last2[:, j])[0, 1]
        print(f"truth[{i}] vs err[{j+2}] -> {coeff:.6f}")

# compute norms
truth_norm = np.linalg.norm(truth_first2, axis=1)
err_norm = np.linalg.norm(err_last2, axis=1)
coeff_norm = np.corrcoef(truth_norm, err_norm)[0, 1]
print(f"\nCorrelation between norms (truth first2 vs err last2): {coeff_norm:.6f}")

# also maybe scatter plot? (optional)
try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 4))
    plt.scatter(truth_norm, err_norm, alpha=0.3)
    plt.xlabel("norm of truth first2")
    plt.ylabel("norm of absolute_errors last2")
    plt.title("scatter truth vs error")
    plt.grid(True)
    plt.savefig("correlation_scatter.png")
    print("Scatter plot saved to correlation_scatter.png")
except ImportError:
    pass
