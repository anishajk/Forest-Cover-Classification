# import matplotlib.pyplot as plt

# f1_scores = {'mlp': 0.4565187667430023, 'rf': 0.5632988263206719, 'lr': 0.08908843599125474, 'knn': 0.9656734748640252}
# accuracies = {'mlp': 0.6598107470425276, 'rf': 0.7528, 'lr': 0.3796065438754118, 'knn': 0.9656734748640252}

# # create a figure with two subplots
# fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

# # create a bar plot for f1_scores
# ax1.bar(f1_scores.keys(), f1_scores.values())
# ax1.set_title('F1 Scores')
# ax1.set_xlabel('Models')
# ax1.set_ylabel('Scores')

# # create a bar plot for accuracies
# ax2.bar(accuracies.keys(), accuracies.values())
# ax2.set_title('Accuracies')
# ax2.set_xlabel('Models')
# ax2.set_ylabel('Scores')

# # adjust the layout and show the plots
# plt.tight_layout()
# plt.show()
# plt.savefig('results/plots/comparison.png')

import matplotlib.pyplot as plt
import numpy as np

f1_scores = {'mlp': 0.4565187667430023, 'rf': 0.5632988263206719, 'lr': 0.08908843599125474, 'knn': 0.9167}
accuracies = {'mlp': 0.6598107470425276, 'rf': 0.7528, 'lr': 0.3796065438754118, 'knn': 0.9656734748640252}

# create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))

# create a bar plot for f1_scores
colors = np.array([0.2, 0.3, 0.1, 0.9])
bar1 = ax1.bar(f1_scores.keys(), f1_scores.values())
ax1.set_title('F1 Scores')
ax1.set_xlabel('Models')
ax1.set_ylabel('Scores')
for i, v in enumerate(f1_scores.values()):
    ax1.text(i, v, str(round(v,2)), ha='center', fontweight='bold')

# create a bar plot for accuracies
colors = np.array([0.2, 0.4, 0.6, 0.8])
bar2 = ax2.bar(accuracies.keys(), accuracies.values())
ax2.set_title('Accuracies')
ax2.set_xlabel('Models')
ax2.set_ylabel('Scores')
for i, v in enumerate(accuracies.values()):
    ax2.text(i, v, str(round(v,2)), ha='center', fontweight='bold')

# adjust the layout and show the plots
plt.tight_layout()
plt.show()
plt.savefig('results/plots/comparison.png')
