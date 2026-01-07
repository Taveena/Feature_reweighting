import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#ASU
fbcsp = [0.32,	0.27,	0.31,	0.15,	0.39,	0.44]
deep = [0.03,	0.09,	0.19,	0.08,	0.08,	0.11]
shallow = [0.49,	0.58,	0.58,	0.47,	0.60,	0.47]
eegnet = [0.09,	0.06,	0.16,	0.05,	0.04,	0.11]
eegnetfusion = [0.06,	0.08,	0.17,	0.12,	-0.01,	0.12]
mieegnet = [0.51,	0.50,	0.51,	0.50,	0.50,	0.60]
tssefnet = [0.17,	0.25,	0.32,	0.25,	0.27,	0.27]
lmda= [0.09,	0.14,	0.27,	0.10,	0.09,	0.26]
proposed= [0.60,	0.60,	0.60,	0.57,	0.64,	0.66]

# fill with colors F-Measure dataset 3
colors = ['lavender', 'pink', 'lightblue', 'lightgreen', 'plum', 'orange', 'salmon', 'lightseagreen', 'cyan'] #'orange', lightblue
all_data = [deep, eegnet, eegnetfusion, tssefnet, lmda, fbcsp, mieegnet, shallow, proposed]
labels = ['Deep', 'EEGNet', 'EEGNet Fusion', 'TS-SEFFNet', 'LMDA', 'FBCSP', 'MI-EEGNet', 'Shallow', 'Proposed']

# Create a DataFrame for seaborn
data = pd.DataFrame({
    'Method': np.repeat(labels, [len(deep), len(eegnet), len(eegnetfusion), len(tssefnet), len(lmda), len(fbcsp), len(mieegnet), len(shallow), len(proposed)]),
    'Accuracy': deep + eegnet + eegnetfusion + tssefnet + lmda + fbcsp + mieegnet + shallow + proposed
})

# Plotting the violin plot
plt.figure(figsize=(10, 7))
sns.violinplot(x='Method', y='Accuracy', data=data, palette=['lavender', 'pink', 'lightblue', 'lightgreen', 'plum', 'orange', 'salmon', 'lightseagreen', 'cyan'])

# Add labels and customize ticks
plt.grid(True)
plt.xlabel('Method', fontsize=20)
plt.ylabel('Accuracy', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Show the plot
plt.show()

