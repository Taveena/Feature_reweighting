import matplotlib.pyplot as plt
import numpy as np

# Random test data
np.random.seed(19680801)


# Speech Imagery
'''
lora= [51.94,49.44,51.67,52.50]#[52.22,51.94,49.44,51.67,52.50]
s2 = [50.56,51.94,51.94,52.50]#[52.22,51.94,49.44,51.67,52.50]
s3 = [54.17,51.94,51.94,52.50]#[51.67,51.94,50.00,52.50,52.22]
s4 = [52.50,51.39,51.39,51.67]#[51.39,52.50,52.50,52.50,51.39]
s5 = [50.83,52.22,52.50,51.39]#[51.11,50.83,51.67,51.94,51.94]


#BCI
Lora=	[68.98,	67.40,	68.02,	67.44]
Dora=	[68.44,	68.33,	68.09,	67.48]
Loha=	[66.74,	66.74,	67.32,	67.90]
Lokr=	[66.98,	65.93,	66.17,	66.47]
Lora_Ens=	[67.55,	68.67,	67.48,	67.52]
AELoRA=	[68.83,	69.83,	68.29,	67.55]
#proposed = [54.17,52.22,52.50,52.50]#[52.78,54.17,52.22,52.50,52.50]
'''
'''
# Motor Imagery
lora = [67.40,68.02,67.44,66.51]#[68.98,67.40,68.02,67.44,66.51]

s2 = [67.90,67.67,68.13,67.59]#[68.44,68.33,68.09,67.48,67.40]
s3 = [68.48,67.75,68.60,67.86]#[66.74,66.74,67.32,67.90,66.90]
s4 = [69.14,68.25,67.90,68.21]#[66.98,65.93,66.17,66.47,66.47]
s5 = [68.13,68.17,68.21,66.90]#[69.21,69.14,68.25,68.60,68.21]
'''
'''
# fill with colors Kappa dataset 1 and 2
colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow', 'cyan']
all_data = [FRNet, Deep, EEGNet, Tangent_RVM, Proposed]
labels = ['FRNet', 'Deep', 'EEGNet', 'RVM','Proposed']


# fill with colors F-Measure dataset 1 and 2
colors = ['lavender', 'pink', 'lightblue', 'lightgreen', 'cyan']
all_data = [EEG_Conformer, FRNet, Deep, EEGNet, Proposed]
labels = ['Conformer','FRNet', 'Deep', 'EEGNet','Proposed']


# fill with colors Kappa dataset 3
colors = ['pink', 'lightblue', 'lightgreen', 'orange', 'cyan']
all_data = [FRNet, Deep, EEGNet, ResNet, Proposed]
labels = ['FRNet', 'Deep', 'EEGNet', 'ResNet','Proposed']
'''

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

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 7))

# rectangular box plot
bplot = ax.boxplot(all_data,
                     vert=True,  # vertical box alignment
                     patch_artist=True,  # fill with color
                     labels=labels)  # will be used to label x-ticks
#ax1.set_title('Rectangular box plot')


for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

# adding horizontal grid lines
#for ax in [ax1, ax2]:
ax.yaxis.grid(True)
ax.set_xlabel('Method', fontsize=20)
ax.set_ylabel('Accuracy', fontsize=20)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=15)
#ax.set_title('Motor Imagery')
plt.show()

#plt.savefig("graphs/ranks_result_MI.pdf") 



