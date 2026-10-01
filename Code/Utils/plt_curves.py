from os.path import join

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"


def make_performance_plots(path):
    plt.plot(np.loadtxt(f'{join(path, "train")}.csv', skiprows=1, usecols=(1,), delimiter=','),
             np.loadtxt(f'{join(path, "train")}.csv', skiprows=1, usecols=(2,), delimiter=','), 'b', label='Entraînement (sans attention)')
    plt.plot(np.loadtxt(f'{join(path, "val")}.csv', skiprows=1, usecols=1, delimiter=','),
             np.loadtxt(f'{join(path, "val")}.csv', skiprows=1, usecols=2, delimiter=','), 'b--', label='Évaluation (sans attention)')
    plt.plot(np.loadtxt(f'{join(path, "att_train")}.csv', skiprows=1, usecols=(1,), delimiter=','),
             np.loadtxt(f'{join(path, "att_train")}.csv', skiprows=1, usecols=(2,), delimiter=','), 'r', label='Entraînement (avec attention)')
    plt.plot(np.loadtxt(f'{join(path, "att_val")}.csv', skiprows=1, usecols=1, delimiter=','),
             np.loadtxt(f'{join(path, "att_val")}.csv', skiprows=1, usecols=2, delimiter=','), 'r--', label='Évaluation (avec attention)')
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Accuracy', fontsize=13)
    plt.legend(fontsize=13)

    plt.tight_layout()
    plt.show()


def compare_plot(path):
    plt.plot(np.loadtxt(f'{join(path, "train")}.csv', skiprows=1, usecols=(1,), delimiter=','),
             np.loadtxt(f'{join(path, "train")}.csv', skiprows=1, usecols=(2,), delimiter=','), 'b',
             label='Entraînement (rotations)')
    plt.plot(np.loadtxt(f'{join(path, "val")}.csv', skiprows=1, usecols=1, delimiter=','),
             np.loadtxt(f'{join(path, "val")}.csv', skiprows=1, usecols=2, delimiter=','), 'b--',
             label='Évaluation (rotations)')
    plt.plot(np.loadtxt(f'{join(path, "loss_train_noDA")}.csv', skiprows=1, usecols=(1,), delimiter=','),
             np.loadtxt(f'{join(path, "loss_train_noDA")}.csv', skiprows=1, usecols=(2,), delimiter=','), 'r',
             label='Entraînement (sans aug. des données)')
    plt.plot(np.loadtxt(f'{join(path, "loss_val_noDA")}.csv', skiprows=1, usecols=1, delimiter=','),
             np.loadtxt(f'{join(path, "loss_val_noDA")}.csv', skiprows=1, usecols=2, delimiter=','), 'r--',
             label='Évaluation (sans aug. des données)')
    plt.plot(np.loadtxt(f'{join(path, "train_da6")}.csv', skiprows=1, usecols=(1,), delimiter=','),
             np.loadtxt(f'{join(path, "train_da6")}.csv', skiprows=1, usecols=(2,), delimiter=','), 'g',
             label='Entraînement (rotations+réflexions)')
    plt.plot(np.loadtxt(f'{join(path, "val_da6")}.csv', skiprows=1, usecols=1, delimiter=','),
             np.loadtxt(f'{join(path, "val_da6")}.csv', skiprows=1, usecols=2, delimiter=','), 'g--',
             label='Évaluation (rotations+réflexions)')
    plt.xlabel('Epoch', fontsize=13)
    plt.ylabel('Loss', fontsize=13)
    plt.legend(fontsize=11)

    plt.tight_layout()
    plt.show()

compare_plot(join('Results', 'Courbes', 'Loss'))