import numpy as np
from matplotlib import pyplot as plt

def compare_mats(true, guess, mask, thresh, skeleton=False):
    fig, ax = plt.subplots(1, 3, figsize=[10, 3])
    ax[0].imshow(guess, cmap='Reds', vmin=0, vmax=guess.max())
    ax[0].imshow(1-mask, vmin=0, vmax=1, cmap='Greys', alpha=0.2)
    ax[0].set_title("Inferred")
    ax[1].imshow(true, cmap='Blues')
    ax[1].imshow(1-mask, vmin=0, vmax=1, cmap='Greys', alpha=0.2)
    ax[1].set_title("True")
    ax[2].imshow(true, vmin=0, vmax=1, cmap='Blues')
    ax[2].imshow(guess > thresh,
                 vmin=0, vmax=1, cmap='Reds', alpha=0.5)
    ax[2].imshow(1-mask, vmin=0, vmax=1, cmap='Greys', alpha=0.2)
    ax[2].set_title("Overlay")

    errs = ((guess > thresh).astype(int) - true) * mask
    misoriented_edge_locs = ((errs + errs.T) == 0) & (errs != 0)
    shd = (errs != 0).sum() - misoriented_edge_locs.sum() / 2
    shd /= 1 + skeleton
    print('n misoriented edges:', misoriented_edge_locs.sum() / 2)
    print('n missing edges:', (errs < 0).sum() - misoriented_edge_locs.sum() / 2)
    print('n extra edges:', (errs > 0).sum() - misoriented_edge_locs.sum() / 2)
    print("SHD is:", shd)
    th_precision = ((guess>thresh) * (true!=0) * mask).sum() / ((guess>thresh) * mask).sum()
    th_recall = ((guess>thresh) * (true!=0) * mask).sum() / (mask * (true != 0)).sum()
    f1 = 2 / ((1 / th_precision) + (1 / th_recall))
    print("Precision is:", th_precision)
    print("Recall is:", th_recall)
    print("F1 is:", f1)

    # tn = ((guess<=thresh) * (true==0) * mask).sum()

    fig, ax = plt.subplots(1, 2, figsize=[8, 3])
    for t in np.r_[np.percentile(guess.flatten(), np.linspace(0, 100, 100)), [thresh]]:
        tested = guess > t
        tp = tested * (true != 0)
        recall = (mask * tp).sum() / (mask * (true != 0)).sum()
        precision = (mask * tp).sum() / (mask * tested).sum()
        ax[0].plot(recall, precision, '.', markersize=5,
                 color='black' if t!=thresh else 'red')
    rand_prec = ((true!=0) * mask).sum()/mask.sum()
    ax[0].plot([0, 1], rand_prec*np.ones(2), color='grey')
    ax[0].set_ylabel("precision")
    ax[0].set_xlabel("recall")
    ax[0].set_ylim(rand_prec, 1)
    ax[0].set_xlim(0, 1)

    for t in np.r_[np.percentile(guess.flatten(), np.linspace(0, 100, 100)), [thresh]]:
        tested = guess > t
        tpr = (tested * (true != 0) * mask).astype(float).sum() / ((true != 0) * mask).sum()
        fpr = (tested * (true == 0) * mask).astype(float).sum() / ((true == 0) * mask).sum()
        ax[1].plot(fpr, tpr, '.', markersize=5,
                 color='black' if t!=thresh else 'red')
    ax[1].plot([0, 1], [0, 1], color='grey')
    ax[1].set_ylabel("TPR")
    ax[1].set_xlabel("FPR")
    ax[1].set_ylim(0, 1)
    ax[1].set_xlim(0, 1)
    return shd, th_precision, th_recall, f1