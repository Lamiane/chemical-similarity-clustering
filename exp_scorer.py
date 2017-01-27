import numpy as np

# define score
def my_binary_score(labels_true, labels_pred):
    # labels true: ((x_i, x_j), sim(x_i, x_j)), sim(.,.) in {-1, 1}
    # labels_pred: list
    true, pred = np.array(labels_true), np.array(labels_pred)
    matched = 0.
    for ((idx_1, idx_2), sim_true) in labels_true:
        if labels_pred[idx_1] == labels_pred[idx_2] and sim_true==1:
            matched +=1.
        elif labels_pred[idx_1] != labels_pred[idx_2] and sim_true==-1:
            matched +=1.
        else:
            pass
    return matched/len(labels_true)