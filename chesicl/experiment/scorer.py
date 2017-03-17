import numpy as np
from collections import Counter

# define score
def my_binary_score(labels_true, labels_pred):
    # rand index for constrained problems
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


def balanced_accuracy(labels_true, labels_pred):
    # balanced accuracy for binary constrained problems (cannot-, must-link)
    # labels true: ((x_i, x_j), sim(x_i, x_j)), sim(.,.) in {-1, 1}
    # labels_pred: list
    
    def _scale(kk):
        # -1 -> 0, 1 -> 1
        return 0 if kk == -1 else kk
        
    n_classes = 2
    arr = np.zeros((n_classes, n_classes))
    for ((idx_1, idx_2), sim_true) in labels_true:
        print ((idx_1, idx_2), sim_true)
        print _scale(sim_true)
        if labels_pred[idx_1] == labels_pred[idx_2]:
            arr[_scale(sim_true), 1] += 1.
            print 'a'
        else:
            arr[_scale(sim_true), 0] += 1.
            print 'b'
        print '\n'
    
    cumulative_sum = 0.
    for c in range(n_classes):
        numerator = arr[c,c]
        denominator = np.sum(arr[c, :])
        # ponizsza linijka sie nie wysypie o ile w kazdej klasie jest co najmniej jeden przyklad
        cumulative_sum += (numerator/denominator)
    print arr
    cumulative_sum = cumulative_sum/n_classes
                   
    return cumulative_sum
            
        
        