#get prediction results

import numpy as np
from scipy.stats import mode

def true_prediction(predict_label,k,image_per_person):
    result=np.zeros(k)
    for i in range(k):
        m=[j+i*image_per_person for j in range(image_per_person)]
        L = [predict_label[t] for t in m]
        mode_L=mode(L)[0]
        count=0
        for j in range(np.size(L)):
            if L[j]==mode_L:
                count = count + 1
        result[i]=count
    return max(result)/image_per_person

if __name__ == '__main__':
    predict_label=[10, 2, 4, 1, 7, 7, 2, 7, 7, 6, 8, 5, 2, 7, 10, 7, 7, 2, 8, 5, 2, 8, 3, 8, 4, 6, 9, 6, 9, 9, 9, 9, 9, 6, 4, 6, 9, 9,
        10, 6, 2, 9, 9, 6, 9, 9, 9, 7, 7, 7, 1, 3, 10, 10, 3, 10, 8, 3, 7, 2, 8, 6, 3, 2, 3, 8, 10, 3, 3, 10, 3, 3, 10, 7,
        7, 7, 4, 4, 5, 4, 4, 4, 5, 7, 4, 7, 5, 6, 4, 5, 5, 5, 7, 3, 7, 3, 4, 2, 5, 7, 1, 3, 6, 6, 5, 7, 1, 5, 1, 7, 8, 6,
        5, 6, 5, 6, 7, 9, 2, 10, 3, 1, 3, 6, 3, 10, 10, 8, 2, 1, 5, 2, 8, 6, 5, 8, 2, 9, 9, 8, 7, 2, 6, 8, 2, 8, 10, 7, 10,
        8, 2, 4, 10, 10, 5, 4, 7, 7, 4, 10, 10, 8, 4, 10, 2, 5, 2, 10, 7, 4, 2, 10, 7, 7, 10, 10, 10, 6, 2, 1, 3, 1, 1, 10,
        4, 7, 4, 10, 4, 6, 6, 2, 6, 10, 5, 6, 6, 1, 6, 2, 4, 4, 7, 7, 10, 2, 7, 4, 10, 4, 7, 7, 10, 7, 2, 7, 10, 4, 7, 7,
        10, 10, 7, 10, 10, 6, 7, 4, 10, 5, 7, 4, 7, 7, 5, 2, 10, 10, 9, 7, 7, 2, 2, 7, 7, 7, 10, 7, 2, 10]
    k=10;image_per_person=25
    true_pred = true_prediction(predict_label, k, image_per_person)
    print("\nthe number of true prediction is ", true_pred)