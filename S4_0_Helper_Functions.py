import numpy as np
import pandas as pd 
from sklearn.metrics import accuracy_score, recall_score, precision_score

"""
The following is a simple method for getting accuracy metrics for our models.
"""

def getAccuracyMetrics(preds, test_y):
    predicted_class_indices=np.argmax(preds,axis=1)

    # convert [x, y, z] style labels to either 0, 1, or 2
    test_y_label = []
    for label in test_y:
        if np.array_equal(label, [0, 1, 0]):
            test_y_label.append(1)
        elif np.array_equal(label, [0, 0, 1]):
            test_y_label.append(2)
        elif np.array_equal(label, [1, 0, 0]):
            test_y_label.append(0)

    accuracy = accuracy_score(test_y_label, predicted_class_indices)
    recall = recall_score(test_y_label, predicted_class_indices, average='micro')
    precision = precision_score(test_y_label, predicted_class_indices, average='micro')
    print('accuracy:', accuracy)
    print('recall:', recall)
    print('precision:', precision)

    return None
