import numpy as np


def confustion_matrix(y_real, y_predicted):
    matrix = np.zeros((2, 2), int)
    y_predicted_classes = [1 if i > 0.5 else 0 for i in y_predicted]
    for i in range(len(y_real)):
        if y_real[i] == 0 and y_predicted_classes[i] == 0:
            matrix[0][0] += 1
        if y_real[i] == 1 and y_predicted_classes[i] == 1:
            matrix[1][1] += 1
        if y_real[i] == 1 and y_predicted_classes[i] == 0:
            matrix[1][0] += 1
        if y_real[i] == 0 and y_predicted_classes[i] == 1:
            matrix[0][1] += 1

    return matrix


def accuracy(y_real, y_predicted):
    matrix = confustion_matrix(y_real, y_predicted)
    tp, tn, fp, fn = (matrix[1][1], matrix[0][0], matrix[0][1], matrix[1][0])
    return (tp + tn) / (tp + tn + fp + fn)
