import numpy as np

def compute_metrics(y_true, y_pred, positive_label="spam", negative_label="ham"):
    """
    Compute evaluation metrics for binary classification.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        positive_label (optional): Value representing the positive label. Default is 1.
        negative_label (optional): Value representing the negative label. Default is 0.

    Returns:
        accuracy (float): Accuracy metric.
        precision (float): Precision metric.
        recall (float): Recall metric.
        f1 (float): F1-score metric.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # accuray calculation
    accuracy = np.mean(y_true == y_pred)
    
    # true positives, false positives, and false negatives calculation
    tp = np.sum(y_true == y_pred)  # true positive count (both true and predicted labels are the same)
    fp = np.sum((y_true != y_pred) & (y_pred == positive_label))  # false positive count (true label is negative, but predicted is positive)
    fn = np.sum((y_true != y_pred) & (y_pred == negative_label))  # flase negative count (true label is positive, but predicted is negative)
    
    # precision, recall, and F1-score calculation
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return accuracy, precision, recall, f1


def confusion_matrix(y_true, y_pred):
    """
    Calculates the confusion matrix based on the ground truth and predicted labels.

    Parameters:
        y_true (list): The ground truth labels.
        y_pred (list): The predicted labels.

    Returns:
        list of lists: The confusion matrix.
    """

    # get unique classes from y_true and y_pred
    classes = list(set(y_true + y_pred))
    classes.sort()

    # total number of unique classes
    num_classes = len(classes)

    # initialize the confusion matrix as a 2D list of zeros
    cm = [[0] * num_classes for _ in range(num_classes)]

    # iterate each pair of true and predicted labels
    for true, pred in zip(y_true, y_pred):
        # find indices of true and predicted classes in the classes list
        true_idx = classes.index(true)
        pred_idx = classes.index(pred)

        # increment corresponding cells in the confusion matrix
        cm[true_idx][pred_idx] += 1

    # return confusion matrix
    return cm
