import numpy as np

def get_precisions_recalls(importances, groundtruth):
    importances = np.reshape(importances, (-1))
    groundtruth = np.reshape(groundtruth, (-1))

    sorted_features = importances.argsort()[:][::-1]
    precisions = []
    recalls = []
    for threshold in range(importances.shape[0]):
        cur_positives = sorted_features[:threshold+1]
        cur_negatives = sorted_features[threshold+1:]
        true_positives = np.sum(groundtruth[cur_positives])
        false_positives = np.sum([1 for el in groundtruth[cur_positives] if (el == 0)])
        false_negatives = np.sum(groundtruth[cur_negatives])
        precisions.append(true_positives/(true_positives + false_positives))
        recalls.append(true_positives/(true_positives + false_negatives))

    return precisions, recalls

def get_area(precisions, recalls):
    area = 0
    old_recall = 0
    old_height = precisions[0]
    for precision,recall in zip(precisions, recalls):
        base = recall - old_recall
        height = precision
        area += (base * height + base*old_height)/2.0
        old_height = height
        old_recall = recall
    return area

def get_aupr(importances, groundtruth):
    precisions, recalls = get_precisions_recalls(importances, groundtruth)
    return get_area(precisions, recalls)

def compute_precision_recalls_for_gene(importances, adjacency_matrix, name = ''):
    precisions = []
    recalls = []
    true_positives = 0
    false_negatives = int(np.sum(adjacency_matrix))
    false_positives = 0
    global_cnt = 0
    for cnt in range(importances.shape[0]-1):
        for _ in range(importances.shape[0]):
            i,j = np.unravel_index(importances.argmax(), importances.shape)
            if adjacency_matrix[i][j] == 1:
                true_positives += 1
                false_negatives -= 1
            else:
                false_positives += 1
            importances[i][j] = -100
            precisions.append(float(true_positives)/(true_positives + false_positives))
            recalls.append(float(true_positives)/(true_positives + false_negatives))
            global_cnt += 1
    return precisions, recalls

def compute_gene_area(importances_matrix, adjacency_matrix, name = ''):
    pr, rec = compute_precision_recalls_for_gene(importances_matrix, adjacency_matrix)
    return get_area(pr, rec)