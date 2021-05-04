import json
import csv
#import pandas as pd
#import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# PyTorch
import torch
import torch.nn as nn


# TODO: 
# ONLY WORK FOR BATCH OF SIZE=1
# FIX TO WORK FOR ANY BATCH SIZE

def test_model(device, model, test_loader, PATH_RESULTS):
    y_probs_list = []
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch)
            
            sigmoid = nn.Sigmoid()
            preds = sigmoid(y_test_pred)
            
            y_probs_list.append(preds.cpu().numpy())
            
            preds = torch.tensor([1.0 if pred > 0.5 else 0.0 for pred in preds]).to(device)
            #preds = torch.reshape(preds, (batch_size_, 1))
                    
            #_, y_pred_tag = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(preds.cpu().numpy())
            y_true_list.append(y_batch.cpu().numpy())

    y_pred_list = [i[0] for i in y_pred_list]
    y_true_list = [i[0] for i in y_true_list]
    y_probs_list = [i[0] for i in y_probs_list]

    
    # Serializing csv
    data = []
    
    for fname, y_true, y_proba, y_pred in zip(test_loader.dataset.samples, y_true_list, y_probs_list, y_pred_list):
        data.append((fname[0], y_true, y_proba[0], y_pred))
        
    with open(PATH_RESULTS,'w') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['filename', 'y_true','y_proba', 'y_pred'])
        for row in data:
            csv_out.writerow(row)
        
        
    return (y_true_list, y_pred_list) 
    
    

def eval_preds(y_true_list,y_pred_list, PATH_RESULTS, idx2class=None):
    '''
    # TODO: check dependencies (sns, pd)
    if idx2class is not None:
        print(classification_report(y_true_list, y_pred_list))
        confusion_matrix_df = pd.DataFrame(confusion_matrix(y_true_list, y_pred_list)).rename(columns=idx2class, index=idx2class)
        fig, ax = plt.subplots(figsize=(7,5))         
        sns.heatmap(confusion_matrix_df, annot=True, ax=ax)
    '''

    tn, fp, fn, tp = confusion_matrix(y_true_list, y_pred_list).ravel()
    print('tn=%d, fp=%d, fn=%d, tp=%d' % (tn, fp, fn, tp))
    accuracy = accuracy_score(y_true_list, y_pred_list)
    print('Accuracy: %f' % accuracy)
    # precision tp / (tp + fp)
    precision = precision_score(y_true_list, y_pred_list)
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_true_list, y_pred_list)
    print('Recall: %f' % recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_true_list, y_pred_list)
    print('F1 score: %f' % f1)
    specificity = tn / (tn + fp)
    print('Specificity: %f ' % specificity)
    
    results = {'tp': float(tp), 'tn': float(tn), 'fn': float(fn), 'fp': float(fp),
        'accuracy': accuracy, 'precision': precision, 'recal': recall, 
        'f1-score': f1, 'specificity': specificity
    }
    
    # Serializing json    
    with open(PATH_RESULTS, "w") as outfile:  
        json.dump(results, outfile)  