import os
import torch
import pandas as po
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score

def load_pretrained_model(model, pretrained_weights_shortcut=None):
     if model == 'bert':
          from transformers import BertModel
          
          if pretrained_weights_shortcut == None:
               model = BertModel.from_pretrained('bert-base-uncased')

          else:
               model = BertModel.from_pretrained(pretrained_weights_shortcut)

          return model

     if model == 'distil_bert':
          from transformers import DistilBertModel
          
          if pretrained_weights_shortcut == None:
               model = DistilBertModel.from_pretrained('distilbert-base-uncased')

          else:
               model = DistilBertModel.from_pretrained(pretrained_weights_shortcut)

          return model

def load_pretrained_tokenizer(model, pretrained_weights_shortcut=None):
     if model == 'bert':
          from transformers import BertTokenizer
          
          if pretrained_weights_shortcut == None:
               tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

          else:
               tokenizer = BertTokenizer.from_pretrained(pretrained_weights_shortcut)

          return tokenizer

     if model == 'distil_bert':
          from transformers import DistilBertTokenizer
          
          if pretrained_weights_shortcut == None:
               tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

          else:
               tokenizer = DistilBertTokenizer.from_pretrained(pretrained_weights_shortcut)

          return tokenizer

def area_under_the_curve(y_true, y_pred):
     return roc_auc_score(y_true, y_pred)

def mean_average_precision(y_true, y_pred):
  return average_precision_score(y_true, y_pred)

def accuracy(y_true, y_pred):
  return accuracy_score(y_true, y_pred)

def f1(y_true, y_pred):
  return f1_score(y_true, y_pred, average='micro')

def evaluate_classifier(model_name, y_true, y_pred, save_path='results.csv'):
    metrics = {'Model' : model_name, 
           'AUC' : area_under_the_curve(y_true, y_pred), 
           'MAP' : mean_average_precision(y_true, y_pred), 
           'Accuracy' : accuracy(y_true, y_pred),
           'F1' : f1(y_true, y_pred)
           }
    return metrics
"""if not os.path.exists(save_path):
       results_df = po.DataFrame(columns = ['Model'])
     else:
       results_df = po.read_csv(save_path)

     results_df = results_df.append(metrics, ignore_index=True)
     results_df.to_csv(save_path, index=False)"""
