import torch
import numpy as np 
device = torch.device('cuda:1')
import pandas as pd
from models.my_models import *
from trainer.pre_train_test_split import train_source_or
from data.mydataset import create_dataset_full
from models.models_config import get_model_config


# hyper parameters
hyper_param={ 'FD001': {'epochs':100,'batch_size':256,'lr':3e-4,'alpha1':0.5,'alpha2':0.5},
                  'FD002': {'epochs':75,'batch_size':256,'lr':3e-4,'alpha1':0.5,'alpha2':0.5},
                  'FD003': {'epochs':150,'batch_size':256,'lr':3e-4,'alpha1':0.5,'alpha2':0.5},
                  'FD004': {'epochs':175,'batch_size':256,'lr':3e-4,'alpha1':0.5,'alpha2':0.5}}


# load dataset
data_path= "/home/furqon/rul/MDAN/processed_data/cmapps_train_test_cross_domain.pt"
my_dataset = torch.load(data_path)

# configuration setup
config = get_model_config('LSTM')
config.update({'num_runs':1, 'save':True, 'iterations':1})

# train source domain
if __name__ == '__main__':
  df=pd.DataFrame();res = [];full_res = []
  print('Training Source Domain')
  for src_id in ['FD001', 'FD002', 'FD003', 'FD004']:
    for run_id in range(config['num_runs']):
      src_train_dl, src_test_dl = create_dataset_full(my_dataset[src_id],batch_size=256)
      train_source_or(Mixup_RUL, src_train_dl, src_test_dl, src_id, config, hyper_param)
      

  
