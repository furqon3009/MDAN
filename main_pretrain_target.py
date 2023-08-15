import torch
import numpy as np 
device = torch.device('cuda:1')
import pandas as pd
from models.my_models import *
from trainer.pre_train_test_split import train_target
from data.mydataset import create_dataset_full
from models.models_config import get_model_config


# hyper parameters
hyper_param={ 'FD001_FD002': {'epochs':100,'batch_size':256,'lr':3e-4,'alpha1':0.5,'alpha2':0.5,'alpha3':1,'alpha4':1,'alpha5':1}, 
                  'FD001_FD003': {'epochs':100,'batch_size':256,'lr':3e-4,'alpha1':0.5,'alpha2':0.5,'alpha3':1,'alpha4':1,'alpha5':1},
                  'FD001_FD004': {'epochs':100,'batch_size':256,'lr':3e-4,'alpha1':0.5,'alpha2':0.5,'alpha3':1,'alpha4':1,'alpha5':1},
                  'FD002_FD001': {'epochs':75,'batch_size':256,'lr':3e-4,'alpha1':0.5,'alpha2':0.5,'alpha3':1,'alpha4':1,'alpha5':1},
                  'FD002_FD003': {'epochs':75,'batch_size':256,'lr':3e-4,'alpha1':0.5,'alpha2':0.5,'alpha3':1,'alpha4':1,'alpha5':1},
                  'FD002_FD004': {'epochs':75,'batch_size':256,'lr':3e-4,'alpha1':0.5,'alpha2':0.5,'alpha3':1,'alpha4':1,'alpha5':1},
                  'FD003_FD001': {'epochs':150,'batch_size':256,'lr':3e-4,'alpha1':0.5,'alpha2':0.5,'alpha3':1,'alpha4':1,'alpha5':1},
                  'FD003_FD002': {'epochs':150,'batch_size':256,'lr':3e-4,'alpha1':0.5,'alpha2':0.5,'alpha3':1,'alpha4':1,'alpha5':1},
                  'FD003_FD004': {'epochs':150,'batch_size':256,'lr':3e-4,'alpha1':0.5,'alpha2':0.5,'alpha3':1,'alpha4':1,'alpha5':1},
                  'FD004_FD001': {'epochs':175,'batch_size':256,'lr':3e-4,'alpha1':0.5,'alpha2':0.5,'alpha3':1,'alpha4':1,'alpha5':1},
                  'FD004_FD002': {'epochs':175,'batch_size':256,'lr':3e-4,'alpha1':0.5,'alpha2':0.5,'alpha3':1,'alpha4':1,'alpha5':1},
                  'FD004_FD003': {'epochs':175,'batch_size':256,'lr':3e-4,'alpha1':0.5,'alpha2':0.5,'alpha3':1,'alpha4':1,'alpha5':1}}

# load dataset
data_path= "/home/furqon/rul/MDAN/processed_data/cmapps_train_test_cross_domain.pt"
my_dataset = torch.load(data_path)

# configuration setup
config = get_model_config('LSTM')
config.update({'num_runs':1, 'save':True, 'iterations':1})


# train target domain
if __name__ == '__main__':
  df=pd.DataFrame();res = [];full_res = []
  print('Training Target Domain')

  for src_id in ['FD001', 'FD002', 'FD003', 'FD004']:
    for tgt_id in ['FD001', 'FD002', 'FD003', 'FD004']:
      if src_id != tgt_id:
        total_loss = []
        total_score = []
        for run_id in range(config['num_runs']):
          train_target(Mixup_RUL, src_id, tgt_id, config, hyper_param, my_dataset)

