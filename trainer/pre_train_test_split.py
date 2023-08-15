##############################################################################
# All the codes about the model construction should be kept in the folder ./models/
# All the codes about the data processing should be kept in the folder ./data/
# All the codes about the loss functions should be kept in the folder ./losses/
# All the source pre-trained checkpoints should be kept in the folder ./trained_models/
# All runs and experiment
# The file ./opts.py stores the options
# The file ./train_eval.py stores the training and test strategy
# The file ./main.py should be simple
#################################################################################
import sys
sys.path.append("..")
import warnings
import torch
from torch import optim
import time
from utils import *
from torch.optim.lr_scheduler import StepLR
from trainer.train_eval import train, evaluate, train_source, train_mix_inter, train_mix_tgt
from data.mydataset import create_dataset_full
import matplotlib.pyplot as plt

fix_randomness(5)
device = torch.device('cuda')


def train_source_or(model, train_dl, test_dl, data_id, config, params):
    # criteierion
    criterion = RMSELoss()
    print(f'Train data {data_id}')
    
    source_model = model(14, 32, 5, 0.01, True, device).to(device)
    optimizer = torch.optim.Adam(source_model.parameters(), lr=params[data_id]['lr'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(params[data_id]['epochs']):
        start_time = time.time()
        # training
        
        train_loss, train_score, train_feat, train_labels = train_source(source_model, train_dl, optimizer, criterion, config, device, params[data_id]['alpha1'], params[data_id]['alpha2'])
        scheduler.step()
        # log time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # printing results
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Score: {train_score:7.3f}')
        # Evaluate on the test set
        if (epoch + 1) % 5 == 0:
            test_loss, test_score, _, _,_,_ = evaluate(source_model, test_dl, criterion, config, device)
            print('=' * 89)
            print(f'\t  Performance on test set:{data_id}::: Loss: {test_loss:.3f} | Score: {test_score:7.3f}')
        if config['tensorboard']:
          writer.add_scalar('Train loss', train_loss, epoch)
          writer.add_scalar('Train Score', train_score, epoch)
          writer.add_scalar('Test loss', test_loss, epoch)
          writer.add_scalar('Test Score', test_score, epoch)
        # saving last epoch model
        if config['save']:
            if (epoch + 1) % 10 == 0:
                checkpoint1 = {'model': source_model,
                               'epoch': epoch,
                               'state_dict': source_model.state_dict(),
                               'optimizer': optimizer.state_dict()}
                torch.save(checkpoint1,
                           f'./trained_models/pretrainedOr_seed_5_epoch_100_75_150_175_dropout_001_alpha1_05_alpha2_05_lr_-4_{config["model_name"]}_{data_id}_new.pt')
    # Evaluate on the test set
    test_loss, test_score, _, _, _, _ = evaluate(source_model, test_dl, criterion, config, device)
    print('=' * 89)
    print(f'\t  Performance on test set:{data_id}::: Loss: {test_loss:.3f} | Score: {test_score:7.3f}')
    print('=' * 89)
    print('| End of Pre-training  |')
    print('=' * 89)
    return source_model, test_loss, test_score


def train_inter(model, src_id, tgt_id, config, params, my_dataset):
    #print('train_source_mix')
    hyper = params[f'{src_id}_{tgt_id}']
    print(f'From_source:{src_id}--->target:{tgt_id}...')
    src_train_dl, src_test_dl = create_dataset_full(my_dataset[src_id],batch_size=hyper['batch_size'])
    tgt_train_dl, tgt_test_dl = create_dataset_full(my_dataset[tgt_id],batch_size=hyper['batch_size'])

    print('Restore source pre_trained model...')
    checkpoint = torch.load(f'/home/furqon/rul/CADA/trained_models/pretrainedOr_seed_5_epoch_100_75_150_175_dropout_001_alpha1_05_alpha2_05_lr_-4_{config["model_name"]}_{src_id}_new.pt')
    
    # criteierion  
    criterion = RMSELoss()
    
    # initialize intermediate model
    inter_model = model(14, 32, 5, 0.01, True, device).to(device)
    inter_model.load_state_dict(checkpoint['state_dict'])
    set_requires_grad(inter_model, requires_grad=True)

    optimizer = torch.optim.Adam(inter_model.parameters(), lr=hyper['lr'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    lbd = np.random.beta(2,2) #lambda
    lbd_mix = lbd

    for epoch in range(hyper['epochs']):
        start_time = time.time()
        
        # training
        train_loss, train_score, train_feat, train_labels, lbd_mix = train_mix_inter(inter_model, src_train_dl, tgt_train_dl, optimizer, criterion, config, device, 
            hyper['alpha3'], lbd_mix)

        scheduler.step()
        # log time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        # printing results
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Score: {train_score:7.3f}')
        # Evaluate on the test set
        if (epoch + 1) % 5 == 0:
            test_loss, test_score, _, _,_,_ = evaluate(inter_model, tgt_test_dl, criterion, config, device)
            print('=' * 89)
            print(f'\t  Performance on test set:{tgt_id}::: Loss: {test_loss:.3f} | Score: {test_score:7.3f}')
        
        # saving last epoch model
        if config['save']:
            if (epoch + 1) % 10 == 0:
                checkpoint1 = {'model': inter_model,
                               'epoch': epoch,
                               'state_dict': inter_model.state_dict(),
                               'optimizer': optimizer.state_dict()}
                
                torch.save(checkpoint1,
                           f'./trained_models/pretrainedIntermediate_seed_5_epoch_100_75_150_175_dropout_001_alpha1_05_alpha3_1_lr_-4_{config["model_name"]}_{src_id}_{tgt_id}_new.pt')
                
    # Evaluate on the test set
    test_loss, test_score, _, _, _, _ = evaluate(inter_model, tgt_test_dl, criterion, config, device)
    print('=' * 89)
    print(f'\t  Performance on test set:{tgt_id}::: Loss: {test_loss:.3f} | Score: {test_score:7.3f}')
    print('=' * 89)
    print('| End of Pre-training  |')
    print('=' * 89)
    return inter_model #, test_loss, test_score

def train_target(model, src_id, tgt_id, config, params, my_dataset):
    loss_history = []
    score_history = []
    pred_labels_history = []
    true_labels_history = []

    hyper = params[f'{src_id}_{tgt_id}']
    print(f'From_source:{src_id}--->target:{tgt_id}...')
    tgt_train_dl, tgt_test_dl = create_dataset_full(my_dataset[tgt_id],batch_size=hyper['batch_size'])

    
    # load source pretrained model
    
    print('Restore pre_trained model...')
    checkpoint = torch.load(f'/home/furqon/rul/CADA/trained_models/pretrainedIntermediate_seed_5_epoch_100_75_150_175_dropout_001_alpha1_05_alpha3_1_lr_-4_{config["model_name"]}_{src_id}_{tgt_id}_new.pt')
    
    # criteierion  
    criterion = RMSELoss()
    
    # initialize target model
    target_model = model(14, 32, 5, 0.01, True, device).to(device)
    target_model.load_state_dict(checkpoint['state_dict'])
    set_requires_grad(target_model, requires_grad=True)

    optimizer = torch.optim.Adam(target_model.parameters(), lr=hyper['lr'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    lbd = np.random.beta(2,2) #lambda

    for epoch in range(hyper['epochs']):
        start_time = time.time()
        
        train_loss, train_score, train_feat, train_labels = train_mix_tgt(target_model, tgt_train_dl, optimizer, criterion, config, device, 
            hyper['alpha4'], hyper['alpha5'], lbd)
        
        scheduler.step()
        # log time
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        testing_loss, testing_score, _, _, pred_labels, true_labels = evaluate(target_model, tgt_test_dl, criterion, config, device)
        
        loss_history.append(testing_loss)
        score_history.append(testing_score)
        pred_labels_history.append(pred_labels)
        true_labels_history.append(true_labels)

        # printing results
        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Score: {train_score:7.3f}')
        # Evaluate on the test set every 5 epoch
        if (epoch + 1) % 5 == 0:
            test_loss, test_score, _, _,_,_ = evaluate(target_model, tgt_test_dl, criterion, config, device)
            print('=' * 89)
            print(f'\t  Performance on test set:{tgt_id}::: Loss: {test_loss:.3f} | Score: {test_score:7.3f}')
            
        # saving last epoch model

        if config['save']:
            if (epoch + 1) % 10 == 0:
                checkpoint1 = {'model': target_model,
                               'epoch': epoch,
                               'state_dict': target_model.state_dict(),
                               'optimizer': optimizer.state_dict()}
                torch.save(checkpoint1,
                           f'./trained_models/pretrainedTarget2_{config["model_name"]}_{src_id}_{tgt_id}_new.pt')

    # Evaluate on the test set
    test_loss, test_score, _, _, _, _ = evaluate(target_model, tgt_test_dl, criterion, config, device)

    print('=' * 89)
    print(f'\t  Performance on test set:{tgt_id}::: Loss: {test_loss:.3f} | Score: {test_score:7.3f}')
    print('=' * 89)
    print('| End of Pre-training  |')
    print('=' * 89)
    print(f'loss:{loss_history}')
    print(f'score:{score_history}')
    return target_model, loss_history, score_history, pred_labels_history, true_labels_history #, test_loss, test_score





