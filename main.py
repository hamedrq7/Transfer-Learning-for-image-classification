import numpy as np
import matplotlib.pyplot as plt
import glob, os
import pickle
from typing import Dict, List

def mkdir(dir: str):
  if not os.path.exists(dir):
    os.makedirs(dir)

def early_exp_six_fig():
    """
    'model_name': self.model_name,
    'freeze_mode': self.freeze_mode,
    'num_epochs': self.num_epochs,
    'lr': self.lr,
    'optim': self.opt_str,
    'val_loss_hist': val_loss_hist,
    'val_acc_hist': val_acc_hist,
    'train_loss_hist': train_loss_hist, 
    'train_acc_hist': train_acc_hist,
    """
    # model_name = 'resnet18'
    # model_name = 'vgg16'
    model_name = 'densenet121'
    
    freeze_modes = [70, 50, 30, 10]
    lr = 0.001
    num_epochs = 5
    optim = 'adam'


    fig, axs = plt.subplots(nrows = 2, ncols = 4, figsize=[20, 9], dpi=150,
                                    gridspec_kw={'hspace': 0.05, 'wspace': 0.15})
    subplot_idx = 0
    
    for freeze in freeze_modes:
        if model_name == 'densenet121' and freeze == 50: continue
        name_to_load = f'model_{model_name}-freeze_{freeze}-optim_{optim}-{lr}-num_epochs_{num_epochs}-stats.pickle'

        with open(f'training_info/early_exp_{model_name}/{name_to_load}', 'rb') as handle:
            temp = pickle.load(handle)
        
        temp['test_loss_hist'] = temp['val_loss_hist']
        temp['test_acc_hist'] = temp['val_acc_hist']
        
        lowest_loss_train_x = np.argmin(np.array(temp['train_loss_hist']))
        lowest_loss_train_y = temp['train_loss_hist'][lowest_loss_train_x]
        lowest_loss_test_x  = np.argmin(np.array(temp['test_loss_hist']))
        lowest_loss_test_y = temp['test_loss_hist'][lowest_loss_test_x]
        # print(axs)

        axs[0][subplot_idx].annotate("{:.4f}".format(lowest_loss_train_y), [lowest_loss_train_x, lowest_loss_train_y])
        axs[0][subplot_idx].annotate("{:.4f}".format(lowest_loss_test_y), [lowest_loss_test_x, lowest_loss_test_y])  

        loss_train_plt = axs[0][subplot_idx].plot(temp['train_loss_hist'], '-x', label = f'train loss', markevery = [lowest_loss_train_x])
        loss_test_plt = axs[0][subplot_idx].plot(temp['test_loss_hist'], '-x', label = f'test loss', markevery = [lowest_loss_test_x])

        axs[0][subplot_idx].set_xlabel(xlabel='epochs')
        axs[0][subplot_idx].set_ylabel(ylabel='loss')
        # axs[0][subplot_idx].set_ylim(bottom=-0.05, top=0.4)
        axs[0][subplot_idx].grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        axs[0][subplot_idx].legend()
        axs[0][subplot_idx].label_outer()
        axs[0][subplot_idx].set_title(f'Freezing {freeze}%')

        # nex_row_indx = (subplot_idx + 4) % 8
        highest_acc_train_x = np.argmax(np.array(temp['train_acc_hist']))
        highest_acc_train_y = temp['train_acc_hist'][highest_acc_train_x]
        highest_acc_test_x = np.argmax(np.array(temp['test_acc_hist']))
        highest_acc_test_y = temp['test_acc_hist'][highest_acc_test_x]
        
        high_acc_train_plt = axs[1][subplot_idx].plot(temp['train_acc_hist'], '-x', label = f'acc loss', markevery = [highest_acc_train_x])
        high_acc_test_plt = axs[1][subplot_idx].plot(temp['test_acc_hist'], '-x', label = f'acc loss', markevery = [highest_acc_test_x])

        axs[1][subplot_idx].annotate("{:.4f}".format(highest_acc_train_y), [highest_acc_train_x, highest_acc_train_y])
        axs[1][subplot_idx].annotate("{:.4f}".format(highest_acc_test_y), [highest_acc_test_x, highest_acc_test_y])  
        
        axs[1][subplot_idx].set_xlabel(xlabel='epochs')
        axs[1][subplot_idx].set_ylabel(ylabel='acc')
        # axs[1][subplot_idx].set_title(f'acc')
        # axs[1][subplot_idx].set_ylim(bottom=0.8, top=1.02)
        axs[1][subplot_idx].grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        axs[1][subplot_idx].legend()
        axs[1][subplot_idx].label_outer()
        # axs[nex_row_indx].set_title(opt)

        subplot_idx += 1

    fig.suptitle(f'early experiments of {model_name} - different freezings')
    base_dir = './images/early_exp' 
    mkdir(base_dir)
    # plt.savefig(f'{base_dir}/{optim}-{hidd}.jpg')
    plt.savefig(f'{base_dir}/{model_name} early_exp.jpg')
    
    plt.clf()
        # exit()


# early_exp_six_fig()

def freeze_100_six_fig():

    fig, axs = plt.subplots(nrows = 2, ncols = 3, figsize=[15, 9], dpi=150,
                                    gridspec_kw={'hspace': 0.05, 'wspace': 0.15})
    subplot_idx = 0
    for model_name in ['resnet18', 'vgg16', 'densenet121']:

        freeze = 100
        lr = 0.001
        num_epochs = 10
        optim = 'sgd'
        if model_name == 'densenet121' and freeze == 50: continue
        warmup_steps = 0 

        # name_to_load = f'model_{model_name}-freeze_{freeze}-optim_{optim}-{lr}-num_epochs_{num_epochs}-stats.pickle'
        name_to_load = f'model_{model_name}-freeze_{freeze}-optim_{optim}-{lr}-num_epochs_{num_epochs}-warmup_{warmup_steps}-stats.pickle'

        with open(f'training_info/{model_name}/{name_to_load}', 'rb') as handle:
            temp = pickle.load(handle)
        
        temp['test_loss_hist'] = temp['val_loss_hist']
        temp['test_acc_hist'] = temp['val_acc_hist']
        
        lowest_loss_train_x = np.argmin(np.array(temp['train_loss_hist']))
        lowest_loss_train_y = temp['train_loss_hist'][lowest_loss_train_x]
        lowest_loss_test_x  = np.argmin(np.array(temp['test_loss_hist']))
        lowest_loss_test_y = temp['test_loss_hist'][lowest_loss_test_x]
        # print(axs)

        axs[0][subplot_idx].annotate("{:.4f}".format(lowest_loss_train_y), [lowest_loss_train_x, lowest_loss_train_y])
        axs[0][subplot_idx].annotate("{:.4f}".format(lowest_loss_test_y), [lowest_loss_test_x, lowest_loss_test_y])  

        loss_train_plt = axs[0][subplot_idx].plot(temp['train_loss_hist'], '-x', label = f'train loss', markevery = [lowest_loss_train_x])
        loss_test_plt = axs[0][subplot_idx].plot(temp['test_loss_hist'], '-x', label = f'test loss', markevery = [lowest_loss_test_x])

        axs[0][subplot_idx].set_xlabel(xlabel='epochs')
        axs[0][subplot_idx].set_ylabel(ylabel='loss')
        # axs[0][subplot_idx].set_ylim(bottom=-0.05, top=0.4)
        axs[0][subplot_idx].grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        axs[0][subplot_idx].legend()
        axs[0][subplot_idx].label_outer()
        axs[0][subplot_idx].set_title(f'Model: {model_name}')

        # nex_row_indx = (subplot_idx + 4) % 8
        highest_acc_train_x = np.argmax(np.array(temp['train_acc_hist']))
        highest_acc_train_y = temp['train_acc_hist'][highest_acc_train_x]
        highest_acc_test_x = np.argmax(np.array(temp['test_acc_hist']))
        highest_acc_test_y = temp['test_acc_hist'][highest_acc_test_x]
        
        high_acc_train_plt = axs[1][subplot_idx].plot(temp['train_acc_hist'], '-x', label = f'acc loss', markevery = [highest_acc_train_x])
        high_acc_test_plt = axs[1][subplot_idx].plot(temp['test_acc_hist'], '-x', label = f'acc loss', markevery = [highest_acc_test_x])

        axs[1][subplot_idx].annotate("{:.4f}".format(highest_acc_train_y), [highest_acc_train_x, highest_acc_train_y])
        axs[1][subplot_idx].annotate("{:.4f}".format(highest_acc_test_y), [highest_acc_test_x, highest_acc_test_y])  
        
        axs[1][subplot_idx].set_xlabel(xlabel='epochs')
        axs[1][subplot_idx].set_ylabel(ylabel='acc')
        # axs[1][subplot_idx].set_title(f'acc')
        # axs[1][subplot_idx].set_ylim(bottom=0.8, top=1.02)
        axs[1][subplot_idx].grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        axs[1][subplot_idx].legend()
        axs[1][subplot_idx].label_outer()
        # axs[nex_row_indx].set_title(opt)

        subplot_idx += 1

    fig.suptitle(f'Freeze mode: {freeze} %')
    base_dir = './images/exp1' 
    mkdir(base_dir)
    plt.savefig(f'{base_dir}/freeze {freeze}.png')
    
    plt.clf()


# freeze_100_six_fig()


def mode_w_warmup(model_name: str):
    lr_dict = {
        70: 0.0005,
        50: 0.0001,
        30: 5e-05,
        10: 1e-05,
    }
    ncols = 4
    if model_name == 'densenet121': ncols -= 1
    fig, axs = plt.subplots(nrows = 2, ncols = ncols, figsize=[15, 9], dpi=150,
                                    gridspec_kw={'hspace': 0.05, 'wspace': 0.15})
    subplot_idx = 0
    for freeze in [70, 50, 30, 10]:

        lr = lr_dict[freeze]
        num_epochs = 10
        optim = 'sgd'
        warmup_steps=1
        if model_name == 'densenet121' and freeze == 50: continue
    
        name_to_load = f'model_{model_name}-freeze_{freeze}-optim_{optim}-{lr}-num_epochs_{num_epochs}-warmup_{warmup_steps}-stats.pickle'

        with open(f'training_info/{model_name}/{name_to_load}', 'rb') as handle:
            temp = pickle.load(handle)
        
        temp['test_loss_hist'] = temp['val_loss_hist']
        temp['test_acc_hist'] = temp['val_acc_hist']
        
        lowest_loss_train_x = np.argmin(np.array(temp['train_loss_hist']))
        lowest_loss_train_y = temp['train_loss_hist'][lowest_loss_train_x]
        lowest_loss_test_x  = np.argmin(np.array(temp['test_loss_hist']))
        lowest_loss_test_y = temp['test_loss_hist'][lowest_loss_test_x]
        # print(axs)

        axs[0][subplot_idx].annotate("{:.4f}".format(lowest_loss_train_y), [lowest_loss_train_x, lowest_loss_train_y])
        axs[0][subplot_idx].annotate("{:.4f}".format(lowest_loss_test_y), [lowest_loss_test_x, lowest_loss_test_y])  

        loss_train_plt = axs[0][subplot_idx].plot(temp['train_loss_hist'], '-x', label = f'train loss', markevery = [lowest_loss_train_x])
        loss_test_plt = axs[0][subplot_idx].plot(temp['test_loss_hist'], '-x', label = f'test loss', markevery = [lowest_loss_test_x])

        axs[0][subplot_idx].set_xlabel(xlabel='epochs')
        axs[0][subplot_idx].set_ylabel(ylabel='loss')
        # axs[0][subplot_idx].set_ylim(bottom=-0.05, top=0.4)
        axs[0][subplot_idx].grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        axs[0][subplot_idx].legend()
        axs[0][subplot_idx].label_outer()
        axs[0][subplot_idx].set_title(f'Freeze Mode: {freeze}%')

        # nex_row_indx = (subplot_idx + 4) % 8
        highest_acc_train_x = np.argmax(np.array(temp['train_acc_hist']))
        highest_acc_train_y = temp['train_acc_hist'][highest_acc_train_x]
        highest_acc_test_x = np.argmax(np.array(temp['test_acc_hist']))
        highest_acc_test_y = temp['test_acc_hist'][highest_acc_test_x]
        
        high_acc_train_plt = axs[1][subplot_idx].plot(temp['train_acc_hist'], '-x', label = f'acc loss', markevery = [highest_acc_train_x])
        high_acc_test_plt = axs[1][subplot_idx].plot(temp['test_acc_hist'], '-x', label = f'acc loss', markevery = [highest_acc_test_x])

        axs[1][subplot_idx].annotate("{:.4f}".format(highest_acc_train_y), [highest_acc_train_x, highest_acc_train_y])
        axs[1][subplot_idx].annotate("{:.4f}".format(highest_acc_test_y), [highest_acc_test_x, highest_acc_test_y])  
        
        axs[1][subplot_idx].set_xlabel(xlabel='epochs')
        axs[1][subplot_idx].set_ylabel(ylabel='acc')
        # axs[1][subplot_idx].set_title(f'acc')
        # axs[1][subplot_idx].set_ylim(bottom=0.8, top=1.02)
        axs[1][subplot_idx].grid(color = 'green', linestyle = '--', linewidth = 0.5, alpha=0.75)
        axs[1][subplot_idx].legend()
        axs[1][subplot_idx].label_outer()
        # axs[nex_row_indx].set_title(opt)

        subplot_idx += 1

    fig.suptitle(f'Model name: {model_name}, warmup_steps: {warmup_steps}')
    base_dir = './images/exp1' 
    mkdir(base_dir)
    plt.savefig(f'{base_dir}/{model_name}.png')
    
    plt.clf()

mode_w_warmup('resnet18')

mode_w_warmup('vgg16')

mode_w_warmup('densenet121')

