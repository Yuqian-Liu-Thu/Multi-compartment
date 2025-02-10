import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
from torch.utils import data
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR,MultiStepLR
import tensorflow.keras as keras

import numpy as np
import math
from copy import deepcopy
import matplotlib.pyplot as plt
from datetime import datetime
import collections
from generate_shd_dataset import *
SWEEP_DURATION = 1.4

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--task", help="choose the task: smnist and psmnist", type=str,default="smnist")
parser.add_argument("--ec_f", help="choose the encode function: rbf, binary, poisson, none", type=str,default='none')
parser.add_argument("--dc_f", help="choose the decode function: adp-mem, adp-spike, integrator", type=str,default='adp-spike')
parser.add_argument("--batch_size", help="set the batch_size", type=int,default=200)
parser.add_argument("--num_encoders", help="set the number of encoder", type=int,default=8)
parser.add_argument("--num_epochs", help="set the number of epoch", type=int,default=200)
parser.add_argument("--learning_rate", help="set the learning rate", type=float,default=1e-3)
parser.add_argument("--len", help="set the length of the gaussian", type=float,default=0.5)
parser.add_argument('--hidden_size', nargs='+', type=int,default=[256,128])
parser.add_argument('--step_pixels', help="set the number of per-step-pixels", type=int,default=1)
parser.add_argument('--flatten_dim', help="set the number of flatten dimension", type=int,default=784)
parser.add_argument("--device", type=str,default="cuda:1")



torch.set_num_threads(2)


## ================================================ ##
##                   LOAD DATASET                   ##
## ================================================ ##
def load_dataset(task='smnist'):
    if 'smnist' in task:
        if task == 'smnist':
            (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
        if task == 'psmnist':
            X_train = np.load('./dataset/ps_data/ps_X_train.npy')
            X_test = np.load('./dataset/ps_data/ps_X_test.npy')
            y_train = np.load('./dataset/ps_data/Y_train.npy')
            y_test = np.load('./dataset/ps_data/Y_test.npy')
        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_train = torch.from_numpy(y_train).long()
        y_test = torch.from_numpy(y_test).long()
        train_dataset = data.TensorDataset(X_train,y_train) # create train datset
        test_dataset = data.TensorDataset(X_test,y_test) # create test datset
    elif task == 'ecg':
        X_train = np.load('./dataset/ecg/X_train_187.npy')*args.input_gain
        X_test = np.load('./dataset/ecg/X_test_187.npy')*args.input_gain
        y_train = np.load('./dataset/ecg/Y_train_187.npy')
        y_test = np.load('./dataset/ecg/Y_test_187.npy')
        X_train = torch.from_numpy(X_train).float()
        X_test = torch.from_numpy(X_test).float()
        y_train = torch.from_numpy(y_train).long()
        y_test = torch.from_numpy(y_test).long()
        train_dataset = data.TensorDataset(X_train,y_train) 
        test_dataset = data.TensorDataset(X_test,y_test) 
    elif task == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda img:img*255),
            transforms.Lambda(lambda img:img.permute(1,2,0)),
            ])
        train_dataset = dsets.CIFAR10(root='./dataset/cifar10', train=True, download=True,transform=transform)
        test_dataset = dsets.CIFAR10(root='./dataset/cifar10', train=False, download=True,transform=transform)
    else:
        print('only two task, -- smnist and psmnist')
        return 0
    return train_dataset,test_dataset





## ================================================ ##
##                ENCODE FUNCTION                   ##
## ================================================ ##
def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

from code_function import *


## ================================================ ##
##                    PRC MODEL                     ##
## ================================================ ##
from MCSRNN_PRCcomponents import *
from MCSRNN_PRCargs import *

def loss_fn(actual_output, desired_output):
    log_softmax_fn = nn.LogSoftmax(dim=1)
    neg_log_lik_fn = nn.NLLLoss()
    m, _ = torch.max(actual_output, 1)
    log_p_y = log_softmax_fn(m)
    loss_val = neg_log_lik_fn(log_p_y, desired_output.long())
    return loss_val

def get_network(net_type,NETWORK_ARCHITECTURE) -> SpikingNetwork:
    somatic_spike_fn = get_spike_fn(threshold=15)
    dendritic_nl_fn = get_default_dendritic_fn(
        threshold=2, sensitivity=10, gain=1
    )
    neuron_params = RecurrentNeuronParameters(
        tau_mem=10e-3,
        tau_syn=5e-3,
        backprop_gain=0.5,
        feedback_strength=15,
        somatic_spike_fn=somatic_spike_fn,
        dendritic_spike_fn=dendritic_nl_fn,
    )

    parallel_params = PRCNeuronParameters(
        tau_mem=10e-3,
        tau_syn=5e-3,
        backprop_gain=0.05,
        feedback_strength=15,
        somatic_spike_fn=somatic_spike_fn,
        dend_na_fn=dendritic_nl_fn,
        dend_ca_fn=get_sigmoid_fn(threshold=4, sensitivity=10, gain=1),
        dend_nmda_fn=dendritic_nl_fn,
        tau_dend_na=5e-3,
        tau_dend_ca=40e-3,
        tau_dend_nmda=80e-3,
    )

    simple_network_architecture = deepcopy(NETWORK_ARCHITECTURE)
    simple_network_architecture.weight_scale_by_layer = (140, 7)
    two_compartment_network_architecture = deepcopy(NETWORK_ARCHITECTURE)
    two_compartment_network_architecture.weight_scale_by_layer = PRCPARS.ws
    parallel_network_architecture = deepcopy(NETWORK_ARCHITECTURE)
    parallel_network_architecture.weight_scale_by_layer = (1, 7)

    nets = {
        'One compartment': SpikingNetwork(
            neuron_params, simple_network_architecture
        ),
        'No BAP': TwoCompartmentSpikingNetwork(
            neuron_params, two_compartment_network_architecture
        ),
        'BAP': RecurrentSpikingNetwork(
            neuron_params, two_compartment_network_architecture
        ),
        'Parallel subunits, no BAP': ParallelSpikingNetwork(
            parallel_params, parallel_network_architecture
        ),
        'Parallel subunits + BAP (full PRC model)': PRCSpikingNetwork(
            parallel_params, parallel_network_architecture
        ),
    }
    return nets[net_type]



## ================================================ ##
##                    BASE SRNN                     ##
## ================================================ ##

b_j0 = .1#0.01  # neural threshold baseline
tau_m = 20  # ms membrane potential constant
R_m = 1  # membrane resistance
dt = 1   #
gamma = .5  # gradient scale
lens = 0.5

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp =  gaussian(input, mu=0., sigma=lens)
        return grad_input * temp.float() * gamma

act_fun_adp = ActFun_adp.apply


def mem_update_adp(inputs, mem, spike, tau_adp,tau_m, b, dt=1, isAdapt=False):
    #     tau_adp = torch.FloatTensor([tau_adp])
    alpha = torch.exp(-1. * dt / tau_m).to(args.device)
    ro = torch.exp(-1. * dt / tau_adp).to(args.device)
    # tau_adp is tau_adaptative which is learnable # add requiregredients
    if isAdapt:
        beta = 1.8
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = b_j0 + beta * b
    #import pdb
    #pdb.set_trace()
    mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    B = b_j0
    inputs_ = mem - B
    spike = act_fun_adp(inputs_)  # act_fun : approximation firing function
    return mem, spike, B, b

def output_Neuron(inputs, mem, tau_m, dt=1):
    # alpha = torch.exp(-1. * dt / torch.FloatTensor([30.])).to(device)
    alpha = torch.exp(-1. * dt / tau_m).to(args.device)
    mem = mem * alpha + (1. - alpha) * R_m * inputs
    return mem

class RNN_custom(nn.Module):
    def __init__(self, input_size, hidden_dims,num_encoders=30,EC_f='rbf',DC_f='mem'):
        super(RNN_custom, self).__init__()

        self.EC_f = EC_f
        self.DC_f = DC_f

        self.num_encoders = num_encoders
        self.hidden_size = hidden_dims[0]
        self.num_decoder = hidden_dims[1]
        self.input_size = input_size
        self.i2h = nn.Linear(self.input_size, self.hidden_size)
        self.h2h = nn.Linear(self.hidden_size, self.hidden_size)
        self.h2d = nn.Linear(self.hidden_size, self.num_decoder)
        self.d2d = nn.Linear(self.num_decoder, self.num_decoder)
        #self.d2o = nn.Linear(self.num_decoder, output_size)

        self.tau_adp_h = nn.Parameter(torch.Tensor(self.hidden_size))
        self.tau_adp_d = nn.Parameter(torch.Tensor(self.num_decoder))
        #self.tau_adp_o = nn.Parameter(torch.Tensor(output_size))
        self.tau_m_h = nn.Parameter(torch.Tensor(self.hidden_size))
        self.tau_m_d = nn.Parameter(torch.Tensor(self.num_decoder))
        #self.tau_m_o = nn.Parameter(torch.Tensor(output_size))
        self.prc_model = get_network(PRCPARS.net_type,NetworkArchitecture((hidden_dims[1],PRCPARS.hidden_size,PRCPARS.output_size)))
        
        if self.EC_f == 'rbf-lc':
            self.threshold_event = nn.Parameter(torch.tensor(0.2,requires_grad=True))
 

        nn.init.orthogonal_(self.h2h.weight, gain=args.ws)
        nn.init.xavier_uniform_(self.i2h.weight, gain=args.ws)
        nn.init.xavier_uniform_(self.h2d.weight, gain=args.ws)
        nn.init.xavier_uniform_(self.d2d.weight, gain=args.ws)
        
        nn.init.constant_(self.i2h.bias, 0)
        nn.init.constant_(self.h2h.bias, 0)
        nn.init.constant_(self.h2d.bias, 0)
        nn.init.constant_(self.d2d.bias, 0)

        nn.init.normal_(self.tau_adp_h, 700,25)
        nn.init.normal_(self.tau_adp_d, 700,25)

        nn.init.normal_(self.tau_m_h, 20,5)
        nn.init.normal_(self.tau_m_d, 15,5)
        self.b_h = self.b_o  = self.b_d  = 0

    def forward(self, input, prc_reset=True):
        self.b_h = self.b_o = self.b_d = b_j0
        hidden_mem = hidden_spike = torch.zeros([args.batch_size, self.hidden_size]).to(args.device)
        h2d_mem = h2d_spike = torch.zeros([args.batch_size, self.num_decoder]).to(args.device)

        if self.EC_f[:3]=='rbf':
            input_EC = rbf_encode(input.view(args.batch_size,args.flatten_dim,1).float(),args.step_pixels,self.num_encoders,device=args.device)
        if self.EC_f == "poisson":
            input_EC = poisson_encode(input.view(args.batch_size,args.flatten_dim,1).float(),args.step_pixels,self.num_encoders,device=args.device)
        if self.EC_f == "binary":
            input_EC = binary_encode(input.view(args.batch_size,args.flatten_dim,1).float(),args.step_pixels,self.num_encoders,device=args.device)
        if self.EC_f == "none":
            input_EC = input

        #import pdb
        #pdb.set_trace()

        seq_num = args.flatten_dim//args.step_pixels
        prc_input=torch.zeros([args.batch_size,seq_num,args.hidden_size[1]])
        for i in range(seq_num):
            if self.EC_f == 'rbf':
                input_x = input_EC[:,i,:]
            elif self.EC_f == 'rbf-lc':
                input_x = input_EC[:,i,:].gt(self.threshold_event).float().to(args.device)
            elif self.EC_f == 'poisson':
                input_x = input_EC[:,i,:]
            elif self.EC_f == 'binary':
                input_x = input_EC[:,i,:]
            elif self.EC_f == 'none':
                input_x = input_EC[:,i,:]
            
            h_input = self.i2h(input_x.float()) + self.h2h(hidden_spike)
            hidden_mem, hidden_spike, theta_h, self.b_h = mem_update_adp(h_input,hidden_mem, hidden_spike, self.tau_adp_h, self.tau_m_h,self.b_h)
            d_input = self.h2d(hidden_spike) + self.d2d(h2d_spike)
            h2d_mem, h2d_spike, theta_d, self.b_d = mem_update_adp(d_input, h2d_mem, h2d_spike, self.tau_adp_d,self.tau_m_d, self.b_d)
            prc_input[:,i,:] = h2d_spike
            
        output_spike = self.prc_model.run_snn(prc_input.to(args.device),prc_reset)[0]
        import gc
        del input_EC
        gc.collect()
        return output_spike, hidden_spike

def save_model(model):
    params_dict = collections.OrderedDict(model.state_dict())
    #params_dict = model.state_dict()
    for i in range(len(model.prc_model.weights_by_layer)):
        params_dict["prc_weight{}".format(i)] = model.prc_model.weights_by_layer[i]
    torch.save(params_dict, './MCSRNN_result/model/model_'+file_name+'.pth')

def train(model, num_epochs,train_loader,test_loader,file_name,MyFile):
    acc = []
    best_accuracy = 0.1
    for epoch in range(num_epochs):
        loss_list=[]
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, args.flatten_dim, 1).requires_grad_().to(args.device)
            labels = labels.long().to(args.device)
            optimizer.zero_grad()
            actual_output, _ = model(images)
            loss = loss_fn(actual_output, labels)
            #print("epoch:{} i:{} loss:{}".format(epoch,i,loss.item()))
            loss.backward()
            if model.i2h.weight.grad.max()==0:
                print("SRNN gradients = 0!")
            if model.prc_model.weights_by_layer[0].grad.max()==0:
                print("PRC gradients = 0!")
            optimizer.step()
            loss_list.append(loss.item())
        if scheduler:
            scheduler.step()
        accuracy = test(model, train_loader)
        ts_acc = test(model,test_loader)
        if ts_acc > best_accuracy and accuracy > best_accuracy:
            save_model(model)
            #torch.save(model, './MCSRNN_result/model/model_'+file_name+'-tau_adp.pth')
            best_accuracy = ts_acc
        acc.append(accuracy)
        res_str = "epoch: {}, loss: {}, accTR: {}, accTS: {}".format(epoch,np.mean(np.array(loss_list)),accuracy*100,ts_acc*100)
        print(res_str)
        MyFile=open('./MCSRNN_result/'+file_name+'.txt','a')
        MyFile.write(res_str)
        MyFile.write('\n')
        MyFile.close()
    return acc


def train_shd(model, num_epochs, file_name,MyFile):
    acc = []
    best_accuracy = 80
    path_to_train_data = os.path.join(CACHE_DIR, CACHE_SUBDIR, 'shd_train.h5')
    path_to_test_data = os.path.join(CACHE_DIR, CACHE_SUBDIR, 'shd_test.h5')
    with Data(path_to_train_data, path_to_test_data) as data:
        test_shd(model,data.x_train,data.y_train)
        for epoch in range(num_epochs):
            loss_list=[]
            for batch_x, batch_y in sparse_data_generator_from_hdf5_spikes(data.x_train, data.y_train, SWEEP_DURATION, shuffle=True,args=args):
                images = batch_x.to_dense()
                labels = batch_y
                outputs, _ = model(images)
                optimizer.zero_grad()
                loss = loss_fn(outputs, labels)
                #print("epoch:{} loss:{}".format(epoch,loss.item()))
                loss.backward()
                loss_list.append(loss.item())
                if model.i2h.weight.grad.max()==0:
                    print("SRNN gradients = 0!")
                if model.prc_model.weights_by_layer[0].grad.max()==0:
                    print("PRC gradients = 0!")
                optimizer.step()
            if scheduler:
                scheduler.step()
            accuracy  = test_shd(model,data.x_train,data.y_train)
            ts_acc = test_shd(model,data.x_test,data.y_test)
            if ts_acc > best_accuracy and accuracy > 80:
                torch.save(model, './MCSRNN_result/model/model_'+file_name+'-tau_adp.pth')
                best_accuracy = ts_acc
            acc.append(accuracy)
            res_str = "epoch: {}, loss: {}, accTR: {}, accTS: {}".format(epoch,np.mean(np.array(loss_list)),accuracy*100,ts_acc*100)
            print(res_str)
            MyFile=open('./MCSRNN_result/'+file_name+'.txt','a')
            MyFile.write(res_str)
            MyFile.write('\n')
            MyFile.close()
        return acc


def test(model, dataloader):
    with torch.no_grad():
        acc_list = []
        for images, labels in dataloader:
            images = images.view(-1, args.flatten_dim, 1).to(args.device)
            labels = labels.long().to(args.device)
            output, _ = model(images)
            m, _ = torch.max(output, 1)  
            _, am = torch.max(m, 1) 
            acc = np.mean((labels == am).detach().cpu().numpy()) 
            acc_list.append(acc)
        return np.mean(np.array(acc_list))



def test_shd(net,x_data,y_data):
    with torch.no_grad():
        acc_list=[]
        for x_local, y_local in sparse_data_generator_from_hdf5_spikes(x_data, y_data, SWEEP_DURATION, shuffle=False,args=args):
            images = x_local.to_dense()
            labels = y_local.long()
            output, _ = model(images)
            m, _ = torch.max(output, 1) 
            _, am = torch.max(m, 1) 
            acc = np.mean((labels == am).detach().cpu().numpy()) 
            acc_list.append(acc)
        return np.mean(np.array(acc_list))


if __name__ == '__main__':
    args = parser.parse_args()
    scheduler=True
    args.ws = (15,7)
    if args.task == 'shd':
        args.num_encoders = 700
        args.step_pixels = 1
        args.nb_steps = 100
        args.flatten_dim = 100
        args.batch_size = 256
        args.ec_f = 'none'
    else:
        if args.task == 'ecg':
            args.input_gain = 100
            PRCPARS.ws = (50,7)
            args.ws = 5
        train_dataset,test_dataset = load_dataset(args.task)
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True,drop_last=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=args.batch_size,shuffle=False,drop_last=True)
    args.input_size = args.step_pixels*args.num_encoders

    import json
    pars_dict0 = args.__dict__.copy()
    pars_dict0['scheduler']=str(scheduler)
    pars_dict1={}
    for item in dir(PRCPARS):
        if item[0]!="_":
            pars_dict1[item]=str(PRCPARS.__dict__[item])
    pars_json=json.dumps({"base_pars":pars_dict0,"prc_pars":pars_dict1},indent=4)
    print(pars_json)

    model = RNN_custom(args.input_size, args.hidden_size, num_encoders=args.num_encoders,EC_f=args.ec_f,DC_f=args.ec_f)
    print("device:",args.device)
    model.to(args.device)
    criterion = nn.CrossEntropyLoss()


    if args.ec_f == 'rbf-lc':
        base_params = [model.i2h.weight, model.i2h.bias, 
                model.h2h.weight, model.h2h.bias, 
                model.h2d.weight, model.h2d.bias,
                model.d2d.weight, model.d2d.bias, 
                model.threshold_event]
    else:
        base_params = [model.i2h.weight, model.i2h.bias, 
                model.h2h.weight, model.h2h.bias, 
                model.h2d.weight, model.h2d.bias,
                model.d2d.weight, model.d2d.bias, 
                ]

    prc_params = model.prc_model.weights_by_layer

    optimizer = torch.optim.Adam([
        {'params': base_params},
        {'params': model.tau_adp_h, 'lr': args.learning_rate * 2},
        {'params': model.tau_adp_d, 'lr': args.learning_rate * 3},
        {'params': model.tau_m_h, 'lr': args.learning_rate * 2},
        {'params': model.tau_m_d, 'lr': args.learning_rate * 2},
        {'params': prc_params, 'lr': PRCPARS.learning_rate, 'betas':(0.9, 0.999)}],
        lr=args.learning_rate)

    if scheduler:
        scheduler = StepLR(optimizer, step_size=25, gamma=.75)
        scheduler = MultiStepLR(optimizer, milestones=[25,50,100,150],gamma=0.5)
    now = datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    
    file_name = dt_string+'||TASK-'+args.task+'||ECF-'+args.ec_f+'||BS-'+str(args.batch_size)\
            +'||NET-'+'{}_{}_{}_{}({})_{}'.format(args.input_size,args.hidden_size[0],args.hidden_size[1],PRCPARS.hidden_size,PRCPARS.net_type,PRCPARS.output_size)\
            +'||LR-'+str(args.learning_rate)
    print(file_name)
    MyFile=open('./MCSRNN_result/'+file_name+'.txt','a')
    MyFile.write(file_name)
    MyFile.write('\n\n =========== Hyperpars ========== \n')
    MyFile.write(pars_json)
    MyFile.write('\n\n ============ Result ============ \n')
    MyFile.close()

    if args.task == 'shd':
        acc = train_shd(model,args.num_epochs,file_name,MyFile)
    else:
        acc = train(model, args.num_epochs,train_loader,test_loader,file_name,MyFile)

# python s_mnist-gpu.py --task smnist --ec_f rbf --dc_f adp-spike --step_pixels 28
# python MCSRNN_main.py --task smnist --ec_f binary --num_encoders 8 --dc_f adp-spike --step_pixels 28 --learning_rate 0.001
# python MCSRNN_main.py --task cifar10 --ec_f poisson --num_encoders 80 --step_pixels 96 --flatten_dim 3072 --learning_rate 0.0005 --hidden_size 1024 512 --num_epochs 1500
# python MCSRNN_main.py --task psmnist --ec_f binary --num_encoders 8 --dc_f adp-spike --step_pixels 28 --learning_rate 0.001 --num_epochs 1500
# python MCSRNN_main.py --task shd --dc_f adp-spike --learning_rate 0.0001 --num_epochs 1500
# python MCSRNN_main.py --task ecg --dc_f adp-spike --learning_rate 0.001 --num_epochs 1500 --flatten_dim 187 --step_pixels 1 --num_encoders 1 --batch_size 20050