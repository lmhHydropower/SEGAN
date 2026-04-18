import os
from time import time
from copy import deepcopy
from typing import Union, List, Tuple, Dict

import torch as t
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
from torch.nn import MSELoss, BCELoss
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor

from tools.EGAN_utils import AdaWeightedLoss
from tools.generate_dataset import GenerateData
from tools.tcn import TemporalConvNet


class RNNEncoder(nn.Module):
    """
    An implementation of Encoder based on Recurrent neural networks.
    """
    def __init__(self, inp_dim, z_dim, hidden_dim, rnn_hidden_dim, num_layers, bidirectional=False, cell='lstm', isLinear = False):
        """
        args:
            inp_dim: dimension of input value
            z_dim: dimension of latent code
            hidden_dim: dimension of fully connection layers
            rnn_hidden_dim: dimension of rnn cell hidden states
            num_layers: number of layers of rnn cell
            bidirectional: whether use BiRNN cell
            cell: one of ['lstm', 'gru', 'rnn']
        """
        super(RNNEncoder, self).__init__()

        self.inp_dim = inp_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.isLinear = isLinear

        if self.isLinear:
            self.linear1 = nn.Linear(inp_dim, hidden_dim)
            
        else:
            self.tcn = TemporalConvNet(inp_dim, kernel_size=3, channel_sizes=[8, 16, hidden_dim], dropout=0.2)
            self.conv1 = weight_norm(nn.Conv1d(hidden_dim, z_dim, kernel_size=3))

        if bidirectional:
            self.linear2 = nn.Linear(self.rnn_hidden_dim * 2, z_dim)
        else:
            self.linear2 = nn.Linear(self.rnn_hidden_dim, z_dim)

        if cell == 'lstm':
            self.rnn = nn.LSTM(hidden_dim,
                               rnn_hidden_dim,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               batch_first = True)
        elif cell == 'gru':
            self.rnn = nn.GRU(hidden_dim,
                              rnn_hidden_dim,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              batch_first = True)
        else:
            self.rnn = nn.RNN(hidden_dim,
                              rnn_hidden_dim,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              batch_first = True)
        
        
    def forward(self, inp):
        # inp shape: [bsz, seq_len, inp_dim]
        self.rnn.flatten_parameters()
        if self.isLinear:
            rnn_inp = t.relu(self.linear1(inp))
            rnn_out, _ = self.rnn(rnn_inp)
            z = t.relu(self.linear2(rnn_out))
        else:
            rnn_inp = self.tcn(inp.permute(0, 2, 1)).permute(0, 2, 1)
            rnn_out, _ = self.rnn(rnn_inp)
            z = t.relu(self.linear2(rnn_out))
        return z, rnn_inp, rnn_out


class RNNDecoder(nn.Module):
    """
    An implementation of Decoder based on Recurrent neural networks.
    """
    def __init__(self, inp_dim, z_dim, hidden_dim, rnn_hidden_dim, num_layers, bidirectional=False, cell='lstm'):
        """
        args:
            Reference argument annotations of RNNEncoder.
        """
        super(RNNDecoder, self).__init__()

        self.inp_dim = inp_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.linear1 = nn.Linear(z_dim, hidden_dim)
        if bidirectional:
            self.linear2 = nn.Linear(self.rnn_hidden_dim * 2, inp_dim)
        else:
            self.linear2 = nn.Linear(self.rnn_hidden_dim, inp_dim)

        if cell == 'lstm':
            self.rnn = nn.LSTM(hidden_dim,
                               rnn_hidden_dim,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               batch_first = True)
        elif cell == 'gru':
            self.rnn = nn.GRU(hidden_dim,
                              rnn_hidden_dim,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              batch_first = True)
        else:
            self.rnn = nn.RNN(hidden_dim,
                              rnn_hidden_dim,
                              num_layers=num_layers,
                              bidirectional=bidirectional,
                              batch_first = True)
        
        
    def forward(self, z):
        # z shape: [bsz, seq_len, z_dim]
        self.rnn.flatten_parameters()
        rnn_inp = t.relu(self.linear1(z))
        rnn_out, _ = self.rnn(rnn_inp)
        re_x = self.linear2(rnn_out)
        return re_x, rnn_inp, rnn_out



class AutoEncoder(nn.Module):
    def __init__(self, inp_dim, z_dim, hidden_dim, rnn_hidden_dim, num_layers, ks = 3, bidirectional=False, cell='lstm'):
        super(AutoEncoder, self).__init__()

        self.encoder = RNNEncoder(inp_dim, z_dim, hidden_dim, rnn_hidden_dim,
                                  num_layers, bidirectional=bidirectional, cell=cell)
        self.decoder = RNNDecoder(inp_dim, z_dim, hidden_dim, rnn_hidden_dim,
                                  num_layers, bidirectional=bidirectional, cell=cell)
        
        # self.weigthnet = WeightNet(hidden_dim, rnn_hidden_dim, inp_dim, ks, bidirectional)

    def forward(self, inp):
        # inp shape: [bsz, seq_len, inp_dim]
        z     , enc_rnn_in, enc_rnn_out = self.encoder(inp)
        re_inp, dec_rnn_in, dec_rnn_out = self.decoder(z)
        # weigth = self.weigthnet(enc_rnn_out - dec_rnn_out, enc_rnn_in - dec_rnn_in, inp - re_inp )
        return re_inp, z#, weigth


class MLPDiscriminator(nn.Module):
    def __init__(self, inp_dim, hidden_dim):
        super(MLPDiscriminator, self).__init__()

        self.dis = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),

            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, inp):
        seq, df = inp.shape
        c = self.dis(inp)
        return c.view(seq)


class EGANModel(object):
    def __init__(self, data_packs, **kwargs):
        self.params = kwargs

        self.only_ae = False

        ae = AutoEncoder(inp_dim        = data_packs['nc'],
                            z_dim          = kwargs['z_dim'],
                            hidden_dim     = kwargs['hidden_dim'],
                            rnn_hidden_dim = kwargs['rnn_hidden_dim'],
                            num_layers     = kwargs['num_layers'],
                            bidirectional  = kwargs['bidirectional'],
                            cell           = kwargs['cell'])

        dis_ar=MLPDiscriminator(inp_dim    = 1,
                                hidden_dim = kwargs['hidden_dim'])


        self.print_param()
        self.print_model(ae, dis_ar)

        self.device = kwargs['device']
        self.lr = kwargs['lr']
        self.epoch = kwargs['epoch']
        self.window_size = kwargs['window_size']
        self.early_stop = kwargs['early_stop']
        self.early_stop_tol = kwargs['early_stop_tol']
        self.if_scheduler = kwargs['if_scheduler']

        self.adv_rate = kwargs['adv_rate']
        self.dis_ar_iter = kwargs['dis_ar_iter']

        self.is_weighted_loss = kwargs['weighted_loss']
        self.strategy = kwargs['strategy']

        self.is_time_sample = kwargs['time_sampling']

        self.ae = ae.to(self.device)
        self.dis_ar = dis_ar.to(self.device)
        self.data_pakage = data_packs

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        self.mse = MSELoss()
        self.bce = BCELoss()
        self.ada_mse = AdaWeightedLoss(self.strategy)
        self.weigth = None

        self.ae_optimizer = Adam(params=self.ae.parameters(), lr=self.lr)
        self.ae_scheduler = lr_scheduler.StepLR(optimizer=self.ae_optimizer,
                                                step_size=kwargs['scheduler_step_size'],
                                                gamma=kwargs['scheduler_gamma'])
        self.ar_optimizer = Adam(params=self.dis_ar.parameters(), lr=self.lr)
        self.ar_scheduler = lr_scheduler.StepLR(optimizer=self.ar_optimizer,
                                                step_size=kwargs['scheduler_step_size'],
                                                gamma=kwargs['scheduler_gamma'])

        self.cur_step = 0
        self.cur_epoch = 0
        self.best_ae = AutoEncoder(inp_dim        = data_packs['nc'],
                            z_dim          = kwargs['z_dim'],
                            hidden_dim     = kwargs['hidden_dim'],
                            rnn_hidden_dim = kwargs['rnn_hidden_dim'],
                            num_layers     = kwargs['num_layers'],
                            bidirectional  = kwargs['bidirectional'],
                            cell           = kwargs['cell']).to(self.device)
        self.best_dis_ar = None
        self.best_val_loss = np.inf
        self.val_loss = None
        self.early_stop_count = 0
        self.re_loss = None
        self.adv_dis_loss = None
        self.time_per_epoch = None



    def train(self) -> Dict[str, List[float]]:
        print('*' * 20 + 'Start training' + '*' * 20)
        re_losses : List[float] = []
        val_losses : List[float] = []
        adv_losses : List[float] = []

        result : Dict[str, List[float]] = {'re_loss':re_losses, 'val_loss':val_losses, 'adv_loss':adv_losses}
        
        for idx_loader in range(len(self.data_pakage['train'])):
            self.train_loader = self.data_pakage['train'][idx_loader]
        
            for i in range(self.epoch):
                self.cur_epoch += 1

                self.train_epoch() 
                self.validate()

                if self.val_loss < self.best_val_loss and self.best_val_loss - self.val_loss >= 1e-4:
                    self.best_val_loss = self.val_loss
                    self.best_ae.load_state_dict(deepcopy(self.ae.state_dict()))
                    self.best_dis_ar = deepcopy(self.dis_ar)
                    self.save_best_model()
                    self.early_stop_count = 0
                elif self.early_stop:
                    self.early_stop_count += 1
                    if self.early_stop_count > self.early_stop_tol:
                        print('*' * 20 + 'Early stop' + '*' * 20)
                        self.early_stop_count = 0
                        break
                        # return result
                else:
                    pass
                
                
                left_time = self.time_per_epoch * ((self.epoch-i-1) + (len(self.data_pakage['train'])-idx_loader-1) * self.epoch)

                left_hours = left_time // 3600
                left_minutes = (left_time % 3600) // 60
                left_seconds = left_time % 60
                print(f'[DataLoader {idx_loader+1:2}/{len(self.data_pakage["train"])}], Epoch {i+1:3}/{self.epoch}] train_loss:{self.re_loss:.5f}, val_loss:{self.val_loss:.5f}, adv loss is {self.adv_dis_loss:.5f}, time per epoch is {self.time_per_epoch:.5f}')

                result['re_loss'].append(self.re_loss.item())
                result['val_loss'].append(self.val_loss.item())
                if not self.only_ae:
                    result['adv_loss'].append(self.adv_dis_loss.item())
                else:
                    result['adv_loss'].append(self.adv_dis_loss)

        return result

    def train_epoch(self):
        start_time = time()
        
        for x in self.train_loader:
            self.cur_step += 1
            x = x.to(self.device)
            
            if not self.only_ae:
                for _ in range(self.dis_ar_iter):
                    self.dis_ar_train(x)
            
            self.ae_train(x)
        
        end_time = time()
        
        self.time_per_epoch = end_time - start_time
        
        if self.if_scheduler:
            if not self.only_ae: self.ar_scheduler.step()
            self.ae_scheduler.step()

    def dis_ar_train(self, x):
        self.ar_optimizer.zero_grad()

        re_x, z  = self.ae(x)

        if self.is_time_sample:
            hard_label = self.temporal_sampling_eliminator(x, re_x)

            re_dis_loss = t.tensor(0.0, dtype=t.float, device=self.device, requires_grad=True)
            actual_dis_loss = t.tensor(0.0, dtype=t.float, device=self.device, requires_grad=True)
        
            for i in range(x.shape[2]):
                single_x = x[:, :, [i]]
                single_re_x = re_x[:, :, [i]]

                actual_normal = single_x[t.where(hard_label[:, :, i] == 0)]
                re_normal = single_re_x[t.where(hard_label[:, :, i] == 0)]

                actual_target = t.ones(size=(actual_normal.shape[0],), dtype=t.float, device=self.device)
                re_target = t.zeros(size=(actual_normal.shape[0],), dtype=t.float, device=self.device)

                re_logits = self.dis_ar(re_normal)
                actual_logits = self.dis_ar(actual_normal)

                re_dis_loss = re_dis_loss + self.bce(input=re_logits, target=re_target)
                actual_dis_loss = actual_dis_loss + self.bce(input=actual_logits, target=actual_target)
        else:

            hard_label = t.ones_like(x, dtype=t.float, device=self.device)
            
            re_dis_loss = t.tensor(0.0, dtype=t.float, device=self.device, requires_grad=True)
            actual_dis_loss = t.tensor(0.0, dtype=t.float, device=self.device, requires_grad=True)
        
            for i in range(x.shape[2]):
                single_x = x[:, :, [i]]
                single_re_x = re_x[:, :, [i]]

                actual_normal = single_x[t.where(hard_label[:, :, i] == 0)]
                re_normal = single_re_x[t.where(hard_label[:, :, i] == 0)]

                actual_target = t.ones(size=(actual_normal.shape[0],), dtype=t.float, device=self.device)
                re_target = t.zeros(size=(actual_normal.shape[0],), dtype=t.float, device=self.device)

                re_logits = self.dis_ar(re_normal)
                actual_logits = self.dis_ar(actual_normal)

                re_dis_loss = re_dis_loss + self.bce(input=re_logits, target=re_target)
                actual_dis_loss = actual_dis_loss + self.bce(input=actual_logits, target=actual_target)

        dis_loss = re_dis_loss + actual_dis_loss
        dis_loss.backward()
        self.ar_optimizer.step()

    def dis_ar_train_no_filter(self, x):
        self.ar_optimizer.zero_grad()

        bsz, seq, fd = x.shape
        re_x, z = self.ae(x)

        re_x = re_x.contiguous().view(bsz * seq, fd)
        x = x.contiguous().view(bsz * seq, fd)

        actual_target = t.ones(size=(x.shape[0],), dtype=t.float, device=self.device)
        re_target = t.zeros(size=(re_x.shape[0],), dtype=t.float, device=self.device)

        re_logits = self.dis_ar(re_x)
        actual_logits = self.dis_ar(x)

        re_dis_loss = self.bce(input=re_logits, target=re_target)
        actual_dis_loss = self.bce(input=actual_logits, target=actual_target)

        dis_loss = re_dis_loss + actual_dis_loss
        dis_loss.backward()
        self.ar_optimizer.step()

    def ae_train(self, x):
        bsz, seq, fd = x.shape
        self.ae_optimizer.zero_grad()

        re_x, z = self.ae(x)
        
        
        # reconstruction loss
        if self.is_weighted_loss:
            # self.re_loss = self.ada_mse(re_x, x, self.cur_step, self.weigth)
            
            self.re_loss = self.weighted_mse_loss(re_x, x, self.weigth)
        else:
            self.re_loss = self.mse(re_x, x)
        
        if not self.only_ae:
            adv_dis_loss = t.tensor(0.0, dtype=t.float, device=self.device, requires_grad=True)
            # adversarial loss
            for i in range(fd):

                ar_inp = re_x[:,:,i].contiguous().view(bsz*seq, 1)
                actual_target = t.ones(size=(ar_inp.shape[0],), dtype=t.float, device=self.device)
                re_logits = self.dis_ar(ar_inp)
                adv_dis_loss = adv_dis_loss + self.bce(input=re_logits, target=actual_target)

            self.adv_dis_loss = adv_dis_loss

            loss = self.re_loss + self.adv_dis_loss * self.adv_rate
        else:
            loss = self.re_loss
            self.adv_dis_loss = 0
    
        loss.backward()
        self.ae_optimizer.step()

    def weighted_mse_loss(self, predictions, targets, weights):
        mse = (predictions - targets) ** 2
        
        if weights is not None:
            weighted_mse = mse * weights
        else:
            weighted_mse = mse
        return weighted_mse.mean()


    def validate(self):
        self.ae.eval()

        val_loss : List[float] = []
        for idx_loader in range(len(self.data_pakage['val_set'])):
            raw_values = self.data_pakage['val_set'][idx_loader].data

            re_values, _, _ = self.value_reconstruction(raw_values, self.window_size)
            val_loss = mean_squared_error(y_true=raw_values, y_pred=re_values)
        
        self.val_loss = np.array(val_loss).mean()

        self.ae.train()

    def test(self, load_from_file : bool =False) -> Dict[str,Tuple[List[np.ndarray], List[np.ndarray]]]:
        if load_from_file:
            self.load_best_model()

        if not next(self.best_ae.parameters()).is_cuda:
            self.best_ae.to(self.device)

        self.best_ae.eval()

        keys = ['train_set', 'val_set', 'test_set']

        results : Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}

        # init the result dict using the keys
        for k in keys:
            re_sets : List[np.ndarray] = []
            ws : List[np.ndarray] = []
            results[k] = (re_sets, ws)

        time_speed = dict()
        for my_set, re_list in results.items():
            
            for idx_set in range(len(self.data_pakage[my_set])):

                data_set : GenerateData = self.data_pakage[my_set][idx_set]
                raw_values : np.ndarray = data_set.data
                
                re_values, ws, times = self.value_reconstruction(raw_values, self.window_size, val=False)

                time_speed[data_set.pile_name] = times
                
                trans : StandardScaler = data_set.trans
                re_values = trans.inverse_transform(re_values)

                re_list[0].append(re_values)
                re_list[1].append(ws)


        return results, time_speed

    def value_reconstruction(self, values, window_size, val=True):
        piece_num = len(values) // window_size

        features = values.shape[1]
        
        reconstructed_values = []
        ws = []

        time_spends = []

        for i in range(piece_num+1):
            raw_values = values[i * window_size:(i + 1) * window_size, :]

            if raw_values.shape[0] == 0: break

            # convert to ndarray
            try:
                raw_np_array = np.array(raw_values).reshape(1, window_size, features)
            except:
                raw_np_array = np.array(raw_values).reshape(1, raw_values.shape[0],features)

            raw_values = t.from_numpy(raw_np_array).float().to(self.device)
            
            start_time = time()
            
            if val:
                reconstructed_value_, z = self.ae(raw_values)
            else:
                reconstructed_value_, z = self.best_ae(raw_values)
            
            end_time = time()

            time_spends.append(end_time - start_time)

            if raw_np_array.shape[1] != window_size:
                reconstructed_value_ = reconstructed_value_[:, :raw_values.shape[1], :]
                # w = w[:, :raw_values.shape[1], :]
            # w = 1 - w
            reconstructed_value_ = reconstructed_value_.squeeze(0).detach().cpu().tolist()
            # w = w.squeeze(0).detach().cpu().tolist()
            reconstructed_values.extend(reconstructed_value_)
            # ws.extend(w)
        return np.array(reconstructed_values), np.array(ws), time_spends



    def temporal_sampling_eliminator(self, values, re_values):
        
        with t.no_grad():
            batch_size, seq_len, feature = values.size()
            
            errors = values - re_values
            weights = t.zeros_like(errors)
            
            for i in range(feature):
                error = errors[:, :, i]  # Shape: (batch_size, seq_len)
                Q1 = t.quantile(error, 0.25, dim=-1, keepdim=True)
                Q3 = t.quantile(error, 0.75, dim=-1, keepdim=True)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                inliers_mask = (error > lower_bound) & (error < upper_bound)
                sampled = t.zeros_like(error)
                for bt in range(batch_size):
                    sampled[bt, :] = self.forgetting_mechanism(error[bt, :])
                mean = t.mean(error, dim=-1, keepdim=True)
                std = t.std(error, dim=-1, keepdim=True)
                z_score = (error - mean) / (std + 1e-8)  # Add epsilon for numerical stability
                z_score = t.sigmoid(z_score)
                
                inliers_float = inliers_mask.float()
                weights[:, :, i] = inliers_float * z_score * sampled
            
            self.weights = weights
            
            return weights
    def forgetting_mechanism(self, error_sequence):
        
        seq_len = error_sequence.size(0)
        
        decay_rate = 0.95
        time_indices = t.arange(seq_len, dtype=t.float32, device=error_sequence.device)
        sampled = decay_rate ** (seq_len - 1 - time_indices)
        # error_abs = t.abs(error_sequence)
        # threshold = t.quantile(error_abs, 0.75)
        # sampled = t.where(error_abs < threshold, 
        #                   t.ones_like(error_abs), 
        #                   t.exp(-error_abs / threshold))
        
        return sampled


        

    def save_best_model(self):
        if not os.path.exists(self.data_pakage['paths']['pt']):
            os.makedirs(self.data_pakage['paths']['pt'])

        t.save(self.best_ae, os.path.join(self.data_pakage['paths']['pt'],
                                          'ae_'+str(self.params['strategy'])+'_'+str(self.params['adv_rate'])+'.pth'))
        t.save(self.best_dis_ar, os.path.join(self.data_pakage['paths']['pt'],
                                              'dis_'+str(self.params['strategy'])+'_'+str(self.params['adv_rate'])+'.pth'))

    def load_best_model(self):
        
        self.best_ae = t.load(os.path.join(self.data_pakage['paths']['pt'], 'ae_'+str(self.params['strategy'])+'_'+str(self.params['adv_rate'])+'.pth'))
        self.best_dis_ar = t.load(os.path.join(self.data_pakage['paths']['pt'], 'dis_'+str(self.params['strategy'])+'_'+str(self.params['adv_rate'])+'.pth'))

    def print_param(self):
        print('*'*20+'parameters'+'*'*20)
        for k, v in self.params.items():
            print(k+' = '+str(v))
        print('*' * 20 + 'parameters' + '*' * 20)

    def print_model(self, ae, dis_ar):
        print(ae)
        print(dis_ar)
