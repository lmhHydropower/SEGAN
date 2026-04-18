import torch as t
import pandas as pd
from tools.gans_utils import load_data_pakages, plot_and_save
from tools.FGANomaly_utils import seed_all
from tools.FGANomaly import FGANomalyModel, RNNAutoEncoder, MLPDiscriminator
from tools.EGAN import EGANModel
from tools.TadGAN import TadGANModel
from tools.my_enums import Stage
from tools.final_results import final_results, statistical_results, process_speed_time

import numpy as np
from typing import List,Dict,Tuple

seed_all(2024)

# set GANs parameters
# params_FGAN params_TadGAN params_EGAN

def main_FGAN(stage : Stage = Stage.STAGE_1, timestamp : int = 0, is_only_get_times : bool = False):
    current_model = 'FGAN'
    data = load_data_pakages(
                     current_model=current_model,
                     val_size = params_FGAN['val_size'],
                     test_size = params_FGAN['test_size'],
                     window_size=params_FGAN['window_size'],
                     batch_size=params_FGAN['batch_size'],
                     idx_slice=params_FGAN['slice'],
                     stage=stage,
                     is_repeat=params_FGAN['is_repeat'],
                     timestamp=timestamp
                     )

    model = FGANomalyModel(ae=RNNAutoEncoder(inp_dim=data['nc'],
                                             z_dim=params_FGAN['z_dim'],
                                             hidden_dim=params_FGAN['hidden_dim'],
                                             rnn_hidden_dim=params_FGAN['rnn_hidden_dim'],
                                             num_layers=params_FGAN['num_layers'],
                                             bidirectional=params_FGAN['bidirectional'],
                                             cell=params_FGAN['cell']),
                           dis_ar=MLPDiscriminator(inp_dim=data['nc'],
                                                   hidden_dim=params_FGAN['hidden_dim']),
                           data_packs=data, **params_FGAN)
    
    if not is_only_get_times:
        losses : Dict[str, List[float]] = model.train()

        full_re_data ,_  = model.test(load_from_file=True)

        plot_and_save(losses, full_re_data, data, current_model)

        return data["paths"]
    else:
        _, testset_times = model.test(load_from_file=True)
        return testset_times

def main_TadGAN(stage : Stage = Stage.STAGE_1, timestamp : int = 0, is_only_get_times : bool = False):
    
    current_model = 'TadGAN'

    data = load_data_pakages(
        current_model = current_model,
        val_size      = params_TadGAN['val_size'],
        test_size     = params_TadGAN['test_size'],
        window_size   = params_TadGAN['window_size'],
        batch_size    = params_TadGAN['batch_size'],
        idx_slice     = params_TadGAN['slice'],
        stage         = stage,
        is_repeat     = params_TadGAN['is_repeat'],
        timestamp     = timestamp
    )

    
    model = TadGANModel(data, **params_TadGAN)

    if not is_only_get_times:
        losses : Dict[str, list]  = model.train()

        full_re_data ,_ = model.test(load_from_file=True)
        # losses : Dict[str, list] = {}
        plot_and_save(losses, full_re_data, data, current_model)

        return data["paths"]
    else:
        _, testset_times = model.test(load_from_file=True)
        return testset_times
    

def main_EGAN(stage : Stage = Stage.STAGE_1, timestamp : int = 0, is_only_get_times : bool = False):
    current_model = 'EGAN'
    data = load_data_pakages(
                     current_model = current_model,
                     val_size      = params_EGAN['val_size'],
                     test_size     = params_EGAN['test_size'],
                     window_size   = params_EGAN['window_size'],
                     batch_size    = params_EGAN['batch_size'],
                     idx_slice     = params_EGAN['slice'],
                     stage         = stage,
                     is_repeat     = params_EGAN['is_repeat'], 
                     is_raw        = False,
                     timestamp     = timestamp
                     )
    data_raw = load_data_pakages(
                        current_model = current_model,
                        val_size      = params_EGAN['val_size'],
                        test_size     = params_EGAN['test_size'],
                        window_size   = params_EGAN['window_size'],
                        batch_size    = params_EGAN['batch_size'],
                        idx_slice     = params_EGAN['slice'],
                        stage         = stage,
                        is_repeat     = params_EGAN['is_repeat'],
                        is_raw        = True,
                        only_get_data = True,
                        timestamp     = timestamp
                        )
    
    model = EGANModel(data_packs=data, **params_EGAN)
    
    if not is_only_get_times:
        losses : Dict[str, List[float]] = model.train()
    
        full_re_data ,_ = model.test(load_from_file=True)
        
        plot_and_save(losses, full_re_data, data, current_model, data_raw)
        
        return data["paths"]
    else:
        _, testset_times = model.test(load_from_file=True)

        return testset_times

    

def main_AE(stage : Stage = Stage.STAGE_1, timestamp : int = 0, is_only_get_times : bool = False):
    current_model = 'AE'
    data = load_data_pakages(
                     current_model = current_model,
                     val_size      = params_EGAN['val_size'],
                     test_size     = params_EGAN['test_size'],
                     window_size   = params_EGAN['window_size'],
                     batch_size    = params_EGAN['batch_size'],
                     idx_slice     = params_EGAN['slice'],
                     stage         = stage,
                     is_repeat     = params_FGAN['is_repeat'],
                     timestamp     = timestamp
                     )

    model = EGANModel(data_packs=data, **params_EGAN)
    model.only_ae = True
    model.is_weighted_loss = False
    
    if not is_only_get_times:
        losses : Dict[str, List[float]] = model.train()

        full_re_data ,_ = model.test(load_from_file=True)

        plot_and_save(losses, full_re_data, data, current_model)

        return data["paths"]
    else:
        _, testset_times = model.test(load_from_file=True)

        return testset_times

if __name__ == '__main__':

    params_EGAN['slice'] = slice(1,2)
    params_TadGAN['slice'] = slice(1,2)
    params_FGAN['slice'] = slice(1,2)

    model_names = ['AE', 'TadGAN', 'FGANomaly', 'SEGAN']

    models_test = [main_AE, main_TadGAN, main_FGAN, main_EGAN]

    all_model_times = pd.DataFrame(index=model_names, columns=['mean', 'std'])

    for i, model_name in enumerate(model_names):
        print(f"{i} : {model_name}")

        testset_times = models_test[i](Stage.STAGE_1, 1, is_only_get_times = True)

        _, model_time_speed = process_speed_time(testset_times, model_name)

        all_model_times.loc[model_name] = model_time_speed

