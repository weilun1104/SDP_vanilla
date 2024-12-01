import math
import pandas as pd
from ax.service.utils.instantiation import ObjectiveProperties
from ax.service.ax_client import AxClient
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from omegaconf import DictConfig
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods.gaussian_likelihood import (
    GaussianLikelihood,
)
from operator import attrgetter
from gpytorch.priors import GammaPrior
from benchmarking.eval_utils import get_model_hyperparameters
from benchmarking.mappings import (
    get_test_function,
    ACQUISITION_FUNCTIONS,
    INITS
)
from benchmarking.gp_priors import (
    MODELS,
    get_covar_module
)
import os
import numpy as np
from os.path import dirname, abspath, join
import json
import hydra
import torch
from time import time
from Model.Point_search.CONBO import plot, save_data

# 定义约束函数
def constraint_1(y):
    return (y[0, 0] > (60)).item()

# def constraint_2(y):
#     return (y[0, 1] < (1e-3 / 1.8)).item()

def constraint_3(y):
    return (y[0, 2] > (60)).item()

def constraint_4(y):
    return (y[0, 3] > (4e6)).item()

def all_constraints(y):
    return constraint_1(y) and constraint_3(y) and constraint_4(y)

def save_data(best_all_y, iter_times, save_path):
    best_all_y1 = [item[0][0].item() for item in best_all_y]
    best_all_y2 = [item[0][1].item() for item in best_all_y]
    best_all_y3 = [item[0][2].item() for item in best_all_y]
    best_all_y4 = [item[0][3].item() for item in best_all_y]
    # 将5个list保存到一个CSV文件中
    df = pd.DataFrame({
        'iter_times': iter_times,
        'gain(db)': best_all_y1,
        'dc_current': best_all_y2,
        'phase': best_all_y3,
        'GBW(MHZ)': best_all_y4,
    })
    df.to_csv(save_path, index=False)

@hydra.main(config_path='./configs', config_name='tonf')
def main(cfg: DictConfig) -> None:
    for m in range(404, 407):
        seed = m
        torch.manual_seed(seed)
        np.random.seed(seed)
        q = cfg.q
        benchmark = cfg.benchmark.name
        if hasattr(cfg.benchmark, 'outputscale'):
            test_function = get_test_function(
                benchmark, float(cfg.benchmark.noise_std), cfg.seed, float(cfg.benchmark.outputscale))

        else:
            test_function = get_test_function(
                name=cfg.benchmark.benchmark,
                noise_std=float(cfg.benchmark.noise_std),
                seed=cfg.seed,
                bounds=cfg.benchmark.bounds,
            )
        
        if cfg.init == "sqrt":
            factor = cfg.init_factor
            num_init = math.ceil(factor * len(cfg.benchmark.bounds) ** 0.5)

        elif isinstance(cfg.benchmark.num_init, int):
            num_init = max(cfg.benchmark.num_init, cfg.q)

        num_bo = cfg.benchmark.num_iters - num_init

        if cfg.acq.name == 'Sampling':
            num_init = cfg.benchmark.num_iters
            num_bo = 0
        acq_func = ACQUISITION_FUNCTIONS[cfg.acq.acq_func]
        bounds = torch.transpose(torch.Tensor(cfg.benchmark.bounds), 1, 0)

        if hasattr(cfg.acq, 'acq_kwargs'):
            acq_func_kwargs = dict(cfg.acq.acq_kwargs)
        else:
            acq_func_kwargs = {}


        model_kwargs = get_covar_module(cfg.model.model_name, len(
            bounds.T), 
            gp_params=cfg.model.get('gp_params', None),
            gp_constraints=cfg.model.get('gp_constraints', {})
        )

        model_enum = Models.BOTORCH_MODULAR
        init_type = INITS['sobol']
        init_kwargs = {"seed": int(cfg.seed)}
        steps = [
            GenerationStep(
                model=init_type,
                num_trials=num_init,
                model_kwargs=init_kwargs,
            )
        ]
        opt_setup = cfg.acq_opt
        model = MODELS[cfg.model.gp]

        bo_step = GenerationStep(
            model=model_enum,
            num_trials=num_bo,
            model_kwargs={  # Kwargs to pass to `BoTorchModel.__init__`
                "surrogate": Surrogate(
                    botorch_model_class=model,
                    covar_module_class=model_kwargs["covar_module_class"],
                    covar_module_options=model_kwargs["covar_module_options"],
                    likelihood_class=model_kwargs["likelihood_class"],
                    likelihood_options=model_kwargs["likelihood_options"],
                ),
                "botorch_acqf_class": acq_func,
                "acquisition_options": {**acq_func_kwargs},
            },
            model_gen_kwargs={"model_gen_options": {  # Kwargs to pass to `BoTorchModel.gen`
                "optimizer_kwargs": dict(opt_setup)},
            },
        )
        steps.append(bo_step)

        best_y = [[6.0105e+1,3.2510e-4,6.2355e+1,9.9196e+6]]
        best_y = torch.tensor(best_y)
        best_all_y = [best_y]
        min_current_value = 3.2510e-4

        def  evaluate(best_all_y,best_y,min_current_value,parameters, seed=None):
            x = torch.tensor(
                [[parameters.get(f"x_{i+1}") for i in range(len(cfg.benchmark.bounds))]])

            if seed is not None:
                bc_eval = test_function.evaluate_true(x, seed=seed).squeeze().tolist()
            else:
                y = test_function(x)
                bc_eval = y[1]

            return {benchmark: (bc_eval,None)},best_all_y,best_y,min_current_value

        gs = GenerationStrategy(
            steps=steps
        )
        # 定义自定义初始点
        initial_point = {"x_1": 35.0, "x_2": 18.0, "x_3": 80.0, "x_4":20.0, "x_5":20.0, "x_6":100.0, "x_7":60.0, "x_8":60.0, "x_9":300.0, "x_10":12, "x_11":12, 
             "x_12":10,"x_13":10,"x_14":10}
        
        # Initialize the client - AxClient offers a convenient API to control the experiment
        ax_client = AxClient(generation_strategy=gs)
        # Setup the experiment
        ax_client.create_experiment(
            name=cfg.experiment_name,
            parameters=[
                {
                    "name": f"x_{i+1}",
                    "type": "range",
                    "bounds": bounds[:, i].tolist(),
                    "value_type": "float" if i <= 8 else "int"  # 前9个参数为 float，后5个参数为 int
                }
                for i in range(len(cfg.benchmark.bounds))
            ],
            objectives={
                benchmark: ObjectiveProperties(minimize=False),
            },
        )
        output_parameters, trial_index = ax_client.attach_trial(parameters=initial_point)
        # true_vals = []
        hyperparameters = {}
        bo_times = []

        total_iters = num_init + num_bo
        total_batches = math.ceil((num_init + num_bo) / q)
        current_count = 0

        for i in range(total_batches):

            current_count = (q * i)+1
            batch_data = []
            if i==0:
                batch_data.append((output_parameters, trial_index))
            q_curr = min(q, total_iters - current_count)
            if current_count >= num_init:
                start_time = time()

            for q_rep in range(q_curr):
                batch_data.append(ax_client.get_next_trial())
            if current_count >= num_init:
                end_time = time()
                bo_times.append(end_time - start_time)
            # Local evaluation here can be replaced with deployment to external system.
            for q_rep in range(q_curr):
                parameters, trial_index = batch_data[q_rep]
                raw_data,best_all_y,best_y,min_current_value=evaluate(best_all_y,best_y,min_current_value,parameters)
                ax_client.complete_trial(
                    trial_index=trial_index, raw_data=raw_data)


            results_df = ax_client.get_trials_data_frame()
            configs = torch.tensor(
                results_df.loc[:, ['x_' in col for col in results_df.columns]].to_numpy())

            if cfg.benchmark.get('synthetic', True):
                for q_idx in range(q_curr):
                    y = test_function.evaluate_true(configs[-q_curr + q_idx].unsqueeze(0))
                    y_tensor = torch.tensor(y)
                    y_tensor = y_tensor.unsqueeze(0)
                    if all_constraints(y_tensor):
                        if y_tensor[0][1].item() < min_current_value:
                                min_current_value = y_tensor[0][1].item()
                                best_y = y_tensor.clone()
                    best_all_y.append(best_y)
                if current_count > num_init:
                    model = ax_client._generation_strategy.model.model.surrogate.model
                    current_data = ax_client.get_trials_data_frame()[benchmark].to_numpy()
                    hps = get_model_hyperparameters(model, current_data)
                    hyperparameters[f'iter_{i}'] = hps
            file_path = ('C:/DAC/vanilla_bo_in_highdim-main/Experiment/exp_under_different_train_sample/exp_design_1/baseline_Model_v1/vanilla/vanilla_40_OTA_two_seed_{}.csv').format(seed)
            # 实验结果计算路径
            cal_path = (
                    "C:\\DAC\\vanilla_bo_in_highdim-main\\Experiment\\exp_under_different_train_sample\\exp_design_1\\baseline_Model_v1\\vanilla\\vanilla_40_OTA_two_seed_")
            # 均值方差计算结果保存路径
            to_path = (
                'C:\\DAC\\vanilla_bo_in_highdim-main\\Experiment\\exp_under_different_train_sample\\exp_design_1_report\\vanilla_40_OTA_two_current_mean_var_strand.csv')

            iter_times = list(range(1, len(best_all_y) + 1))
            save_data(best_all_y, iter_times, file_path)

            plot(cal_path, to_path, [403,404, 402, 405, 406])


if __name__ == '__main__':
    main()