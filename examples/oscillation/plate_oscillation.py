# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 10:13:07 2021

@author: KRD2RNG
"""

import os
import numpy as np
import pandas as pd
from scipy import signal as sg
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
from torchphysics.problem.datacreator import InnerDataCreator
from torchphysics.problem import Variable
from torchphysics.setting import Setting
from torchphysics.problem.domain import Interval, Rectangle
from torchphysics.problem.condition import (DirichletCondition,
                                            NeumannCondition, 
                                              DiffEqCondition)
from torchphysics.models.fcn import SimpleFCN
from torchphysics import PINNModule
from torchphysics.utils import laplacian, grad
from torchphysics.utils.plot import _plot, _create_domain
from torchphysics.utils.evaluation import (get_min_max_inside,
                                             get_min_max_boundary)


os.environ["CUDA_VISIBLE_DEVICES"] = "0" # select GPUs to use

#pl.seed_everything(43) # set a global seed
torch.cuda.is_available()
# matplotlib.style.use('default')
density = 7e-6
e_modulus = 2e5
PoissonRatio = 0.3
thickness = 5
plate_stiffness = e_modulus * thickness**3 / ( 12 * ( 1 - PoissonRatio**2 ) )
# time
SampleFrequency = 1024
t_end = 1
t = np.linspace(0, t_end, t_end * SampleFrequency)
# geometry
width = 50
length = 150
n_length = 100


eval_points = 264
#%%
# def analytical_dof_1(x0):
#     delta = calc_delta(damping, mass)
#     omega_0 = calc_omega_0(stiffness_array, mass)
#     omega_d = np.sqrt(omega_0**2 - delta**2)
#     y = np.exp(-delta[np.newaxis,:] * t[:,np.newaxis]) * \
#         (((x0[1] + delta[np.newaxis,:] * x0[0]) / omega_d[np.newaxis,:]) * np.sin(omega_d[np.newaxis,:] * t[:,np.newaxis]) \
#          + x0[0] * np.cos(omega_d[np.newaxis,:] * t[:,np.newaxis]))
#     return pd.DataFrame(data=y, index=t, columns= np.ceil(stiffness_array))

#%% reference solution
# dirac = np.array([sg.unit_impulse(len(t), idx=0)])
# x0 = np.array([[1], [0]])
# y = analytical_dof_1(x0)
# y.plot(legend=False)
#%% PINN approach

# Variables
norm = torch.nn.MSELoss() #  #L1Loss

time = Variable(name='time',
              order=1,
              domain=Interval(low_bound=0,
                              up_bound=t_end),
              train_conditions={},
              val_conditions={})
x = Variable(name='x',
             order=2,
             domain=Rectangle(corner_dl=[0, 0],
                              corner_dr=[width, 0],
                              corner_tl=[0, length]),
             train_conditions={},
             val_conditions={})

# BC/IC

def time_dirichlet_fun(time): # (time)
    return np.ones_like(time)

def time_neumann_fun(time):
    return np.zeros_like(time)

def x_dirichlet_fun(x):
    return np.zeros_like(x)

time.add_train_condition(DirichletCondition(dirichlet_fun=time_dirichlet_fun,
                                          name='dirichlet_t',
                                          norm=norm,
                                          weight = 20,
                                          dataset_size=1,
                                          boundary_sampling_strategy='lower_bound_only',
                                          data_plot_variables=True))

time.add_train_condition(NeumannCondition(neumann_fun=time_neumann_fun,
                                          name='neumann_t',
                                          norm=norm,
                                          weight=50,
                                          dataset_size=1,
                                          boundary_sampling_strategy='lower_bound_only',
                                          data_plot_variables=True))

x.add_train_condition(DirichletCondition(dirichlet_fun=x_dirichlet_fun,
                                         name='dirichlet_x',
                                         sampling_strategy='random',
                                         boundary_sampling_strategy='random',
                                         norm=norm,
                                         weight=1.0,
                                         dataset_size=n_length,
                                         data_plot_variables=('x','t')))

def pde_plate(u, x, time):
    f = laplacian(laplacian(u, x), x) + (density * thickness / plate_stiffness) * laplacian(u, time) 
    return f

train_cond = DiffEqCondition(pde=pde_plate,
                              name='pde_plate',
                              norm=norm,
                              sampling_strategy='grid',
                              weight=1,
                              dataset_size=eval_points,
                              data_plot_variables=("time"))#)('time'))True
#%%
setup = Setting(variables=(time, x), 
                train_conditions={'pde_plate': train_cond},
                val_conditions={},
                solution_dims={'u': 2},
                n_iterations=50)
#%%
solver = PINNModule(model=SimpleFCN(variable_dims=setup.variable_dims,
                                    solution_dims=setup.solution_dims,
                                    depth=4,
                                    width=15,
                                    activation_func=torch.nn.Mish()),
                    optimizer=torch.optim.Adam, # Adam
                    lr=1e-3,
                    # log_plotter=plotter
                    )
#%%
trainer = pl.Trainer(gpus='-1' if torch.cuda.is_available() else None,
                      # logger=False,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      check_val_every_n_epoch=2,
                      log_every_n_steps=10,
                      max_epochs=12,
                      checkpoint_callback=False
                      )
#%%
trainer.fit(solver, setup)
#%%
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device =  'cpu'
solver = solver.to('cpu')
plot_type = "contour_surface" # 'contour_surface'
fig = _plot(model=solver.model, solution_name="u", plot_variables=x, points=500,
            dic_for_other_variables={'time' : 0}, plot_type=plot_type,
            device=device, angle=[30, 30]) 
fig.axes[0].set_box_aspect(1/2)