import numpy as np 
import torch 
from snas.genotypes import * 
from utils.visualize_ftn import * 
from snas.option.default_option import TrainOptions
opt = TrainOptions()

epoch = int(70)

temperature = opt.initial_temp * np.exp(opt.anneal_rate * epoch)
alpha_normal = np.load('alpha_npy/alpha_normal_' + str(epoch) + '.npy')
alpha_reduce = np.load('alpha_npy/alpha_reduce_' + str(epoch) + '.npy')


m_normal = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
    torch.tensor([temperature]), torch.tensor(alpha_normal)) 
m_reduce = torch.distributions.relaxed_categorical.RelaxedOneHotCategorical(
    torch.tensor([temperature]) , torch.tensor(alpha_reduce)) 

alpha_normal = m_normal.sample().cpu().numpy()
alpha_reduce = m_reduce.sample().cpu().numpy()
ex = genotype(alpha_normal,alpha_reduce)
plot(ex.normal,'normal.pdf')
plot(ex.reduce,'reduce.pdf')
