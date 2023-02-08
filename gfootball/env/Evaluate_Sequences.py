import pandas as pd
from glob import glob
from PIL import Image
import cv2
import torch.utils.data
import numpy as np
from PIL import Image
import cv2
from gfootball.env.Conditional_VRNN import VRNN
from gfootball.env.img_to_epv import colormap2arr, calculate_epv, midpoint_double1
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from gfootball.env.ssim import pytorch_ssim
from gfootball.env.PitchSeq_Classifier import classifier
from gfootball.env import PitchSeq_Classifier
import pickle
from scipy.stats import norm

batch_size_sample = 1
h_dim = 1024
z_dim = 256
n_layers = 1

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

EPV = np.loadtxt('/home/aaron/pymarl2/EPV_grid.csv', delimiter=',')
state_dict = torch.load('/home/aaron/pymarl2/vrnn.pth')
model = VRNN(h_dim, z_dim, n_layers)
model.load_state_dict(state_dict)
model.to(device)


def process_img(path):
    img_1 = Image.open(path)
    img_1 = img_1.crop((21, 17, 639, 417))
    img_1 = img_1.convert('RGB')
    img_1 = cv2.resize(np.array(img_1), (96, 64))
    img_2 = img_1.reshape(1, 3, 64, 96)
    img_2 = img_2 / 255.0
    return img_1, img_2


def decision_to_label(decision, t0, t1):
    if decision == 'push':
        label = torch.cat([t1, t0], 1)
    if decision == 'back':
        label = torch.cat([t0, t1], 1)
    if decision == 'stay':
        label = torch.cat([t0, t0], 1)
    return label


def get_best_seq(sample_t0, seq_len):
    sample = []
    # initial lstm state
    h = (torch.zeros([1, batch_size_sample, model.h_dim], device=device),
         torch.zeros([1, batch_size_sample, model.h_dim], device=device))

    # conditions control
    t0 = torch.zeros([batch_size_sample, 1], device=device)
    t1 = torch.ones([batch_size_sample, 1], device=device)

    label = decision_to_label('stay', t0, t1)

    # encoder
    phi_x_t = model.encoder(torch.from_numpy(sample_t0).type(torch.FloatTensor).to(device))
    phi_x_t = phi_x_t.reshape(batch_size_sample, 64 * 4 * 4)

    prior = model.prior_1(h[-1][-1])
    prior = model.prior_2(torch.cat([prior, label], 1))
    prior_mean_t = model.prior_mean(prior)
    prior_std_t = model.prior_std(prior)

    # sampling and reparameterization
    epsilon = torch.empty(size=prior_std_t.size(), device=device, dtype=torch.float).normal_()
    z = epsilon.mul(prior_std_t).add_(prior_mean_t)
    phi_z_t = model.phi_z(z)

    # recurrence
    _, h = model.lstm_general(torch.cat([phi_x_t, phi_z_t], 1).reshape(batch_size_sample, 1, h_dim + h_dim), h)

    for t in range(1, seq_len):
        img_for_decision = []
        h_for_decision = []

        image_buffer = []

        for i in range(3):
            # condition contro
            decision = ['push', 'back', 'stay'][i]
            label = decision_to_label(decision, t0, t1)

            # prior
            prior = model.prior_1(h[-1][-1])
            prior = model.prior_2(torch.cat([prior, label], 1))
            prior_mean_t = model.prior_mean(prior)
            prior_std_t = model.prior_std(prior)

            # sampling and reparameterization
            epsilon = torch.empty(size=prior_std_t.size(), device=device, dtype=torch.float).normal_()
            z = epsilon.mul(prior_std_t).add_(prior_mean_t)
            phi_z_t = model.phi_z(z)

            # decoder
            dec_t = model.f_decoder(torch.cat([phi_z_t, h[-1][-1]], 1))
            dec_t = dec_t.reshape(batch_size_sample, 64, 4, 4)
            dec_mean_t = model.decoder(dec_t)

            # encoder
            phi_x_t = model.encoder(dec_mean_t)
            phi_x_t = phi_x_t.reshape(batch_size_sample, 64 * 4 * 4)

            # recurrence
            # if decision == 'push':
            #     _, h = model.lstm_push(torch.cat([phi_x_t, phi_z_t], 1).reshape(batch_size_sample, 1, h_dim + h_dim), h)
            # if decision == 'back':
            #     _, h = model.lstm_back(torch.cat([phi_x_t, phi_z_t], 1).reshape(batch_size_sample, 1, h_dim + h_dim), h)
            # if decision == 'stay':
            #     _, h = model.lstm_stay(torch.cat([phi_x_t, phi_z_t], 1).reshape(batch_size_sample, 1, h_dim + h_dim), h)
            _, h = model.lstm_general(torch.cat([phi_x_t, phi_z_t], 1).reshape(batch_size_sample, 1, h_dim + h_dim), h)

            # plt.imsave('/home/aaron/Desktop/img_for_decision' + str(i) + '.png',
            #            dec_mean_t.reshape(64, 96, 3).cpu().detach().numpy())

            image_buffer.append(dec_mean_t.reshape(64, 96, 3).cpu().detach().numpy())

            img_for_decision.append(dec_mean_t.data)
            h_for_decision.append(h)

        EPV_Value_log = []
        for i in range(3):
            # arr = Image.open('/home/aaron/Desktop/img_for_decision' + str(i) + '.png')
            # arr = np.array(arr) / 255.0

            arr = image_buffer[i]           
            ones = np.ones((64,96,1))
            arr = np.concatenate((arr, ones), axis=2)
            
            value = colormap2arr(arr, cm.bwr)
            value = value.reshape(96, 64)
            EPV_Value = midpoint_double1(calculate_epv, value, -48, 48, -32, 32, 48, 32, EPV)
            EPV_Value_log.append(EPV_Value)

        # ideal_img_index = EPV_Value_log.index(min(EPV_Value_log))
        # h = h_for_decision[ideal_img_index]
        # ideal_img = img_for_decision[ideal_img_index]
        # sample.append(ideal_img)

        sample.append(max(EPV_Value_log))
    return sample


def ssim_loss(x_out, x_in):
    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    return ssim_loss(x_out, x_in)
