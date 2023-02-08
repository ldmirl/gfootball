import math
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from gfootball.env.ssim import pytorch_ssim
import numpy as np
from torch.nn import functional as F

from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# changing device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPS = torch.finfo(torch.float).eps  # numerical logs
batch_size = 64
h_dim = 1024
z_dim = 256
z_dim_2 = 64
timespan = 2

class classifier(nn.Module):
    def __init__(self, h_dim, z_dim, n_layers, bias=False):
        super(classifier, self).__init__()

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        # Encoder
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, kernel_size=(7, 7), stride=(2, 2), padding=(1, 1)),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),

                                     nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 3), padding=(1, 0)),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),

                                     nn.Conv2d(32, 48, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
                                     nn.BatchNorm2d(48),
                                     nn.ReLU(),

                                     nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(3, 3), padding=(0, 0)),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU())

        # recurrence
        self.lstm = nn.LSTM(input_size=z_dim_2, hidden_size=h_dim, num_layers=self.n_layers, bias=True,
                            batch_first=True)

        self.fc = nn.Sequential(nn.Linear(z_dim, z_dim), nn.ReLU(),
                                nn.Linear(z_dim, z_dim_2), nn.ReLU())

        self.fc_2 = nn.Sequential(nn.Linear(timespan * h_dim, h_dim), nn.ReLU(),
                                  nn.Linear(h_dim, h_dim // 2), nn.ReLU(),
                                  nn.Linear(h_dim // 2, h_dim // 2), nn.ReLU(),
                                  nn.Linear(h_dim // 2, 2))
        # self.fc = nn.Sequential(nn.Linear(10*h_dim, 5*h_dim),nn.ReLU(),
        #                         nn.Linear(5*h_dim, 5*h_dim), nn.ReLU(),
        #                         nn.Linear(5*h_dim, h_dim), nn.ReLU(),
        #                         nn.Linear(h_dim, 3), nn.Sigmoid())

    def forward(self, x):
        input_shape = x.shape
        x = x.reshape(x.shape[0] * x.shape[1], 3, 64, 96)

        # input shape [b,t,w,h,c]
        # for t in range(x.size(1)):
        # encoder
        phi_x = self.encoder(x)

        phi_x = phi_x.reshape(batch_size, input_shape[1], 64 * 2 * 2)
        phi_x = self.fc(phi_x)
        # print('phixshape', phi_x.shape)
        # if t == 0:
        #     encoded_seq = phi_x
        # else:
        #     encoded_seq = torch.cat([encoded_seq, phi_x], 1)

        # recurrence
        out, _ = self.lstm(phi_x)
        # print(phi_x.shape)
        out = out.reshape(batch_size, timespan * h_dim)
        out = self.fc_2(out)
        # print('outshape', out.shape)
        # print(out.shape)
        return out
        
class VRNN(nn.Module):
    def __init__(self, h_dim, z_dim, n_layers, bias=False):
        super(VRNN, self).__init__()

        self.h_dim = h_dim
        self.z_dim = z_dim
        self.n_layers = n_layers

        # Prior
        # self.prior = nn.Sequential(
        #     nn.Linear(h_dim, z_dim),
        #     nn.ReLU())
        # self.prior_mean = nn.Linear(z_dim + 2, z_dim)
        # self.prior_std = nn.Sequential(
        #     nn.Linear(z_dim + 2, z_dim),
        #     nn.Softplus())

        self.prior_1 = nn.Sequential(
            nn.Linear(h_dim, z_dim),
            nn.ReLU())
        self.prior_2 = nn.Sequential(
            nn.Linear(z_dim+2, z_dim),
            nn.ReLU())
        self.prior_mean = nn.Linear(z_dim, z_dim)
        self.prior_std = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Softplus())

        # self.prior = nn.Sequential(
        #     nn.Linear(h_dim, z_dim),
        #     nn.ReLU())
        # self.prior_mean = nn.Linear(z_dim, z_dim)
        # self.prior_std = nn.Sequential(
        #     nn.Linear(z_dim, z_dim),
        #     nn.Softplus())

        # Encoder
        self.encoder = nn.Sequential(nn.Conv2d(3, 16, kernel_size=(7, 7), stride=(2, 2), padding=(1, 1)),
                                     nn.BatchNorm2d(16),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),

                                     nn.Conv2d(16, 32, kernel_size=(5, 5), stride=(2, 3), padding=(1, 0)),
                                     nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),

                                     nn.Conv2d(32, 48, kernel_size=(5, 5), stride=(2, 2), padding=(1, 1)),
                                     nn.BatchNorm2d(48),
                                     nn.ReLU(),

                                     nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
                                     nn.BatchNorm2d(64),
                                     nn.ReLU()

                                     # nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(3, 3), padding=(0, 0)),
                                     # nn.BatchNorm2d(64),
                                     # nn.ReLU()
                                     )

        # Posterior
        # self.posterior = nn.Sequential(nn.Linear(h_dim + h_dim, z_dim),
        #                                nn.ReLU(),
        #                                nn.Linear(z_dim, z_dim),
        #                                nn.ReLU())
        # self.posterior_mean = nn.Linear(z_dim + 2, z_dim)
        # self.posterior_std = nn.Sequential(
        #     nn.Linear(z_dim + 2, z_dim),
        #     nn.Softplus())

        self.posterior_1 = nn.Sequential(nn.Linear(h_dim, z_dim),
                                       nn.ReLU(),
                                       nn.Linear(z_dim, z_dim),
                                       nn.ReLU())
        self.posterior_2 = nn.Sequential(nn.Linear(z_dim + 2, z_dim),
                                       nn.ReLU())
        self.posterior_3 = nn.Sequential(nn.Linear(z_dim + h_dim, z_dim),
                                         nn.ReLU(),
                                         nn.Linear(z_dim, z_dim),
                                         nn.ReLU())
        self.posterior_mean = nn.Linear(z_dim, z_dim)
        self.posterior_std = nn.Sequential(
            nn.Linear(z_dim, z_dim),
            nn.Softplus())

        # self.posterior = nn.Sequential(nn.Linear(h_dim + h_dim, z_dim),
        #                                nn.ReLU(),
        #                                nn.Linear(z_dim, z_dim),
        #                                nn.ReLU())
        # self.posterior_mean = nn.Linear(z_dim, z_dim)
        # self.posterior_std = nn.Sequential(
        #     nn.Linear(z_dim, z_dim),
        #     nn.Softplus())

        # phi_z
        self.phi_z = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())

        # reverse-posterior
        self.f_decoder = nn.Sequential(
            nn.Linear(h_dim + h_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim),
            nn.ReLU())

        # decoder
        self.decoder = nn.Sequential(
                                     # nn.ConvTranspose2d(64, 48, kernel_size=(3, 3), stride=(3, 3), padding=(0, 0)),
                                     # # nn.BatchNorm2d(48),
                                     # nn.ReLU(),
                                     # # nn.Dropout(0.2),

                                     nn.ConvTranspose2d(64, 48, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
                                     nn.ReLU(),

                                     nn.ConvTranspose2d(48, 32, kernel_size=(5, 5), stride=(2, 2), padding=(0, 0)),
                                     # nn.BatchNorm2d(32),
                                     nn.ReLU(),
                                     # nn.Dropout(0.2),

                                     nn.ConvTranspose2d(32, 16, kernel_size=(5, 5), stride=(2, 3), padding=(1, 0)),
                                     # nn.BatchNorm2d(16),
                                     nn.ReLU(),

                                     nn.ConvTranspose2d(16, 3, kernel_size=(6, 6), stride=(2, 2), padding=(1, 1)),
                                     # nn.BatchNorm2d(3),
                                     nn.Sigmoid()

                                     )

        # recurrence
        # self.lstm_push = nn.LSTM(input_size=h_dim + h_dim, hidden_size=h_dim, num_layers=self.n_layers, bias=bias,
        #                     batch_first=True)
        # self.lstm_back = nn.LSTM(input_size=h_dim + h_dim, hidden_size=h_dim, num_layers=self.n_layers, bias=bias,
        #                       batch_first=True)
        # self.lstm_stay = nn.LSTM(input_size=h_dim + h_dim, hidden_size=h_dim, num_layers=self.n_layers, bias=bias,
        #                       batch_first=True)

        self.lstm_general = nn.LSTM(input_size=h_dim + h_dim, hidden_size=h_dim, num_layers=self.n_layers, bias=bias,
                              batch_first=True)
        self.classifier = classifier(128,256,1)
        self.classifier.load_state_dict(torch.load('/home/aaron/pymarl2/classifier.pth'))

    def forward(self, x):
        # input shape [b,t,w,h,c]
        x = x / 255.0
        all_enc_mean, all_enc_std = [], []
        all_dec_mean, all_dec_std = [], []
        kld_loss = 0
        rec_loss = 0
        rec_diff = 0

        # initialize lstm state
        h = (torch.zeros([1, batch_size, self.h_dim], device=device),
             torch.zeros([1, batch_size, self.h_dim], device=device))

        # load classifier and get label

        for t in range(x.size(1)):
            if t > 0:
                pred_seq_label = []
                self.classifier.eval()
                with torch.no_grad():
                    for n in range(int(batch_size / 4)):
                        pred_label = self.classifier(x[n * 4:n * 4 + 4, t - 1:t + 1, :, :, :])
                        if n == 0:
                            pred_seq_label = pred_label
                        else:
                            pred_seq_label = torch.vstack((pred_seq_label, pred_label))
                    pred_seq_label = torch.nn.functional.softmax(pred_seq_label, dim=1)
                    for i in range(pred_seq_label.shape[0]):
                        label_list = pred_seq_label[i, :]
                        min_index = torch.argmin(label_list).cpu().numpy()
                        max_index = torch.argmax(label_list).cpu().numpy()
                        if min(label_list) < 0.1:
                            pred_seq_label[i, min_index] = 0
                            pred_seq_label[i, max_index] = 1
                        else:
                            pred_seq_label[i, min_index] = 0
                            pred_seq_label[i, max_index] = 0
            else:
                pred_seq_label = torch.zeros([batch_size, 2], device=device)

            # prior

            # prior = self.prior(h[-1][-1])
            # prior_mean = self.prior_mean(torch.cat([prior, pred_seq_label], 1))
            # prior_std = self.prior_std(torch.cat([prior, pred_seq_label], 1))

            prior = self.prior_1(h[-1][-1])
            prior = self.prior_2(torch.cat([prior, pred_seq_label], 1))
            prior_mean = self.prior_mean(prior)
            prior_std = self.prior_std(prior)

            # prior = self.prior(h[-1][-1])
            # prior_mean = self.prior_mean(prior)
            # prior_std = self.prior_std(prior)

            # encoder
            phi_x = self.encoder(x[:, t, :, :, :])

            phi_x = phi_x.reshape(batch_size, 64 * 4 * 4)

            # posterior

            # posterior = self.posterior(torch.cat([phi_x, h[-1][-1]], 1))
            # posterior_mu = self.posterior_mean(torch.cat([posterior, pred_seq_label], 1))
            # posterior_sigma = self.posterior_std(torch.cat([posterior, pred_seq_label], 1))

            posterior = self.posterior_1(phi_x)
            posterior = self.posterior_2(torch.cat([posterior, pred_seq_label],1))
            posterior = self.posterior_3(torch.cat([posterior, h[-1][-1]], 1))
            posterior_mu = self.posterior_mean(posterior)
            posterior_sigma = self.posterior_std(posterior)

            # posterior = self.posterior(torch.cat([phi_x, h[-1][-1]], 1))
            # posterior_mu = self.posterior_mean(posterior)
            # posterior_sigma = self.posterior_std(posterior)

            # sampling and reparameterization
            epsilon = torch.empty(size=posterior_sigma.size(), device=device, dtype=torch.float).normal_()
            z = epsilon.mul(posterior_sigma).add_(posterior_mu)
            phi_z = self.phi_z(z)
            # if t == 0:
            #     latent_log = posterior_mu.reshape(1, batch_size, self.z_dim)
            # else:
            #     latent_log = torch.cat((latent_log, posterior_mu.reshape(1, batch_size, self.z_dim)), dim=0)

            # decoder
            decoder_out = self.f_decoder(torch.cat([phi_z, h[-1][-1]], 1))
            decoder_out = decoder_out.reshape(batch_size, 64, 4, 4)
            x_out = self.decoder(decoder_out)

            # lstm transition
            # general trainsition without conditons
            # _, h = self.lstm_general(torch.cat([phi_x, phi_z], 1).reshape(batch_size, 1, h_dim + h_dim), h)

            # generate transition with conditions
            push_idx = []
            back_idx = []
            stay_idx = []
            for i in range(batch_size):
                if pred_seq_label[i,0] == 0:
                    if pred_seq_label[i,1] == 0:
                        stay_idx.append(i)
                    else:
                        back_idx.append(i)
                else:
                    push_idx.append(i)
            phi_x_push = phi_x[push_idx, :]
            phi_x_back = phi_x[back_idx, :]
            phi_x_stay = phi_x[stay_idx, :]
            phi_z_push = phi_z[push_idx, :]
            phi_z_back = phi_z[back_idx, :]
            phi_z_stay = phi_z[stay_idx, :]
            h_0 = h[0][-1]
            h_1 = h[1][-1]
            h_0_push = h_0[push_idx, :]
            h_0_back = h_0[back_idx, :]
            h_0_stay = h_0[stay_idx, :]
            h_1_push = h_1[push_idx, :]
            h_1_back = h_1[back_idx, :]
            h_1_stay = h_1[stay_idx, :]

            if len(push_idx) != 0:
                h_push = [h_0_push.reshape(1, len(push_idx), h_dim), h_1_push.reshape(1, len(push_idx), h_dim)]
                _, h_push = self.lstm_push(torch.cat([phi_x_push, phi_z_push], 1).reshape(len(push_idx), 1, h_dim + h_dim), h_push)
            if len(back_idx) != 0:
                h_back = [h_0_back.reshape(1, len(back_idx), h_dim), h_1_back.reshape(1, len(back_idx), h_dim)]
                _, h_back = self.lstm_back(torch.cat([phi_x_back, phi_z_back], 1).reshape(len(back_idx), 1, h_dim + h_dim), h_back)
            if len(stay_idx) != 0:
                h_stay = [h_0_stay.reshape(1, len(stay_idx), h_dim), h_1_stay.reshape(1, len(stay_idx), h_dim)]
                _, h_stay = self.lstm_stay(torch.cat([phi_x_stay, phi_z_stay], 1).reshape(len(stay_idx), 1, h_dim + h_dim), h_stay)

            h_new_0 = []
            h_new_1 = []
            for i in range(batch_size):
                if i == 0:
                    if i in push_idx:
                        h_new_0 = h_push[0][:, push_idx.index(i):push_idx.index(i)+1, :]
                        h_new_1 = h_push[1][:, push_idx.index(i):push_idx.index(i)+1, :]
                    if i in back_idx:
                        h_new_0 = h_back[0][:, back_idx.index(i):back_idx.index(i)+1, :]
                        h_new_1 = h_back[1][:, back_idx.index(i):back_idx.index(i)+1, :]
                    if i in stay_idx:
                        h_new_0 = h_stay[0][:, stay_idx.index(i):stay_idx.index(i)+1, :]
                        h_new_1 = h_stay[1][:, stay_idx.index(i):stay_idx.index(i)+1, :]
                else:
                    if i in push_idx:
                        h_new_0 = torch.cat([h_new_0, h_push[0][:, push_idx.index(i):push_idx.index(i)+1, :]], dim=1)
                        h_new_1 = torch.cat([h_new_1, h_push[1][:, push_idx.index(i):push_idx.index(i)+1, :]], dim=1)
                    if i in back_idx:
                        h_new_0 = torch.cat([h_new_0, h_back[0][:, back_idx.index(i):back_idx.index(i)+1, :]], dim=1)
                        h_new_1 = torch.cat([h_new_1, h_back[1][:, back_idx.index(i):back_idx.index(i)+1, :]], dim=1)
                    if i in stay_idx:
                        h_new_0 = torch.cat([h_new_0, h_stay[0][:, stay_idx.index(i):stay_idx.index(i)+1, :]], dim=1)
                        h_new_1 = torch.cat([h_new_1, h_stay[1][:, stay_idx.index(i):stay_idx.index(i)+1, :]], dim=1)

            h = [h_new_0, h_new_1]

            # computing losses

            kld_loss += self.kl_gaussgauss(prior_mean, prior_std, posterior_mu, posterior_sigma)

            # rec_loss += self.cross_entropy(x_out, x[:, t, :, :, :])
            rec_loss += 1 - self.ssim_loss(x_out, x[:, t, :, :, :])
            # rec_loss += self.mse_loss(x_out, x[:, t, :, :, :])

        return torch.mean(kld_loss), torch.mean(rec_loss)
        # , latent_log.reshape(1,6,batch_size,z_dim)

    def sample(self, seq_len):
        batch_size_sample = 1

        sample = []
        # sample_df = np.load('/home/aarongu/Downloads/VRNN_VAL/data.npy')
        # sample_t0 = sample_df[0:1, 0, :, :, :]/255.0
        # plt.imshow(sample_t0.reshape(64, 96, 3))
        img_1 = Image.open('/home/aarongu/Downloads/event16_0Chelsea.png')
        img_1 = img_1.crop((21, 17, 639, 417))
        img_1 = img_1.convert('RGB')
        img_1 = cv2.resize(np.array(img_1), (96, 64))
        sample_t0 = img_1.reshape(1,3,64,96)
        print(sample_t0.shape)
        plt.imshow(img_1)
        plt.show()

        # initial lstm state
        h = (torch.zeros([1, batch_size_sample, self.h_dim], device=device),
             torch.zeros([1, batch_size_sample, self.h_dim], device=device))

        # conditions control
        t0 = torch.zeros([batch_size_sample, 1], device=device)
        t1 = torch.ones([batch_size_sample, 1], device=device)
        label = torch.cat([t1, t0], 1)

        # encoder
        phi_x_t = self.encoder(torch.from_numpy(sample_t0).type(torch.FloatTensor).to(device))
        phi_x_t = phi_x_t.reshape(batch_size_sample, 64 * 4 * 4)

        # prior
        # prior_t = self.prior(h[-1][-1])
        # prior_t = torch.cat([prior_t, label], 1)
        # prior_mean_t = self.prior_mean(prior_t)
        # prior_std_t = self.prior_std(prior_t)

        prior = self.prior_1(h[-1][-1])
        prior = self.prior_2(torch.cat([prior, label], 1))
        prior_mean_t = self.prior_mean(prior)
        prior_std_t = self.prior_std(prior)

        # prior = self.prior(h[-1][-1])
        # prior_mean_t = self.prior_mean(prior)
        # prior_std_t = self.prior_std(prior)

        # sampling and reparameterization
        epsilon = torch.empty(size=prior_std_t.size(), device=device, dtype=torch.float).normal_()
        z = epsilon.mul(prior_std_t).add_(prior_mean_t)
        phi_z_t = self.phi_z(z)

        # decoder
        # dec_t = self.f_decoder(torch.cat([phi_z_t, h[-1][-1]], 1))
        # dec_t = dec_t.reshape(batch_size_sample, 64, 4, 4)
        # dec_mean_t = self.decoder(dec_t)

        # recurrence
        _, h = self.lstm_general(torch.cat([phi_x_t, phi_z_t], 1).reshape(batch_size_sample, 1, h_dim + h_dim), h)

        for t in range(1, seq_len):

            # # conditions control
            # t0 = torch.zeros([batch_size_sample, 1], device=device)
            # t1 = torch.ones([batch_size_sample, 1], device=device)
            # label = torch.cat([t0, t1], 1)

            # prior
            # prior_t = self.prior(h[-1][-1])
            # prior_t = torch.cat([prior_t, label],1)
            # prior_mean_t = self.prior_mean(prior_t)
            # prior_std_t = self.prior_std(prior_t)

            prior = self.prior_1(h[-1][-1])
            prior = self.prior_2(torch.cat([prior, label], 1))
            prior_mean_t = self.prior_mean(prior)
            prior_std_t = self.prior_std(prior)

            # prior = self.prior(h[-1][-1])
            # prior_mean_t = self.prior_mean(prior)
            # prior_std_t = self.prior_std(prior)

            # sampling and reparameterization
            epsilon = torch.empty(size=prior_std_t.size(), device=device, dtype=torch.float).normal_()
            z = epsilon.mul(prior_std_t).add_(prior_mean_t)
            phi_z_t = self.phi_z(z)

            # decoder
            dec_t = self.f_decoder(torch.cat([phi_z_t, h[-1][-1]], 1))
            dec_t = dec_t.reshape(batch_size_sample, 64, 4, 4)
            dec_mean_t = self.decoder(dec_t)

            # encoder
            phi_x_t = self.encoder(dec_mean_t)
            phi_x_t = phi_x_t.reshape(batch_size_sample, 64 * 4 * 4)

            # recurrence
            _, h = self.lstm_general(torch.cat([phi_x_t, phi_z_t], 1).reshape(batch_size_sample, 1, h_dim + h_dim), h)

            sample.append(dec_mean_t.data)

        return sample

    def kld_loss(self, mean_1, std_1, mean_2, std_2):
        """Using std to compute KLD"""
        kld_element = (2 * torch.log(std_2 + EPS) - 2 * torch.log(std_1 + EPS) +
                       (std_1.pow(2) + (mean_1 - mean_2).pow(2)) /
                       std_2.pow(2) - 1)
        return 0.5 * torch.sum(kld_element)

    def kl_gaussgauss(self, mu_1, sigma_1, mu_2, sigma_2):
        return torch.sum(
            torch.log(sigma_2) - torch.log(sigma_1) + (sigma_1 ** 2 + (mu_1 - mu_2) ** 2) /
            (2 * ((sigma_2) ** 2)) - 0.5, dim=1)

    def ssim_loss(self, x_out, x_in):
        ssim_loss = pytorch_ssim.SSIM(window_size=11)
        return ssim_loss(x_out, x_in)

    def ssim_diff(self, x_out, x_in):
        ssim_diff = pytorch_ssim.ssim(x_out, x_in)
        return ssim_diff

    def cross_entropy(self, y_prediction, y):
        prediction_loss = y * torch.log(1e-7 + y_prediction) + (1 - y) * torch.log(1e-7 + 1 - y_prediction)
        return -torch.mean(prediction_loss, dim=[1, 2, 3])

    def reset_parameters(self, stdv=1e-1):
        for weight in self.parameters():
            weight.data.normal_(0, stdv)

    def _init_weights(self, stdv):
        pass

    def mse_loss(self, x_out, x_in):
        return F.mse_loss(x_out, x_in)