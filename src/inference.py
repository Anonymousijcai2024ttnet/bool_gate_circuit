from copy import copy, deepcopy
from ctypes import c_uint8, c_uint64
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Function
from sympy import symbols
from sympy.logic.boolalg import truth_table
from sympy.parsing.sympy_parser import parse_expr
from torchvision.datasets.mnist import read_image_file, read_label_file
from tqdm import tqdm
import os


def get_mapping_filter(args):
    mapping_filter = {}
    input_dim = args.preprocessing_CNN[0]
    for block in range(len(args.Blocks_filters_output)):
        mapping_filter[block] = {}
        if block == 0:
            inter_filter_coef = int(args.Blocks_filters_output[block] / input_dim)
        else:
            inter_filter_coef = int(args.Blocks_filters_output[block] * args.groups_per_block[block] / (
            args.Blocks_filters_output[block - 1]))
        for filterblockici_out in range(args.Blocks_filters_output[block]):
            mapping_filter[block][filterblockici_out] = args.groups_per_block[block] * (
                        filterblockici_out // inter_filter_coef)
    return mapping_filter, input_dim


def get_mapping_filter_cnn(args):
    mapping_filter = {}
    input_dim = args.preprocessing_CNN[0]
    for block in range(len(args.Blocks_filters_output)):
        mapping_filter[block] = {}
        if block == 0:
            inter_filter_coef = int(args.Blocks_filters_output[block] / input_dim)
        else:
            inter_filter_coef = int(args.Blocks_filters_output[block] * args.groups_per_block[block] / (
            args.Blocks_filters_output[block - 1]))
        for filterblockici_out in range(args.Blocks_filters_output[block]):
            if block == 0:
                # mapping_filter[block][filterblockici_out] = args.groups_per_block[block]*int(args.groups_per_block[block]*(filterblockici_out // inter_filter_coef)//4)
                mapping_filter[block][filterblockici_out] = args.groups_per_block[block] * (
                            filterblockici_out // inter_filter_coef)
            else:
                mapping_filter[block][filterblockici_out] = args.groups_per_block[block] * (
                            filterblockici_out // inter_filter_coef)
    return mapping_filter, input_dim


def get_mapping_filter_cnn_multihead(args):
    mapping_filter1 = {}
    mapping_filter2 = {}
    input_dim = args.preprocessing_CNN[0]
    for block in range(3):  # len(args.Blocks_filters_output)):
        mapping_filter1[block] = {}
        # if block==0:
        inter_filter_coef1 = int(args.Blocks_filters_output[0] / input_dim)

        for filterblockici_out in range(args.Blocks_filters_output[0]):
            if block == 0 or block == 1:
                mapping_filter1[block][filterblockici_out] = 1 * (filterblockici_out // inter_filter_coef1)
            else:
                mapping_filter1[block][filterblockici_out] = 6 * (filterblockici_out // 6)
    for block in range(1):  # len(args.Blocks_filters_output)):
        mapping_filter2[block] = {}
        # if block==0:
        inter_filter_coef2 = 6  # int(2*args.Blocks_filters_output[0]/(4*args.Blocks_filters_output[block]))
        print(inter_filter_coef2)
        for filterblockici_out in range(2 * args.Blocks_filters_output[0]):
            mapping_filter2[block][filterblockici_out] = 2 * (filterblockici_out)
    return mapping_filter1, mapping_filter2


class BN_eval_CIFAR10(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, scale, bias):
        super(BN_eval_CIFAR10, self).__init__()
        #print(scale.shape, bias.shape)
        self.scale = torch.Tensor(scale).unsqueeze(2).unsqueeze(3)#.unsqueeze(3)
        # print(torch.Tensor(scale).shape)
        self.bias = torch.Tensor(bias).unsqueeze(2).unsqueeze(2)#.unsqueeze(3)

    def forward(self, x):
        device = x.device
        # if x.shape[1]==3 and x.shape[-1]==32:
        #    x = torch.cat((x - 2 / 255, x, x + 2 / 255), dim=1)
        # elif x.shape[1]==1 and x.shape[-1]==28:
        # k    x = torch.cat((x - 0.1, x, x + 0.1), dim=1)
        #print(self.scale.shape, x.shape, self.bias.shape)
        return self.scale.to(device) * x + self.bias.to(device)


class BN_eval_MNIST(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, scale, bias):
        super(BN_eval_MNIST, self).__init__()
        #print(scale.shape, bias.shape)
        self.scale = torch.Tensor(scale).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # print(torch.Tensor(scale).shape)
        self.bias = torch.Tensor(bias).unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def forward(self, x):
        device = x.device
        # if x.shape[1]==3 and x.shape[-1]==32:
        #    x = torch.cat((x - 2 / 255, x, x + 2 / 255), dim=1)
        # elif x.shape[1]==1 and x.shape[-1]==28:
        # k    x = torch.cat((x - 0.1, x, x + 0.1), dim=1)
        #print(self.scale.shape, x.shape, self.bias.shape)
        return self.scale.to(device) * x + self.bias.to(device)

import json
import itertools
def extract(output_strings):
    template = '{{"values": [{}]}}'
    parse_func = lambda x: json.loads(template.format(x))
    return list(itertools.chain(*[parse_func(x)["values"] for x in output_strings]))
def BitsToIntAFast(bits):
    m,n = bits.shape # number of columns is needed, not bits.size
    a = 2**np.arange(n)[::-1]  # -1 reverses array of powers of 2 of same length as bits
    return bits @ a  # this matmult is the key line of code

def TerToIntAFast(bits):
    m,n = bits.shape # number of columns is needed, not bits.size
    #print(m,n)
    a = 3**np.arange(n)[::-1]  # -1 reverses array of powers of 2 of same length as bits
    return bits @ a  # this matmult is the key line of code

class BN_eval_CNN(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, scale, bias):
        super(BN_eval_CNN, self).__init__()
        self.scale = torch.Tensor(scale).unsqueeze(0).unsqueeze(2).transpose(1, 3)
        self.bias = torch.Tensor(bias).unsqueeze(0).unsqueeze(2).transpose(1, 3)

    def forward(self, x):
        device = x.device
        # if x.shape[1]==3 and x.shape[-1]==32:
        #    x = torch.cat((x - 2 / 255, x, x + 2 / 255), dim=1)
        # elif x.shape[1]==1 and x.shape[-1]==28:
        # k    x = torch.cat((x - 0.1, x, x + 0.1), dim=1)
        # print(self.scale.shape, x.shape, self.bias.shape)
        return self.scale.to(device) * x + self.bias.to(device)


class InputQuantizer(nn.Module):
    """quantize input in the range ``[0, 1]`` to be a multiple of given ``step``
    """

    class RoundFn(Function):
        @staticmethod
        def forward(ctx, inp):
            return torch.round(inp)

        @staticmethod
        def backward(ctx, g_out):
            return g_out

    def __init__(self, step: float):
        super().__init__()
        self.step = step

    def forward(self, x):
        xint = self.RoundFn.apply(x / self.step)
        return xint * self.step

    def __repr__(self):
        return f'{type(self).__name__}({self.step})'


g_bingrad_soft_tanh_scale = 1


class Binarize01Act(nn.Module):
    class Fn(Function):
        @staticmethod
        def forward(ctx, inp, T, scale=None):
            """:param scale: scale for gradient computing"""
            if scale is None:
                ctx.save_for_backward(inp)
            else:
                ctx.save_for_backward(inp, scale)
            all_ones = 1.0 * (inp >= T / 2)
            maks = 1.0 * (inp < T / 2) - 1.0 * (inp < -T / 2)
            random = torch.randint_like(inp, 2).to(inp.dtype)
            res = all_ones + maks * random

            return res.to(inp.dtype)

        @staticmethod
        def backward(ctx, g_out):
            if len(ctx.saved_tensors) == 2:
                inp, scale = ctx.saved_tensors
            else:
                inp, = ctx.saved_tensors
                scale = 1

            if g_bingrad_soft_tanh_scale is not None:
                scale = scale * g_bingrad_soft_tanh_scale
                tanh = torch.tanh_(inp * scale)
                return (1 - tanh.mul_(tanh)).mul_(g_out), None, None

            # grad as sign(hardtanh(x))
            g_self = (inp.abs() <= 1).to(g_out.dtype)
            return g_self.mul_(g_out), None, None

    def __init__(self, T=0, grad_scale=1):
        super().__init__()
        self.T = T
        self.register_buffer(
            'grad_scale',
            torch.tensor(float(grad_scale), dtype=torch.float32))

    def forward(self, x):
        grad_scale = getattr(self, 'grad_scale', None)
        f = lambda x: self.Fn.apply(x, self.T, grad_scale)  # thr_bin_act=self.thr_bin_act)
        return f(x)


def load_cnf_dnf(args):
    if args.with_contradiction:
        path_save_modelvf = args.path_save_model + '/thr_' + str(args.thr_bin_act_test[1:]).replace(" ",
                                                                                                    "") + '/avec_contradiction/'
    else:
        path_save_modelvf = args.path_save_model + '/thr_' + str(args.thr_bin_act_test[1:]).replace(" ",
                                                                                                    "") + '/sans_contradiction/'
    print()
    print("Load CNF / DNF from ", path_save_modelvf, " only work for k = 9 ou 8")
    print()
    # load DNF
    all_dnf = {}
    all_cnf = {}
    nogolist = []
    # if k ==9:
    w1, x1, y1, v1, w2, x2, y2, v2, w3 = symbols('x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8')
    varsici9 = [w1, x1, y1, v1, w2, x2, y2, v2, w3]
    varsici4 = [w1, x1, y1, v1]
    # else:
    w1, x1, y1, v1, w2, x2, y2, v2 = symbols('x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7')
    varsici8 = [w1, x1, y1, v1, w2, x2, y2, v2]
    for block_occurence in tqdm(range(0, len(args.Blocks_filters_output))):
        kici = args.kernel_size_per_block[block_occurence] * args.kernel_size_per_block[block_occurence] * \
               args.groups_per_block[block_occurence]
        if kici == 9:
            varsici = varsici9
        elif kici == 4:
            varsici = varsici4
        else:
            varsici = varsici8

        all_dnf[block_occurence] = {}
        all_cnf[block_occurence] = {}
        for filteroccurence in range(args.Blocks_filters_output[block_occurence]):
            try:
                with open(path_save_modelvf + 'DNF_expression_block' + str(block_occurence) + '_filter_' + str(
                        filteroccurence) + '_coefdefault_1.0_sousblock_None.txt') as file:
                    dnf = parse_expr(str(file.read()), evaluate=False)
                    # print(dnf)

                    values = truth_table(dnf, varsici, input=False)
                    values2 = list(values)
                    values3 = []
                    for x in values2:
                        if str(x) == "True":
                            values3.append(1)
                        else:
                            values3.append(0)
                    all_dnf[block_occurence][filteroccurence] = values3
                with open(path_save_modelvf + 'CNF_expression_block' + str(block_occurence) + '_filter_' + str(
                        filteroccurence) + '_coefdefault_1.0_sousblock_None.txt') as file:
                    cnf = parse_expr(str(file.read()), evaluate=False)
                    values = truth_table(cnf, varsici, input=False)
                    values2 = list(values)
                    values3 = []
                    for x in values2:
                        if str(x) == "True":
                            values3.append(1)
                        else:
                            values3.append(0)
                    all_cnf[block_occurence][filteroccurence] = values3
            except:
                all_dnf[block_occurence][filteroccurence] = 1
                all_cnf[block_occurence][filteroccurence] = 1
                nogolist.append((block_occurence, filteroccurence))

    return all_cnf, all_dnf, nogolist


import torchvision
import torchvision.transforms as transforms


def load_data(args):
    nclass = 10
    transform_test = transforms.Compose([
        transforms.ToTensor()])
    if args.dataset == "MNIST":
        testset = torchvision.datasets.MNIST("~/datasets/mnist", transform=transform_test, train=False, download=True)
        nclass = 10
    elif args.dataset == "CIFAR10":
        testset = torchvision.datasets.CIFAR10("~/datasets/CIFAR10", transform=transform_test, train=False,
                                               download=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=(args.batch_size_test), shuffle=False,
                                             num_workers=(args.num_workers))
    dataloaders = {"val": testloader}
    return dataloaders, testset, nclass


import numpy as np


def BitsToIntAFast(bits):
    m, n = bits.shape  # number of columns is needed, not bits.size
    a = 2 ** np.arange(n)[::-1]  # -1 reverses array of powers of 2 of same length as bits
    return bits @ a  # this matmult is the key line of code


def unpacking(output_0):
    output_array = []
    for i in range(48):
        output_array.append((output_0 >> (47 - i)) & 0x01)
    output_array = np.array(output_array)
    # print(output_array)
    return output_array


import concurrent.futures


def infer_normal(inputs, preprocessing, device, unfold_all, args, mapping_filter, W_LR, b_LR, cache, c_func,
                 block_1_functions, items):
    with torch.no_grad():
        imgs_debut = preprocessing(inputs.to(device)).to(device)
    batch_size_test = inputs.shape[0]
    res_all_tensorinput_block = {}
    res_all_tensoroutput_block = {}
    shape_all_tensorinput_block = {}
    shape_all_tensoroutput_block = {}
    for block_occurence in range(len(args.Blocks_filters_output)):  ##Each layer
        iterici = 0
        filtericitot = args.Blocks_filters_output[block_occurence]
        unfold_block = unfold_all[block_occurence][iterici]
        if block_occurence == 0:
            input_vu_par_cnn_avant_unfold = imgs_debut[:,  # batch
                                            mapping_filter[block_occurence][0]  # channel
                                            : mapping_filter[block_occurence][0] + int(
                                                args.groups_per_block[block_occurence]),  # channel
                                            :,
                                            :]
            input_vu_par_cnn_et_sat_starting = unfold_block(input_vu_par_cnn_avant_unfold).cpu().numpy().astype("i")
            input_vu_par_cnn_et_sat = input_vu_par_cnn_et_sat_starting.transpose(1, 0, 2).reshape(9, -1)
            input_vu_par_cnn_et_sat_T = input_vu_par_cnn_et_sat.transpose()
            input_vu_par_cnn_et_sat_T_int = BitsToIntAFast(input_vu_par_cnn_et_sat_T)
            input_unfold_unique0, inverse_indices = np.unique(input_vu_par_cnn_et_sat_T_int,
                                                              return_inverse=True)  # ,axis=1)
            input_var_unfold = np.array([cache[xx] for xx in input_unfold_unique0]).transpose()
            output_filters = np.zeros((48, input_var_unfold.shape[1]))
            for col in range(input_var_unfold.shape[1]):
                input_var = input_var_unfold[:, col]
                c_func.TTnet_block_0.argtypes = c_uint8, c_uint8, c_uint8, c_uint8, c_uint8, c_uint8, c_uint8, c_uint8, c_uint8
                c_func.TTnet_block_0.restype = c_uint64
                output_0 = c_func.TTnet_block_0(input_var[0], input_var[1], input_var[2], input_var[3], input_var[4],
                                                input_var[5], input_var[6], input_var[7],
                                                input_var[8])  # note 1 -> (1,0,0,0,0,0,0,0,0)
                output_0 = unpacking(output_0)
                output_filters[:, col] = output_0
            output_var_unfold = output_filters[:, inverse_indices]

            res_all_tensorinput_block[block_occurence] = deepcopy(imgs_debut)
            shape_all_tensorinput_block[block_occurence] = [imgs_debut.shape[1:]]
            del imgs_debut
            imgs_debut = torch.Tensor(output_var_unfold.reshape(48, batch_size_test, 13, 13).transpose(1, 0, 2, 3)).to(
                device)
            res_all_tensoroutput_block[block_occurence] = deepcopy(imgs_debut)
            shape_all_tensoroutput_block[block_occurence] = [imgs_debut.shape[1:]]
        elif block_occurence == 1:
            def function_one_filter(filter_occurencefunction):
                input_vu_par_cnn_avant_unfold = imgs_debut[:,
                                                mapping_filter[1][filter_occurencefunction]
                                                : mapping_filter[1][filter_occurencefunction] + int(
                                                    args.groups_per_block[1]),
                                                :,
                                                :]
                input_vu_par_cnn_et_sat_starting = unfold_block(input_vu_par_cnn_avant_unfold).cpu().numpy().astype("i")
                input_vu_par_cnn_et_sat = input_vu_par_cnn_et_sat_starting.transpose(1, 0, 2).reshape(9, -1)
                input_vu_par_cnn_et_sat_T = input_vu_par_cnn_et_sat.transpose()
                input_vu_par_cnn_et_sat_T_int = BitsToIntAFast(input_vu_par_cnn_et_sat_T)
                input_unfold_unique0, inverse_indices = np.unique(input_vu_par_cnn_et_sat_T_int,
                                                                  return_inverse=True)  # ,axis=1)
                input_unfold_unique = np.array([cache[xx] for xx in input_unfold_unique0]).transpose()
                TT_C_function = block_1_functions[filter_occurencefunction]
                output_filters = np.zeros(input_unfold_unique.shape[1])
                for col in range(input_unfold_unique.shape[1]):
                    input_var = input_unfold_unique[:, col]
                    TT_C_function.argtypes = c_uint8, c_uint8, c_uint8, c_uint8, c_uint8, c_uint8, c_uint8, c_uint8, c_uint8
                    TT_C_function.restype = c_uint8
                    output_1 = TT_C_function(input_var[0], input_var[1], input_var[2], input_var[3],
                                             input_var[4], input_var[5], input_var[6], input_var[7],
                                             input_var[8])  # note 1 -> (1,0,0,0,0,0,0,0,0)

                    output_filters[col] = int(output_1)
                output_var_unfold = output_filters[inverse_indices]
                return output_var_unfold

            with concurrent.futures.ThreadPoolExecutor() as executor:
                features_per_filter_output_list = executor.map(function_one_filter, items)
            features_per_filter_output_list = np.vstack(list(features_per_filter_output_list))
            imgs_fin = features_per_filter_output_list.reshape(48, batch_size_test, 11, 11).transpose(1, 0, 2, 3)

            res_all_tensorinput_block[block_occurence] = deepcopy(imgs_debut)
            res_all_tensoroutput_block[block_occurence] = deepcopy(imgs_fin)
            shape_all_tensorinput_block[block_occurence] = [imgs_debut.shape[1:]]
            shape_all_tensoroutput_block[block_occurence] = [imgs_fin.shape[1:]]

    feature_vector = imgs_fin.reshape(batch_size_test, -1).astype(
        'i').transpose()
    V_ref = np.dot(W_LR, feature_vector).transpose() + b_LR
    predicted = np.argmax(V_ref, axis=-1)
    return predicted, res_all_tensorinput_block, res_all_tensoroutput_block, \
           shape_all_tensorinput_block, shape_all_tensoroutput_block


def infer_normal_withPYTHON(inputs, preprocessing, device, unfold_all, args, mapping_filter, W_LR, b_LR,
                            array_block_0, array_block_1, items, putawayliteral=[],
                            bit1_I=[],
                            bit1_L=[],
                            bit1_F=[]):
    #max_value_B0_thr_flow = [170,170,167,22,171,170,161,38,159,168,160,19,32,40,161,166,156,172,31,36,166,40,162,173,26,160,35,None]+[173,177,161,154,161,158,157,39,38,171,158,32,40,17,22,157,34,161,31,40]
    #max_value_B1_thr_flow = [21, 13, 12, 14, 7, 39, 8, 45, 38, 5, 31, 27, 35, 37, 14, 20, 35, 11, 7, 30, 36, 31, 30, 10,
    #                         32, 39, 17, 22, 8, 10, 10, None, 39, 8, 7, 36, 15, 15, 16, 32, 10, 30, 32, 44, 47, 38, 11,
    #                         9]

    with torch.no_grad():
        imgs_debut = preprocessing(inputs.to(device)).to(device)
    #for x in bit1_I:
    #    input_value = imgs_debut[:, x[0], x[1], x[2]]
    #input_value = imgs_debut[:, 0, 16, 17]
    batch_size_test = inputs.shape[0]
    res_all_tensorinput_block = {}
    res_all_tensorinput_block_unfold = {}
    res_all_tensoroutput_block = {}
    shape_all_tensorinput_block = {}
    shape_all_tensoroutput_block = {}
    for block_occurence in range(len(args.Blocks_filters_output)):  ##Each layer
        nSize = args.kernel_size_per_block[block_occurence] ** 2 * args.groups_per_block[block_occurence]
        iterici = 0
        filtericitot = args.Blocks_filters_output[block_occurence]
        unfold_block = unfold_all[block_occurence][iterici]
        if block_occurence == 0:
            res_all_tensorinput_block_unfold[block_occurence] = {}
            doit = int(args.preprocessing_CNN[0] / args.groups_per_block[block_occurence])
            coef_multi = int(args.Blocks_filters_output[0] / doit)
            output_var_unfold_vf = None
            for filter_occurenceb0 in range(args.Blocks_filters_output[0]):
                # print(filter_occurenceb0)
                # print(coef_multi * filter_occurenceb0)
                # print(mapping_filter[block_occurence][ coef_multi * filter_occurenceb0] + int(
                #                                    args.groups_per_block[block_occurence]))
                input_vu_par_cnn_avant_unfold = imgs_debut[:,  # batch
                                                mapping_filter[0][filter_occurenceb0]:
                                                mapping_filter[0][filter_occurenceb0]+ int(
                                                args.groups_per_block[1]),
                                                :,
                                                :]
                input_vu_par_cnn_et_sat_starting = unfold_block(input_vu_par_cnn_avant_unfold).cpu().numpy().astype("i")
                H = int(np.sqrt(input_vu_par_cnn_et_sat_starting.shape[-1]))
                input_vu_par_cnn_et_sat = input_vu_par_cnn_et_sat_starting.transpose(1, 0, 2).reshape(nSize, -1)
                #for kkb0 in range(nSize):
                #    if (block_occurence, filter_occurenceb0, kkb0) in putawayliteral:
                #        input_vu_par_cnn_et_sat[kkb0, :] = 0
                res_all_tensorinput_block_unfold[block_occurence][filter_occurenceb0] = input_vu_par_cnn_et_sat
                input_vu_par_cnn_et_sat_T = input_vu_par_cnn_et_sat.transpose()
                input_vu_par_cnn_et_sat_T_int = BitsToIntAFast(input_vu_par_cnn_et_sat_T)
                input_unfold_unique0, inverse_indices = np.unique(input_vu_par_cnn_et_sat_T_int,
                                                                  return_inverse=True)  # ,axis=1)
                # output_filters = np.zeros((coef_multi, input_unfold_unique0.shape[0]))
                # print(coef_multi*filter_occurenceb0, coef_multi*filter_occurenceb0+coef_multi)
                output_filters_ici = array_block_0[filter_occurenceb0:filter_occurenceb0 + 1, input_unfold_unique0]

                # print(output_filters_ici.shape)
                output_var_unfold = output_filters_ici[:, inverse_indices]
                #print(" Block 0, Filter ", filter_occurenceb0, " sum: ", np.sum(output_var_unfold==1), " sum (%) ", np.sum(output_var_unfold==1)/(imgs_debut.shape[0])
                #      , " sum2 (%) ", np.sum(output_var_unfold==1)/(imgs_debut.shape[0]*14*14))
                if output_var_unfold_vf is None:
                    output_var_unfold_vf = output_var_unfold
                else:
                    output_var_unfold_vf = np.concatenate((output_var_unfold_vf, output_var_unfold), axis=0)
            res_all_tensorinput_block[block_occurence] = deepcopy(imgs_debut)

            shape_all_tensorinput_block[block_occurence] = [imgs_debut.shape[1:]]
            del imgs_debut
            imgs_debut = torch.Tensor(
                output_var_unfold_vf.reshape(args.Blocks_filters_output[block_occurence], batch_size_test,
                                             H,
                                             H).transpose(1, 0, 2, 3)).to(
                device)


            # for FB0 in tqdm(range(0,48,4)):
            #     if max_value_B0_thr_flow[FB0]>14*14/2:
            #         offset = -10
            #     else:
            #         offset = 10
            #     boundB0ici = max_value_B0_thr_flow[FB0]+offset
            #     if boundB0ici is not None:
            #         bound = torch.sum(imgs_debut[:, FB0, :, :] == 1, dim=(1, 2))
            #         for batchici in range(batch_size_test):
            #             if bound[batchici]!=boundB0ici:
            #                 nbre2foistoadd = abs(bound[batchici] - boundB0ici)
            #                 if boundB0ici > bound[batchici]:
            #                     value2add = 1
            #                 else:
            #                     value2add = 0
            #                 #while nbre2foistoadd!=0:
            #                     #print(nbre2foistoadd)
            #                 for xiterici in range(2,12):
            #                     if nbre2foistoadd != 0:
            #                         for yiterici in range(2,12):
            #                         #print(nbre2foistoadd)
            #                             if nbre2foistoadd !=0:
            #                                 if imgs_debut[batchici, FB0, xiterici, yiterici]!=value2add:
            #                                     imgs_debut[batchici, FB0, xiterici, yiterici] =value2add
            #                                     nbre2foistoadd=nbre2foistoadd-1
            #                             else:
            #                                 break
            #                     else:
            #                         break



            res_all_tensoroutput_block[block_occurence] = deepcopy(imgs_debut)
            shape_all_tensoroutput_block[block_occurence] = [imgs_debut.shape[1:]]
        elif block_occurence == 1:
            # allbit1_L = [(2, 6, 8), (2, 7, 9), (2, 11, 9), (3, 7, 4), (3, 7, 8), (3, 10, 4), (5, 8, 9), (7, 6, 8),
            #              (15, 7, 5),
            #              (16, 6, 10), (18, 6, 9), (20, 4, 8), (29, 8, 9), (41, 6, 8), (41, 8, 8)]
            # for x in allbit1_L:
            #     imgs_debut[:, x[0], x[1], x[2]] = input_value
            res_all_tensorinput_block_unfold[block_occurence] = {}
            input_vu_par_cnn_avant_unfold = imgs_debut[:,
                                            mapping_filter[1][0]
                                            : mapping_filter[1][0] + int(
                                                args.groups_per_block[1]),
                                            :,
                                            :]
            input_vu_par_cnn_et_sat_starting = unfold_block(input_vu_par_cnn_avant_unfold).cpu().numpy().astype("i")
            H2 = int(np.sqrt(input_vu_par_cnn_et_sat_starting.shape[-1]))
            del input_vu_par_cnn_avant_unfold, input_vu_par_cnn_et_sat_starting

            def function_one_filter(filter_occurencefunction):
                input_vu_par_cnn_avant_unfold = imgs_debut[:,
                                                mapping_filter[1][filter_occurencefunction]
                                                : mapping_filter[1][filter_occurencefunction] + int(
                                                    args.groups_per_block[1]),
                                                :,
                                                :]
                input_vu_par_cnn_et_sat_starting = unfold_block(input_vu_par_cnn_avant_unfold).cpu().numpy().astype("i")
                # H2 = int(np.sqrt(input_vu_par_cnn_et_sat_starting.shape[-1]))
                # global H2
                input_vu_par_cnn_et_sat = input_vu_par_cnn_et_sat_starting.transpose(1, 0, 2).reshape(nSize, -1)

                #for kkb0 in range(nSize):
                #    if (block_occurence, filter_occurencefunction, kkb0) in putawayliteral:
                #        input_vu_par_cnn_et_sat[kkb0, :] = 0
                res_all_tensorinput_block_unfold[block_occurence][filter_occurencefunction] = input_vu_par_cnn_et_sat
                input_vu_par_cnn_et_sat_T = input_vu_par_cnn_et_sat.transpose()
                input_vu_par_cnn_et_sat_T_int = BitsToIntAFast(input_vu_par_cnn_et_sat_T)
                input_unfold_unique0, inverse_indices = np.unique(input_vu_par_cnn_et_sat_T_int,
                                                                  return_inverse=True)  # ,axis=1)
                output_filters = array_block_1[filter_occurencefunction, input_unfold_unique0]
                output_var_unfold = output_filters[inverse_indices]
                #print(" Block 1, Filter ", filter_occurencefunction, " sum: ", np.sum(output_var_unfold == 1), " sum (%) ",
                #      np.sum(output_var_unfold == 1) / (imgs_debut.shape[0])
                #, " sum2 (%) ", np.sum(output_var_unfold == 1) / (imgs_debut.shape[0] * 7 * 7))
                return output_var_unfold

            with concurrent.futures.ThreadPoolExecutor() as executor:
                features_per_filter_output_list = executor.map(function_one_filter, items)
            features_per_filter_output_list = np.vstack(list(features_per_filter_output_list))
            imgs_fin = features_per_filter_output_list.reshape(args.Blocks_filters_output[block_occurence],
                                                               batch_size_test,
                                                               H2,
                                                               H2).transpose(1, 0, 2, 3)


            # for FB0 in tqdm(range(0,48,4)):
            #     if max_value_B1_thr_flow[FB0]>14*14/2:
            #         offset = 0
            #     else:
            #         offset = 0
            #     boundB0ici = max_value_B1_thr_flow[FB0]+offset
            #     if boundB0ici is not None:
            #         bound = np.sum(imgs_fin[:, FB0, :, :] == 1, axis=(1, 2))
            #         for batchici in range(batch_size_test):
            #             if bound[batchici]!=boundB0ici:
            #                 nbre2foistoadd = abs(bound[batchici] - boundB0ici)
            #                 if boundB0ici > bound[batchici]:
            #                     value2add = 1
            #                 else:
            #                     value2add = 0
            #                 #while nbre2foistoadd!=0:
            #                     #print(nbre2foistoadd)
            #                 for xiterici in range(2,6):
            #                     if nbre2foistoadd != 0:
            #                         for yiterici in range(2,6):
            #                         #print(nbre2foistoadd)
            #                             if nbre2foistoadd !=0:
            #                                 if imgs_fin[batchici, FB0, xiterici, yiterici]!=value2add:
            #                                     imgs_fin[batchici, FB0, xiterici, yiterici] =value2add
            #                                     nbre2foistoadd=nbre2foistoadd-1
            #                             else:
            #                                 break
            #                     else:
            #                         break





            res_all_tensorinput_block[block_occurence] = deepcopy(imgs_debut)
            res_all_tensoroutput_block[block_occurence] = deepcopy(imgs_fin)
            shape_all_tensorinput_block[block_occurence] = [imgs_debut.shape[1:]]
            shape_all_tensoroutput_block[block_occurence] = [imgs_fin.shape[1:]]

    feature_vector = imgs_fin.reshape(batch_size_test, -1).astype(
        'i').transpose()
    #bit1_F = [17, 79, 164, 184, 213, 278, 416, 571, 767, 858, 1157, 1741, 2034, 2088]

    #for xposval1 in bit1_F:
    #    feature_vector[xposval1, :] = input_value

    V_ref = np.dot(W_LR, feature_vector).transpose() + b_LR
    predicted = np.argmax(V_ref, axis=-1)
    return predicted, res_all_tensorinput_block, res_all_tensoroutput_block, \
           shape_all_tensorinput_block, shape_all_tensoroutput_block, res_all_tensorinput_block_unfold


def infer_normal_withPYTHON_without_preprocess(imgs_debut, device, unfold_all, args, mapping_filter, W_LR, b_LR,
                                               array_block_0, array_block_1, items):
    # with torch.no_grad():
    #    imgs_debut = preprocessing(inputs.to(device)).to(device)
    batch_size_test = imgs_debut.shape[0]
    res_all_tensorinput_block = {}
    res_all_tensoroutput_block = {}
    shape_all_tensorinput_block = {}
    shape_all_tensoroutput_block = {}
    for block_occurence in range(len(args.Blocks_filters_output)):  ##Each layer
        nSize = args.kernel_size_per_block[block_occurence] ** 2 * args.groups_per_block[block_occurence]
        iterici = 0
        filtericitot = args.Blocks_filters_output[block_occurence]
        unfold_block = unfold_all[block_occurence][iterici]
        if block_occurence == 0:
            doit = int(args.preprocessing_CNN[0] / args.groups_per_block[block_occurence])
            coef_multi = int(args.Blocks_filters_output[0] / doit)
            output_var_unfold_vf = None
            # print(doit, coef_multi)
            for filter_occurenceb0 in range(doit):
                # print(filter_occurenceb0)
                # print(coef_multi * filter_occurenceb0)
                # print(mapping_filter[block_occurence][ coef_multi * filter_occurenceb0] + int(
                #                                    args.groups_per_block[block_occurence]))
                input_vu_par_cnn_avant_unfold = imgs_debut[:,  # batch
                                                mapping_filter[block_occurence][
                                                    coef_multi * filter_occurenceb0]  # channel
                                                : mapping_filter[block_occurence][
                                                      coef_multi * filter_occurenceb0] + int(
                                                    args.groups_per_block[block_occurence]),  # channel
                                                :,
                                                :]
                input_vu_par_cnn_et_sat_starting = unfold_block(input_vu_par_cnn_avant_unfold).cpu().numpy().astype("i")
                H = int(np.sqrt(input_vu_par_cnn_et_sat_starting.shape[-1]))
                input_vu_par_cnn_et_sat = input_vu_par_cnn_et_sat_starting.transpose(1, 0, 2).reshape(nSize, -1)
                input_vu_par_cnn_et_sat_T = input_vu_par_cnn_et_sat.transpose()
                input_vu_par_cnn_et_sat_T_int = BitsToIntAFast(input_vu_par_cnn_et_sat_T)
                input_unfold_unique0, inverse_indices = np.unique(input_vu_par_cnn_et_sat_T_int,
                                                                  return_inverse=True)  # ,axis=1)
                # output_filters = np.zeros((coef_multi, input_unfold_unique0.shape[0]))
                # print(coef_multi*filter_occurenceb0, coef_multi*filter_occurenceb0+coef_multi)
                output_filters_ici = array_block_0[
                                     coef_multi * filter_occurenceb0:coef_multi * filter_occurenceb0 + coef_multi,
                                     input_unfold_unique0]
                # print(output_filters_ici.shape)
                output_var_unfold = output_filters_ici[:, inverse_indices]
                # print(output_var_unfold.shape)
                if output_var_unfold_vf is None:
                    output_var_unfold_vf = output_var_unfold
                else:
                    output_var_unfold_vf = np.concatenate((output_var_unfold_vf, output_var_unfold), axis=0)
            shape_all_tensorinput_block[block_occurence] = [imgs_debut.shape[1:]]
            del imgs_debut
            imgs_debut = torch.Tensor(
                output_var_unfold_vf.reshape(args.Blocks_filters_output[block_occurence], batch_size_test,
                                             H,
                                             H).transpose(1, 0, 2, 3)).to(
                device)
            res_all_tensoroutput_block[block_occurence] = deepcopy(imgs_debut)
            shape_all_tensoroutput_block[block_occurence] = [imgs_debut.shape[1:]]
        elif block_occurence == 1:
            input_vu_par_cnn_avant_unfold = imgs_debut[:,
                                            mapping_filter[1][0]
                                            : mapping_filter[1][0] + int(
                                                args.groups_per_block[1]),
                                            :,
                                            :]
            input_vu_par_cnn_et_sat_starting = unfold_block(input_vu_par_cnn_avant_unfold).cpu().numpy().astype("i")
            H2 = int(np.sqrt(input_vu_par_cnn_et_sat_starting.shape[-1]))
            del input_vu_par_cnn_avant_unfold, input_vu_par_cnn_et_sat_starting

            def function_one_filter(filter_occurencefunction):
                input_vu_par_cnn_avant_unfold = imgs_debut[:,
                                                mapping_filter[1][filter_occurencefunction]
                                                : mapping_filter[1][filter_occurencefunction] + int(
                                                    args.groups_per_block[1]),
                                                :,
                                                :]
                input_vu_par_cnn_et_sat_starting = unfold_block(input_vu_par_cnn_avant_unfold).cpu().numpy().astype("i")
                # H2 = int(np.sqrt(input_vu_par_cnn_et_sat_starting.shape[-1]))
                # global H2
                input_vu_par_cnn_et_sat = input_vu_par_cnn_et_sat_starting.transpose(1, 0, 2).reshape(nSize, -1)
                input_vu_par_cnn_et_sat_T = input_vu_par_cnn_et_sat.transpose()
                input_vu_par_cnn_et_sat_T_int = BitsToIntAFast(input_vu_par_cnn_et_sat_T)
                input_unfold_unique0, inverse_indices = np.unique(input_vu_par_cnn_et_sat_T_int,
                                                                  return_inverse=True)  # ,axis=1)
                output_filters = array_block_1[filter_occurencefunction, input_unfold_unique0]
                output_var_unfold = output_filters[inverse_indices]
                return output_var_unfold

            with concurrent.futures.ThreadPoolExecutor() as executor:
                features_per_filter_output_list = executor.map(function_one_filter, items)
            features_per_filter_output_list = np.vstack(list(features_per_filter_output_list))
            imgs_fin = features_per_filter_output_list.reshape(args.Blocks_filters_output[block_occurence],
                                                               batch_size_test,
                                                               H2,
                                                               H2).transpose(1, 0, 2, 3)

            res_all_tensorinput_block[block_occurence] = deepcopy(imgs_debut)
            res_all_tensoroutput_block[block_occurence] = deepcopy(imgs_fin)
            shape_all_tensorinput_block[block_occurence] = [imgs_debut.shape[1:]]
            shape_all_tensoroutput_block[block_occurence] = [imgs_fin.shape[1:]]

    feature_vector = imgs_fin.reshape(batch_size_test, -1).astype(
        'i').transpose()
    V_ref = np.dot(W_LR, feature_vector).transpose() + b_LR
    predicted = np.argmax(V_ref, axis=-1)
    return predicted, res_all_tensorinput_block, res_all_tensoroutput_block, \
           shape_all_tensorinput_block, shape_all_tensoroutput_block


def infer_normal_withPYTHONjeudelavie(inputs, preprocessing, device, unfold_all, args, mapping_filter, W_LR, b_LR,
                                      array_block_0, array_block_1, items, l3, l4):
    with torch.no_grad():
        imgs_debut = preprocessing(inputs.to(device)).to(device)
    batch_size_test = inputs.shape[0]
    res_all_tensorinput_block = {}
    res_all_tensoroutput_block = {}
    shape_all_tensorinput_block = {}
    shape_all_tensoroutput_block = {}
    for block_occurence in range(len(args.Blocks_filters_output)):  ##Each layer
        nSize = args.kernel_size_per_block[block_occurence] ** 2 * args.groups_per_block[block_occurence]
        iterici = 0
        filtericitot = args.Blocks_filters_output[block_occurence]
        unfold_block = unfold_all[block_occurence][iterici]
        if block_occurence == 0:
            doit = int(args.preprocessing_CNN[0] / args.groups_per_block[block_occurence])
            coef_multi = int(args.Blocks_filters_output[0] / doit)
            output_var_unfold_vf = None
            # print(doit, coef_multi)
            for filter_occurenceb0 in range(doit):
                # print(filter_occurenceb0)
                # print(coef_multi * filter_occurenceb0)
                # print(mapping_filter[block_occurence][ coef_multi * filter_occurenceb0] + int(
                #                                    args.groups_per_block[block_occurence]))
                input_vu_par_cnn_avant_unfold = imgs_debut[:,  # batch
                                                mapping_filter[block_occurence][
                                                    coef_multi * filter_occurenceb0]  # channel
                                                : mapping_filter[block_occurence][
                                                      coef_multi * filter_occurenceb0] + int(
                                                    args.groups_per_block[block_occurence]),  # channel
                                                :,
                                                :]
                input_vu_par_cnn_et_sat_starting = unfold_block(input_vu_par_cnn_avant_unfold).cpu().numpy().astype("i")
                H = int(np.sqrt(input_vu_par_cnn_et_sat_starting.shape[-1]))
                input_vu_par_cnn_et_sat = input_vu_par_cnn_et_sat_starting.transpose(1, 0, 2).reshape(nSize, -1)
                input_vu_par_cnn_et_sat_T = input_vu_par_cnn_et_sat.transpose()
                input_vu_par_cnn_et_sat_T_int = BitsToIntAFast(input_vu_par_cnn_et_sat_T)
                input_unfold_unique0, inverse_indices = np.unique(input_vu_par_cnn_et_sat_T_int,
                                                                  return_inverse=True)  # ,axis=1)
                # output_filters = np.zeros((coef_multi, input_unfold_unique0.shape[0]))
                # print(coef_multi*filter_occurenceb0, coef_multi*filter_occurenceb0+coef_multi)
                output_filters_ici = array_block_0[
                                     coef_multi * filter_occurenceb0:coef_multi * filter_occurenceb0 + coef_multi,
                                     input_unfold_unique0]
                # print(output_filters_ici.shape)
                output_var_unfold = output_filters_ici[:, inverse_indices]
                # print(output_var_unfold.shape)
                if output_var_unfold_vf is None:
                    output_var_unfold_vf = output_var_unfold
                else:
                    output_var_unfold_vf = np.concatenate((output_var_unfold_vf, output_var_unfold), axis=0)
            res_all_tensorinput_block[block_occurence] = deepcopy(imgs_debut)
            shape_all_tensorinput_block[block_occurence] = [imgs_debut.shape[1:]]
            del imgs_debut
            imgs_debut = torch.Tensor(
                output_var_unfold_vf.reshape(args.Blocks_filters_output[block_occurence], batch_size_test,
                                             H,
                                             H).transpose(1, 0, 2, 3)).to(
                device)
            res_all_tensoroutput_block[block_occurence] = deepcopy(imgs_debut)
            shape_all_tensoroutput_block[block_occurence] = [imgs_debut.shape[1:]]
        elif block_occurence == 1:
            input_vu_par_cnn_avant_unfold = imgs_debut[:,
                                            mapping_filter[1][0]
                                            : mapping_filter[1][0] + int(
                                                args.groups_per_block[1]),
                                            :,
                                            :]
            input_vu_par_cnn_et_sat_starting = unfold_block(input_vu_par_cnn_avant_unfold).cpu().numpy().astype("i")
            H2 = int(np.sqrt(input_vu_par_cnn_et_sat_starting.shape[-1]))
            del input_vu_par_cnn_avant_unfold, input_vu_par_cnn_et_sat_starting

            def function_one_filter(filter_occurencefunction):
                input_vu_par_cnn_avant_unfold = imgs_debut[:,
                                                mapping_filter[1][filter_occurencefunction]
                                                : mapping_filter[1][filter_occurencefunction] + int(
                                                    args.groups_per_block[1]),
                                                :,
                                                :]
                input_vu_par_cnn_et_sat_starting = unfold_block(input_vu_par_cnn_avant_unfold).cpu().numpy().astype("i")
                # H2 = int(np.sqrt(input_vu_par_cnn_et_sat_starting.shape[-1]))
                # global H2
                input_vu_par_cnn_et_sat = input_vu_par_cnn_et_sat_starting.transpose(1, 0, 2).reshape(nSize, -1)
                input_vu_par_cnn_et_sat_T = input_vu_par_cnn_et_sat.transpose()
                input_vu_par_cnn_et_sat_T_int = BitsToIntAFast(input_vu_par_cnn_et_sat_T)
                input_vu_par_cnn_et_sat_T_int_jeu2lavie = []
                for x in input_vu_par_cnn_et_sat_T_int:
                    if x not in l3 and x not in l4:
                        input_vu_par_cnn_et_sat_T_int_jeu2lavie.append(x)
                    elif x in l3:
                        input_vu_par_cnn_et_sat_T_int_jeu2lavie.append(0)
                    elif x in l4:
                        input_vu_par_cnn_et_sat_T_int_jeu2lavie.append(512)

                input_unfold_unique0, inverse_indices = np.unique(input_vu_par_cnn_et_sat_T_int_jeu2lavie,
                                                                  return_inverse=True)  # ,axis=1)
                output_filters = array_block_1[filter_occurencefunction, input_unfold_unique0]
                output_var_unfold = output_filters[inverse_indices]
                return output_var_unfold

            with concurrent.futures.ThreadPoolExecutor() as executor:
                features_per_filter_output_list = executor.map(function_one_filter, items)
            features_per_filter_output_list = np.vstack(list(features_per_filter_output_list))
            imgs_fin = features_per_filter_output_list.reshape(args.Blocks_filters_output[block_occurence],
                                                               batch_size_test,
                                                               H2,
                                                               H2).transpose(1, 0, 2, 3)

            res_all_tensorinput_block[block_occurence] = deepcopy(imgs_debut)
            res_all_tensoroutput_block[block_occurence] = deepcopy(imgs_fin)
            shape_all_tensorinput_block[block_occurence] = [imgs_debut.shape[1:]]
            shape_all_tensoroutput_block[block_occurence] = [imgs_fin.shape[1:]]

    feature_vector = imgs_fin.reshape(batch_size_test, -1).astype(
        'i').transpose()
    V_ref = np.dot(W_LR, feature_vector).transpose() + b_LR
    predicted = np.argmax(V_ref, axis=-1)

    return predicted, res_all_tensorinput_block, res_all_tensoroutput_block, shape_all_tensorinput_block, shape_all_tensoroutput_block


def infer_normal_withPYTHON_multihead(inputs, preprocessing, device, unfold_all, args, mapping_filter,
                                      W_LR, b_LR, array_block_0,
                                      array_block_1, items):
    with torch.no_grad():
        imgs_debut = preprocessing(inputs.to(device)).to(device)
    batch_size_test = inputs.shape[0]
    res_all_tensorinput_block = {}
    res_all_tensoroutput_block = {}
    shape_all_tensorinput_block = {}
    shape_all_tensoroutput_block = {}
    for block_occurence in range(len(args.Blocks_filters_output)):  ##Each layer
        nSize = args.kernel_size_per_block[block_occurence] ** 2 * args.groups_per_block[block_occurence]
        iterici = 0
        filtericitot = args.Blocks_filters_output[block_occurence]
        unfold_block = unfold_all[block_occurence][iterici]
        if block_occurence == 0:
            doit = int(args.preprocessing_CNN[0] / args.groups_per_block[block_occurence])
            coef_multi = int(args.Blocks_filters_output[0] / doit)
            output_var_unfold_vf = None
            # print(doit, coef_multi)
            for filter_occurenceb0 in range(doit):
                # print(filter_occurenceb0)
                # print(coef_multi * filter_occurenceb0)
                # print(mapping_filter[block_occurence][ coef_multi * filter_occurenceb0] + int(
                #                                    args.groups_per_block[block_occurence]))
                input_vu_par_cnn_avant_unfold = imgs_debut[:,  # batch
                                                mapping_filter[block_occurence][
                                                    coef_multi * filter_occurenceb0]  # channel
                                                : mapping_filter[block_occurence][
                                                      coef_multi * filter_occurenceb0] + int(
                                                    args.groups_per_block[block_occurence]),  # channel
                                                :,
                                                :]
                input_vu_par_cnn_et_sat_starting = unfold_block(input_vu_par_cnn_avant_unfold).cpu().numpy().astype("i")
                H = int(np.sqrt(input_vu_par_cnn_et_sat_starting.shape[-1]))
                input_vu_par_cnn_et_sat = input_vu_par_cnn_et_sat_starting.transpose(1, 0, 2).reshape(nSize, -1)
                input_vu_par_cnn_et_sat_T = input_vu_par_cnn_et_sat.transpose()
                input_vu_par_cnn_et_sat_T_int = BitsToIntAFast(input_vu_par_cnn_et_sat_T)
                input_unfold_unique0, inverse_indices = np.unique(input_vu_par_cnn_et_sat_T_int,
                                                                  return_inverse=True)  # ,axis=1)
                # output_filters = np.zeros((coef_multi, input_unfold_unique0.shape[0]))
                # print(coef_multi*filter_occurenceb0, coef_multi*filter_occurenceb0+coef_multi)
                output_filters_ici = array_block_0[
                                     coef_multi * filter_occurenceb0:coef_multi * filter_occurenceb0 + coef_multi,
                                     input_unfold_unique0]
                # print(output_filters_ici.shape)
                output_var_unfold = output_filters_ici[:, inverse_indices]
                # print(output_var_unfold.shape)
                if output_var_unfold_vf is None:
                    output_var_unfold_vf = output_var_unfold
                else:
                    output_var_unfold_vf = np.concatenate((output_var_unfold_vf, output_var_unfold), axis=0)
            res_all_tensorinput_block[block_occurence] = deepcopy(imgs_debut)
            shape_all_tensorinput_block[block_occurence] = [imgs_debut.shape[1:]]
            del imgs_debut
            imgs_debut = torch.Tensor(
                output_var_unfold_vf.reshape(args.Blocks_filters_output[block_occurence], batch_size_test,
                                             H,
                                             H).transpose(1, 0, 2, 3)).to(
                device)
            res_all_tensoroutput_block[block_occurence] = deepcopy(imgs_debut)
            shape_all_tensoroutput_block[block_occurence] = [imgs_debut.shape[1:]]
        elif block_occurence == 1:
            input_vu_par_cnn_avant_unfold = imgs_debut[:,
                                            mapping_filter[1][0]
                                            : mapping_filter[1][0] + int(
                                                args.groups_per_block[1]),
                                            :,
                                            :]
            input_vu_par_cnn_et_sat_starting = unfold_block(input_vu_par_cnn_avant_unfold).cpu().numpy().astype("i")
            H2 = int(np.sqrt(input_vu_par_cnn_et_sat_starting.shape[-1]))
            del input_vu_par_cnn_avant_unfold, input_vu_par_cnn_et_sat_starting

            def function_one_filter(filter_occurencefunction):
                input_vu_par_cnn_avant_unfold = imgs_debut[:,
                                                mapping_filter[1][filter_occurencefunction]
                                                : mapping_filter[1][filter_occurencefunction] + int(
                                                    args.groups_per_block[1]),
                                                :,
                                                :]
                input_vu_par_cnn_et_sat_starting = unfold_block(input_vu_par_cnn_avant_unfold).cpu().numpy().astype("i")
                # H2 = int(np.sqrt(input_vu_par_cnn_et_sat_starting.shape[-1]))
                # global H2
                input_vu_par_cnn_et_sat = input_vu_par_cnn_et_sat_starting.transpose(1, 0, 2).reshape(nSize, -1)
                input_vu_par_cnn_et_sat_T = input_vu_par_cnn_et_sat.transpose()
                input_vu_par_cnn_et_sat_T_int = BitsToIntAFast(input_vu_par_cnn_et_sat_T)
                input_unfold_unique0, inverse_indices = np.unique(input_vu_par_cnn_et_sat_T_int,
                                                                  return_inverse=True)  # ,axis=1)
                output_filters = array_block_1[filter_occurencefunction, input_unfold_unique0]
                output_var_unfold = output_filters[inverse_indices]
                return output_var_unfold

            with concurrent.futures.ThreadPoolExecutor() as executor:
                features_per_filter_output_list = executor.map(function_one_filter, items)
            features_per_filter_output_list = np.vstack(list(features_per_filter_output_list))
            imgs_fin = features_per_filter_output_list.reshape(args.Blocks_filters_output[block_occurence],
                                                               batch_size_test,
                                                               H2,
                                                               H2).transpose(1, 0, 2, 3)

            res_all_tensorinput_block[block_occurence] = deepcopy(imgs_debut)
            res_all_tensoroutput_block[block_occurence] = deepcopy(imgs_fin)
            shape_all_tensorinput_block[block_occurence] = [imgs_debut.shape[1:]]
            shape_all_tensoroutput_block[block_occurence] = [imgs_fin.shape[1:]]

    feature_vector = imgs_fin.reshape(batch_size_test, -1).astype(
        'i').transpose()
    V_ref = np.dot(W_LR, feature_vector).transpose() + b_LR
    predicted = np.argmax(V_ref, axis=-1)
    return predicted, res_all_tensorinput_block, res_all_tensoroutput_block, \
           shape_all_tensorinput_block, shape_all_tensoroutput_block


import pickle


def get_refs_all2(res_all_tensorinput_block, res_all_tensoroutput_block, coeff=1, coef_start=0):
    block_ref_all_inputs = {}
    block_ref_all_inputs_inverse = {}
    block_ref_all_outputs = {}
    cpt = coef_start
    poscpt = []
    for block in range(len(list(res_all_tensorinput_block.keys()))):
        block_ref_all_inputs[block] = {}
        block_ref_all_inputs_inverse[block] = {}
        allshapeici = res_all_tensorinput_block[block][0]
        block_ref_all_inputs[block] = torch.zeros(allshapeici)
        print(allshapeici)
        if block == 0:
            for k in range(coeff):
                for i in range(block_ref_all_inputs[block].shape[1]):
                    for j in range(block_ref_all_inputs[block].shape[2]):
                        cpt += 1
                        block_ref_all_inputs[block][k][i][j] = cpt
                        if cpt == 466:
                            print(k, i, j)
            for k in range(coeff, block_ref_all_inputs[block].shape[0]):
                k2 = k % coeff
                block_ref_all_inputs[block][k] = deepcopy(block_ref_all_inputs[block][k2])

        else:
            for k in range(block_ref_all_inputs[block].shape[0]):
                for i in range(block_ref_all_inputs[block].shape[1]):
                    for j in range(block_ref_all_inputs[block].shape[2]):
                        cpt += 1
                        block_ref_all_inputs[block][k][i][j] = cpt
                        # print(cpt)
                        if cpt in [917, 918, 1449, 1506, 2989, 3029, 4642, 4838, 5577, 6602, 7726, 8124, 9164, 9948]:
                            poscpt.append((k, i, j))
    print(poscpt)
    for block in range(len(list(res_all_tensoroutput_block.keys())) - 1):
        block_ref_all_outputs[block] = block_ref_all_inputs[block + 1]
    # print(block_ref_all_outputs)
    # print()
    blockf = len(list(res_all_tensoroutput_block.keys()))
    block_ref_all_outputs[blockf - 1] = {}
    allshapeici = res_all_tensoroutput_block[blockf - 1][0]
    for block in range(1, len(list(res_all_tensorinput_block.keys())) - 1):
        block_ref_all_outputs[block - 1] = deepcopy(block_ref_all_inputs_inverse[block])
    # print(block_ref_all_outputs)
    # print(ok)

    block_ref_all_outputs[blockf - 1] = torch.zeros(allshapeici)
    for k in range(block_ref_all_outputs[blockf - 1].shape[0]):
        for i in range(block_ref_all_outputs[blockf - 1].shape[1]):
            for j in range(block_ref_all_outputs[blockf - 1].shape[2]):
                cpt += 1
                block_ref_all_outputs[blockf - 1][k][i][j] = cpt

    return block_ref_all_inputs, block_ref_all_outputs, cpt


def get_dictionnary_ref(args, input_binary_ref, output_binary_ref, unfold_block, mapping_filter, soublock=1):
    dictionnary_ref = {}
    for block_occurence in range(len(args.Blocks_filters_output)):
        dictionnary_ref[block_occurence] = {}
        if len(input_binary_ref[block_occurence]) == 1:
            iterici = 0
        else:
            iterici = 0
        # print(output_binary_ref, block_occurence, input_binary_ref[block_occurence])
        imgs_debut = deepcopy(input_binary_ref[block_occurence]).unsqueeze(0)
        imgs_fin = output_binary_ref[block_occurence].unsqueeze(0)
        print("Image shape input output block ", block_occurence, imgs_debut.shape, imgs_fin.shape)
        shapeici_out = imgs_fin.shape[-1] ** 2
        shapeici_out2 = imgs_fin.shape[-1]
        unfold_blocklocal = unfold_block[block_occurence][iterici]
        for filter_occurence in range(args.Blocks_filters_output[block_occurence]):
            dictionnary_ref[block_occurence][filter_occurence] = {}
            # print(imgs_debut.shape, filter_occurence, mapping_filter[block_occurence][filter_occurence], mapping_filter[block_occurence][filter_occurence] + int(args.groups_per_block[block_occurence]))
            input_vu_par_cnn_avant_unfold = imgs_debut[:,
                                            mapping_filter[block_occurence][filter_occurence]
                                            : mapping_filter[block_occurence][filter_occurence] + int(
                                                args.groups_per_block[block_occurence]),
                                            :,
                                            :]
            # print(input_vu_par_cnn_avant_unfold.shape)
            input_vu_par_cnn_et_sat = unfold_blocklocal(input_vu_par_cnn_avant_unfold)
            # print(input_vu_par_cnn_et_sat.shape)
            for xy_pixel in range(shapeici_out):
                input_varref = deepcopy(input_vu_par_cnn_et_sat[0, :, xy_pixel].numpy().astype("i").tolist())
                x_pixel = int(xy_pixel // shapeici_out2)
                y_pixel = int(xy_pixel % shapeici_out2)
                output_binary_ref_vf = imgs_fin[0, filter_occurence, x_pixel, y_pixel]
                dictionnary_ref[block_occurence][filter_occurence][xy_pixel] = (
                    input_varref, int(output_binary_ref_vf.item()))

    return dictionnary_ref


def get_refs_all_cnn(res_all_tensorinput_block, res_all_tensoroutput_block, coeff=1, coef_start=0):
    block_ref_all_inputs = {}
    block_ref_all_inputs_inverse = {}
    block_ref_all_outputs = {}
    cpt = coef_start
    for block in range(len(list(res_all_tensorinput_block.keys()))):
        block_ref_all_inputs[block] = {}
        block_ref_all_inputs_inverse[block] = {}
        allshapeici = res_all_tensorinput_block[block][0]
        block_ref_all_inputs[block] = torch.zeros(allshapeici)
        print(allshapeici)
        if block == 0:
            for k in range(block_ref_all_inputs[block].shape[0]):
                for i in range(block_ref_all_inputs[block].shape[1]):
                    for j in range(block_ref_all_inputs[block].shape[2]):
                        cpt += 1
                        block_ref_all_inputs[block][k][i][j] = cpt
            # for k in range(coeff, block_ref_all_inputs[block].shape[0]):
            #    k2 = k % coeff
            #    block_ref_all_inputs[block][k] = deepcopy(block_ref_all_inputs[block][k2])

        else:
            for k in range(block_ref_all_inputs[block].shape[0]):
                for i in range(block_ref_all_inputs[block].shape[1]):
                    for j in range(block_ref_all_inputs[block].shape[2]):
                        cpt += 1
                        block_ref_all_inputs[block][k][i][j] = cpt
    for block in range(len(list(res_all_tensoroutput_block.keys())) - 1):
        block_ref_all_outputs[block] = block_ref_all_inputs[block + 1]
    # print(block_ref_all_outputs)
    # print()
    blockf = len(list(res_all_tensoroutput_block.keys()))
    block_ref_all_outputs[blockf - 1] = {}
    allshapeici = res_all_tensoroutput_block[blockf - 1][0]
    for block in range(1, len(list(res_all_tensorinput_block.keys())) - 1):
        block_ref_all_outputs[block - 1] = deepcopy(block_ref_all_inputs_inverse[block])
    # print(block_ref_all_outputs)
    # print(ok)

    block_ref_all_outputs[blockf - 1] = torch.zeros(allshapeici)
    for k in range(block_ref_all_outputs[blockf - 1].shape[0]):
        for i in range(block_ref_all_outputs[blockf - 1].shape[1]):
            for j in range(block_ref_all_outputs[blockf - 1].shape[2]):
                cpt += 1
                block_ref_all_outputs[blockf - 1][k][i][j] = cpt

    return block_ref_all_inputs, block_ref_all_outputs, cpt


def load_TT_TTnoise(args):
    nogolist = []
    if args.with_contradiction:
        path_save_modelvf = args.path_save_model + '/thr_' + str(args.thr_bin_act_test[1:]).replace(" ",
                                                                                                    "") + '/avec_contradiction/'
    else:
        path_save_modelvf = args.path_save_model + '/thr_' + str(args.thr_bin_act_test[1:]).replace(" ",
                                                                                                    "") + '/sans_contradiction/'
    print()
    print("Load TT / TT_noise from ", path_save_modelvf)
    print()

    # load DNF
    all_TT = {}
    for block_occurence in tqdm(range(0, len(args.Blocks_filters_output))):
        all_TT[block_occurence] = {}
        for filteroccurence in range(args.Blocks_filters_output[block_occurence]):
            try:
                with open(path_save_modelvf + 'all_noise_table_propagation_block_' + str(
                        block_occurence) + '_filter_' + str(filteroccurence) + '.pickle', 'rb') as f:
                    all_TT[block_occurence][filteroccurence] = pickle.load(f)
                    # all_TT[block_occurence][filteroccurence]["mask"].reshape(-1)
            except:
                all_TT[block_occurence][filteroccurence] = 1
                nogolist.append((block_occurence, filteroccurence))
    all_TT_noiseonly = {}
    for block_occurence in tqdm(range(0, len(args.Blocks_filters_output))):
        all_TT_noiseonly[block_occurence] = {}
        for filteroccurence in range(args.Blocks_filters_output[block_occurence]):
            try:
                with open(path_save_modelvf + 'noise_only_table_propagation_block_' + str(
                        block_occurence) + '_filter_' + str(filteroccurence) + '.pickle', 'rb') as f:
                    all_TT_noiseonly[block_occurence][filteroccurence] = pickle.load(f)
                    # print(block_occurence, filteroccurence, all_TT_noiseonly[block_occurence][filteroccurence].shape)
            except:
                all_TT_noiseonly[block_occurence][filteroccurence] = 1
                nogolist.append((block_occurence, filteroccurence))

    return all_TT, all_TT_noiseonly, nogolist


import re


def transform_cnf(inputref_vu_par_cnn_et_sat_ref, output_binary_ref, TTici, cnf_general, k=3):
    TTici = re.sub(" ", "", str(TTici))
    TTici = re.sub("~", "-", TTici)
    TTici = re.sub("y", str(output_binary_ref), TTici)
    for xici in range(k):
        TTici = re.sub("x_" + str(xici), str(inputref_vu_par_cnn_et_sat_ref[xici]), TTici)
    exp_CNF3strlist = TTici.split("],[")
    for clause in exp_CNF3strlist:
        clause = str(clause).replace("[", "")
        clause2 = re.sub(']', "", clause)
        clause3 = clause2.split(",")
        cnf_general.append(list(map(int, clause3)))
    return cnf_general


def transform_cnf2(TTici):
    exp_CNF3strlist = TTici.split(']","[')
    cnf_general = []
    for indexc, clause in enumerate(exp_CNF3strlist):
        if indexc == 0:
            clause = str(clause).replace('"[', "")
        if indexc == len(exp_CNF3strlist) - 1:
            clause = re.sub(']"', "", clause)
        clause = clause.split(",")
        cnf_general.append(list(map(int, clause)))
    return cnf_general


import math
from functools import reduce


def find_gcd(list):
    x = reduce(math.gcd, list)
    return x


from pysat.formula import CNF

from pysat.solvers import Lingeling, Glucose3, Glucose4, Minisat22, Cadical, MapleChrono, MapleCM, Maplesat, Solver, \
    Minicard, MinisatGH

from threading import Timer
import time


def interrupt(s):
    s.interrupt()


def solve_cnf(args, cnf_general):
    if args.sat_solver == "Minicard":
        l = Minicard()
    elif args.sat_solver == "Glucose3":
        l = Glucose3()
    elif args.sat_solver == "Glucose4":
        l = Glucose4()
    elif args.sat_solver == "Minisat22":
        l = Minisat22()
    elif args.sat_solver == "Lingeling":
        l = Lingeling()
    elif args.sat_solver == "CaDiCaL":
        l = Cadical()
    elif args.sat_solver == "MapleChrono":
        l = MapleChrono()
    elif args.sat_solver == "MapleCM":
        l = MapleCM()
    elif args.sat_solver == "Maplesat":
        l = Maplesat()
    elif args.sat_solver == "Mergesat3":
        l = Solver("mergesat3")
    elif args.sat_solver == "MinisatGH":
        l = MinisatGH()



    else:
        raise "PB"
    # print(cnf_general)
    l.append_formula(cnf_general)
    timer = Timer(args.time_out, interrupt, [l])
    timer.start()
    start = time.time()
    flag2 = l.solve_limited(expect_interrupt=True)
    end0 = time.time()
    sol = None
    # print(flag2)
    if flag2:
        # print("ATTACK", indexicicici)
        # print(end0-start)
        sol = l.get_model()
    del l

    return flag2, sol, end0 - start


def complete_solving(UNKNOWN1, V_ref, features1_ref, labelref, cnf_general, args, W, nclass=10):
    litsici2 = features1_ref[UNKNOWN1].tolist()
    Wf_l = 1.0 * W[labelref, UNKNOWN1]
    flag_attack = False
    cnf_general2 = deepcopy(cnf_general)
    flag2, sol, timesatsolve = None, None, None
    for aconcurant in range(nclass):
        if not flag_attack:
            if aconcurant != labelref:
                Wf_a = 1.0 * W[aconcurant, UNKNOWN1]
                V = V_ref[aconcurant] - V_ref[labelref]
                Wf_diff = Wf_l - Wf_a
                weightsici = Wf_diff.astype("i").tolist()
                litsici3 = []
                weightsici2 = []
                for index_litteral, litteral in enumerate(litsici2):
                    if weightsici[index_litteral] != 0:
                        litsici3.append(litteral)
                        weightsici2.append(weightsici[index_litteral])
                if len(litsici3) > 0:
                    if len(weightsici2) == 1:
                        gscdw = max(abs(weightsici2[0]), 1)
                    else:
                        gscdw = max(find_gcd(weightsici2), 1)
                    Vfinal = np.floor((V - 1) / gscdw)
                    weightsici3 = [int(x / gscdw) for x in weightsici2]
                    from pysat.pb import PBEnc
                    cnflr = PBEnc.leq(lits=litsici3, weights=weightsici3, bound=int(Vfinal),
                                      encoding=args.encoding_type).clauses
                    save_flag = True
                else:
                    cnflr = [[]]
                    save_flag = False
                if save_flag:
                    cnf_general3 = cnflr + cnf_general2
                    cnf_general3_var = []
                    for y in cnf_general3:
                        for x in y:
                            cnf_general3_var.append(abs(x))
                    flag2, sol, timesatsolve = solve_cnf(args, cnf_general3)
                    if flag2:
                        flag_attack = True
                    del cnflr, PBEnc
    return flag2, sol, timesatsolve


from pysat.pb import PBEnc


def complete_solving_accelerate(UNKNOWN1, V_ref, features1_ref, labelref, solver, cnfG, args, W, nclass=10):
    # start = time.time()
    litsici2 = features1_ref[UNKNOWN1].tolist()
    Wf_l = 1.0 * W[labelref, UNKNOWN1]
    flag_attack = False
    flag2, sol, timesatsolve = False, None, None
    # print("F1 ", time.time()-start)
    cpt = -1
    time_to_remove = 0
    time_to_solve = 0
    for aconcurant in range(nclass):
        if not flag_attack:
            if aconcurant != labelref:
                cpt += 1
                # start = time.time()
                solverici = solver[cpt]
                Wf_a = 1.0 * W[aconcurant, UNKNOWN1]
                V = V_ref[aconcurant] - V_ref[labelref]
                Wf_diff = Wf_l - Wf_a
                weightsici = Wf_diff.astype("f").tolist()
                litsici3 = []
                weightsici2 = []
                for index_litteral, litteral in enumerate(litsici2):
                    if weightsici[index_litteral] != 0:
                        litsici3.append(litteral)
                        weightsici2.append(weightsici[index_litteral])
                if len(litsici3) > 0:
                    # if len(weightsici2) == 1:
                    gscdw = abs(weightsici2[0])
                    # else:
                    #    gscdw = max(find_gcd(weightsici2), 1)
                    Vfinal = int(np.floor(V) - 1) + 1
                    weightsici3 = [int(x / gscdw) for x in weightsici2]
                    list_m1 = np.sum(np.array(weightsici3) == -1)
                    list_p1 = np.sum(np.array(weightsici3) == 1)
                    if int(Vfinal) < 0 and abs(int(Vfinal)) > list_m1:
                        pass
                    elif int(Vfinal) > 0 and abs(int(Vfinal)) > list_p1:
                        flag2 = True
                        flag_attack = True
                        # print(ok)
                    else:
                        # print("F2 ", time.time() - start)
                        # start = time.time()
                        # print(np.unique(weightsici3))
                        cnflr = PBEnc.leq(lits=litsici3, weights=weightsici3, bound=int(Vfinal),
                                          encoding=args.encoding_type, top_id=int(max(litsici2) + 1)).clauses
                        # print("F3A ", time.time() - start)
                        start2 = time.time()
                        solverici.append_formula(cnfG)
                        solverici.append_formula(cnflr)
                        # print("F3B ", time.time() - start2)
                        start3 = time.time()
                        flag2 = solverici.solve()
                        if flag2:
                            # print("ATTACK", indexicicici)
                            # print(end0-start)
                            sol = solverici.get_model()
                        else:
                            sol = None
                        time_to_remove += time.time() - start2
                        # print("F3C ", time.time() - start2)
                        time_to_solve += time.time() - start3
                        if flag2:
                            flag_attack = True
                        # print("F3 ", time.time() - start)

    return flag2, sol, time_to_remove, time_to_solve


def complete_solving_accelerate_v2(dico_clause, labelref, solver, cnfG, nclass=10):
    flag_attack = False
    attack, new_lab = None, None
    flag2, sol, timesatsolve = False, None, None
    cpt = -1
    time_to_remove = 0
    time_to_solve = 0
    for aconcurant in range(nclass):
        if not flag_attack:
            if aconcurant != labelref:
                cpt += 1
                start2 = time.time()
                solverici = solver[cpt]
                cnf_LR = dico_clause[aconcurant]
                solverici.append_formula(cnfG)
                solverici.append_formula(cnf_LR)
                start3 = time.time()
                flag2 = solverici.solve()
                time_to_remove += time.time() - start2
                time_to_solve += time.time() - start3
                if flag2:
                    flag_attack = True
                    new_lab = aconcurant
                    attack = solverici.get_model()

    return flag2, time_to_remove, time_to_solve, attack, new_lab


def complete_solving_accelerate_v3(dico_clause, labelref, solver, cnfG, cnfpre, nclass=10, path_save_modelvf_str=""):
    flag_attack = False
    attack, new_lab = None, None
    flag2, sol, timesatsolve = False, None, None
    cpt = -1
    time_to_remove = 0
    time_to_solve = 0
    start2 = time.time()
    # for aconcurant in range(nclass):
    # if aconcurant != labelref:
    # cnf_LR = dico_clause[aconcurant]
    # if cnf_LR is not None:
    # cnfici = CNF(from_clauses=cnfpre + cnfG + cnf_LR)
    # with open(path_save_modelvf_str + str(aconcurant) + '.cnf', 'w') as fp:
    #    cnfici.to_fp(fp)  # writing to the file pointer
    # del cnfici
    time_to_remove += time.time() - start2
    for aconcurant in range(nclass):
        if not flag_attack:
            if aconcurant != labelref:
                cnf_LR = dico_clause[aconcurant]
                cpt += 1
                if cnf_LR is not None:
                    start2 = time.time()
                    solverici = solver[cpt]
                    # print(cpt, solverici.nof_clauses())
                    # print(solverici.nof_vars())
                    solverici.append_formula(cnfpre)
                    solverici.append_formula(cnfG)
                    solverici.append_formula(cnf_LR)
                    # print(solverici.nof_clauses())
                    # print(solverici.nof_vars())
                    start3 = time.time()
                    flag2 = solverici.solve()
                    time_to_remove += time.time() - start2
                    time_to_solve += time.time() - start3
                    if flag2:
                        flag_attack = True
                        new_lab = aconcurant
                        attack = solverici.get_model()
                    solverici.delete()

    return flag2, time_to_remove, time_to_solve, attack, new_lab


def save_cnf_accelerate(UNKNOWN1, V_ref, features1_ref, labelref, cnf, args, W, path_save_modelvf, batchici, nclass=10):
    # start = time.time()
    litsici2 = features1_ref[UNKNOWN1].tolist()
    Wf_l = 1.0 * W[labelref, UNKNOWN1]
    flag_attack = False
    # cnf_general2 = deepcopy(cnf_general)

    path_save_modelvf_str = path_save_modelvf + "cnf2verify_" + str(args.attack_eps) + "_BATCH_" + str(
        batchici) + "_ACONCURANT_"

    flag2, sol, timesatsolve = False, None, None
    # print(" F1- ", time.time()-start)
    for aconcurant in range(nclass):
        if not flag_attack:
            if aconcurant != labelref:
                cnfici = deepcopy(cnf)
                # start = time.time()
                Wf_a = 1.0 * W[aconcurant, UNKNOWN1]
                V = V_ref[aconcurant] - V_ref[labelref]
                Wf_diff = Wf_l - Wf_a
                weightsici = Wf_diff.astype("i").tolist()
                litsici3 = []
                weightsici2 = []
                for index_litteral, litteral in enumerate(litsici2):
                    if weightsici[index_litteral] != 0:
                        litsici3.append(litteral)
                        weightsici2.append(weightsici[index_litteral])
                # print(" F2- ", time.time() - start)
                if len(litsici3) > 0:
                    # start = time.time()
                    if len(weightsici2) == 1:
                        gscdw = max(abs(weightsici2[0]), 1)
                    else:
                        gscdw = max(find_gcd(weightsici2), 1)
                    Vfinal = np.floor((V - 1) / gscdw)
                    weightsici3 = [int(x / gscdw) for x in weightsici2]
                    from pysat.pb import PBEnc
                    list_m1 = np.sum(np.array(weightsici3) == -1)
                    list_p1 = np.sum(np.array(weightsici3) == 1)
                    if int(Vfinal) < 0 and abs(int(Vfinal)) > list_m1:
                        pass
                    elif int(Vfinal) > 0 and abs(int(Vfinal)) > list_p1:
                        flag2 = True
                        flag_attack = True
                    else:
                        cnflr = PBEnc.leq(lits=litsici3, weights=weightsici3, bound=int(Vfinal),
                                          encoding=args.encoding_type).clauses
                        # print(" F4- ", time.time() - start)
                        # start = time.time()
                        # print("solver initial t-1", aconcurant, solver.nof_clauses())
                        cnfici.extend(cnflr)
                        # with open(path_save_modelvf_str+str(aconcurant)+'.cnf', 'w') as fp:
                        #    cnfici.to_fp(fp)  # writing to the file pointer


class MNIST_1_7(torchvision.datasets.MNIST):

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        data = read_image_file(os.path.join(self.raw_folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        targets = read_label_file(os.path.join(self.raw_folder, label_file))

        # print(targets, len(targets))
        targets_1 = (targets == 1).clone()
        targets_7 = (targets == 7).clone()
        targets[targets_1] = 0
        targets[targets_7] = 1
        # print(torch.sum(targets_1), torch.sum(targets_7), torch.sum(targets_1)+torch.sum(targets_7))
        targets_m = (1.0 * targets_1 + 1.0 * targets_7) > 0
        # print(torch.sum(targets_m))
        targets_m_np = targets_m.detach().cpu().numpy()
        # print(data.shape, targets.shape)
        data = data[targets_m_np, :]
        targets = targets[targets_m_np]
        # print(data.shape, targets.shape, targets)
        # print(ok)

        return data, targets

def load_cnf_dnf_block(args):
    if args.with_contradiction:
        path_save_modelvf = args.path_save_model + '/thr_' + str(args.thr_bin_act_test[1:]).replace(" ",
                                                                                                    "") + '/avec_contradiction/'
    else:
        path_save_modelvf = args.path_save_model + '/thr_' + str(args.thr_bin_act_test[1:]).replace(" ",
                                                                                                    "") + '/sans_contradiction/'
    print()
    print("Load CNF / DNF from ", path_save_modelvf, " only work for k = 9 ou 8")
    print()
    # load DNF
    all_dnf_b0 = []
    all_dnf_b1 = []
    nogolist = []
    for block_occurence in tqdm(range(0, len(args.Blocks_filters_output))):
        #if block_occurence == 0:
        resnumpy_all = np.loadtxt(path_save_modelvf + 'Array_all_block_' +
                   str(block_occurence) + ".txt")
        print(resnumpy_all.shape)
        for filteroccurence in range(args.Blocks_filters_output[block_occurence]):
            #print(filteroccurence)
            try:
                dnf = pd.read_csv(path_save_modelvf+"Truth_Table_block"+ str(block_occurence)+"_filter_"+ str(filteroccurence)+"_coefdefault_1.0_sousblock_None.csv")
                values3 = dnf["Filter_"+str(filteroccurence)+"_Value_1"]
                if block_occurence == 0:
                    all_dnf_b0.append(values3)
                else:
                    all_dnf_b1.append(values3)
            except:
                int_ici = resnumpy_all[0, filteroccurence]
                if block_occurence == 0:
                    all_dnf_b0.append([int_ici]*len(values3))
                else:
                    all_dnf_b1.append([int_ici]*len(values3))
                nogolist.append((block_occurence, filteroccurence))



    return np.array(all_dnf_b0), np.array(all_dnf_b1), nogolist