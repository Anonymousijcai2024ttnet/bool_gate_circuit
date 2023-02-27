import copy
from collections import OrderedDict
import os
#from eevbnn.net_bin import SeqBinModelHelper, BinLinearPos, Binarize01Act, g_weight_binarizer, BinConv2d, setattr_inplace, BatchNormStatsCallbak, g_use_scalar_scale_last_layer
import numpy as np
from torch.autograd import Function
import weakref
import functools

from eevbnn.net_bin import g_bingrad_soft_tanh_scale, AbstractTensor, MultiSampleTensor, g_weight_binarizer, \
    g_weight_decay, g_weight_mask_std, g_use_scalar_scale_last_layer, binarize_weights, BinLinear, \
    PositiveInputCombination, SeqBinModelHelper, BinLinearPos, setattr_inplace, BatchNormStatsCallbak, \
    TernaryWeightWithMaskFn, BinConv2d, BinConv2d_Pos, Quant3WeightFn, Classifier_scale, Classifier_scale_all
from eevbnn.utils import ModelHelper, Flatten
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
from sympy import symbols, SOPform, POSform, simplify_logic
#from src.sat.get_TT import get_exp_with_y

g_weight_binarizer = TernaryWeightWithMaskFn
g_weight_binarizer3 = Quant3WeightFn

class Binarize01Act(nn.Module):

    class Fn(Function):
        @staticmethod
        def forward(ctx, inp, T, scale=None):
            """:param scale: scale for gradient computing"""
            if scale is None:
                ctx.save_for_backward(inp)
            else:
                ctx.save_for_backward(inp, scale)
            all_ones = 1.0*(inp >= T/2)
            maks = 1.0*(inp < T/2) - 1.0*( inp < -T/2)
            #print(torch.sum(maks) / (maks.shape[0]*maks.shape[1]*maks.shape[2]*maks.shape[3]))
            random = torch.randint_like(inp, 2).to(inp.dtype)
            res = all_ones + maks*random

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

    def __init__(self, T=0,grad_scale=1):
        super().__init__()
        self.T = T
        self.register_buffer(
            'grad_scale',
            torch.tensor(float(grad_scale), dtype=torch.float32))


    def forward(self, x):
        grad_scale = getattr(self, 'grad_scale', None)
        f = lambda x: self.Fn.apply(x, self.T, grad_scale)#thr_bin_act=self.thr_bin_act)

        def rsloss(x, y):
            return (1 - torch.tanh(1 + x * y)).sum()

        if type(x) is AbstractTensor:
            loss = rsloss(x.vmin, x.vmax)
            loss += x.loss * AbstractTensor.loss_layer_decay
            vmin = f(x.vmin)
            vmax = f(x.vmax)
            return AbstractTensor(vmin, vmax, loss)
        elif type(x) is MultiSampleTensor:
            rv = x.as_expanded_tensor()
            loss = rsloss(rv[-1], rv[-2])
            return x.apply_batch(
                f,
                loss=x.loss * MultiSampleTensor.loss_layer_decay + loss
            )
        else:
            return f(x)

class Binarize01Act_robustness_benchmark(nn.Module):

    class Fn_benchmark(Function):
        @staticmethod
        def forward(ctx, inp, T, random, scale=None):
            """:param scale: scale for gradient computing"""
            if scale is None:
                ctx.save_for_backward(inp)
            else:
                ctx.save_for_backward(inp, scale)
            all_ones = 1.0*(inp >= T/2)
            maks = 1.0*(inp < T/2) - 1.0*( inp < -T/2)
            if random is None:
                random = torch.randint_like(inp, 2).to(inp.dtype)
            #print(maks, random)
            #print(maks.cpu() * random)
            #print((maks.cpu()*random).to(inp.device))
            #print(all_ones)
            res = all_ones + (maks.cpu()*random).to(inp.device)
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
                return (1 - tanh.mul_(tanh)).mul_(g_out), None, None, None

            # grad as sign(hardtanh(x))
            g_self = (inp.abs() <= 1).to(g_out.dtype)
            return g_self.mul_(g_out), None, None, None

    def __init__(self, T=0, grad_scale=1):
        super().__init__()
        self.T = T
        self.register_buffer(
            'grad_scale',
            torch.tensor(float(grad_scale), dtype=torch.float32))


    def forward(self, x, random= None):
        grad_scale = getattr(self, 'grad_scale', None)
        #if random is None:
            #random = torch.randint_like(x, 2).to(x.dtype)
        f = lambda x: self.Fn_benchmark.apply(x, self.T, random, grad_scale)#thr_bin_act=self.thr_bin_act)

        def rsloss(x, y):
            return (1 - torch.tanh(1 + x * y)).sum()

        if type(x) is AbstractTensor:
            loss = rsloss(x.vmin, x.vmax)
            loss += x.loss * AbstractTensor.loss_layer_decay
            vmin = f(x.vmin)
            vmax = f(x.vmax)
            return AbstractTensor(vmin, vmax, loss)
        elif type(x) is MultiSampleTensor:
            rv = x.as_expanded_tensor()
            loss = rsloss(rv[-1], rv[-2])
            return x.apply_batch(
                f,
                loss=x.loss * MultiSampleTensor.loss_layer_decay + loss
            )
        else:
            return f(x)



class BinConv2dici(nn.Conv2d):
    """conv with binarized weights; no bias is allowed"""

    rounding = False

    class RoundFn(Function):
        """apply rounding to compensate computation errors in float conv"""

        @staticmethod
        def forward(ctx, inp):
            return torch.round_(inp)

        @staticmethod
        def backward(ctx, g_out):
            return g_out

        @classmethod
        def g_apply(cls, inp):
            """general apply with handling of :class:`AbstractTensor`"""
            f = cls.apply
            if type(inp) is AbstractTensor:
                return inp.apply_elemwise_mono(f)
            if type(inp) is MultiSampleTensor:
                return inp.apply_batch(f)
            return f(inp)

    def __init__(self, weight_binarizer, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1, bias=False, rounding=True, maksmeme=False):
        """:param rounding: whether the output should be rounded to integer to
            compensate float computing errors. It should be set when input is
            guaranteed to be int.
        """
        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, groups=groups, bias=bias)
        self.weight_binarizer = weight_binarizer
        binarize_weights(self)  # create weight_mask
        self.rounding = rounding
        self.maksmeme = maksmeme

    def _do_forward(self, x, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight_bin  # * self.weight

        # else:
        # weight = self.weight * abs(self.weight_bin)

        def do_conv(x, w):
            return F.conv2d(x, w, bias, self.stride,
                            self.padding, self.dilation, self.groups)

        if type(x) is AbstractTensor:
            return x.apply_linear(weight, do_conv)

        if type(x) is MultiSampleTensor:
            return x.apply_batch(lambda d: do_conv(d, weight))

        return do_conv(x, weight)

    def forward(self, x):
        y = self._do_forward(x)
        if self.rounding:
            y = self.RoundFn.g_apply(y)
        return y

    @property
    def weight_bin(self):
        if self.maksmeme:
            return binarize_weights(self)
        else:
            return binarize_weights(self)

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.normal_(std=0.01)

    def __repr__(self):
        kh, kw = self.weight.shape[2:]
        return (
            f'{type(self).__name__}({self.in_channels}, {self.out_channels}, '
            f'bin={self.weight_binarizer.__name__}, '
            f'kern=({kh}, {kw}), stride={self.stride}, padding={self.padding})'
        )





"""

class BinConv2d_Pos(nn.Conv2d):

    rounding = False

    class RoundFn(Function):


        @staticmethod
        def forward(ctx, inp):
            return torch.round_(inp)

        @staticmethod
        def backward(ctx, g_out):
            return g_out

        @classmethod
        def g_apply(cls, inp):

            f = cls.apply
            if type(inp) is AbstractTensor:
                return inp.apply_elemwise_mono(f)
            if type(inp) is MultiSampleTensor:
                return inp.apply_batch(f)
            return f(inp)

    def __init__(self, weight_binarizer, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, groups=1, bias=False, rounding=True, maksmeme=False):

        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding, groups=groups, bias=bias)
        self.weight_binarizer = weight_binarizer
        binarize_weights(self)  # create weight_mask
        self.rounding = rounding
        self.maksmeme = maksmeme

    def _do_forward(self, x, *, weight=None, bias=None):
        if weight is None:
            weight = self.weight_bin.abs()  # * self.weight

        # else:
        # weight = self.weight * abs(self.weight_bin)

        def do_conv(x, w):
            return F.conv2d(x, w, bias, self.stride,
                            self.padding, self.dilation, self.groups)

        if type(x) is AbstractTensor:
            return x.apply_linear(weight, do_conv)

        if type(x) is MultiSampleTensor:
            return x.apply_batch(lambda d: do_conv(d, weight))

        return do_conv(x, weight)

    def forward(self, x):
        y = self._do_forward(x)
        if self.rounding:
            y = self.RoundFn.g_apply(y)
        return y

    @property
    def weight_bin(self):
        if self.maksmeme:
            return binarize_weights(self)
        else:
            return binarize_weights(self)

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.normal_(std=0.01)

    def __repr__(self):
        kh, kw = self.weight.shape[2:]
        return (
            f'{type(self).__name__}({self.in_channels}, {self.out_channels}, '
            f'bin={self.weight_binarizer.__name__}, '
            f'kern=({kh}, {kw}), stride={self.stride}, padding={self.padding})'
        )



"""



from sympy.logic.boolalg import to_cnf

class BinLinearPosv2(BinLinear, PositiveInputCombination):
    def _do_forward(self, x):
        weight_bin = (self.weight_bin.abs())
        #print(weight_bin)
        return super()._do_forward(
            x, weight=weight_bin,
            bias=self.bias_from_bin_weight(weight_bin))




class Block_TT(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, k=3, t=8, padding=1, stride=1,
                 groupsici=1, quant_flag="float", blockici=0, T=0):
        super(Block_TT, self).__init__()
        self.k = k
        #self.qm = QuineMcCluskey()
        self.blockici = blockici
        self.final_mask_noise = None
        self.in_planes = in_planes
        self.groupsici = groupsici
        self.pad1 = None
        if k ==6:
            k = (3,2)
        if padding != 0:
            self.pad1 = nn.ConstantPad2d(padding, 0)
        if quant_flag == "bin":
            wb = g_weight_binarizer
            self.conv1 = BinConv2dici(wb, in_planes, t * in_planes, kernel_size=k,
                               stride=stride, padding=0, groups=groupsici, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, t * in_planes, kernel_size=k,
                                  stride=stride, padding=0, groups=groupsici, bias=False)
        self.bn1 = nn.BatchNorm2d(t * in_planes)
        self.conv2 = nn.Conv2d(t * in_planes, out_planes, kernel_size=1, stride=1, padding=0,
                               groups=groupsici, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.stride = stride
        self.act = Binarize01Act(T=T)

    def forward(self, x, compute_final_mask_noise = True):
        if self.final_mask_noise is not None and compute_final_mask_noise:
           x = self.final_mask_noise*x
        self.input_layer = x.clone()
        if self.pad1 is not None:
            x = self.pad1(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        #out = F.gelu(self.conv1(x))
        out  = self.bn2(self.conv2(out))#.clone().to(x.device)
        #out = self.conv2(out)
        self.output_layer_pre_activ = out.clone()
        out = self.act(out)
        self.output_layer = out.clone()
        return out

    def get_TT_block_all_filter(self, device, blockici, sousblockici):
        self.blockici = blockici
        self.sousblockici = sousblockici
        with torch.no_grad():
            nbrefilter = self.in_planes
            chanel_interest = int(nbrefilter/self.groupsici)
            #print(self.k, chanel_interest)
            if self.k == (3,2) or self.k == (2,3):
                self.n = 6
            else:
                self.n = self.k ** 2 * chanel_interest
            c_a_ajouter = int((nbrefilter - chanel_interest)/chanel_interest)
            l = [[int(y) for y in format(x, 'b').zfill(self.n)] for x in range(2 ** self.n)]
            df = pd.DataFrame(l)
            self.df = df.reset_index()
            if self.k ==(3,2):
                x_input_f2 = torch.Tensor(l).reshape(2 ** self.n, chanel_interest, 3,2)
            elif self.k ==(2,3):
                x_input_f2 = torch.Tensor(l).reshape(2 ** self.n, chanel_interest, 2,3)
            else:
                x_input_f2 = torch.Tensor(l).reshape(2 ** self.n, chanel_interest, self.k,
                                                     self.k)
            y = x_input_f2.detach().clone()
            padding = torch.autograd.Variable(y)
            for itera in range(c_a_ajouter):
                x_input_f2 = torch.cat((x_input_f2, padding), 1)  # .type(torch.ByteTensor)
            del padding
            #print(x_input_f2.shape)
            #ok
            #if self.pad1 is not None:
            #    x_input_f2 = self.pad1(x_input_f2)
            out = F.gelu(self.bn1(self.conv1(x_input_f2.to(device))))
            # out = F.gelu(self.conv1(x))
            out = self.bn2(self.conv2(out))  # .clone().to(x.device)
            # out = self.conv2(out)
            self.output_layer_pre_activ = out.clone()
            out = self.act(out)
            self.res_numpy = out.squeeze(-1).squeeze(-1).detach().cpu().clone().numpy()
            #print(self.res_numpy.shape)
            self.res_numpy_pre_activ = self.output_layer_pre_activ.squeeze(-1).squeeze(-1).detach().cpu().clone().numpy()
            #print(self.res_numpy_pre_activ )
            self.dontcares = 1.0*(self.res_numpy_pre_activ > -(self.act.T )/2) - 1.0*(self.res_numpy_pre_activ > (self.act.T)/2)
            #print(self.dontcares, self.act.T)
        return self.res_numpy

    def get_TT_block_1filter(self, filterici, path_save_exp):
        self.filterici = filterici
        self.path_save_exp = path_save_exp
        resici = self.res_numpy[:, filterici]
        dontcaresici  = self.dontcares[:, filterici]
        unique = np.unique(resici)
        #print(unique)
        if len(unique) == 1:
            # s'il n'y a qune seule valeur, enregistre la valeur
            self.save_cnf_dnf(resici[0], str(resici[0]))
            table = np.chararray((2 ** self.n, 2 ** self.n), itemsize=3)
            table[:][:] = str(resici[0])
            np.save(self.path_save_exp + 'table_outputblock_' +
                    str(self.blockici) + '_filter_' + str(self.filterici) +
                    '_value_' + str(resici[0]) + '_coefdefault_' +
                    str(resici[0]) + '.npy', table)
            exp_CNF, exp_DNF, exp_CNF3 = None, None, None
        else:
            # sinon on cherche la formule SAT
            exp_CNF, exp_DNF, exp_CNF3 = self.iterate_over_filter(resici, unique, dontcaresici)
        return exp_CNF, exp_DNF, exp_CNF3

    def save_cnf_dnf(self, coef, exp_CNF3, exp_DNF=None, exp_CNF=None):
        #exp_CNF3 = str(coef)
        with open(self.path_save_exp + 'table_outputblock_' +
                  str(self.blockici) + '_filter_' + str(self.filterici) +
                  '_coefdefault_' +
                  str(coef) + ".txt", 'w') as f:
            f.write(str(exp_CNF3))
        if exp_CNF is not None:
            with open(self.path_save_exp + 'CNF_expression_block' +
                  str(self.blockici) + '_filter_' + str(self.filterici) +
                  '_coefdefault_' +
                  str(coef) + "_sousblock_" + str(self.sousblockici) + ".txt", 'w') as f:
                f.write(str(exp_CNF))
            with open(self.path_save_exp + 'DNF_expression_block' +
                  str(self.blockici) + '_filter_' + str(self.filterici) +
                  '_coefdefault_' +
                  str(coef) + "_sousblock_" + str(self.sousblockici) + ".txt", 'w') as f:
                f.write(str(exp_DNF))


    def iterate_over_filter(self, resici, unique, dontcaresici):
        coef_default = unique[0]
        unique2 = unique[1:]
        #print(coef_default, unique)
        for unq2 in unique2:
            #self.for_1_filter(unq2, resici)
            exp_CNF, exp_DNF, exp_CNF3  = self.for_1_filter(unq2, resici, dontcaresici)
            self.save_cnf_dnf(unq2, exp_CNF3, exp_DNF, exp_CNF)
        return exp_CNF, exp_DNF, exp_CNF3

    def for_1_filter(self, unq2, resici, dontcaresici):
        answer = resici == unq2
        #print(answer.shape)
        dfres = pd.DataFrame(answer)
        dc =  dontcaresici == 1
        dfres.columns = ["Filter_" + str(self.filterici) + "_Value_" + str(int(unq2))]
        dfdontcare = pd.DataFrame(dc)
        dfdontcare.columns = ["Filter_" + str(self.filterici) + "_dontcares_" + str(int(unq2))]

        df2 = pd.concat([self.df, dfres, dfdontcare], axis=1)
        # print(df2)
        df2.to_csv(self.path_save_exp + 'Truth_Table_block' +
                   str(self.blockici) + '_filter_' + str(self.filterici) +
                   '_coefdefault_' +
                   str(unq2) + "_sousblock_" + str(self.sousblockici) + '.csv')
        condtion_filter = df2["index"].values[answer].tolist()
        dc_filter = df2["index"].values[dc].tolist()

        # condtion_filter_cnf = df2["index"].values[answer_cnf].tolist()
        exp_DNF, exp_CNF = self.get_expresion_methode1(condtion_filter, dc_filter = dc_filter)
        exp_CNF3 = get_exp_with_y(exp_DNF, exp_CNF)
        return exp_CNF, exp_DNF, exp_CNF3


    def get_expresion_methode1(self, condtion_filter, dc_filter=None):

        if dc_filter is not None:
            dc_filtervf = dc_filter + self.dontcares_train
            condtion_filter_vf = [x for x in condtion_filter if x not in dc_filtervf]
            print(len(condtion_filter_vf), len(dc_filtervf), len(dc_filtervf) / 2 ** self.n)
        else:
            condtion_filter_vf = condtion_filter

        if self.n == 4:
            w1, x1, y1, v1 = symbols('x_0, x_1, x_2, x_3')
            exp_DNF = SOPform([w1, x1, y1, v1], minterms=condtion_filter_vf, dontcares=dc_filtervf)
            if self.with_contradiction:
                exp_CNF = POSform([w1, x1, y1, v1], minterms=condtion_filter_vf, dontcares=dc_filtervf)
            else:
                exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
        elif self.n == 8:
            w1, x1, y1, v1, w2, x2, y2, v2 = symbols('x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7')
            exp_DNF = SOPform([w1, x1, y1, v1, w2, x2, y2, v2], minterms=condtion_filter_vf, dontcares=dc_filtervf)
            if self.with_contradiction:
                exp_CNF = POSform([w1, x1, y1, v1, w2, x2, y2, v2], minterms=condtion_filter_vf, dontcares=dc_filtervf)
            else:
                exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
        elif self.n == 6:
            w1, x1, y1, v1, w2, x2 = symbols('x_0, x_1, x_2, x_3, x_4, x_5')
            exp_DNF = SOPform([w1, x1, y1, v1, w2, x2], minterms=condtion_filter_vf, dontcares=dc_filtervf)
            if self.with_contradiction:
                exp_CNF = POSform([w1, x1, y1, v1, w2, x2], minterms=condtion_filter_vf, dontcares=dc_filtervf)
            else:
                exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
        elif self.n == 9:
            w1, x1, y1, v1, w2, x2, y2, v2, w3 = symbols('x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8')
            exp_DNF = SOPform([w1, x1, y1, v1, w2, x2, y2, v2, w3], minterms=condtion_filter_vf, dontcares =dc_filtervf)
            if self.with_contradiction:
                exp_CNF = POSform([w1, x1, y1, v1, w2, x2, y2, v2, w3], minterms=condtion_filter_vf, dontcares =dc_filtervf)
            else:
                exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
        else:
            w1, x1, y1, v1, w2, x2, y2, v2, w10, x10, y10, v10, w20, x20, y20, v20 = symbols('x_0, x_1, x_2, x_3, x_4, x_5, x_6, x_7,x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15')
            exp_DNF = SOPform([w1, x1, y1, v1, w2, x2, y2, v2, w10, x10, y10, v10, w20, x20, y20, v20], minterms=condtion_filter_vf, dontcares=dc_filtervf)
            if self.with_contradiction:
                exp_CNF = POSform([w1, x1, y1, v1, w2, x2, y2, v2, w10, x10, y10, v10, w20, x20, y20, v20], minterms=condtion_filter_vf, dontcares=dc_filtervf)
            else:
                exp_CNF = to_cnf(exp_DNF, simplify=True, force=True)
        return exp_DNF, exp_CNF

class Block_TT2(Block_TT):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, k=3, t=8, padding=1, stride=1,
                 groupsici=1, quant_flag="float", blockici=0, T=0):
        super(Block_TT, self).__init__()
        self.k = k
        #self.qm = QuineMcCluskey()
        self.blockici = blockici
        self.final_mask_noise = None
        self.in_planes = in_planes
        self.groupsici = groupsici
        self.pad1 = None
        if padding != 0:
            self.pad1 = nn.ConstantPad2d(padding, 0)
        if quant_flag == "bin":
            wb = g_weight_binarizer
            self.conv1 = BinConv2dici(wb, in_planes, t * in_planes, kernel_size=k,
                               stride=stride, padding=0, groups=groupsici, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_planes, t * in_planes, kernel_size=k,
                                  stride=stride, padding=0, groups=groupsici, bias=False)
        self.bn1 = nn.BatchNorm2d(t * in_planes)
        self.conv2 = nn.Conv2d(t * in_planes, out_planes, kernel_size=1, stride=1, padding=0,
                               groups=groupsici, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.stride = stride
        self.act = Binarize01Act(T=T)
        self.Block_conv4 = nn.AvgPool2d(2)

    def forward(self, x, compute_final_mask_noise = True):
        if self.final_mask_noise is not None and compute_final_mask_noise:
           x = self.final_mask_noise*x
        self.input_layer = x.clone()
        if self.pad1 is not None:
            x = self.pad1(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out  = self.bn2(self.conv2(out))#.clone().to(x.device)
        self.output_layer_pre_activ = out.clone()
        out = self.act(out)
        self.output_layer = out.clone()

        if self.stride == 2:
            #pass
            out2 = self.act(self.Block_conv4(x)-0.5)
        else:
            out2 = x
        padding = out2.clone()
        if out2.shape[1]!= out.shape[1]:
            for _ in range(int((out.shape[1]-out2.shape[1])/out2.shape[1])):
                out2 = torch.cat((out2, padding), 1)

        #if out.shape[-1]!=out2.shape[-1]:
        #print(out.shape, out2.shape, x.shape)

        if out2.shape[-1] != out.shape[-1]:
            inttouse = int(abs(out2.shape[-1] - out.shape[-1]))
            out2 = out2[:, :, :-inttouse+1, :-inttouse]


        #if (x.shape[-1] == 32) or (x.shape[-1] == 14):
        #    out2 = out2[:, :, :-1, :-1]
        #print(out.shape, out2.shape, x.shape)

        #padding = torch.autograd.Variable(y)


        outf = torch.cat((out, out2), dim=1)
        n, c, w, h = outf.shape
        outf = outf.view(n, 2, int(c / 2), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)

        return outf


class Block_resnet_multihead_general_BN_vf_small_v3(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1, T=0.0, last=False):
        super(Block_resnet_multihead_general_BN_vf_small_v3, self).__init__()
        self.cpt = 0
        self.pad0 = nn.ZeroPad2d((1,0,1,0))
        self.groups = [1,1,8,1]
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(self.groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                    #pass
                    self.Block_conv1 = Block_TT(in_planes, in_planes,  k = (3,2), stride=stride, padding=1,
                                   groupsici=int(in_planes / g), T=T)    # int(in_planes/1))
                    #self.Block_conv1 = nn.AvgPool2d(2)
                    self.cpt += 1

                elif index_g == 1:
                    self.Block_conv2 = Block_TT(in_planes, in_planes,  k=(2,3), stride=stride,
                                                       padding=1,
                                                       groupsici=int(in_planes / g), T=T)    # int(in_planes/2))
                    #print(int(in_planes / g), g, in_planes)
                    self.cpt += 1
                    g2 = g + 2
                elif index_g == 2:
                    self.Block_conv3 = Block_TT(in_planes, in_planes,  k=1, stride=1,
                                                       padding=0,
                                                       groupsici=int(in_planes / g), T=T)    # int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3 and stride == 1:
                    pass
                    #self.Block_conv4 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=1, stride=stride,
                    #                                   padding=0,
                    #                                   groupsici=int(in_planes / g))  # int(in_planes/2))

                    #self.Block_conv4 =nn.AvgPool2d(2)
                    self.cpt += 1
                elif index_g == 3 and stride == 2:
                    self.Block_conv4 = nn.AvgPool2d(2)
                    self.cpt += 1

        self.stride = stride
        groupvf = 8
        print(self.cpt * in_planes, int(self.cpt * in_planes / groupvf), self.cpt, groupvf, out_planes)
        if last:
            self.Block_convf = Block_TT(self.cpt * in_planes, 2*in_planes, k=1, stride=1,
                                           padding=0,
                                           groupsici=int(self.cpt * in_planes / groupvf), T=T)  # int(4*in_planes/4))
        else:
            self.Block_convf = Block_TT(self.cpt * in_planes, 2*out_planes, k=1, stride=1,
                                           padding=0,
                                           groupsici=int(self.cpt * in_planes / groupvf), T=T)  # int(4*in_planes/4))


    def forward(self, x):
        """out1, out2, out3, out4 = None, None, None, None
        if self.Block_conv4 is not None:
            out4 = self.Block_conv4(x)
        if self.Block_conv3 is not None:
            out3 = self.Block_conv3(x)
        if self.Block_conv2 is not None:
            out2 = self.Block_conv2(x)
        if self.Block_conv1 is not None:
            out1 = self.Block_conv1(x)"""
        out3 = self.Block_conv3(x)
        out2 = self.Block_conv2(x)
        out1 = self.Block_conv1(x)
        if self.stride == 2:
            #pass
            out4 = self.Block_conv1.act(self.Block_conv4(x)-0.5)
            out3 = self.Block_conv1.act(self.Block_conv4(out3) - 0.5)
        else:
            out4 = x
        pad = nn.ZeroPad2d((0,1,0,1))
        pad2 = nn.ZeroPad2d((0, 0, 0, 2))
        pad3 = nn.ZeroPad2d((0, 2, 0, 0))
        pad21 = nn.ZeroPad2d((0, 0, 0, 1))
        pad31 = nn.ZeroPad2d((0, 1, 0, 0))
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        if (x.shape[-1] == 12) and out1.shape[-1]==13: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
            #pass
            out1 = pad21(out1)
            out2 = pad31(out2)
            out3 = pad(out3)
            out4 = pad(out4)
        elif (x.shape[-1] == 14):# and out1.shape[-1] == 13:  # or (x.shape[-1] == 9) or (x.shape[-1] == 10):
                # pass
                out1 = pad21(out1)
                out2 = pad31(out2)
                out3 = pad(out3)
                out4 = pad(out4)
            #out1 = pad(out1)
        elif (x.shape[-1] == 16) and out2.shape[-1]==16:  # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
                # pass
            out1 = pad21(out1)
            out2 = pad31(out2)
            out3 = pad(out3)
            out4 = pad(out4)
        elif (x.shape[-1] == 16):  # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
                # pass
            out1 = pad(out1)
            #out2 = pad31(out2)
            out3 = pad(out3)
            out4 = pad(out4)
            #out1 = pad(out1)
            #x = pad(x)
            #out1 = out1[:,:,:-1,:-1]
            #out1 = out1[:, :, :-1, :-1]
            #out3 = out3[:, :, :-1, :-1]
            #out2 = out2[:, :, :-1, :-1]

        #elif (x.shape[-1] == 8) and out1.shape[-1]==8: # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
            #out1 = pad2(out1)
            #out2 = pad3(out2)

        elif (x.shape[-1] == 4) and out1.shape[-1]==4: # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
            out1 = pad2(out1)
            out2 = pad3(out2)

        elif (x.shape[-1] == 17):
            #out1 = pad(out1)
            out3 = pad(out3)
            out4 = pad(out4)
            #out1 = pad(out1)
            #out3 = pad(out3)
        elif (x.shape[-1] == 9):
            out4 = pad(out4)
            out3 = pad(out3)
        elif (x.shape[-1] == 7):
            out4 = pad(out4)
            out3 = pad(out3)
            #out1 = pad2(out1)
            #out2 = pad2(out2)
        elif (x.shape[-1] == 17) or (x.shape[-1] == 9)or(x.shape[-1] == 20)or(self.stride == 2 and x.shape[-1] == 10)or (self.stride == 2 and x.shape[-1] == 6):# or (self.stride == 2 and x.shape[-1] == 5):
            pass
            #out1 = out1[:, :, :-1, :-1]
            #out2 = out2[:, :, :-1, :-1]
            #out3 = out3[:, :, :-1, :-1]
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        outf = torch.cat((out1, out2, out3, out4), axis=1)

        n, c, w, h = outf.shape
        outf = outf.view(n, self.cpt, int(c / self.cpt), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        outff = self.Block_convf(outf)
        return outff #torch.cat((outff, out4 ), axis = 1)


class Block_resnet_multihead_general_BN_vf_small_v3_xmini(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1, T=0.0, last=False):
        super(Block_resnet_multihead_general_BN_vf_small_v3_xmini, self).__init__()
        self.cpt = 0
        self.pad0 = nn.ZeroPad2d((1,0,1,0))
        self.groups = [1,1,4,1]
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(self.groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                    #pass
                    self.Block_conv1 = Block_TT(in_planes, in_planes,  k = (2,2), stride=stride, padding=1,
                                   groupsici=int(in_planes / g), T=T)    # int(in_planes/1))
                    #self.Block_conv1 = nn.AvgPool2d(2)
                    self.cpt += 1

                elif index_g == 1:
                    self.Block_conv2 = Block_TT(in_planes, in_planes,  k=(2,2), stride=stride,
                                                       padding=1,
                                                       groupsici=int(in_planes / g), T=T)    # int(in_planes/2))
                    #print(int(in_planes / g), g, in_planes)
                    self.cpt += 1
                    g2 = g + 2
                elif index_g == 2:
                    self.Block_conv3 = Block_TT(in_planes, in_planes,  k=1, stride=1,
                                                       padding=0,
                                                       groupsici=int(in_planes / g), T=T)    # int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3 and stride == 1:
                    pass
                    #self.Block_conv4 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=1, stride=stride,
                    #                                   padding=0,
                    #                                   groupsici=int(in_planes / g))  # int(in_planes/2))

                    #self.Block_conv4 =nn.AvgPool2d(2)
                    self.cpt += 1
                elif index_g == 3 and stride == 2:
                    self.Block_conv4 = nn.AvgPool2d(2)
                    self.cpt += 1

        self.stride = stride
        groupvf = 4
        print(self.cpt * in_planes, int(self.cpt * in_planes / groupvf), self.cpt, groupvf, out_planes)
        if last:
            self.Block_convf = Block_TT(self.cpt * in_planes, 2*in_planes, k=1, stride=1,
                                           padding=0,
                                           groupsici=int(self.cpt * in_planes / groupvf), T=T)  # int(4*in_planes/4))
        else:
            self.Block_convf = Block_TT(self.cpt * in_planes, 2*out_planes, k=1, stride=1,
                                           padding=0,
                                           groupsici=int(self.cpt * in_planes / groupvf), T=T)  # int(4*in_planes/4))


    def forward(self, x):
        """out1, out2, out3, out4 = None, None, None, None
        if self.Block_conv4 is not None:
            out4 = self.Block_conv4(x)
        if self.Block_conv3 is not None:
            out3 = self.Block_conv3(x)
        if self.Block_conv2 is not None:
            out2 = self.Block_conv2(x)
        if self.Block_conv1 is not None:
            out1 = self.Block_conv1(x)"""
        out3 = self.Block_conv3(x)
        out2 = self.Block_conv2(x)
        out1 = self.Block_conv1(x)
        if self.stride == 2:
            #pass
            out4 = self.Block_conv1.act(self.Block_conv4(x)-0.5)
            out3 = self.Block_conv1.act(self.Block_conv4(out3) - 0.5)
        else:
            out4 = x
        pad = nn.ZeroPad2d((0,1,0,1))
        pad2 = nn.ZeroPad2d((0, 0, 0, 2))
        pad3 = nn.ZeroPad2d((0, 2, 0, 0))
        pad21 = nn.ZeroPad2d((0, 0, 0, 1))
        pad31 = nn.ZeroPad2d((0, 1, 0, 0))
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        if (x.shape[-1] == 12) and out1.shape[-1]==13: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
            #pass
            out1 = pad21(out1)
            out2 = pad31(out2)
            out3 = pad(out3)
            out4 = pad(out4)
        elif (x.shape[-1] == 14):# and out1.shape[-1] == 13:  # or (x.shape[-1] == 9) or (x.shape[-1] == 10):
                # pass
                out1 = pad21(out1)
                out2 = pad31(out2)
                out3 = pad(out3)
                out4 = pad(out4)
        elif (x.shape[-1] == 28):# and out1.shape[-1] == 13:  # or (x.shape[-1] == 9) or (x.shape[-1] == 10):
                # pass
                #out1 = pad21(out1)
                #out2 = pad31(out2)
                out3 = pad(out3)
                out4 = pad(out4)
            #out1 = pad(out1)
        elif (x.shape[-1] == 16) and out2.shape[-1]==16:  # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
                # pass
            out1 = pad21(out1)
            out2 = pad31(out2)
            out3 = pad(out3)
            out4 = pad(out4)
        elif (x.shape[-1] == 16):  # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
                # pass
            out1 = pad(out1)
            #out2 = pad31(out2)
            out3 = pad(out3)
            out4 = pad(out4)
            #out1 = pad(out1)
            #x = pad(x)
            #out1 = out1[:,:,:-1,:-1]
            #out1 = out1[:, :, :-1, :-1]
            #out3 = out3[:, :, :-1, :-1]
            #out2 = out2[:, :, :-1, :-1]

        #elif (x.shape[-1] == 8) and out1.shape[-1]==8: # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
            #out1 = pad2(out1)
            #out2 = pad3(out2)

        elif (x.shape[-1] == 4) and out1.shape[-1]==4: # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
            out1 = pad2(out1)
            out2 = pad3(out2)

        elif (x.shape[-1] == 17):
            #out1 = pad(out1)
            out3 = pad(out3)
            out4 = pad(out4)
            #out1 = pad(out1)
            #out3 = pad(out3)
        elif (x.shape[-1] == 9):
            out4 = pad(out4)
            out3 = pad(out3)
        elif (x.shape[-1] == 7):
            out4 = pad(out4)
            out3 = pad(out3)
            #out1 = pad2(out1)
            #out2 = pad2(out2)
        elif (x.shape[-1] == 17) or (x.shape[-1] == 9)or(x.shape[-1] == 20)or(self.stride == 2 and x.shape[-1] == 10)or (self.stride == 2 and x.shape[-1] == 6):# or (self.stride == 2 and x.shape[-1] == 5):
            pass
            #out1 = out1[:, :, :-1, :-1]
            #out2 = out2[:, :, :-1, :-1]
            #out3 = out3[:, :, :-1, :-1]
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        outf = torch.cat((out1, out3, out4), axis=1)

        #n, c, w, h = outf.shape
        #outf = outf.view(n, self.cpt, int(c / self.cpt), w, h)
        #outf = outf.transpose_(1, 2).contiguous()
        #outf = outf.view(n, c, w, h)
        #outff = self.Block_convf(outf)
        return outf #torch.cat((outff, out4 ), axis = 1)



class Block_resnet_multihead_general_BN_vf_small_v3_mini(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1, T=0.0, last=False):
        super(Block_resnet_multihead_general_BN_vf_small_v3_mini, self).__init__()
        self.cpt = 0
        self.pad0 = nn.ZeroPad2d((1,0,1,0))
        self.groups = [1,1,6,1]
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(self.groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                    #pass
                    self.Block_conv1 = Block_TT(in_planes, in_planes,  k = (3,2), stride=stride, padding=1,
                                   groupsici=int(in_planes / g), T=T)    # int(in_planes/1))
                    #self.Block_conv1 = nn.AvgPool2d(2)
                    self.cpt += 1

                elif index_g == 1:
                    self.Block_conv2 = Block_TT(in_planes, in_planes,  k=(2,3), stride=stride,
                                                       padding=1,
                                                       groupsici=int(in_planes / g), T=T)    # int(in_planes/2))
                    #print(int(in_planes / g), g, in_planes)
                    self.cpt += 1
                    g2 = g + 2
                elif index_g == 2:
                    self.Block_conv3 = Block_TT(in_planes, in_planes,  k=1, stride=1,
                                                       padding=0,
                                                       groupsici=int(in_planes / g), T=T)    # int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3 and stride == 1:
                    pass
                    #self.Block_conv4 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=1, stride=stride,
                    #                                   padding=0,
                    #                                   groupsici=int(in_planes / g))  # int(in_planes/2))

                    #self.Block_conv4 =nn.AvgPool2d(2)
                    self.cpt += 1
                elif index_g == 3 and stride == 2:
                    self.Block_conv4 = nn.AvgPool2d(2)
                    self.cpt += 1

        self.stride = stride
        groupvf = 6
        print(self.cpt * in_planes, int(self.cpt * in_planes / groupvf), self.cpt, groupvf, out_planes)
        if last:
            self.Block_convf = Block_TT(self.cpt * in_planes, 2*in_planes, k=1, stride=1,
                                           padding=0,
                                           groupsici=int(self.cpt * in_planes / groupvf), T=T)  # int(4*in_planes/4))
        else:
            self.Block_convf = Block_TT(self.cpt * in_planes, 2*out_planes, k=1, stride=1,
                                           padding=0,
                                           groupsici=int(self.cpt * in_planes / groupvf), T=T)  # int(4*in_planes/4))


    def forward(self, x):
        """out1, out2, out3, out4 = None, None, None, None
        if self.Block_conv4 is not None:
            out4 = self.Block_conv4(x)
        if self.Block_conv3 is not None:
            out3 = self.Block_conv3(x)
        if self.Block_conv2 is not None:
            out2 = self.Block_conv2(x)
        if self.Block_conv1 is not None:
            out1 = self.Block_conv1(x)"""
        out3 = self.Block_conv3(x)
        out2 = self.Block_conv2(x)
        out1 = self.Block_conv1(x)
        if self.stride == 2:
            #pass
            out4 = self.Block_conv1.act(self.Block_conv4(x)-0.5)
            out3 = self.Block_conv1.act(self.Block_conv4(out3) - 0.5)
        else:
            out4 = x
        pad = nn.ZeroPad2d((0,1,0,1))
        pad2 = nn.ZeroPad2d((0, 0, 0, 2))
        pad3 = nn.ZeroPad2d((0, 2, 0, 0))
        pad21 = nn.ZeroPad2d((0, 0, 0, 1))
        pad31 = nn.ZeroPad2d((0, 1, 0, 0))
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        if (x.shape[-1] == 12) and out1.shape[-1]==13: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
            #pass
            out1 = pad21(out1)
            out2 = pad31(out2)
            out3 = pad(out3)
            out4 = pad(out4)
        elif (x.shape[-1] == 14):# and out1.shape[-1] == 13:  # or (x.shape[-1] == 9) or (x.shape[-1] == 10):
                # pass
                out1 = pad21(out1)
                out2 = pad31(out2)
                out3 = pad(out3)
                out4 = pad(out4)
            #out1 = pad(out1)
        elif (x.shape[-1] == 16) and out2.shape[-1]==16:  # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
                # pass
            out1 = pad21(out1)
            out2 = pad31(out2)
            out3 = pad(out3)
            out4 = pad(out4)
        elif (x.shape[-1] == 16):  # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
                # pass
            out1 = pad(out1)
            #out2 = pad31(out2)
            out3 = pad(out3)
            out4 = pad(out4)
            #out1 = pad(out1)
            #x = pad(x)
            #out1 = out1[:,:,:-1,:-1]
            #out1 = out1[:, :, :-1, :-1]
            #out3 = out3[:, :, :-1, :-1]
            #out2 = out2[:, :, :-1, :-1]

        #elif (x.shape[-1] == 8) and out1.shape[-1]==8: # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
            #out1 = pad2(out1)
            #out2 = pad3(out2)

        elif (x.shape[-1] == 4) and out1.shape[-1]==4: # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
            out1 = pad2(out1)
            out2 = pad3(out2)

        elif (x.shape[-1] == 17):
            #out1 = pad(out1)
            out3 = pad(out3)
            out4 = pad(out4)
            #out1 = pad(out1)
            #out3 = pad(out3)
        elif (x.shape[-1] == 9):
            out4 = pad(out4)
            out3 = pad(out3)
        elif (x.shape[-1] == 7):
            out4 = pad(out4)
            out3 = pad(out3)
            #out1 = pad2(out1)
            #out2 = pad2(out2)
        elif (x.shape[-1] == 17) or (x.shape[-1] == 9)or(x.shape[-1] == 20)or(self.stride == 2 and x.shape[-1] == 10)or (self.stride == 2 and x.shape[-1] == 6):# or (self.stride == 2 and x.shape[-1] == 5):
            pass
            #out1 = out1[:, :, :-1, :-1]
            #out2 = out2[:, :, :-1, :-1]
            #out3 = out3[:, :, :-1, :-1]
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        outf = torch.cat((out1, out2, out3, out4), axis=1)

        n, c, w, h = outf.shape
        outf = outf.view(n, self.cpt, int(c / self.cpt), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        outff = self.Block_convf(outf)
        return outff #torch.cat((outff, out4 ), axis = 1)



class Block_resnet_multihead_general_BN_vf_small_v3_big(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1, T=0.0, last=False):
        super(Block_resnet_multihead_general_BN_vf_small_v3_big, self).__init__()
        self.cpt = 0
        self.pad0 = nn.ZeroPad2d((1,0,1,0))
        self.groups = [1,2,8,1]
        self.Block_conv1, self.Block_conv2, self.Block_conv3, self.Block_conv4 = None, None, None, None
        for index_g, g in enumerate(self.groups):
            #print(groups)
            if g is not None:
                if index_g == 0:
                    #pass
                    self.Block_conv1 = Block_TT(in_planes, in_planes,  k = (3), stride=stride, padding=1,
                                   groupsici=int(in_planes / g), T=T)    # int(in_planes/1))
                    #self.Block_conv1 = nn.AvgPool2d(2)
                    self.cpt += 1

                elif index_g == 1:
                    self.Block_conv2 = Block_TT(in_planes, in_planes,  k=(2), stride=stride,
                                                       padding=1,
                                                       groupsici=int(in_planes / g), T=T)    # int(in_planes/2))
                    #print(int(in_planes / g), g, in_planes)
                    self.cpt += 1
                    g2 = g + 2
                elif index_g == 2:
                    self.Block_conv3 = Block_TT(in_planes, in_planes,  k=1, stride=1,
                                                       padding=0,
                                                       groupsici=int(in_planes / g), T=T)    # int(in_planes/2))
                    self.cpt += 1
                    g2 = g
                elif index_g == 3 and stride == 1:
                    pass
                    #self.Block_conv4 = Block_resnet_BN(in_planes, in_planes, Abit_inter=Abit_inter, k=1, stride=stride,
                    #                                   padding=0,
                    #                                   groupsici=int(in_planes / g))  # int(in_planes/2))

                    #self.Block_conv4 =nn.AvgPool2d(2)
                    self.cpt += 1
                elif index_g == 3 and stride == 2:
                    self.Block_conv4 = nn.AvgPool2d(2)
                    self.cpt += 1

        self.stride = stride
        groupvf = 8
        print(self.cpt * in_planes, int(self.cpt * in_planes / groupvf), self.cpt, groupvf, out_planes)
        if last:
            self.Block_convf = Block_TT(self.cpt * in_planes, 2*in_planes, k=1, stride=1,
                                           padding=0,
                                           groupsici=int(self.cpt * in_planes / groupvf), T=T)  # int(4*in_planes/4))
        else:
            self.Block_convf = Block_TT(self.cpt * in_planes, 2*out_planes, k=1, stride=1,
                                           padding=0,
                                           groupsici=int(self.cpt * in_planes / groupvf), T=T)  # int(4*in_planes/4))


    def forward(self, x):
        """out1, out2, out3, out4 = None, None, None, None
        if self.Block_conv4 is not None:
            out4 = self.Block_conv4(x)
        if self.Block_conv3 is not None:
            out3 = self.Block_conv3(x)
        if self.Block_conv2 is not None:
            out2 = self.Block_conv2(x)
        if self.Block_conv1 is not None:
            out1 = self.Block_conv1(x)"""
        out3 = self.Block_conv3(x)
        out2 = self.Block_conv2(x)
        out1 = self.Block_conv1(x)
        if self.stride == 2:
            #pass
            out4 = self.Block_conv1.act(self.Block_conv4(x)-0.5)
            out3 = self.Block_conv1.act(self.Block_conv4(out3) - 0.5)
        else:
            out4 = x
        pad = nn.ZeroPad2d((0,1,0,1))
        pad2 = nn.ZeroPad2d((0, 0, 0, 2))
        pad3 = nn.ZeroPad2d((0, 2, 0, 0))
        pad21 = nn.ZeroPad2d((0, 0, 0, 1))
        pad31 = nn.ZeroPad2d((0, 1, 0, 0))
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        if (x.shape[-1] == 12) and out1.shape[-1]==13: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
            #pass
            out1 = pad21(out1)
            out2 = pad31(out2)
            out3 = pad(out3)
            out4 = pad(out4)
        elif (x.shape[-1] == 14):# and out1.shape[-1] == 13:  # or (x.shape[-1] == 9) or (x.shape[-1] == 10):
                # pass
                out1 = pad21(out1)
                out2 = pad31(out2)
                out3 = pad(out3)
                out4 = pad(out4)
            #out1 = pad(out1)
        elif (x.shape[-1] == 16) and out2.shape[-1]==16:  # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
                # pass
            out1 = pad21(out1)
            out2 = pad31(out2)
            out3 = pad(out3)
            out4 = pad(out4)
        elif (x.shape[-1] == 16):  # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
                # pass
            out1 = pad21(out1)
            out2 = pad31(out2)
            #out3 = pad(out3)
            #out4 = pad(out4)
            #out1 = pad(out1)
            #x = pad(x)
            #out1 = out1[:,:,:-1,:-1]
            #out1 = out1[:, :, :-1, :-1]
            #out3 = out3[:, :, :-1, :-1]
            #out2 = out2[:, :, :-1, :-1]

        #elif (x.shape[-1] == 8) and out1.shape[-1]==8: # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
            #out1 = pad2(out1)
            #out2 = pad3(out2)

        elif (x.shape[-1] == 4) and out1.shape[-1]==4: # and out1.shape[-1]==16: #or (x.shape[-1] == 9) or (x.shape[-1] == 10):
            out1 = pad2(out1)
            out2 = pad3(out2)

        elif (x.shape[-1] == 17):
            #out1 = pad(out1)
            out3 = pad(out3)
            out4 = pad(out4)
            #out1 = pad(out1)
            #out3 = pad(out3)
        elif (x.shape[-1] == 9):
            out4 = pad(out4)
            out3 = pad(out3)
        elif (x.shape[-1] == 7):
            out4 = pad(out4)
            out3 = pad(out3)
            #out1 = pad2(out1)
            #out2 = pad2(out2)
        elif (x.shape[-1] == 17) or (x.shape[-1] == 9)or(x.shape[-1] == 20)or(self.stride == 2 and x.shape[-1] == 10)or (self.stride == 2 and x.shape[-1] == 6):# or (self.stride == 2 and x.shape[-1] == 5):
            pass
            #out1 = out1[:, :, :-1, :-1]
            #out2 = out2[:, :, :-1, :-1]
            #out3 = out3[:, :, :-1, :-1]
        #print(out1.shape, out2.shape, out3.shape, out4.shape, x.shape)
        outf = torch.cat((out1, out2, out3, out4), axis=1)

        n, c, w, h = outf.shape
        outf = outf.view(n, self.cpt, int(c / self.cpt), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        outff = self.Block_convf(outf)
        return outff #torch.cat((outff, out4 ), axis = 1)



class Block_TT_multihead(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1, t=8, k1=4, k2=2, k3=2, p1=1, p2=1, p3=1, g1=1, g2=4, g3=4):
        super(Block_TT_multihead, self).__init__()
        self.stride = stride
        self.pad1, self.pad2, self.pad3 = None, None, None
        #if p1!=0:
        #    self.pad1 = nn.ConstantPad2d(p1, 0)
        self.Block_conv1 = Block_TT(in_planes, in_planes, t=t, k = 3, stride=stride, padding=1,
                                  groupsici=int(in_planes / 1), T=0.4)
        #if p2!=0:
        #    self.pad2 = nn.ConstantPad2d(p2, 0)
        self.Block_conv2 = Block_TT(in_planes, in_planes, t=t, k = 2, stride = stride, padding=1,
                                  groupsici = int(in_planes/2), T=0.4)
        #if p3!=0:
            #self.pad3 = nn.ConstantPad2d(p3, 0)
        #g3 = 3
        self.Block_convf = Block_TT(3*in_planes, out_planes, t=t, k=2, stride=1, padding=1,
                                    groupsici=int(3*in_planes / 3), T=0.4)  # int(in_planes/2))
        self.Block_conv3 = None
        if stride == 2:
            self.Block_conv3 = nn.AvgPool2d(2)


    def forward(self, x):
        self.input_layer1_2 = x.clone()
        if self.pad1 is not None:
            out1 = self.Block_conv1(self.pad1(x))
        else:
            out1 = self.Block_conv1(x)
        if self.pad2 is not None:
            out2 = self.Block_conv2(self.pad2(x))
        else:
            out2 = self.Block_conv2(x)

        if self.Block_conv3 is not None:
            out3 = self.Block_conv1.act(self.Block_conv3(x)-0.5)
        else:
            out3 = x.clone()
        #print(out1.shape, out2.shape, out3.shape, x.shape)

        if (x.shape[-1] == 16) or (x.shape[-1] == 32):
            #out1 = out1[:,:,:-1,:-1]
            #out1 = out1[:, :, :-1, :-1]
            out2 = out2[:, :, :-1, :-1]
        elif (x.shape[-1] == 33) or (x.shape[-1] == 9)or(x.shape[-1] == 20)or(x.shape[-1] == 17)or (self.stride == 2 and x.shape[-1] == 6):# or (self.stride == 2 and x.shape[-1] == 5):
            out1 = out1[:, :, :-1, :-1]
            out2 = out2[:, :, :-1, :-1]

        #print(out1.shape, out2.shape, out3.shape, x.shape)

        #out2 = self.Block_conv2(x)#[:,:,:-1,:-1]
        #out1 = self.Block_conv1(x)
        #print(out1.shape, out2.shape)
        self.output_layer1 = out1.clone()
        self.output_layer2 = out2.clone()
        self.output_layer3 = out3.clone()
        outf = torch.cat((out1, out2, out3), dim=1)
        n, c, w, h = outf.shape
        outf = outf.view(n, 3, int(c / 3), w, h)
        outf = outf.transpose_(1, 2).contiguous()
        outf = outf.view(n, c, w, h)
        self.input_layer3 = outf.clone()
        if self.pad3 is not None:
            self.out_layer3 = self.Block_convf(self.pad3(outf))
        else:
            self.out_layer3 = self.Block_convf(outf)
        #self.out_layer3 = self.Block_convf(outf).clone()
        return self.out_layer3








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
        if type(x) is AbstractTensor:
            return AbstractTensor(self.forward(x.vmin), self.forward(x.vmax),
                                  x.loss)

        if type(x) is MultiSampleTensor:
            return x.apply_batch(self.forward)

        xint = self.RoundFn.apply(x / self.step)
        return xint * self.step

    def __repr__(self):
        return f'{type(self).__name__}({self.step})'



class BNs(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes):
        super(BNs, self).__init__()
        self.Block_BN = []
        #self.device = device
        for out_plane in range(out_planes):
            self.Block_BN.append(nn.BatchNorm2d(in_planes))#.to(device))


    def forward(self, x):
        outf = None
        device = x.device
        for BN in self.Block_BN:
            BN = BN.to(device)
            out = BN(x.to(device))#.to(device)
            if outf is None:
                outf = out
            else:
                outf = torch.cat((outf, out), dim=1)
        return outf




class BNs_v2(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes):
        super(BNs_v2, self).__init__()
        self.Block_BN = []
        #self.device = device
        for out_plane in range(out_planes):
            self.Block_BN.append(nn.BatchNorm2d(3*in_planes))#.to(device))
        #print(self.Block_BN)


    def forward(self, x):
        outf = None
        device = x.device
        #print(x.shape)
        #if x.shape[1]==3 and x.shape[-1]==32:
        #    x = torch.cat((x - 2 / 255, x, x + 2 / 255), dim=1)
        #elif x.shape[1]==1 and x.shape[-1]==28:
        #    x = torch.cat((x - 0.1, x, x + 0.1), dim=1)
        #print(x.shape)
        #x = torch.cat((x-2/255, x, x+2/255), dim=1)
        for BN in self.Block_BN:
            BN = BN.to(device)
            #print(x.shape)
            out = BN(x.to(device))#.to(device)
            if outf is None:
                outf = out
            else:
                outf = torch.cat((outf, out), dim=1)
        return outf

class BN_eval(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, scale, bias):
        super(BN_eval, self).__init__()
        self.scale = torch.Tensor(scale).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        #print(torch.Tensor(scale).shape)
        self.bias = torch.Tensor(bias).unsqueeze(0).unsqueeze(2).unsqueeze(3)


    def forward(self, x):
        device = x.device
        #if x.shape[1]==3 and x.shape[-1]==32:
        #    x = torch.cat((x - 2 / 255, x, x + 2 / 255), dim=1)
        #elif x.shape[1]==1 and x.shape[-1]==28:
        #k    x = torch.cat((x - 0.1, x, x + 0.1), dim=1)
        #print(self.scale, x, self.bias)
        return self.scale.to(device) * x + self.bias.to(device)



class BN_cifar10(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, scale=None, bias=None):
        super(BN_cifar10, self).__init__()
        self.scale = scale
        if scale is not None:
            self.scale = torch.Tensor(scale).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.bias = bias
        if bias is not None:
            self.bias = torch.Tensor(bias).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        self.Block_BN = []
        # self.device = device
        for out_plane in range(3):
            self.Block_BN.append(nn.BatchNorm2d(1))  # .to(device))


    def forward(self, x):
        device = x.device
        if self.scale is not None:
            outf = self.scale.to(device) * x + self.bias.to(device)
        else:
            outf = None
            for index_bn, BN in enumerate(self.Block_BN):
                BN = BN.to(device)
                out = BN(x[:, index_bn : index_bn+1 , :,:].to(device))  # .to(device)
                if outf is None:
                    outf = out
                else:
                    outf = torch.cat((outf, out), dim=1)
        return outf





class Classifier_continu(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, fcsize, nclass, lin2, wb, size_sum = 2500):
        super(Classifier_continu, self).__init__()
        self.fcsize = fcsize
        self.nclass = nclass
        self.pourcentage_sum = size_sum/fcsize
        self.size_sum = size_sum
        self.size_lr = fcsize - self.size_sum
        print("FEATURES SIZE ", self.fcsize, self.pourcentage_sum, self.size_lr, self.size_sum )
        assert self.size_sum%self.nclass==0
        if self.size_lr !=0:
            self.lin = lin2(wb, self.size_lr, nclass)
        self.shape = int(self.size_sum/nclass)




    def forward(self, x):
        device = x.device
        outf = None
        for iterici in range(self.nclass):
            sum_x = torch.sum(x[:, iterici * self.shape:(iterici + 1) * self.shape].to(device), dim=1).unsqueeze(1)
            if outf is None:
                outf = sum_x
            else:
                outf = torch.cat((outf, sum_x), dim=1)
        if self.size_lr != 0:
            outf2 = self.lin(x[:, self.nclass * self.shape:].to(device))
            outf = outf + outf2
        return outf #+ outf2




class Polynome_ACT(nn.Module):
    def __init__(self, alpha=0.47, beta=0.50, gamma=0.09):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


    def forward(self, input):
        #print(input[0], input[0, :160], input[0,160:165])
        #out = self.alpha+ self.beta * input  + self.gamma* input**2 #- 1.7e-10 * input**3 #self.alpha * self.h_function((input / self.gamma) + self.beta) - self.alpha * self.h2_function(self.beta)
        out = 0.47 + 0.5*input + 0.09*input**2
        #print(out[0], out[0,:160], out[160:165])
        #ok
        return out








class TTnet(SeqBinModelHelper, nn.Module, ModelHelper):

    CLASS2NAME = tuple(map(str, range(10)))

    def __init__(self, args):
        super().__init__()
        self._setup_network(args)




    def _setup_network(self, args):
        self.make_small_network(self, args)

    @classmethod
    def make_small_network(
            cls, self,args):

        nclass = 10
        lin = BinLinearPos
        lin2 =BinLinearPosv2
        wb = g_weight_binarizer
        wb3 = g_weight_binarizer3
        Quant3WeightFn
        #poly_act = Polynome_ACT
        #wb3 = g_weight_binarizer3
        act = Binarize01Act
        if args.dataset =="MNIST":
            in_channels = 1
        elif args.dataset =="CIFAR10":
            in_channels = 3
        else:
            raise 'PB'
        liste_fonctions = []#OrderedDict([])
        liste_fonctions.append(InputQuantizer(args.input_quantization))
        if args.first_layer=="bin":
            #liste_fonctions.append(act(T=args.thr_bin_act[0]))
            #liste_fonctions.append(nn.BatchNorm2d(args.preprocessing_CNN[0]))
            liste_fonctions.append(BinConv2d(wb,
                                             in_channels = in_channels,
                                             out_channels = args.preprocessing_CNN[0],
                                             kernel_size = args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[3],
                                             rounding=False))
            self.feature_start = len(liste_fonctions)-1
            liste_fonctions.append(setattr_inplace(BatchNormStatsCallbak(self, args.preprocessing_CNN[0]),
                            'bias_regularizer_coeff', 0))
            #liste_fonctions.append(nn.BatchNorm2d(args.preprocessing_CNN[0]))
            input_channel_here = args.preprocessing_CNN[0]
        if args.first_layer=="binpos":
            liste_fonctions.append(BinConv2d_Pos(wb,
                                             in_channels = in_channels,
                                             out_channels = args.preprocessing_CNN[0],
                                             kernel_size = args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[3],
                                             rounding=False))
            self.feature_start = len(liste_fonctions)-1
            liste_fonctions.append(setattr_inplace(BatchNormStatsCallbak(self, args.preprocessing_CNN[0]),
                            'bias_regularizer_coeff', 0))
            liste_fonctions.append(nn.BatchNorm2d(args.preprocessing_CNN[0]))
            input_channel_here = args.preprocessing_CNN[0]
        elif args.first_layer == "float":
            liste_fonctions.append(nn.Conv2d(in_channels=in_channels,
                                             out_channels=args.preprocessing_CNN[0],
                                             kernel_size=args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[3]))
            self.feature_start = len(liste_fonctions) - 1
            liste_fonctions.append(nn.BatchNorm2d(args.preprocessing_CNN[0]))
            input_channel_here = args.preprocessing_CNN[0]
        elif args.first_layer == "BNs":
            #if args.dataset == "MNIST":
            liste_fonctions.append(BNs(in_channels,args.preprocessing_CNN[0]))
            #liste_fonctions.append(nn.BatchNorm2d(int(in_channels*args.preprocessing_CNN[0])))
            #elif args.dataset == "CIFAR10":
            #    liste_fonctions.append(BN_cifar10())
            #    liste_fonctions.append(BN_cifar10())
            #else:
            #    raise "PB"

            self.feature_start = len(liste_fonctions) - 1
            input_channel_here = int(in_channels*args.preprocessing_CNN[0])
        elif args.first_layer == "BNs_v2":
            #if args.dataset == "MNIST":
            liste_fonctions.append(BNs_v2(in_channels,args.preprocessing_CNN[0]))
            liste_fonctions.append(nn.BatchNorm2d(int(3*in_channels*args.preprocessing_CNN[0])))
            #elif args.dataset == "CIFAR10":
            #    liste_fonctions.append(BN_cifar10())
            #    liste_fonctions.append(BN_cifar10())
            #else:
            #    raise "PB"

            self.feature_start = len(liste_fonctions) - 1
            input_channel_here = int(3*in_channels*args.preprocessing_CNN[0])
        elif args.first_layer == "GroupNorm":
            liste_fonctions.append(nn.GroupNorm(1, in_channels, affine =False))
            liste_fonctions.append(nn.BatchNorm2d(int(in_channels * args.preprocessing_CNN[0])))
            self.feature_start = len(liste_fonctions) - 1
            input_channel_here = args.preprocessing_CNN[0]
        # liste_fonctions.append(nn.BatchNorm2d(args.preprocessing_CNN[0]))
        liste_fonctions.append(act(T=args.thr_bin_act[0]))
        #input_channel_here = args.preprocessing_CNN[0]
        #print(args.Blocks_filters_output)
        for numblock, _ in enumerate(args.Blocks_filters_output):
            print(args.type_blocks[numblock])
            if args.type_blocks[numblock] == "multihead_TTblock":
                liste_fonctions.append(Block_resnet_multihead_general_BN_vf_small_v3(input_channel_here, args.Blocks_filters_output[numblock],
                                       stride=args.Blocks_strides[numblock]))
            elif args.type_blocks[numblock] == "multihead_TTblock_mini":
                liste_fonctions.append(Block_resnet_multihead_general_BN_vf_small_v3_mini(input_channel_here, args.Blocks_filters_output[numblock],
                                       stride=args.Blocks_strides[numblock]))
            elif args.type_blocks[numblock] == "multihead_TTblock_big":
                liste_fonctions.append(Block_resnet_multihead_general_BN_vf_small_v3_big(input_channel_here, args.Blocks_filters_output[numblock],
                                       stride=args.Blocks_strides[numblock]))
            elif args.type_blocks[numblock] == "multihead_TTblock_xmini":
                liste_fonctions.append(Block_resnet_multihead_general_BN_vf_small_v3_xmini(input_channel_here,
                                                                                         args.Blocks_filters_output[
                                                                                             numblock],
                                                                                         stride=args.Blocks_strides[
                                                                                             numblock]))


            elif args.type_blocks[numblock] == "TTblock":
                liste_fonctions.append(Block_TT(input_channel_here, args.Blocks_filters_output[numblock],
                                                          stride=args.Blocks_strides[numblock],
                                                          t=args.Blocks_amplifications[numblock],
                                                          k=args.kernel_size_per_block[numblock],
                                                          padding=args.padding_per_block[numblock],
                                                          groupsici=int(input_channel_here / args.groups_per_block[numblock]),
                                                            quant_flag=args.type_first_layer_block,
                                                            T=args.thr_bin_act[numblock+1]))
                input_channel_here = args.Blocks_filters_output[numblock]
            elif args.type_blocks[numblock] == "TTblock2":
                liste_fonctions.append(Block_TT2(input_channel_here, args.Blocks_filters_output[numblock],
                                                stride=args.Blocks_strides[numblock],
                                                t=args.Blocks_amplifications[numblock],
                                                k=args.kernel_size_per_block[numblock],
                                                padding=args.padding_per_block[numblock],
                                                groupsici=int(input_channel_here / args.groups_per_block[numblock]),
                                                quant_flag=args.type_first_layer_block,
                                                T=args.thr_bin_act[numblock + 1]))

                input_channel_here = 2*args.Blocks_filters_output[numblock]
        #CLASSIFICATION
        #liste_fonctions.append(nn.MaxPool2d(2))
        liste_fonctions.append(Flatten())
        self.features_before_LR = nn.Sequential(*liste_fonctions)
        print(self.features_before_LR)
        fcsize = self.linear_input_neurons(args)
        del self.features_before_LR
        if args.last_layer == "bin":
            liste_fonctions.append(lin(wb, fcsize, nclass))
            liste_fonctions.append(setattr_inplace(
                BatchNormStatsCallbak(
                    self, nclass,
                    use_scalar_scale=g_use_scalar_scale_last_layer),
                'bias_regularizer_coeff', 0)
            )
            self.feature_pos = len(liste_fonctions)
            if args.g_remove_last_bn == "True":
                self.features = self.features[:-1]
                self.feature_pos = None
        if args.last_layer == "MLPbin":

            #liste_fonctions.append(Classifier_scale_all(fcsize, lin2))
            #liste_fonctions.append(Classifier_scale(fcsize, lin))

            liste_fonctions.append(lin(wb, fcsize, 100))
            liste_fonctions.append(setattr_inplace(
                BatchNormStatsCallbak(
                     self, 100,
                    use_scalar_scale=g_use_scalar_scale_last_layer),
                'bias_regularizer_coeff', 0)
            )
            liste_fonctions.append(Polynome_ACT(alpha=0.47, beta=0.50, gamma=0.09))
            liste_fonctions.append(lin(wb, 100, nclass))
            liste_fonctions.append(setattr_inplace(
                 BatchNormStatsCallbak(
                     self, nclass,
                     use_scalar_scale=g_use_scalar_scale_last_layer),
                 'bias_regularizer_coeff', 0)
             )
            self.feature_pos = len(liste_fonctions)
            if args.g_remove_last_bn == "True":
                self.features = self.features[:-1]
                self.feature_pos = None
        elif args.last_layer == "MLPbinpos":

            #liste_fonctions.append(Classifier_scale_all(fcsize, lin2))
            #liste_fonctions.append(Classifier_scale(fcsize, lin))

            liste_fonctions.append(lin2(wb, fcsize, 100))
            liste_fonctions.append(setattr_inplace(
                BatchNormStatsCallbak(
                     self, 100,
                    use_scalar_scale=g_use_scalar_scale_last_layer),
                'bias_regularizer_coeff', 0)
            )
            liste_fonctions.append(Polynome_ACT(alpha=0.47, beta=0.50, gamma=0.09))
            liste_fonctions.append(lin2(wb, 100, nclass))
            liste_fonctions.append(setattr_inplace(
                 BatchNormStatsCallbak(
                     self, nclass,
                     use_scalar_scale=g_use_scalar_scale_last_layer),
                 'bias_regularizer_coeff', 0)
             )
            self.feature_pos = len(liste_fonctions)
            if args.g_remove_last_bn == "True":
                self.features = self.features[:-1]
                self.feature_pos = None
        elif args.last_layer == "binpos":
            liste_fonctions.append(lin2(wb, fcsize, nclass))
            liste_fonctions.append(setattr_inplace(
                BatchNormStatsCallbak(
                    self, nclass,
                    use_scalar_scale=g_use_scalar_scale_last_layer),
                'bias_regularizer_coeff', 0)
            )
            self.feature_pos = len(liste_fonctions)
            if args.g_remove_last_bn == "True":
                self.features = self.features[:-1]
                self.feature_pos = None
        elif args.last_layer == "continu":
            liste_fonctions.append(Classifier_continu(fcsize, nclass, lin2, wb))
            #liste_fonctions.append(setattr_inplace(
            #    BatchNormStatsCallbak(
            #        self, nclass,
            #        use_scalar_scale=g_use_scalar_scale_last_layer),
            #    'bias_regularizer_coeff', 0)
            #)
            self.feature_pos = len(liste_fonctions)
            #if args.g_remove_last_bn == "True":
            #    self.features = self.features[:-1]
            #    self.feature_pos = None
        else:
            #liste_fonctions.append(nn.Dropout(0.2))
            liste_fonctions.append(nn.Linear(fcsize, nclass))
            liste_fonctions.append(nn.BatchNorm1d(nclass))

            self.feature_pos = None
        self.features = nn.Sequential(*liste_fonctions)


    @classmethod
    def make_dataset_loader(cls, args, train: bool):
        if args.dataset == "MNIST":
            dataset = torchvision.datasets.MNIST(
            root=args.data, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        elif args.dataset == "CIFAR10":
            if args.data_augmentation == "True":

                if train:

                    transform = transforms.Compose([
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    #transforms.Normalize((0.4914, 0.4822, 0.4465),
                    #                     (0.2023, 0.1994, 0.2010)),
                ])
                else:
                    transform = transforms.Compose([
                        transforms.ToTensor(),
                        #transforms.Normalize((0.4914, 0.4822, 0.4465),
                        #                 (0.2023, 0.1994, 0.2010)),
                    ])

                dataset = torchvision.datasets.CIFAR10(
                root=args.data, train=train, download=True,
                transform=transform)
            else:
                dataset = torchvision.datasets.CIFAR10(
                root=args.data, train=train, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                ]))
        else:
            raise "PB"
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batchsize, shuffle=train,
            num_workers=args.workers if train else 0)
        return loader


    def linear_input_neurons(self, args):
        if args.dataset =="CIFAR10":
            size = self.features_before_LR(torch.rand(1, 3, 32, 32)).shape[1]  # image size: 64x32
        elif args.dataset =="MNIST":
            size = self.features_before_LR(torch.rand(1, 1, 28, 28)).shape[1]  # image size: 64x32
        else:
            raise 'PB'

        return int(size)


    def get_key_online(self, inputs):
        self.T = 2.2
        with torch.no_grad():

            imagesPLUSv1 = inputs.clone() + float(8/255)
            imagesMOINSv1 = inputs.clone() - float(8/255)
            #print("imagesPLUSv1", imagesPLUSv1[3])
            #print("imagesMOINSv1", imagesMOINSv1[3])
            for layer, module in enumerate(self.features):
                if layer == 0:
                    respv1 = module(imagesPLUSv1.to(inputs.device))
                    resmv1 = module(imagesMOINSv1.to(inputs.device))
                elif layer <= self.feature_start:
                    respv1 = module(respv1)
                    resmv1 = module(resmv1)
                else:
                    #print(module)
                    all_onespv1 = 1.0 * (respv1 >= self.T / 2)
                    maksrmv1 = 1.0 * (resmv1 < self.T / 2) - 1.0 * (resmv1 < -self.T / 2)
                    key = (all_onespv1 * maksrmv1).clone().detach()#.cpu().numpy()
                    #outp = module(respv1).detach().cpu().clone()
                    #outm = module(resmv1).detach().cpu().clone()
                    #input_acomplter = (outm * (outm == outp).float() + -1 * (outm != outp))
                    #image_with_U = np.sum(input_acomplter.detach().cpu().clone().numpy() == -1, axis=(1, 2, 3))
                    #assert np.sum(image_with_U) == 0

                    #print("respv1",respv1[3])
                    #print("all_onespv1",all_onespv1[3])
                    #print("resmv1", resmv1[3])
                    #print("maksrmv1", maksrmv1[3])

                    return key





class MyRobustModel(nn.Module):

    def __init__(self, TT_base_model, T, eps_infer, device):
        super().__init__()
        self.T = T
        TT_base_model.features[3] = Binarize01Act_robustness_benchmark(T=self.T)
        self.model = TT_base_model.eval()
        self.eps_infer = eps_infer
        self.device = device
        self.feature_start = TT_base_model.feature_start
        self.key = None
        self.input_memory = None
        self.resnoraml = None

    def forward(self, inputs):
        #print(inputs.shape)
        if self.input_memory is None:
            self.input_memory = inputs.clone()
            self.key = self.get_key_online(inputs)
            self.resnoraml = None
            flag = False
            #print("OK1")
        else:
            #print(torch.abs(inputs.clone()-self.input_memory)>8/255)
            #print(torch.sum(torch.abs(inputs.clone()-self.input_memory)>8/255))
            if inputs.shape[0]==self.input_memory.shape[0]:
                flag = (torch.sum(torch.abs(inputs.clone()-self.input_memory)>self.eps_infer)>1).item()
            else:
                flag = True
            #print("OK2")

        #print(flag)

        if flag:
            self.input_memory = inputs.clone()
            self.key = self.get_key_online(inputs)
            self.resnoraml = None
            #print("OK3")
            #print(self.key[3])
        for layer, module in enumerate(self.model.features):
            if layer == 0:
                resnoraml = module(inputs.to(self.device))
            elif layer <= self.feature_start:
                resnoraml = module(resnoraml)
            elif layer == self.feature_start + 1:
                #print(self.key)
                resnoraml = module(resnoraml, self.key)
                if self.resnoraml is None:
                    self.resnoraml = resnoraml.clone()
                if False in (resnoraml == self.resnoraml):
                    print(resnoraml == self.resnoraml)
                assert False not in (resnoraml == self.resnoraml)
                #print(resnoraml == self.resnoraml)
                #print(resnoraml[3])
                #print(resnoraml[3][0][1][0])
                self.res_before_actibvib = resnoraml.clone()
            else:
                resnoraml = module(resnoraml)
        return resnoraml

    def get_key_online(self, inputs):
        with torch.no_grad():

            imagesPLUSv1 = inputs.clone() + float(8/255)
            imagesMOINSv1 = inputs.clone() - float(8/255)
            #print("imagesPLUSv1", imagesPLUSv1[3])
            #print("imagesMOINSv1", imagesMOINSv1[3])
            for layer, module in enumerate(self.model.features):
                if layer == 0:
                    respv1 = module(imagesPLUSv1.to(self.device))
                    resmv1 = module(imagesMOINSv1.to(self.device))
                elif layer <= self.feature_start:
                    respv1 = module(respv1)
                    resmv1 = module(resmv1)
                else:
                    all_onespv1 = 1.0 * (respv1 >= self.T / 2)
                    maksrmv1 = 1.0 * (resmv1 < self.T / 2) - 1.0 * (resmv1 < -self.T / 2)
                    key = (all_onespv1 * maksrmv1).clone().detach().cpu().numpy()
                    outp = module(respv1, key).detach().cpu().clone()
                    outm = module(resmv1, key).detach().cpu().clone()
                    input_acomplter = (outm * (outm == outp).float() + -1 * (outm != outp))
                    image_with_U = np.sum(input_acomplter.detach().cpu().clone().numpy() == -1, axis=(1, 2, 3))
                    assert np.sum(image_with_U) == 0

                    #print("respv1",respv1[3])
                    #print("all_onespv1",all_onespv1[3])
                    #print("resmv1", resmv1[3])
                    #print("maksrmv1", maksrmv1[3])

                    return key
