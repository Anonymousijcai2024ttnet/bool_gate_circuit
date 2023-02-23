import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config.config import Config, two_args_str_float, transform_input_filters_multiple, \
    transform_input_filters2, transform_input_thr
from config.config import str2bool, two_args_str_int, str2list, \
    transform_input_filters, transform_input_lr
from src.inference import BN_eval_MNIST, BN_eval_CIFAR10, load_cnf_dnf_block
from src.inference import get_mapping_filter, Binarize01Act, InputQuantizer, load_data, \
    infer_normal_withPYTHON
from src.inference import load_TT_TTnoise

config_general = Config(path="config/")
if config_general.dataset == "CIFAR10":
    config = Config(path="config/cifar10/")
elif config_general.dataset == "MNIST":
    config = Config(path="config/mnist/")
else:
    raise 'PB'

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default=config_general.dataset)

parser.add_argument("--seed", default=config.general.seed, type=two_args_str_int, choices=[i for i in range(100)])
parser.add_argument("--device", default=config.general.device, choices=["cuda", "cpu"])
parser.add_argument("--device_ids", default=config.general.device_ids, type=str2list)
parser.add_argument("--models_path", default=config.general.models_path)
parser.add_argument("--num_workers", default=config.general.num_workers, type=int)
parser.add_argument("--quant_step", default=config.model.quant_step, type=two_args_str_float)
parser.add_argument("--famille", default=config.model.famille)
parser.add_argument("--cbd", default=config.model.cbd)
parser.add_argument("--first_layer", default=config.model.first_layer, choices=["float", "bin"])
parser.add_argument("--preprocessing_CNN", default=config.model.preprocessing_CNN, type=transform_input_filters)
parser.add_argument("--g_remove_last_bn", default=config.model.g_remove_last_bn)
parser.add_argument("--type_blocks", default=config.model.type_blocks, type=transform_input_filters2)
parser.add_argument("--last_layer", default=config.model.last_layer, choices=["float", "bin"])
parser.add_argument("--Blocks_filters_output", default=config.model.Blocks_filters_output, type=transform_input_filters)
parser.add_argument("--Blocks_amplifications", default=config.model.Blocks_amplifications, type=transform_input_filters)
parser.add_argument("--Blocks_strides", default=config.model.Blocks_strides, type=transform_input_filters)
parser.add_argument("--type_first_layer_block", default=config.model.type_first_layer_block, choices=["float", "bin"])
parser.add_argument("--kernel_size_per_block", default=config.model.kernel_size_per_block, type=transform_input_filters)
parser.add_argument("--groups_per_block", default=config.model.groups_per_block, type=transform_input_filters)
parser.add_argument("--padding_per_block", default=config.model.padding_per_block, type=transform_input_filters)
parser.add_argument("--kernel_size_per_block_multihead", default=config.model.kernel_size_per_block_multihead,
                    type=transform_input_filters_multiple)
parser.add_argument("--groups_per_block_multihead", default=config.model.groups_per_block_multihead,
                    type=transform_input_filters_multiple)
parser.add_argument("--paddings_per_block_multihead", default=config.model.paddings_per_block_multihead,
                    type=transform_input_filters_multiple)

parser.add_argument("--adv_epsilon", default=config.train.adv_epsilon)
parser.add_argument("--adv_step", default=config.train.adv_step)
parser.add_argument("--n_epoch", default=config.train.n_epoch, type=two_args_str_int)
parser.add_argument("--lr", default=config.train.lr, type=transform_input_lr)

parser.add_argument("--batch_size_test", default=config.eval.batch_size_test, type=two_args_str_int)
parser.add_argument("--jeudelavie", default=config.eval.jeudelavie, type=str2bool)
parser.add_argument("--pruning", default=config.eval.pruning, type=str2bool)
parser.add_argument("--coef_mul", default=config.eval.coef_mul, type=two_args_str_int)
parser.add_argument("--path_save_model", default=config.eval.path_load_model, type=two_args_str_int)

parser.add_argument("--Transform_normal_model", default=config.NN2TT.Transform_normal_model, type=str2bool)
parser.add_argument("--Transform_pruned_model", default=config.NN2TT.Transform_pruned_model, type=str2bool)
parser.add_argument("--Transform_normal_model_with_filtering",
                    default=config.NN2TT.Transform_normal_model_with_filtering, type=str2bool)
parser.add_argument("--Transform_pruned_model_with_filtering",
                    default=config.NN2TT.Transform_pruned_model_with_filtering, type=str2bool)
parser.add_argument("--filter_occurence", default=config.NN2TT.filter_occurence, type=two_args_str_int)
parser.add_argument("--block_occurence", default=config.NN2TT.block_occurence, type=two_args_str_int)

parser.add_argument("--Add_noise", default=config.NN2TT.Add_noise, type=str2bool)
parser.add_argument("--proportion", default=config.NN2TT.proportion, type=two_args_str_float)
parser.add_argument("--proba", default=config.NN2TT.proba, type=two_args_str_float)

parser.add_argument("--model_to_eval", default=config.verify.model_to_eval)
parser.add_argument("--type_verification", default=config.verify.type_verification)
parser.add_argument("--mode_verification", default=config.verify.mode_verification)
parser.add_argument("--ratio_incomplet", default=config.verify.ratio_incomplet, type=two_args_str_float)
parser.add_argument("--attack_eps", default=config.verify.attack_eps, type=two_args_str_float)
parser.add_argument("--coef_multiplicateur_data", default=config.verify.coef_multiplicateur_data, type=two_args_str_int)
parser.add_argument("--offset", default=config.verify.offset, type=two_args_str_int)
parser.add_argument("--encoding_type", default=config.verify.encoding_type, type=two_args_str_int)
parser.add_argument("--sat_solver", default=config.verify.sat_solver)
parser.add_argument("--time_out", default=config.verify.time_out, type=two_args_str_int)
parser.add_argument("--thr_bin_act", default=config.model.thr_bin_act, type=transform_input_thr)
parser.add_argument("--thr_bin_act_test", default=config.eval.thr_bin_act_test, type=transform_input_thr)
parser.add_argument("--method_verify_incomplete", default=config.verify.method_verify_incomplete,
                    choices=["DP", "formula"])

parser.add_argument("--with_contradiction", default=config.NN2TT.with_contradiction, type=str2bool)

args = parser.parse_args()
args.preprocessing_BN = 1
args.batch_size_test = 1000

args.path_save_model = args.path_save_model + "/"
args.path_load_model = args.path_save_model + "/"
device = "cpu"  # torch.device("cuda:" + str(args.device_ids[0]) if torch.cuda.is_available() else "cpu")
seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if config_general.dataset == "CIFAR10":
    args.attack_eps = args.attack_eps / 255

print(args)

print()
print("-" * 100)
print()
print("START EVALUATION ")
print()

print(args.path_save_model)
dataloaders, testset, nclass = load_data(args)

all_CNFG = {i: {j: [] for j in range(10)} for i in range(10)}

liste_fonctions1 = []
liste_fonctions1.append(InputQuantizer(args.quant_step))
preprocessing_inpQ = nn.Sequential(*liste_fonctions1).eval()
print(preprocessing_inpQ)

# Preprocessing
liste_fonctions = []
liste_fonctions2 = []
liste_fonctions_P = []
liste_fonctions_N = []
liste_fonctions.append(InputQuantizer(args.quant_step))
liste_fonctions2.append(InputQuantizer(args.quant_step))
liste_fonctions_P.append(InputQuantizer(args.quant_step))
liste_fonctions_N.append(InputQuantizer(args.quant_step))

scale = np.loadtxt(args.path_load_model + "/preprocessing_BN_scale.txt")
bias = np.loadtxt(args.path_load_model + "/preprocessing_BN_bias.txt")

if config_general.dataset == "MNIST":
    liste_fonctions.append(BN_eval_MNIST(np.array([scale]), np.array([bias])).to(device))
else:
    liste_fonctions.append(BN_eval_CIFAR10(np.array([scale]), np.array([bias])).to(device))
# liste_fonctions.append(BN_eval(np.array([scale]),np.array([bias])).to(device))
if os.path.isfile(args.path_load_model + "/preprocessing_BN_scale_2.txt"):
    scale2 = np.loadtxt(args.path_load_model + "/preprocessing_BN_scale_2.txt")
    bias2 = np.loadtxt(args.path_load_model + "/preprocessing_BN_bias_2.txt")
    if config_general.dataset == "MNIST":
        liste_fonctions.append(BN_eval_MNIST(np.array([scale2]), np.array([bias2])).to(device))
    else:
        liste_fonctions.append(BN_eval_CIFAR10(np.array([scale2]), np.array([bias2])).to(device))

# ok

putawayliteral = []

act = Binarize01Act
liste_fonctions.append(act(T=args.thr_bin_act_test[0]))
preprocessing = nn.Sequential(*liste_fonctions).eval()
preprocessing_withoutact = nn.Sequential(*liste_fonctions2).eval()
preprocessing_withoutact_P = nn.Sequential(*liste_fonctions_P).eval()
preprocessing_withoutact_N = nn.Sequential(*liste_fonctions_N).eval()

# Last layer
Wbin_scale = 1.0 * (np.loadtxt(args.path_load_model + "/Wbin_scale.txt").astype("f"))
W_LR = 1.0 * (np.loadtxt(args.path_load_model + "/Wbin.txt").astype("f"))
scale_WLR = 1.0 * (np.loadtxt(args.path_load_model + "/gamma_Wbin.txt").astype("f"))
b_LR = 1.0 * (np.loadtxt(args.path_load_model + "/biais.txt").astype("f"))
# ok

# Unfold and Mapping
unfold_all = {}
for numblockici in range(len(args.type_blocks)):
    unfold_all[numblockici] = [
        torch.nn.Unfold(kernel_size=args.kernel_size_per_block[numblockici], stride=args.Blocks_strides[numblockici],
                        padding=args.padding_per_block[numblockici])]

mapping_filter, input_dim = get_mapping_filter(args)
print(mapping_filter)
if config_general.dataset == "CIFAR10":
    for i in range(16):
        mapping_filter[0][i] = 0
    for i in range(16):
        mapping_filter[0][i + 16] = 1
    for i in range(16):
        mapping_filter[0][i + 32] = 2

print(mapping_filter)
putawayliteral = []

all_TT_vold, all_TT_noiseonly, _ = load_TT_TTnoise(args)
print("START LOADING TT")
array_block_0, array_block_1, nogolist = load_cnf_dnf_block(args)
print(nogolist)
tk0 = tqdm(dataloaders["val"], total=int(len(dataloaders["val"])))
items = [filter_no for filter_no in range(args.Blocks_filters_output[1])]
tot = 0
correct = 0

with torch.no_grad():
    for indexicicici, data in enumerate(tk0):
        if int(args.coef_multiplicateur_data) * (int(args.offset) + 1) > indexicicici >= int(
                args.coef_multiplicateur_data) * int(args.offset):
            nSize = args.kernel_size_per_block[0] ** 2 * args.groups_per_block[0]
            inputs, labels = data
            predicted, res_all_tensorinput_block, res_all_tensoroutput_block, \
                shape_all_tensorinput_block, shape_all_tensoroutput_block, \
                res_all_tensorinput_block_unfold = infer_normal_withPYTHON(inputs, preprocessing,
                                                                           device, unfold_all, args,
                                                                           mapping_filter, Wbin_scale, b_LR,
                                                                           array_block_0,
                                                                           array_block_1, items, putawayliteral)
            tot += labels.shape[0]
            predicted = torch.Tensor(predicted).to(device)
            imagev2 = inputs[(predicted == labels.to(device)), :, :, :]
            labelrefv2 = labels[(predicted == labels.to(device))].clone().detach().cpu().numpy().astype("i")
            correct += labelrefv2.shape[0]

print("Accuracy: ", correct / tot)
