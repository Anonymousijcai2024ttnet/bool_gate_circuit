import argparse
import os
import random
from eevbnn.utils import ModelHelper
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
    infer_normal_withPYTHON, quantization_int
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
#parser.add_argument("--jeudelavie", default=config.eval.jeudelavie, type=str2bool)
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
args.batch_size_test = 100

args.path_save_model = "./models_res/CIFAR10_BIG/"#args.path_save_model + "/"
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

model = (ModelHelper.
         create_with_load(args.path_save_model + "/last.pth").
         to(device).
         eval())
print(model)
import copy
model_train = copy.deepcopy(model)
del model
model_train.to(device)
model_train.eval()




tk0 = tqdm(dataloaders["val"], total=int(len(dataloaders["val"])))
items = [filter_no for filter_no in range(args.Blocks_filters_output[1])]
tot = 0
correct = 0

bit_q = 8
W = quantization_int(model_train.features[6].weight.data, bit_q)
b = quantization_int(model_train.features[6].bias.data, bit_q)
np.savetxt(args.path_save_model  + "/W_q.txt", W)
np.savetxt(args.path_save_model  + "/b_q.txt", b)
bit_q = 8
W2 = quantization_int(model_train.features[10].weight.data, bit_q)
b2 = quantization_int(model_train.features[10].bias.data, bit_q)
np.savetxt(args.path_save_model  + "/W_q2.txt", W2)
np.savetxt(args.path_save_model  + "/b_q2.txt", b2)
del W, b, W2, b2

W = np.loadtxt(args.path_save_model  + "/W_q.txt")
b = np.loadtxt(args.path_save_model  + "/b_q.txt")
W2 = np.loadtxt(args.path_save_model  + "/W_q2.txt")
b2 = np.loadtxt(args.path_save_model  + "/b_q2.txt")

lin_4bit = torch.nn.Linear(W.shape[0], W.shape[1], bias=True).eval()
lin_4bit.weight.data= torch.Tensor(W)
lin_4bit.bias.data= torch.Tensor(b)
lin_4bit2 = torch.nn.Linear(W2.shape[0], W2.shape[1], bias=True).eval()
lin_4bit2.weight.data= torch.Tensor(W2)
lin_4bit2.bias.data= torch.Tensor(b2)

print(torch.sum(model_train.features[1].weight_bin.abs().data, dim=(1,2,3)))
ok

with torch.no_grad():
    for indexicicici, data in enumerate(tk0):
        nSize = args.kernel_size_per_block[0] ** 2 * args.groups_per_block[0]
        inputs, labels = data
        X = model_train.features[0](inputs)
        for i in range(1,6):
            X = model_train.features[i](X)
        X = lin_4bit(X)
        X = model_train.features[8](X)
        X = model_train.features[9](X)
        X = lin_4bit2(X)
        p = model_train.features[12](X)
        pred = torch.argmax(p, dim=1)
        tot += labels.shape[0]
        imagev2 = inputs[(pred == labels.to(device)), :, :, :]
        labelrefv2 = labels[(pred == labels.to(device))].clone().detach().cpu().numpy().astype("i")
        correct += labelrefv2.shape[0]

print("Accuracy: ", correct / tot)
