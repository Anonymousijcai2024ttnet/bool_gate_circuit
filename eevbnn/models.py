from collections import OrderedDict
import os
from eevbnn.net_bin import SeqBinModelHelper, BinLinearPos, Binarize01Act, g_weight_binarizer, activation_quantize_fn2, \
    BinConv2d, g_weight_binarizer3, setattr_inplace, BatchNormStatsCallbak, g_use_scalar_scale_last_layer, \
    InputQuantizer, g_weight_binarizer2
from eevbnn.utils import ModelHelper, Flatten
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F


class Block_TT(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, k=3, t=8, padding=1, stride=1, groupsici=1, big=False):
        super(Block_TT, self).__init__()
        #print(in_planes, groupsici)
        wb3 = g_weight_binarizer3
        self.conv1 = BinConv2d(wb3, in_planes, t * in_planes, kernel_size=k,
                               stride=stride, padding=padding, groups=groupsici, bias=False)
        self.bn1 = nn.BatchNorm2d(t * in_planes)
        self.big = big
        if self.big:
            self.conv1b = nn.Conv2d(t * in_planes, t * in_planes, kernel_size=1,
                                stride=1, padding=0, groups=groupsici, bias=False)
            self.bn1b = nn.BatchNorm2d(t * in_planes)
        self.conv2 = nn.Conv2d(t * in_planes, out_planes, kernel_size=1, stride=1, padding=0,
                               groups=groupsici, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.stride = stride
        self.act = Binarize01Act()

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        if self.big:
            out = F.gelu(self.bn1b(self.conv1b(out)))
        out = self.act(self.bn2(self.conv2(out)))
        return out

class Block_TT_general(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1, big=False):
        super(Block_TT_general, self).__init__()
        self.Block_conv1 = Block_TT(in_planes, in_planes, k = 3, stride=stride, padding=0,
                                  groupsici=int(in_planes / 1), big = big)  # int(in_planes/1))

        self.act = Binarize01Act()
        print(in_planes, out_planes, int(2 * in_planes / 2))
        self.Block_convf = Block_TT(in_planes, out_planes, k=3, stride=2, padding=0,
                                  groupsici=int(in_planes), big=big)  # int(4*in_planes/4))
        self.stride = stride

    def forward(self, x):
        out2 = self.Block_conv1(x)
        #out1 = self.Block_conv1(x)
        #print(out1.shape, out2.shape, x.shape)
        #if x.shape[-1]==8:
        #   out2 = out2[:,:,:-1,:-1]
        #outf = torch.cat((out1, out2), axis=1)
        #n, c, w, h = outf.shape
        #outf = outf.view(n, 2, int(c / 2), w, h)
        #outf = outf.transpose_(1, 2).contiguous()
        #outf = outf.view(n, c, w, h)
        return self.Block_convf(out2)

class Block_TT_multihead_general(nn.Module):
    '''Depthwise conv + Pointwise conv'''

    def __init__(self, in_planes, out_planes, stride=1, big=False):
        super(Block_TT_multihead_general, self).__init__()
        self.Block_conv1 = Block_TT(in_planes, 2*8*in_planes, k = 3, stride=stride, padding=0,
                                  groupsici=int(in_planes / 1), big = big)  # int(in_planes/1))
        self.Block_conv2 = Block_TT(in_planes, 8*in_planes, k=2, stride=stride, padding=0,
                                        groupsici=int(in_planes / 3), big=big)  # int(in_planes/1))
        self.act = Binarize01Act()
        print(in_planes, out_planes, int(2 * in_planes / 2))
        self.Block_convf = Block_TT(8*2* in_planes, out_planes, k=3, stride=2, padding=0,
                                  groupsici=int(8*2 * in_planes / 1), big=big)  # int(4*in_planes/4))
        self.stride = stride

    def forward(self, x):
        #out2 = self.Block_conv2(x)
        out1 = self.Block_conv1(x)
        #print(out1.shape, out2.shape, x.shape)
        #if x.shape[-1]==8:
        #   out2 = out2[:,:,:-1,:-1]
        #outf = torch.cat((out1, out2), axis=1)
        #n, c, w, h = outf.shape
        #outf = outf.view(n, 2, int(c / 2), w, h)
        #outf = outf.transpose_(1, 2).contiguous()
        #outf = outf.view(n, c, w, h)
        return self.Block_convf(out1)

class model_general_new_version(SeqBinModelHelper, nn.Module, ModelHelper):

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
        wb = g_weight_binarizer
        wb2 = g_weight_binarizer2
        wb3 = g_weight_binarizer3
        act = Binarize01Act
        if args.dataset =="MNIST":
            in_channels = 1
        elif args.dataset =="CIFAR10":
            in_channels = 3
        else:
            raise 'PB'


        liste_fonctions = []#OrderedDict([])
        #Preprocessing
        print(args)
        if args.type_weigths_preprocessing_CNN == "ter":
            liste_fonctions.append(BinConv2d(wb3,
                                             in_channels = in_channels,
                                             out_channels = args.preprocessing_CNN[0],
                                             kernel_size = args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[2],
                                             rounding=False))
        elif args.type_weigths_preprocessing_CNN == "bin":
            liste_fonctions.append(BinConv2d(wb,
                                             in_channels = in_channels,
                                             out_channels = args.preprocessing_CNN[0],
                                             kernel_size = args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[2],
                                             rounding=False))
        elif args.type_weigths_preprocessing_CNN == "float":
            liste_fonctions.append(BinConv2d(wb2,
                                             in_channels=in_channels,
                                             out_channels=args.preprocessing_CNN[0],
                                             kernel_size=args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[2],
                                             rounding=False))
        else:
            raise 'PB'
        liste_fonctions.append(nn.BatchNorm2d(args.preprocessing_CNN[0]))
        liste_fonctions.append(act())

        # Blocks
        input_channel_here = args.preprocessing_CNN[0]
        #for numblock, _ in enumerate(args.filters):
        liste_fonctions.append(nn.Conv2d( 3, 48*2, kernel_size=3,
                               stride=2, padding=0, groups=3, bias=False))
        liste_fonctions.append(nn.BatchNorm2d(48*2))
        liste_fonctions.append(nn.GELU())
        liste_fonctions.append(nn.Conv2d(48*2, 48, kernel_size=1, stride=1, padding=0,
                               groups=3, bias=False))
        liste_fonctions.append(nn.BatchNorm2d(48))
        liste_fonctions.append(act())
        liste_fonctions.append(nn.Conv2d( 48, 48 * 6, kernel_size=3,
                                         stride=2, padding=0, groups=48, bias=False))
        liste_fonctions.append(nn.BatchNorm2d(48 * 6))
        liste_fonctions.append(nn.GELU())
        liste_fonctions.append(nn.Conv2d(48 * 6, 48, kernel_size=1, stride=1, padding=0,
                                         groups=48, bias=False))
        liste_fonctions.append(nn.BatchNorm2d(48))
        liste_fonctions.append(act())

        #input_channel_here = args.filters[numblock]
        #CLASSIFICATION
        liste_fonctions.append(Flatten())
        self.features_before_LR = nn.Sequential(*liste_fonctions)
        fcsize = self.linear_input_neurons(args)
        del self.features_before_LR
        if args.type_weigths_final_LR == "ter":
            liste_fonctions.append(lin(wb3, fcsize, nclass))
            self.feature_pos = len(liste_fonctions)
            liste_fonctions.append(setattr_inplace(
                BatchNormStatsCallbak(
                    self, nclass,
                    use_scalar_scale=g_use_scalar_scale_last_layer),
                'bias_regularizer_coeff', 0)
            )
        elif args.type_weigths_final_LR == "bin":
            liste_fonctions.append(lin(wb, fcsize, nclass))
            self.feature_pos = len(liste_fonctions)
            liste_fonctions.append(setattr_inplace(
                BatchNormStatsCallbak(
                    self, nclass,
                    use_scalar_scale=g_use_scalar_scale_last_layer),
                'bias_regularizer_coeff', 0)
            )
        elif args.type_weigths_final_LR == "float":
            liste_fonctions.append(nn.Linear(fcsize, nclass))


        self.features = nn.Sequential(*liste_fonctions)

        if args.g_remove_last_bn=="True":
            self.features = self.features[:-1]
            self.feature_pos = None



    @classmethod
    def make_dataset_loader(cls, args, train: bool):
        if args.dataset == "MNIST":
            dataset = torchvision.datasets.MNIST(
            root=args.data, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        elif args.dataset == "CIFAR10":
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





class model_general(SeqBinModelHelper, nn.Module, ModelHelper):

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
        wb = g_weight_binarizer
        wb3 = g_weight_binarizer3
        act = Binarize01Act
        act3 = activation_quantize_fn2
        if args.dataset =="MNIST":
            in_channels = 1
        elif args.dataset =="CIFAR10":
            in_channels = 3
        else:
            raise 'PB'

        wb2 = g_weight_binarizer2

        liste_fonctions = []#OrderedDict([])
        #Preprocessing
        print(args)
        if args.type_weigths_preprocessing_CNN == "ter":
            liste_fonctions.append(BinConv2d(wb3,
                                             in_channels = in_channels,
                                             out_channels = args.preprocessing_CNN[0],
                                             kernel_size = args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[2],
                                             rounding=False))
        elif args.type_weigths_preprocessing_CNN == "bin":
            liste_fonctions.append(BinConv2d(wb,
                                             in_channels = in_channels,
                                             out_channels = args.preprocessing_CNN[0],
                                             kernel_size = args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[2],
                                             rounding=False))
        elif args.type_weigths_preprocessing_CNN == "float":
            liste_fonctions.append(BinConv2d(wb2,in_channels = in_channels,
                                             out_channels = args.preprocessing_CNN[0],
                                             kernel_size = args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[2]))
        else:
            raise 'PB'
        liste_fonctions.append(nn.BatchNorm2d(args.preprocessing_CNN[0]))
        liste_fonctions.append(act())

        # Blocks
        input_channel_here = args.preprocessing_CNN[0]
        for numblock, _ in enumerate(args.filters):
            liste_fonctions.append(BinConv2d(wb2,input_channel_here,
                                            args.filters[numblock] * args.amplifications[numblock],
                                            args.kernelsizes[numblock],
                                            stride=args.strides[numblock],
                                            padding=args.paddings[numblock],
                                            groups=args.groups[numblock]))
            liste_fonctions.append(nn.BatchNorm2d(args.filters[numblock] * args.amplifications[numblock]))
            liste_fonctions.append(nn.ReLU())
            liste_fonctions.append(BinConv2d(wb2,args.filters[numblock] * args.amplifications[numblock],
                                            args.filters[numblock],
                                            1,
                                            stride=1,
                                            padding=0,
                                            groups=args.groups[numblock]))
            liste_fonctions.append(nn.BatchNorm2d(args.filters[numblock]))
            liste_fonctions.append(act())
            input_channel_here = args.filters[numblock]
        #CLASSIFICATION
        liste_fonctions.append(Flatten())
        self.features_before_LR = nn.Sequential(*liste_fonctions)
        fcsize = self.linear_input_neurons(args)
        del self.features_before_LR

        if args.type_weigths_final_LR == "ter":
            liste_fonctions.append(lin(wb3, fcsize, nclass))
        elif args.type_weigths_final_LR == "bin":
            liste_fonctions.append(lin(wb, fcsize, nclass))
        elif args.type_weigths_final_LR == "float":
            liste_fonctions.append(nn.Linear(fcsize, nclass))
        self.feature_pos = len(liste_fonctions)
        liste_fonctions.append(setattr_inplace(
            BatchNormStatsCallbak(
                self, nclass,
                use_scalar_scale=g_use_scalar_scale_last_layer),
            'bias_regularizer_coeff', 0)
        )
        self.features = nn.Sequential(*liste_fonctions)

        if args.g_remove_last_bn=="True":
            self.features = self.features[:-1]
            self.feature_pos = None



    @classmethod
    def make_dataset_loader(cls, args, train: bool):
        if args.dataset == "MNIST":
            dataset = torchvision.datasets.MNIST(
            root=args.data, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        elif args.dataset == "CIFAR10":
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


class model_FHE(SeqBinModelHelper, nn.Module, ModelHelper):

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
        wb = g_weight_binarizer
        wb3 = g_weight_binarizer3
        act = Binarize01Act
        act3 = activation_quantize_fn2
        wb2 = g_weight_binarizer2
        if args.dataset =="MNIST":
            in_channels = 1
        elif args.dataset =="CIFAR10":
            in_channels = 3
        else:
            raise 'PB'


        liste_fonctions = []#OrderedDict([])
        #Preprocessing
        print(args)
        if args.type_weigths_preprocessing_CNN == "ter":
            liste_fonctions.append(BinConv2d(wb3,
                                             in_channels = in_channels,
                                             out_channels = args.preprocessing_CNN[0],
                                             kernel_size = args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[2],
                                             rounding=False))
        elif args.type_weigths_preprocessing_CNN == "bin":
            liste_fonctions.append(BinConv2d(wb,
                                             in_channels = in_channels,
                                             out_channels = args.preprocessing_CNN[0],
                                             kernel_size = args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[2],
                                             rounding=False))
        elif args.type_weigths_preprocessing_CNN == "float":
            liste_fonctions.append(BinConv2d(wb2,in_channels = in_channels,
                                             out_channels = args.preprocessing_CNN[0],
                                             kernel_size = args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[2]))
        else:
            raise 'PB'
        liste_fonctions.append(nn.BatchNorm2d(args.preprocessing_CNN[0]))
        liste_fonctions.append(act())

        # Blocks
        input_channel_here = args.preprocessing_CNN[0]
        for numblock, _ in enumerate(args.filters):
            liste_fonctions.append(BinConv2d(wb2,input_channel_here,
                                            args.filters[numblock] * args.amplifications[numblock],
                                            args.kernelsizes[numblock],
                                            stride=args.strides[numblock],
                                            padding=args.paddings[numblock],
                                            groups=args.groups[numblock]))
            liste_fonctions.append(nn.BatchNorm2d(args.filters[numblock] * args.amplifications[numblock]))
            liste_fonctions.append(nn.ReLU())
            liste_fonctions.append(BinConv2d(wb2,args.filters[numblock] * args.amplifications[numblock],
                                            args.filters[numblock],
                                            1,
                                            stride=1,
                                            padding=0,
                                            groups=args.groups[numblock]))
            liste_fonctions.append(nn.BatchNorm2d(args.filters[numblock]))
            liste_fonctions.append(act())
            input_channel_here = args.filters[numblock]
        #CLASSIFICATION
        liste_fonctions.append(Flatten())
        self.features_before_LR = nn.Sequential(*liste_fonctions)
        fcsize = self.linear_input_neurons(args)
        del self.features_before_LR

        if args.type_weigths_final_LR == "ter":
            liste_fonctions.append(lin(wb3, fcsize, nclass))
        elif args.type_weigths_final_LR == "bin":
            liste_fonctions.append(lin(wb, fcsize, nclass))
        elif args.type_weigths_final_LR == "float":
            liste_fonctions.append(nn.Linear(fcsize, nclass))
        self.feature_pos = len(liste_fonctions)
        liste_fonctions.append(setattr_inplace(
            BatchNormStatsCallbak(
                self, nclass,
                use_scalar_scale=g_use_scalar_scale_last_layer),
            'bias_regularizer_coeff', 0)
        )
        self.features = nn.Sequential(*liste_fonctions)

        if args.g_remove_last_bn=="True":
            self.features = self.features[:-1]
            self.feature_pos = None



    @classmethod
    def make_dataset_loader(cls, args, train: bool):
        if args.dataset == "MNIST":
            dataset = torchvision.datasets.MNIST(
            root=args.data, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        elif args.dataset == "CIFAR10":
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


class model_general_new_version_small(SeqBinModelHelper, nn.Module, ModelHelper):

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
        wb = g_weight_binarizer
        wb3 = g_weight_binarizer3
        act = Binarize01Act
        wb2 = g_weight_binarizer2
        if args.dataset =="MNIST":
            in_channels = 1
        elif args.dataset =="CIFAR10":
            in_channels = 3
        else:
            raise 'PB'


        liste_fonctions = []#OrderedDict([])
        #Preprocessing
        print(args)
        if args.type_weigths_preprocessing_CNN == "ter":
            liste_fonctions.append(BinConv2d(wb3,
                                             in_channels = in_channels,
                                             out_channels = args.preprocessing_CNN[0],
                                             kernel_size = args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[2],
                                             rounding=False))
        elif args.type_weigths_preprocessing_CNN == "bin":
            liste_fonctions.append(BinConv2d(wb,
                                             in_channels = in_channels,
                                             out_channels = args.preprocessing_CNN[0],
                                             kernel_size = args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[2],
                                             rounding=False))
        elif args.type_weigths_preprocessing_CNN == "float":
            liste_fonctions.append(BinConv2d(wb2,in_channels = in_channels,
                                             out_channels = args.preprocessing_CNN[0],
                                             kernel_size = args.preprocessing_CNN[1],
                                             stride=args.preprocessing_CNN[2],
                                             padding=args.preprocessing_CNN[2]))
        else:
            raise 'PB'
        #liste_fonctions.append(nn.BatchNorm2d(args.preprocessing_CNN[0]))
        liste_fonctions.append(act())

        # Blocks
        input_channel_here = args.preprocessing_CNN[0]
        for numblock, _ in enumerate(args.filters):
            liste_fonctions.append(Block_TT_general(input_channel_here, args.filters[numblock], stride=args.strides[numblock],
                                                              big=False))
            input_channel_here = args.filters[numblock]
        #CLASSIFICATION
        liste_fonctions.append(Flatten())
        self.features_before_LR = nn.Sequential(*liste_fonctions)
        fcsize = self.linear_input_neurons(args)
        del self.features_before_LR
        if args.type_weigths_final_LR == "ter":
            liste_fonctions.append(lin(wb3, fcsize, nclass))
        elif args.type_weigths_final_LR == "bin":
            liste_fonctions.append(lin(wb, fcsize, nclass))
        elif args.type_weigths_final_LR == "float":
            liste_fonctions.append(nn.Linear(fcsize, nclass))
        self.feature_pos = len(liste_fonctions)
        liste_fonctions.append(setattr_inplace(
            BatchNormStatsCallbak(
                self, nclass,
                use_scalar_scale=g_use_scalar_scale_last_layer),
            'bias_regularizer_coeff', 0)
        )
        self.features = nn.Sequential(*liste_fonctions)

        if args.g_remove_last_bn=="True":
            self.features = self.features[:-1]
            self.feature_pos = None



    @classmethod
    def make_dataset_loader(cls, args, train: bool):
        if args.dataset == "MNIST":
            dataset = torchvision.datasets.MNIST(
            root=args.data, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
            ]))
        elif args.dataset == "CIFAR10":
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