import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
from torchvision.models import densenet121
import torch
from torch.autograd import Function
import torch.nn.utils.weight_norm as weightNorm
import models.backbone as backbone
import numpy as np

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True

__all__ = ['ResNet','resnet18', 'resnet50','resnet101', 'MEDM', 'MEDMR50']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


"""Domain Adversarial"""


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def grad_reverse(x, lambd=1.0):
    return ReverseLayerF.apply(x, lambd)


class Discriminator(nn.Module):
    def __init__(self, inc=4096, ndomains=1):
        super(Discriminator, self).__init__()

        self.fc1_1 = nn.Linear(inc, 1024)
        self.bn1_1 = nn.BatchNorm1d(1024)
        self.relu1_1 = nn.ReLU(True)
        self.fc2_1 = nn.Linear(1024, ndomains)
        self.log_soft = nn.LogSoftmax(dim=1)

    def forward(self, x, reverse=True, alpha=1.0):

        if reverse:
            x = grad_reverse(x, alpha)
        x = self.relu1_1(self.bn1_1(self.fc1_1(x)))
        x = self.fc2_1(x)
        out = self.log_soft(x)
        return out


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, 'decay_mult': 2}]


"""ResNet"""


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.baselayer = [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet50(pretrained=False, **kwargs):
    
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model


def resnet101(pretrained=False, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


class BnLabelClassifier(nn.Module):

    def __init__(self, num_classes=12, feature_dim=2048, softmax=True):
        super(BnLabelClassifier, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.5)
        self.fc1 = nn.Linear(feature_dim, 1000)
        self.bn1 = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, num_classes)
        """TO-DO"""
        self.softmax = softmax

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc3(x)
        if self.softmax:
            x = F.softmax(x, dim=1)
        else:
            pass
        return x


class MEDM(nn.Module):
    def __init__(self, num_classes=12):
        super(MEDM, self).__init__()
        self.sharedNet = resnet101(False)
        self.cls_fc = BnLabelClassifier(num_classes=num_classes)

    def forward(self, x):
        x = self.sharedNet(x)
        clabel_pred = self.cls_fc(x)
        return clabel_pred


class MEDMR50(nn.Module):
    def __init__(self, num_classes=12):
        super(MEDMR50, self).__init__()
        self.sharedNet = resnet50(False)
        self.cls_fc = BnLabelClassifier(num_classes=num_classes)

    def forward(self, x):
        x = self.sharedNet(x)
        clabel_pred = self.cls_fc(x)
        return clabel_pred


class MEDM_any(nn.Module):
    def __init__(self, num_classes=12, backbone=None, softmax=True):
        super(MEDM_any, self).__init__()

        if backbone == 'resnet101':
            self.sharedNet = resnet101(False)
            feature_dim = 2048
        elif backbone == 'resnet50':
            self.sharedNet = resnet50(False)
            feature_dim = 2048
        elif backbone == 'resnet18':
            self.sharedNet = resnet18(False)
            feature_dim = 512
        elif backbone == 'densenet121':
            self.sharedNet = densenet121(pretrained=False).features
            feature_dim = 1024
        self.backbone = backbone
        self.cls_fc = BnLabelClassifier(num_classes=num_classes, feature_dim=feature_dim, softmax=softmax)

        self.param_f = self.sharedNet.parameters()
        self.param_c = self.cls_fc.parameters()

    def forward(self, x, return_f=False):
        x = self.sharedNet(x)
        if 'densenet' in self.backbone:
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        clabel_pred = self.cls_fc(x)
        if return_f:
            return clabel_pred, x
        else:
            return clabel_pred


class MEDM_any_2c(nn.Module):
    def __init__(self, num_classes=12, backbone=None):
        super(MEDM_any_2c, self).__init__()

        if backbone == 'resnet101':
            self.sharedNet = resnet101(False)
            feature_dim = 2048
        elif backbone == 'resnet50':
            self.sharedNet = resnet50(False)
            feature_dim = 2048
        elif backbone == 'resnet18':
            self.sharedNet = resnet18(False)
            feature_dim = 512
        elif backbone == 'densenet121':
            self.sharedNet = densenet121(pretrained=False).features
            feature_dim = 1024
        self.backbone = backbone
        self.cls_fc_1 = BnLabelClassifier(num_classes=num_classes, feature_dim=feature_dim, softmax=False)
        self.cls_fc_2 = BnLabelClassifier(num_classes=num_classes, feature_dim=feature_dim, softmax=False)

        self.param_f = self.sharedNet.parameters()
        self.param_c = list(self.cls_fc_1.parameters()) + list(self.cls_fc_2.parameters())

    def forward(self, x, return_f=False):
        x = self.sharedNet(x)
        if 'densenet' in self.backbone:
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)

        clabel_pred_1 = self.cls_fc_1(x)
        clabel_pred_2 = self.cls_fc_2(x)
        if return_f:
            return clabel_pred_1, clabel_pred_2, x
        else:
            return clabel_pred_1, clabel_pred_2


class MEDM_prototype(nn.Module):
    def __init__(self, num_classes=12, feature_dim=2048, bottleneck_dim=256, backbone='resnet101', dropout=0.):
        super(MEDM_prototype, self).__init__()

        if backbone == 'resnet101':
            self.sharedNet = resnet101(False)
        elif backbone == 'resnet50':
            self.sharedNet = resnet50(False)
        elif backbone == 'resnet18':
            self.sharedNet = resnet18(False)
            feature_dim = 512
        elif backbone == 'densenet121':
            self.sharedNet = densenet121(pretrained=False).features
            feature_dim = 1024
        self.backbone = backbone

        self.prototype = nn.Linear(feature_dim, bottleneck_dim)
        self.prototype_bn = nn.BatchNorm1d(bottleneck_dim, affine=True)

        self.cls_fc = weightNorm(nn.Linear(bottleneck_dim, num_classes), name="weight")
        self.cls_fc.apply(init_weights)

        self.dropout = dropout
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

    def forward(self, x, get_f=False):

        x = self.sharedNet(x)
        f = x

        if 'densenet' in self.backbone:
            x = F.relu(x, inplace=True)
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)

        x = self.prototype_bn(self.prototype(x))
        f2 = x
        if self.dropout > 0:
            x = self.drop(x)
        clabel_pred = self.cls_fc(x)
        if get_f:
            return F.softmax(clabel_pred, dim=1), clabel_pred, f, f2
        else:
            return F.softmax(clabel_pred, dim=1), clabel_pred


class MEDM_prototype_saito(nn.Module):
    def __init__(self, num_classes=12, feature_dim=2048, bottleneck_dim=256, backbone='resnet101', dropout=0., temp=0.05):
        super(MEDM_prototype_saito, self).__init__()
        if backbone == 'resnet101':
            self.sharedNet = resnet101(False)
        elif backbone == 'resnet50':
            self.sharedNet = resnet50(False)
        self.prototype = nn.Linear(feature_dim, bottleneck_dim)
        self.prototype_bn = nn.BatchNorm1d(bottleneck_dim, affine=True)

        self.cls_fc = weightNorm(nn.Linear(bottleneck_dim, num_classes), name="weight")
        self.cls_fc.apply(init_weights)
        self.temp = temp

        self.dropout = dropout
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

    def forward(self, x, get_f=False):

        x = self.sharedNet(x)
        f = x
        x = self.prototype_bn(self.prototype(x))
        x = F.normalize(x)
        f2 = x
        if self.dropout > 0:
            x = self.drop(x)
        clabel_pred = self.cls_fc(x) / self.temp
        if get_f:
            return F.softmax(clabel_pred, dim=1), clabel_pred, f, f2
        else:
            return F.softmax(clabel_pred, dim=1), clabel_pred


class MEDM_prototype_rot(nn.Module):
    def __init__(self, num_classes=12, feature_dim=2048, bottleneck_dim=256, backbone='resnet101', dropout=0.):
        super(MEDM_prototype_rot, self).__init__()
        if backbone == 'resnet101':
            self.sharedNet = resnet101(False)
        elif backbone == 'resnet50':
            self.sharedNet = resnet50(False)
        self.prototype = nn.Linear(feature_dim, bottleneck_dim)
        self.prototype_bn = nn.BatchNorm1d(bottleneck_dim, affine=True)

        self.cls_fc = weightNorm(nn.Linear(bottleneck_dim, num_classes), name="weight")
        self.cls_fc.apply(init_weights)

        self.rot_fc = weightNorm(nn.Linear(bottleneck_dim * 2, 4), name="weight")
        self.rot_fc.apply(init_weights)

        self.dropout = dropout
        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

    def forward(self, x, only_f=False):

        x = self.sharedNet(x)
        x = self.prototype_bn(self.prototype(x))
        if self.dropout > 0:
            x = self.drop(x)
        clabel_pred = self.cls_fc(x)

        if only_f:
            return x
        else:
            return F.softmax(clabel_pred, dim=1), clabel_pred


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.1)
        m.bias.data.fill_(0)


class GradientReverseLayer(torch.autograd.Function):
    def __init__(self, iter_num=0, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0):
        self.iter_num = iter_num
        self.alpha = alpha
        self.low_value = low_value
        self.high_value = high_value
        self.max_iter = max_iter

    @staticmethod
    def forward(self, input):
        self.iter_num += 1
        output = input * 1.0
        return output

    @staticmethod
    def backward(self, grad_output):
        self.coeff = np.float(
            2.0 * (self.high_value - self.low_value) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iter)) - (
                    self.high_value - self.low_value) + self.low_value)
        return -self.coeff * grad_output


class MDDNet(nn.Module):
    def __init__(self, base_net='ResNet50', use_bottleneck=True, bottleneck_dim=1024, width=1024, class_num=31):
        super(MDDNet, self).__init__()
        ## set base network
        self.base_net = base_net
        if base_net == 'densenet121':
            self.base_network = densenet121(pretrained=False).features
        else:
            self.base_network = backbone.network_dict[base_net]()

        self.use_bottleneck = use_bottleneck
        # self.grl_layer = GradientReverseLayer()
        self.grl_layer = ReverseLayerF()
        self.bottleneck_layer_list = [nn.Linear(width, bottleneck_dim),
                                      nn.BatchNorm1d(bottleneck_dim), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*self.bottleneck_layer_list)
        self.classifier_layer_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                      nn.Linear(width, class_num)]
        self.classifier_layer = nn.Sequential(*self.classifier_layer_list)
        self.classifier_layer_2_list = [nn.Linear(bottleneck_dim, width), nn.ReLU(), nn.Dropout(0.5),
                                        nn.Linear(width, class_num)]
        self.classifier_layer_2 = nn.Sequential(*self.classifier_layer_2_list)
        self.softmax = nn.Softmax(dim=1)

        ## initialization
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        for dep in range(2):
            self.classifier_layer_2[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer_2[dep * 3].bias.data.fill_(0.0)
            self.classifier_layer[dep * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[dep * 3].bias.data.fill_(0.0)

        ## collect parameters
        self.parameter_list = [{"params": self.base_network.parameters(), "lr": 0.1},
                               {"params": self.bottleneck_layer.parameters(), "lr": 1},
                               {"params": self.classifier_layer.parameters(), "lr": 1},
                               {"params": self.classifier_layer_2.parameters(), "lr": 1}]

    def forward(self, inputs, alpha=1.0):
        features = self.base_network(inputs)

        if 'densenet' in self.base_net:
            features = F.relu(features, inplace=True)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)

        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        features_adv = self.grl_layer.apply(features, alpha)
        outputs_adv = self.classifier_layer_2(features_adv)

        outputs = self.classifier_layer(features)
        softmax_outputs = self.softmax(outputs)

        return features, outputs, softmax_outputs, outputs_adv


class MDD(object):
    def __init__(self, base_net='ResNet50', width=1024, class_num=31, use_bottleneck=True, use_gpu=True, srcweight=3):
        self.c_net = MDDNet(base_net, use_bottleneck, width, width, class_num)

        self.use_gpu = use_gpu
        self.is_train = False
        self.iter_num = 0
        self.class_num = class_num
        if self.use_gpu:
            self.c_net = self.c_net.cuda()
        self.srcweight = srcweight

    def get_loss(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()

        _, outputs, _, outputs_adv = self.c_net(inputs)

        classifier_loss = class_criterion(outputs.narrow(0, 0, labels_source.size(0)), labels_source)

        target_adv = outputs.max(1)[1]
        target_adv_src = target_adv.narrow(0, 0, labels_source.size(0))
        target_adv_tgt = target_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0))

        classifier_loss_adv_src = class_criterion(outputs_adv.narrow(0, 0, labels_source.size(0)), target_adv_src)

        logloss_tgt = torch.log(
            1 - F.softmax(outputs_adv.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0)), dim=1))
        classifier_loss_adv_tgt = F.nll_loss(logloss_tgt, target_adv_tgt)

        transfer_loss = self.srcweight * classifier_loss_adv_src + classifier_loss_adv_tgt

        self.iter_num += 1

        total_loss = classifier_loss + transfer_loss

        return total_loss

    def get_loss_baseline(self, inputs, labels_source):
        class_criterion = nn.CrossEntropyLoss()

        _, outputs, _, outputs_adv = self.c_net(inputs)
        total_loss = class_criterion(outputs.narrow(0, 0, labels_source.size(0)), labels_source)

        return total_loss

    def get_loss_baseline_ent(self, inputs, labels_source, w_ent):
        class_criterion = nn.CrossEntropyLoss()

        _, outputs, _, outputs_adv = self.c_net(inputs)

        loss_ent = entropy(outputs.narrow(0, labels_source.size(0), inputs.size(0) - labels_source.size(0)),
                           lamda=w_ent, use_softmax=True)

        total_loss = class_criterion(outputs.narrow(0, 0, labels_source.size(0)), labels_source) + loss_ent

        return total_loss

    def predict(self, inputs, return_f=False):
        f, _, softmax_outputs, _ = self.c_net(inputs)
        if return_f:
            return softmax_outputs, f
        else:
            return softmax_outputs

    def get_parameter_list(self):
        return self.c_net.parameter_list

    def set_train(self, mode):
        self.c_net.train(mode)
        self.is_train = mode

