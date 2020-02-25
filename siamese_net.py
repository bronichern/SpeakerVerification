import torch
from torch import nn
import torch.nn.functional as F
import enum

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.08)


class MetricEnum(enum.Enum):
    CosineSim = 0
    L1Dist = 1
    Euclidean = 2

class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, emb_dim):
        super(EncoderGRU, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        bidirectional = False
        if bidirectional:
            self.hidden_size *=2
        self.rnn = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Linear(self.hidden_size,emb_dim)
        self.linear.apply(init_weights)
        #self.linear2 = nn.Linear(int(self.hidden_size/2), emb_dim)
        self.dp = nn.Dropout(0.01)


    def forward(self, input):
        # self.rnn.flatten_parameters()
        # input = input.view(input.size(0), input.size(1)*input.size(2), input.size(3))
        # input = input.permute(2,0,1)
        speaker_representation, _ = self.rnn(input)
        # speaker_representation = speaker_representation.permute(1,2,0)
        speaker_representation = torch.mean(speaker_representation, 1)
        speaker_representation = speaker_representation.view(speaker_representation.size(0), -1)
        return F.relu(self.linear(speaker_representation))#


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 200, kernel_size=5)
        self.conv2 = nn.Conv2d(200, 32, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(18496, 100)
        # self.conv1 = nn.Conv2d(1, 128, kernel_size=3)
        # self.conv2 = nn.Conv2d(128, 64, kernel_size=3)
        # self.conv2_drop = nn.Dropout2d()
        # self.fc1 = nn.Linear(4032, 100)
        # self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = x.unsqueeze(1)#.unsqueeze(1)
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # x = F.dropout(x,0.3, training=self.training)
        # x = x.view(x.size(0), -1)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, training=self.training)
        return x

class EncoderNet(nn.Module):
    '''
    Networks Code is taken from here: https://github.com/oscarknagg/voicemap/blob/master/voicemap/models.py
    and converted to pytorch.
    '''
    def __init__(self,filters, emb_dim, dropout=0.05):
        super(EncoderNet, self).__init__()
        self.filters = filters
        self.emb_dim = emb_dim
        self.drop = dropout

        self.conv1 = nn.ReLU(nn.Conv2d(1,self.filters, 3))
        self.bn1 = nn.BatchNorm1d(100)
        self.dp = nn.Dropout(p=self.drop)
        self.max1 = nn.MaxPool1d(kernel_size=4, stride=4)

        self.conv2 = nn.ReLU(nn.Conv2d(self.filters,2*self.filters, 3))
        self.bn2 = nn.BatchNorm1d(100)
        self.dp2 = nn.Dropout2d(p=self.drop)
        self.max2 = nn.MaxPool1d(kernel_size=2)

        self.conv3 = nn.ReLU(nn.Conv2d(2*self.filters,3*self.filters, 3))
        self.bn3 = nn.BatchNorm1d(100)
        self.dp3 = nn.Dropout2d(p=self.drop)
        self.max3 = nn.MaxPool1d(kernel_size=2)

        self.conv4= nn.ReLU(nn.Conv2d(3*self.filters,4*self.filters, 3))
        self.bn4 = nn.BatchNorm1d(100)
        self.dp4 = nn.Dropout2d(p=self.drop)
        self.max4 = nn.MaxPool1d(kernel_size=2)
        # binary classification
        #TODO: rep 100 with outdim
        self.linear = nn.Linear(200,self.emb_dim)

    def forward(self, input):
        input = self.max1(self.dp(self.bn1(self.conv1(input))))
        input = self.max2(self.dp2(self.bn2(self.conv2(input)).permute(0, 2, 1)).permute(0, 2, 1))
        input = self.max3(self.dp3(self.bn3(self.conv3(input)).permute(0, 2, 1)).permute(0, 2, 1))
        input = self.max4(self.dp4(self.bn4(self.conv4(input)).permute(0, 2, 1)).permute(0, 2, 1))

        # from (batch_size, steps, features) -> (batch_size, features)
        # input = nn.AdaptiveMaxPool1d(input.size(1))
        input = input.view(input.size(0), -1)
        input = self.linear(input)
        return input

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 31744, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.dropout(out, training=self.training, p=0.3)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

class SiameseNet(nn.Module):
    def __init__(self, distance_metric=MetricEnum.CosineSim):
        super(SiameseNet, self).__init__()
        filters = 128#[16, 32, 64, 128]
        emb_dim = 64#[32, 64, 128, 256, 512]
        dropout = 0.01
        self.encoder = ResNet18(10)
        #self.encoder = EncoderGRU(80, 150, 1, emb_dim)
        # self.lenet = LeNet()
        self.distance_metric = distance_metric
        self.linear = nn.Linear(1,2)

    def forward(self, input1, input2):
        #encoder1 = self.encoder(input1)
        #encoder2 = self.encoder(input2)
        encoder1 = self.encoder(input1.unsqueeze(1))
        encoder2 = self.encoder(input2.unsqueeze(1))

        if self.distance_metric == MetricEnum.L1Dist:
            emb_dist = torch.abs(encoder1.sub(encoder2))
        elif self.distance_metric == MetricEnum.Euclidean:
            emb_dist = torch.sqrt(torch.sum(torch.pow(encoder1.sub(encoder2), 2)))
        elif self.distance_metric == MetricEnum.CosineSim:
            emb_dist = F.cosine_similarity(encoder1.t(),encoder2.t(), dim=0)

        # return F.log_softmax(self.linear(emb_dist), dim=1)
        return self.linear(emb_dist.unsqueeze(1))
