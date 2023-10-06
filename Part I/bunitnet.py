import torch
import torch.nn as nn
import numpy as np
from thop import profile
from basic import *
np.random.seed(30)

UNIT_MAP = {'1-2-1': BasicUnit_1_2_1, '1-2-2': BasicUnit_1_2_2, '1-2-3': BasicUnit_1_2_3,
            '2-1-2': BasicUnit_2_1_2, '2-1-1': BasicUnit_2_1_1, '3-2-1': BasicUnit_3_2_1} # key represents the number of output channel

NODE_MAP = {'1-2-1': 1, '1-2-2': 1, '1-2-3': 1, '2-1-2': 2, '2-1-1': 2, '3-2-1': 3}

class BunitNet(nn.Module):
    def __init__(self, input_node, unit_num, class_num):
        super(BunitNet, self).__init__()
        self.input_node = input_node
        self.unit_num = unit_num # the number of basic unit in each tissue layer
        # assert self.unit_num >= 6, print('The number of basic unit should >= 6!')
    
        # build layer with various functional neurons
        self.unit_idx_1, self.node_idx_1, self.unit_layer_1 = self.build_layer(self.unit_num[0])
        self.unit_idx_2, self.node_idx_2, self.unit_layer_2 = self.build_layer(self.unit_num[1])

        # print(self.unit_idx_1)
        # print(self.unit_idx_2)

        self.node_in_1, self.hidden_node_1, self.output_node_1 = self.node_num(self.unit_idx_1)
        self.node_in_2, self.hidden_node_2, self.output_node_2 = self.node_num(self.unit_idx_2)

        # full connected layer
        self.norm_layer_1 = norm_layer(self.input_node, self.node_in_1)
        self.norm_layer_2 = norm_layer(self.output_node_1, self.node_in_2)

        # self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.layer_m = nn.MaxPool2d(kernel_size=2, stride=2)

        # save feature to visualize
        self.fea_dict = {}

        # NormalNet
        # self.layer_2 = norm_layer(self.node_in_1, self.hidden_node_1)
        # self.layer_3 = norm_layer(self.hidden_node_1, self.output_node_1)
        # self.layer_4 = norm_layer(self.output_node_1, self.node_in_2)
        # self.layer_5 = norm_layer(self.node_in_2, self.hidden_node_2)
        # self.layer_6 = norm_layer(self.hidden_node_2, self.output_node_2)

        # self.fc_out = nn.Linear(self.output_node_2, class_num)
        self.fc_out = nn.Linear(7936, class_num)

    # create one layer with various basic units
    def build_layer(sefl, unit_num):
        unit_idx = np.random.choice(list(UNIT_MAP.keys()), size=unit_num) # units with different position and number
        layer = []
        node_idx = []
        for i in unit_idx:
            layer.append(UNIT_MAP[i]())
            node_idx.append(NODE_MAP[i]) # calculate in_node
        layers = nn.ModuleList(layer)
        return unit_idx, node_idx, layers

    def node_num(self, unit_idx_list):
        in_node = unit_idx_list.tolist().count('1-2-1') * 1 + unit_idx_list.tolist().count('1-2-2') * 1 + \
                  unit_idx_list.tolist().count('1-2-3') * 1 + unit_idx_list.tolist().count('2-1-2') * 2 + \
                  unit_idx_list.tolist().count('2-1-1') * 2 + unit_idx_list.tolist().count('3-2-1') * 3
        hid_node = unit_idx_list.tolist().count('1-2-1') * 2 + unit_idx_list.tolist().count('1-2-2') * 2 + \
                   unit_idx_list.tolist().count('1-2-3') * 2 + unit_idx_list.tolist().count('2-1-2') * 1 + \
                   unit_idx_list.tolist().count('2-1-1') * 1 + unit_idx_list.tolist().count('3-2-1') * 2
        out_node = unit_idx_list.tolist().count('1-2-1') * 1 + unit_idx_list.tolist().count('1-2-2') * 2 + \
                   unit_idx_list.tolist().count('1-2-3') * 3 + unit_idx_list.tolist().count('2-1-2') * 2 + \
                   unit_idx_list.tolist().count('2-1-1') * 1 + unit_idx_list.tolist().count('3-2-1') * 1
        return in_node, hid_node, out_node

    # concat the feature of each basic unit output
    def concat(self, x, node_idx, one_layer):
        fea_output = []
        fea_split = torch.split(x, node_idx, dim=1)
        for i in range(len(node_idx)):
            fea_output.append(one_layer[i](fea_split[i]))
        x = torch.cat((fea_output), 1) # the impact of different position of each sub-output
        return x

    def forward(self, x):
        x = self.norm_layer_1(x) # node_in_1 
        self.fea_dict['normal_1'] = x

        # TissueNet
        x = self.concat(x, self.node_idx_1, self.unit_layer_1)
        self.fea_dict['unit_1'] = x

        x = self.layer_m(x)

        x = self.norm_layer_2(x)
        self.fea_dict['normal_2'] = x

        x = self.concat(x, self.node_idx_2, self.unit_layer_2)
        self.fea_dict['unit_2'] = x

        # # NormalNet
        # x = self.layer_2(x)
        # x = self.layer_3(x)
        # x = self.layer_4(x)
        # x = self.layer_5(x)
        # x = self.layer_6(x)

        # common
        # x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc_out(x)
        return x

