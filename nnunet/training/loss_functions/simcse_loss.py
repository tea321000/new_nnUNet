#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
from nnunet.training.loss_functions.dice_loss import SoftDiceLoss


class SimCSELoss(nn.Module):
    def __init__(self, temper = 0.07):
        super(SimCSELoss, self).__init__()
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **{'batch_dice': False, 'smooth': 1e-5, 'do_bg': False})
        self.temper = temper

    def forward(self, sup_outputs, unsup_outputs):
        pos_loss = torch.exp(self.dc(sup_outputs[1][-1], sup_outputs[0][-1]))/self.temper
        neg_loss = 0
        for i in range(len(unsup_outputs)):
            neg_loss += torch.exp(self.dc(unsup_outputs[i], sup_outputs[0][-1]))/self.temper
        return -torch.log(pos_loss/neg_loss)
