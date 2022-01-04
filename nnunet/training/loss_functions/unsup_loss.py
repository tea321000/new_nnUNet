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


from torch import nn


class UnsupervisedLoss(nn.Module):
    def __init__(self):
        super(UnsupervisedLoss, self).__init__()
        self.loss = nn.SmoothL1Loss()

    def forward(self, consistency_outputs):
        assert isinstance(consistency_outputs, (tuple, list)), "consistency_outputs must be either tuple or list"
        all_loss = 0
        for i in range(len(consistency_outputs)):
            for j in range(i + 1, len(consistency_outputs)):
                all_loss += 1 / (len(consistency_outputs) * (len(consistency_outputs) - 1) / 2) * self.loss(
                    consistency_outputs[i], consistency_outputs[j])
        return all_loss
