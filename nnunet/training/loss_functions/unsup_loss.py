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
        self.l1 = nn.L1Loss()

    def forward(self, output, consistency_counts):
        assert isinstance(consistency_counts, (tuple, list)), "consistency_counts must be either tuple or list"
        l1_loss = 0
        for i in range(len(consistency_counts)):
            for j in range(i + 1, len(consistency_counts)):
                l1_loss += 1 / (len(consistency_counts) * (len(consistency_counts) - 1) / 2) * self.l1(
                    consistency_counts[i], consistency_counts[j])
        return l1_loss
