# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np


LEVIR_PALETTE = np.array([
    [0, 0, 0], [255, 0, 0]], dtype=np.uint8)

WHU_PALETTE = np.array([
    [0, 0, 0], [255, 0, 0]], dtype=np.uint8)


def get_palette(dataset):
    if dataset == 'levir':
        return LEVIR_PALETTE
    elif dataset == 'whu':
        return WHU_PALETTE
    else:
        raise ValueError(dataset)