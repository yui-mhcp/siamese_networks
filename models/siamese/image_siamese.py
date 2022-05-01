
# Copyright (C) 2022 yui-mhcp project's author. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from custom_architectures import get_architecture
from models.siamese.siamese_network import SiameseNetwork
from models.interfaces.base_image_model import BaseImageModel

class ImageSiamese(BaseImageModel, SiameseNetwork):
    def __init__(self, input_size, ** kwargs):
        self._init_image(input_size = input_size, ** kwargs)
        
        super().__init__(** kwargs)
    
    def build_encoder(self, depth = 64, embedding_dim = 28, normalize = None, ** kwargs):
        """ Create a simple cnn architecture with default config fitted for MNIST """
        cnn_config = {
            'architecture_name'     : 'simple_cnn', 
            'input_shape'   : self.input_size,
            'output_shape'  : embedding_dim,
            'n_conv'    : 5,
            'filters'   : [
                depth, 
                depth * 2, 
                [depth * 2, depth * 4],
                [depth * 4, depth * 4],
                [depth * 8, depth * 4]
            ],
            'strides'   : [2, 2, 1, [2, 1], 1],
            'kernel_size'   : [
                7, 5, 3, [3, 1], [3, 1]
            ],
            'residual'      : False,
            'flatten'       : True,
            'flatten_type'  : 'avg',
            'dense_as_final' : True,
            'name'  : 'Encoder',
            ** kwargs
        }
        if cnn_config['architecture_name'] != 'simple_cnn':
            cnn_config['final_activation'] = 'l2_norm'
        return get_architecture(** cnn_config)
    
    @property
    def encoder_input_shape(self):
        return (None, ) + self.input_size
    
    @property
    def training_hparams(self):
        return super().training_hparams(** self.training_hparams_image)

    def __str__(self):
        return super().__str__() + self._str_image()
    
    def get_input(self, data, ** kwargs):
        return self.get_image(data)
    
    def augment_input(self, image):
        return self.augment_image(image)
    
    def preprocess_input(self, image):
        return self.preprocess_image(image)
    
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update(self.get_config_image())
        
        return config
