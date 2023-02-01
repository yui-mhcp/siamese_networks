
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

import pandas as pd

from utils import normalize_filename
from custom_architectures import get_architecture
from custom_architectures.modified_resnet_arch import from_clip_pretrained
from custom_architectures.transformers_arch import get_pretrained_transformer_encoder
from models.siamese.base_comparator import BaseComparator
from models.interfaces.base_text_model import BaseTextModel
from models.interfaces.base_image_model import BaseImageModel

class CLIP(BaseComparator, BaseImageModel, BaseTextModel):
    a_renaming  = 'image'
    b_renaming  = 'text'
    
    def __init__(self,
                 lang,
                 input_size,
                 pretrained     = 'RN50',
                 distance_metric    = 'dp',
                 embed_distance     = False,
                 resize_kwargs      = {'method' : 'bicubic', 'antialias' : True},
                 image_normalization    = 'clip',
                 ** kwargs
                ):
        if pretrained: kwargs.setdefault('text_encoder', 'clip')
        self._init_image(
            input_size = input_size, image_normalization = image_normalization,
            resize_kwargs = resize_kwargs, ** kwargs
        )
        self._init_text(lang = lang, ** kwargs)
        
        kwargs.setdefault('pretrained_name', pretrained)
        kwargs.setdefault('normalize', True)
        kwargs.update({'distance_metric' : distance_metric, 'embed_distance' : embed_distance})
        super().__init__(pretrained = pretrained, ** kwargs)
        
        if hasattr(self.encoder_b, 'set_tokens'): self.encoder_b.set_tokens(** self.model_tokens)
    
    def build_encoder_image(self, embedding_dim = None, normalize = False, pretrained = None,
                            ** kwargs):
        kwargs.update({'output_normalize' : normalize})
        if pretrained is not None:
            if 'rn' in pretrained.lower():
                return from_clip_pretrained(** kwargs)
            elif 'vit' in pretrained.lower():
                return get_pretrained_transformer_encoder(
                    pretrained_name = pretrained, class_name = 'VisualTransformer', ** kwargs
                )
            raise ValueError('Unsupported pretrained CLIP image encoder : {}'.format(pretrained))
        assert embedding_dim, 'You must provide `embedding_dim` argument !'
        return get_architecture(** kwargs)
        
    def build_encoder_text(self, embedding_dim = None, normalize = False, pretrained = None,
                           ** kwargs):
        kwargs.update({'output_normalize' : normalize})
        if pretrained is not None:
            return get_pretrained_transformer_encoder(
                pretrained_name = pretrained, class_name = 'clip', ** kwargs
            )
        assert embedding_dim, 'You must provide `embedding_dim` argument !'
        return get_architecture(** kwargs)
    
    def _build_model(self, normalize = True, pretrained = None, ** kwargs):
        if pretrained is None:
            super()._build_model(normalize = normalize, ** kwargs)
        else:
            super(BaseComparator, self)._build_model(
                comparator  = {
                    'architecture_name' : 'clip',
                    'normalize' : normalize,
                    'distance_metric'   : self.distance_metric,
                    'pretrained_name'   : pretrained,
                    ** kwargs
                }
            )
            self.input_size = self.comparator.input_shape[0][1:]
    
    @property
    def encoder_image_input_signature(self):
        return self.image_signature
    
    @property
    def encoder_text_input_signature(self):
        return self.text_signature

    @property
    def training_hparams(self):
        return super().training_hparams(
            ** self.training_hparams_image, ** self.training_hparams_text
        )

    def __str__(self):
        return super().__str__() + self._str_image() + self._str_text()
    
    def get_input_image(self, data, ** kwargs):
        return self.get_image(data, ** kwargs)
    
    def get_input_text(self, data, ** kwargs):
        if isinstance(data, pd.DataFrame):
            return [self.get_input_text(row, ** kwargs) for idx, row in data.iterrows()]
        elif isinstance(data, list):
            return [self.get_input_text(data_i, ** kwargs) for data_i in data]
        
        return self.tf_encode_text(data, ** kwargs)
    
    def augment_input_image(self, image, ** kwargs):
        return self.augment_image(image, ** kwargs)
    
    def augment_input_text(self, inp, ** kwargs):
        return self.augment_text(inp, ** kwargs)

    def preprocess_input_image(self, image):
        return self.preprocess_image(image)
    
    def embed_image(self, data, ** kwargs):
        data = normalize_filename(data)
        return self.embed(data, use_encoder_a = True, ** kwargs)
    
    def embed_text(self, data, ** kwargs):
        return self.embed(data, use_encoder_a = False, ** kwargs)

    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config.update(self.get_config_image())
        config.update(self.get_config_text())
        
        return config
