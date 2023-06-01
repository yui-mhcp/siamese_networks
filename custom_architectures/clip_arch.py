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

import os
import tensorflow as tf

from utils import download_file

CLIP_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}

_clip_loaded    = {}

def load_clip(pretrained_name = 'RN50', pretrained = None, ** kwargs):
    global _clip_loaded
    
    if pretrained is None:
        from models import _pretrained_models_folder
        
        if pretrained_name not in _clip_loaded:
            import torch

            if pretrained_name not in CLIP_MODELS:
                raise ValueError('Unknown pretrained CLIP model !\n  Accepted : {}\n  Got : {}'.format(
                    tuple(CLIP_MODELS.keys()), pretrained_name
                ))

            filename = download_file(
                CLIP_MODELS[pretrained_name],
                directory = os.path.join(_pretrained_models_folder, 'pretrained_weights')
            )

            if filename is None:
                raise RuntimeError('filename is None, an error has occured while loading it')

            pretrained  = torch.jit.load(filename, map_location = 'cpu').state_dict()
            _clip_loaded[pretrained_name] = pretrained
        
        pretrained = _clip_loaded[pretrained_name]
    
    state_dict = pretrained if isinstance(pretrained, dict) else pretrained.state_dict()
    
    return state_dict

def CLIP(pretrained_name,
         pretrained     = None,
         normalize      = True,
         distance_metric    = 'dp',
         name = 'CLIP',
         ** kwargs
        ):
    """
        Builds a CLIP model as a `tf.keras.Model` instance
        The input signature is :
            - (input_dim, input_dim, 3) : for the input image (input_dim is determined by the pretrained model's input shape)
            - (seq_len, ) with tf.int32 : for the input tokens
    """
    from custom_architectures.simple_models import comparator
    from custom_architectures.modified_resnet_arch import from_clip_pretrained
    from custom_architectures.transformers_arch.clip_encoders_arch import (
        CLIPTextEncoder, CLIPImageEncoder
    )
    
    state_dict  = load_clip(pretrained_name, pretrained = pretrained)

    kwargs.update({'output_normalize' : normalize})
    if 'rn' in pretrained_name.lower():
        visual_encoder  = from_clip_pretrained(
            pretrained = state_dict, name = 'image_encoder', ** kwargs
        )
        input_dim   = visual_encoder.input_shape[1]
    elif 'vit' in pretrained_name.lower():
        visual_encoder  = CLIPImageEncoder.from_pretrained(
            pretrained_name = pretrained_name,
            pretrained  = state_dict,
            name = 'image_encoder',
            ** kwargs
        )
        input_dim   = visual_encoder.input_dim

    text_encoder    = CLIPTextEncoder.from_pretrained(
        pretrained_name = pretrained_name,
        pretrained  = state_dict,
        name    = 'text_encoder',
        ** kwargs
    )

    clip   = comparator(
        encoder_a   = visual_encoder,
        encoder_b   = text_encoder,
        input_signature_a   = tf.TensorSpec(
            shape = (None, input_dim, input_dim, 3), dtype = tf.float32
        ),
        input_signature_b   = tf.TensorSpec(shape = (None, None), dtype = tf.int32),
        distance_metric = distance_metric,
        name    = name
    )
    if hasattr(text_encoder, 'set_tokens'):
        clip.set_tokens = text_encoder.set_tokens
    
    clip.layers[-1].set_weights([
        tf.reshape(state_dict['logit_scale'].exp().detach().numpy(), [1, 1]),
        tf.zeros_like(clip.layers[-1].get_weights()[1])
    ])
    
    return clip

custom_functions    = {'CLIP' : CLIP}