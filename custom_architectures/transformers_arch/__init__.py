
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
import glob

def __load():
    for module_name in glob.glob(os.path.join('custom_architectures', 'transformers_arch/*.py')):
        if os.path.basename(module_name) in ['__init__.py']: continue
        module_name = module_name.replace(os.path.sep, '.')[:-3]

        module = __import__(
            module_name, fromlist = ['custom_objects', 'custom_functions', '_encoders', '_decoders']
        )
        if hasattr(module, 'custom_objects'):
            custom_objects.update(module.custom_objects)
        if hasattr(module, 'custom_functions'):
            custom_functions.update(module.custom_functions)
        if hasattr(module, '_encoders'):
            _encoders.update(module._encoders)
        if hasattr(module, '_decoders'):
            _decoders.update(module._decoders)

def get_pretrained_transformer_encoder(pretrained_name, ** kwargs):
    for name, encoder_class in _encoders.items():
        if name.lower() in pretrained_name:
            return encoder_class.from_pretrained(pretrained_name = pretrained_name, ** kwargs)
    
    raise ValueError("Unknown pretrained class for encoder name {} !".format(pretrained_name))

def get_pretrained_transformer_decoder(pretrained_name, ** kwargs):
    for name, decoder_class in _decoders.items():
        if name.lower() in pretrained_name:
            return decoder_class.from_pretrained(pretrained_name = pretrained_name, ** kwargs)
    
    raise ValueError("Unknown pretrained class for decoder name {} !".format(pretrained_name))

        
custom_objects = {}
custom_functions = {}

_encoders   = {}
_decoders   = {}

__load()

