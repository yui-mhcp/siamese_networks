
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

from custom_train_objects.generators.siamese_generator import SiameseGenerator
from utils.audio.audio_io import load_audio, load_mel

class AudioSiameseGenerator(SiameseGenerator):
    def __init__(self,
                 dataset,
                 rate,
                 mel_fn = None,
                 ** kwargs
                ):
        self.rate       = rate
        self.mel_fn     = mel_fn
        
        file_col = 'wavs_{}'.format(rate)
        if file_col in dataset.columns: kwargs.setdefault('file_column', file_col)
        kwargs.setdefault('processed_column', 'audio' if mel_fn is None else 'mel')
        
        super().__init__(dataset, ** kwargs)
    
    @property
    def processed_output_shape(self):
        return [None] if self.mel_fn is None else [None, self.mel_fn.n_mel_channels]
    
    def load_file(self, filename):
        if self.mel_fn is not None:
            return load_mel(filename, self.mel_fn)
        return load_audio(filename, self.rate)
