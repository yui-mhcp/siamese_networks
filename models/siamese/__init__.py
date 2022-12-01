
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

from models.siamese.clip import CLIP
from models.siamese.audio_encoder import AudioEncoder
from models.siamese.audio_siamese import AudioSiamese
from models.siamese.image_siamese import ImageSiamese
from models.siamese.text_siamese import TextSiamese

_models = {
    'CLIP'  : CLIP,
    'AudioEncoder'  : AudioEncoder,
    'AudioSiamese'  : AudioSiamese,
    'ImageSiamese'  : ImageSiamese,
    'TextSiamese'   : TextSiamese
}

