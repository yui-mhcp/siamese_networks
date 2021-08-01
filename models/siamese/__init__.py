from models.siamese.audio_siamese import AudioSiamese
from models.siamese.image_siamese import ImageSiamese
from models.siamese.text_siamese import TextSiamese

_models = {
    'AudioSiamese'  : AudioSiamese,
    'ImageSiamese'  : ImageSiamese,
    'TextSiamese'   : TextSiamese
}

