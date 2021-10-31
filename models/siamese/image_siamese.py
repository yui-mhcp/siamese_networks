import pandas as pd
import tensorflow as tf

from custom_architectures import get_architecture
from models.siamese.siamese_network import SiameseNetwork

class ImageSiamese(SiameseNetwork):
    def __init__(self, input_size, ** kwargs):
        self.input_size = input_size
        
        super().__init__(** kwargs)
    
    def build_encoder(self, embedding_dim = 28, normalize = None, ** kwargs):
        """ Create a simple cnn architecture with default config fitted for MNIST """
        cnn_config = {
            'architecture_name' : 'simple_cnn',
            'input_shape'   : self.input_size,
            'output_shape'  : embedding_dim,

            'n_conv'    : 3,
            'filters'   : [16, 32, 64],
            'kernel_size'   : 3,
            'strides'       : 1,
            'drop_rate'     : 0.,
            'activation'    : 'relu',
            'pooling'       : 'max',
            'bnorm'         : 'never',
                
            'dense_as_final'    : True,
            
            'final_bias'        : True,
            'final_activation'  : None,
            
            'name'  : 'encoder',
            ** kwargs
        }
        return get_architecture(** cnn_config)
    
    @property
    def encoder_input_shape(self):
        input_shape = self.input_size
        if not isinstance(input_shape, (list, tuple)): input_shape = (input_shape, input_shape, 1)
        return (None, ) + tuple(input_shape)
    
    def get_input(self, data):
        """
            Return image / list of images based on the type of `data` 
            
            This function currently not supports loading image from filename but a better image-loading support will come in future version
        """
        if isinstance(data, pd.DataFrame):
            return [self.get_input(d) for _, d in data.iterrows()]
        elif isinstance(data, (list, tuple)):
            return [self.get_input(d) for d in data]
        
        if isinstance(data, (dict, pd.Series)):
            image = tf.cast(data['image'], tf.float32)
        else:
            image = data
        
        if image.dtype != tf.float32:
            image = tf.cast(image, tf.float32)
        if tf.reduce_sum(image) > 1.:
            image = image / 255.
        return image
        
    def get_config(self, *args, ** kwargs):
        config = super().get_config(*args, **kwargs)
        config['input_size'] = self.input_size
        
        return config