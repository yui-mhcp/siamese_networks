import tensorflow as tf

from hparams.hparams import HParams
from custom_layers import get_activation
from custom_architectures.current_blocks import _get_layer, _get_pooling_layer

HParamsEmbeddingHead   = HParams(
    output_dim = -1,
    process_first_token = False,
    
    hidden_dim  = 0,
    hidden_activation   = None,
    hidden_drop_rate    = 0.1,
    hidden_layer_type   = 'bi_lstm',
    hidden_layer_kwargs = {},
    
    final_pooling   = None,
    use_final_dense = True,
    final_name      = 'output_layer',
    final_activation    = None,
    normalize   = False     # for embedding l2-normalization support (siamese networks)
)

class EmbeddingHead(tf.keras.layers.Layer):
    def __init__(self, output_dim, name = None, ** kwargs):
        super().__init__(name = name)
        kwargs.update({'output_dim' : output_dim})
        self.hparams = HParamsEmbeddingHead.extract(kwargs)

        # Layers are applied in this order (initialized only if required)
        self.hidden_layer   = None
        self.hidden_act_layer   = None
        self.hidden_drop_layer  = None
        self.final_pooling  = None
        self.concat_layer   = None
        self.final_dense    = None
        self.final_act_layer    = None
        
        if not self.hparams.use_final_dense or self.hparams.hidden_dim > 0:
            if self.hparams.use_final_dense:
                units   = self.hparams.hidden_dim
                name    = 'hidden_layer'
                act     = self.hparams.hidden_activation
                drop    = self.hparams.hidden_drop_rate
            else:
                units, name, act, drop = output_dim, self.hparams.final_name, self.hparams.final_activation, 0.

            if units > 0:
                self.hidden_layer = _get_layer(
                    self.hparams.hidden_layer_type, units, name = name, ** self.hparams.hidden_layer_kwargs
                )
                self.hidden_act_layer   = get_activation(act)
                if drop > 0: self.hidden_drop_layer = tf.keras.layers.Dropout(drop)
        
        if self.hparams.final_pooling:
            self.final_pooling = _get_pooling_layer(
                self.hparams.final_pooling, dim = '1d', global_pooling = True
            )
            if isinstance(self.hparams.final_pooling, (list, tuple)):
                self.concat_layer   = tf.keras.layers.Concatenate(axis = -1)
        
        if self.hparams.use_final_dense and output_dim > 0:
            self.final_dense    = tf.keras.layers.Dense(output_dim, name = self.hparams.final_name)
            self.final_act_layer    = get_activation(self.hparams.final_activation)

    def call(self, inputs, mask = None, training = False):
        output = inputs if not self.hparams.process_first_token else inputs[:, 0, :]
        
        if mask is not None and not self.hparams.process_first_token:
            mask = tf.cast(1. - tf.reshape(mask, [tf.shape(output)[0], tf.shape(output)[1]]), tf.bool)
        
        if self.hidden_layer is not None:
            if self.hparams.hidden_layer_type != 'dense':
                output  = self.hidden_layer(output, mask = mask, training = training)
            else:
                output = self.hidden_layer(output, training = training)
            if self.hidden_act_layer is not None:
                output = self.hidden_act_layer(output)
            if self.hidden_drop_layer is not None:
                output = self.hidden_drop_layer(output, training = training)
        
        
        if self.final_pooling is not None:
            if isinstance(self.final_pooling, (list, tuple)):
                pooled = []
                for pool_layer, pool_type in zip(self.final_pooling, self.hparams.final_pooling):
                    masking = {} if pool_type == 'max' else {'mask' : mask}
                    pooled.append(pool_layer(output, ** masking))
                output = self.concat_layer(pooled)
            else:
                masking = {} if self.hparams.final_pooling == 'max' else {'mask' : mask}
                output = self.final_pooling(output, ** masking)
        
        
        if self.final_dense is not None: output = self.final_dense(output)
        if self.final_act_layer is not None: output = self.final_act_layer(output)
        
        if self.hparams.normalize: output = tf.math.l2_normalize(output, axis = -1)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        return (self.hparams + config).get_config()
