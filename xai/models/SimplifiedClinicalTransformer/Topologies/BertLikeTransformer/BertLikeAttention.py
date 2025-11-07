import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
import numpy as np

from ...utils import TokenEmbeddingLayer
from ...utils import MultiHeadAttention
from ...utils import point_wise_feed_forward_network
from ...utils import EncoderLayer
from ...utils import Encoder

def create_padding_mask(seq):
    ''' Mask the pad variables of the input. 
        It ensure that the model does not treat padding as the input. 
        The mask indicates where pad value 0 is present: it outpus
        1 at those locations and 0 otherwise.
        
        This is useful for self training
        
    '''
    seq = tf.cast(tf.math.equal(seq, 0.), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]

def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.

    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
    output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    
    # if mask is not None:
    #     attention_weights = tf.transpose(
    #         tf.transpose(attention_weights, perm=(0, 1, 3, 2))*(1-mask),
    #         perm = (0, 1, 3, 2)
    #     )
    
    output = tf.matmul(attention_weights, v, transpose_a=False)  # (..., seq_len_q, depth_v)

    return output, attention_weights

class NumericalEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size=64):
        super(NumericalEmbeddingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.y = tf.keras.layers.Dense(embedding_size, use_bias=True)
        
    def call(self, x):
        return self.y(x)

class PriorEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_size=64):
        super(PriorEmbeddingLayer, self).__init__()
        self.embedding_size = embedding_size
        self.y = tf.keras.layers.Dense(embedding_size, use_bias=True)
        
    def call(self, x):
        return self.y(x)

class SurvivalTransformer(tf.keras.Model):
    '''

    This architecture does a modification to the attention layers. 
    It adds 0's to the padding attention weights in the columns and rows. 
    This ensures that the model does not look at the <pad> tokens. 

    model = SurvivalTransformer(
        num_layers=2, d_model=8, num_heads=2, dff=128, num_classes=1, num_features=5, embedding_size=8, max_features=5, rate=0.3, masking=True, with_prior=False
    )
    
    X = [
        tf.Variable([[1, 1, 1, 0, 0.]]), 
        tf.Variable([[1, 2, 0, 0, 0.]]), 
        tf.Variable([[]])
    ]
    
    mask = create_padding_mask(X[1])
    O, W, V, Q, K = model(X, mask=mask, training=True)
    pd.DataFrame(W[0][0, 0, :, :].numpy())
    pd.DataFrame(V[0].numpy())
    
    '''
    
    def __init__(self, num_layers, d_model, num_heads, dff, num_classes, num_features, embedding_size, max_features, rate=0.3, masking=True, with_prior=True):

        super(SurvivalTransformer, self).__init__()
        
        self.with_prior = with_prior
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, num_features, embedding_size, max_features, rate, with_prior=self.with_prior)
        
        self.final_layer = tf.keras.layers.Dense(num_classes, use_bias=False, activation='linear')
        # self.risk_layer = tf.keras.layers.Dense(num_classes, use_bias=False)
        
        self.masking = masking
    
    def call(self, inp, training=True):
        
        if self.masking:
            mask = create_padding_mask(inp[1])
        else:
            mask = None
        
        enc_output, attention_weights, q, k,  = self.encoder(inp, training=training, mask=mask)  # (batch_size, inp_seq_len, d_model)
        
        # risks = self.risk_layer(enc_output)[:, 0, :] # take first "feature / token <surv>"
        # final_output= self.final_layer(risks)  # (batch_size, tar_seq_len, target_vocab_size)

        # bert like output directly from embeddings
        risks = enc_output[:, 0, :]
        final_output = self.final_layer(risks)
        
        return final_output, attention_weights, enc_output, risks, k
    
    def train_step(self, data):
        X, y = data

        with tf.GradientTape() as tape:
            y_pred, _, _, _, _ = self(X, training=True)
            
            y_predx = tf.math.divide(
                tf.math.unsorted_segment_sum(y_pred, X[3][:, 0], X[4][0]),
                2.0
            )
            
            loss = self.compiled_loss(y, y_predx, regularization_losses=self.losses)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_predx)
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        X, y = data
            
        y_pred, _, _, _, _ = self(X, training=False)
        y_predx = tf.math.divide(
            tf.math.unsorted_segment_sum(y_pred, X[3][:, 0], X[4][0]),
            2.0
        )

        self.compiled_loss(y, y_predx, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_predx)
        
        return {m.name: m.result() for m in self.metrics}

class SelfSupervisedTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, num_classes, num_features, embedding_size, max_features, rate=0.3, masking=False, with_prior=True):

        super(SelfSupervisedTransformer, self).__init__()
        
        self.with_prior = with_prior
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, num_features, embedding_size, max_features, rate, with_prior=self.with_prior)
        
        self.final_regression = tf.keras.layers.Dense(1, activation='linear', name='value_output')
        self.final_layer= tf.keras.layers.Dense(num_classes)
        self.out = tf.keras.layers.Softmax(name='token_output')
        
        self.masking = masking
    
    def call(self, inp, training):
        
        if self.masking:
            mask = create_padding_mask(inp[1])
        else:
            mask = None
        
        enc_output, attention_weights, q, k = self.encoder(inp, training=training, mask=mask)  # (batch_size, inp_seq_len, d_model)
        final_output= self.out( self.final_layer( enc_output ), )  # (batch_size, tar_seq_len, target_vocab_size)

        regression_output = self.final_regression( enc_output, )

        # token prediction, value prediction, attention weights, embeddings from encoder
        return final_output, regression_output, attention_weights, enc_output
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        X, y = data
        
        with tf.GradientTape() as tape:
            y_pred, v_pred, _, _, = self(X, training=True)  # Forward pass

            # loss = self.compiled_loss(y, y_pred)
            pred = tf.concat([y_pred, v_pred], 2)
            loss = self.compiled_loss(y, pred)
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        # self.compiled_metrics.update_state(y[:, :, 1], y_pred)
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        X, y = data
            
        y_pred, v_pred, _, _, = self(X, training=False)
        pred = tf.concat([y_pred, v_pred], 2)
        loss = self.compiled_loss(y, pred)
                
        return {m.name: m.result() for m in self.metrics}

class ValueMaskedSelfSupervisedTransformer(tf.keras.Model):
    def __init__(
        self,
        num_layers,
        d_model,
        num_heads,
        dff,
        num_classes,
        num_features,
        embedding_size,
        max_features,
        rate=0.3,
        masking=True,
        with_prior=True,
    ):

        super(ValueMaskedSelfSupervisedTransformer, self).__init__()

        self.with_prior = with_prior
        self.encoder = Encoder(
            num_layers,
            d_model,
            num_heads,
            dff,
            num_features,
            embedding_size,
            max_features,
            rate,
            with_prior=self.with_prior,
        )

        self.value_head = tf.keras.layers.Dense(
            1, activation="linear", name="value_output"
        )
        self.masking = masking

    def _build_value_attention_mask(self, value_mask):
        vm = tf.cast(value_mask, tf.float32)
        vm_q = vm[:, :, tf.newaxis]  # (batch, seq, 1)
        vm_k = vm[:, tf.newaxis, :]  # (batch, 1, seq)
        pair_mask = vm_q * vm_k  # (batch, seq, seq)
        return pair_mask[:, tf.newaxis, :, :]

    def call(self, inp, training):
        value_mask = inp[-1]
        base_inp = list(inp[:-1])

        values = base_inp[0]
        masked_values = tf.where(
            tf.equal(value_mask, 1.0), tf.zeros_like(values), values
        )
        base_inp[0] = masked_values

        mask = self._build_value_attention_mask(value_mask)
        if self.masking:
            pad_mask = create_padding_mask(base_inp[1])
            mask = mask + pad_mask

        enc_output, attention_weights, q, k = self.encoder(
            base_inp, training=training, mask=mask
        )

        value_output = self.value_head(enc_output)
        return value_output, attention_weights, enc_output

    def train_step(self, data):
        X, y = data

        with tf.GradientTape() as tape:
            value_pred, _, _ = self(X, training=True)
            loss = self.compiled_loss(y, value_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        for metric in self.metrics:
            metric.update_state(
                y[:, :, 1:], value_pred, sample_weight=y[:, :, 0]
            )

        logs = {m.name: m.result() for m in self.metrics}
        logs["loss"] = loss
        return logs

    def test_step(self, data):
        X, y = data
        value_pred, _, _ = self(X, training=False)
        loss = self.compiled_loss(y, value_pred)

        for metric in self.metrics:
            metric.update_state(
                y[:, :, 1:], value_pred, sample_weight=y[:, :, 0]
            )

        logs = {m.name: m.result() for m in self.metrics}
        logs["loss"] = loss
        return logs

class ClassifierTransformer(tf.keras.Model):
    '''
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, num_classes, num_features, embedding_size, max_features, rate=0.3, masking=False, with_prior=True):

        super(ClassifierTransformer, self).__init__()

        self.with_prior = with_prior
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, num_features, embedding_size, max_features, rate, with_prior=self.with_prior)
        # self.flat = tf.keras.layers.Flatten()
        self.final_layer= tf.keras.layers.Dense(num_classes)
        self.out = tf.keras.layers.Softmax()
        
        self.masking = masking
    
    def call(self, inp, training):

        if self.masking:
            mask = create_padding_mask(inp[1])
        else:
            mask = None

        enc_output, attention_weights, q, k = self.encoder(inp, training=training, mask=mask)  # (batch_size, inp_seq_len, d_model)
        final_output= self.out( self.final_layer( enc_output ) )
    
        return final_output[:, 0, :], attention_weights, enc_output # prediction, attention, embeddings


    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        X, y = data
        
        with tf.GradientTape() as tape:
            y_pred, _, _, = self(X, training=True)  # Forward pass
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            # print(y.shape, y_pred.shape)S
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred[:, 1:])
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        X, y = data
            
        y_pred, _, _, = self(X, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred[:, 1:])
        
        return {m.name: m.result() for m in self.metrics}


#################################################
# Leave this section for experimental models only.
# Experimental models
class SurvivalAndClassifierTransformer(tf.keras.Model):
    '''

    This architecture does a modification to the attention layers. 
    It adds 0's to the padding attention weights in the columns and rows. 
    This ensures that the model does not look at the <pad> tokens. 

    model = SurvivalTransformer(
        num_layers=2, d_model=8, num_heads=2, dff=128, num_classes=1, num_features=5, embedding_size=8, max_features=5, rate=0.3, masking=True, with_prior=False
    )
    
    X = [
        tf.Variable([[1, 1, 1, 0, 0.]]), 
        tf.Variable([[1, 2, 0, 0, 0.]]), 
        tf.Variable([[]])
    ]
    
    mask = create_padding_mask(X[1])
    O, W, V, Q, K = model(X, mask=mask, training=True)
    pd.DataFrame(W[0][0, 0, :, :].numpy())
    pd.DataFrame(V[0].numpy())
    
    '''
    
    def __init__(self, num_layers, d_model, num_heads, dff, num_classes, num_features, embedding_size, max_features, rate=0.3, masking=True, with_prior=True):

        super(SurvivalAndClassifierTransformer, self).__init__()
        
        self.with_prior = with_prior
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, num_features, embedding_size, max_features, rate, with_prior=self.with_prior)
        
        self.final_layer = tf.keras.layers.Dense(num_classes, use_bias=False, activation='linear')
        self.dense_classifier = tf.keras.layers.Dense(2, use_bias=True, activation='relu')
        self.out_classifier = tf.keras.layers.Softmax()
        
        self.masking = masking
    
    def call(self, inp, training=True):
        
        if self.masking:
            mask = create_padding_mask(inp[1])
        else:
            mask = None
        
        enc_output, attention_weights, q, k,  = self.encoder(inp, training=training, mask=mask)  # (batch_size, inp_seq_len, d_model)
        
        # risks = self.risk_layer(enc_output)[:, 0, :] # take first "feature / token <surv>"
        # final_output= self.final_layer(risks)  # (batch_size, tar_seq_len, target_vocab_size)

        # bert like output directly from embeddings
        risks = enc_output[:, 0, :]
        
        classifier = self.out_classifier(self.dense_classifier(risks))
        final_output = self.final_layer(risks)

        return final_output, classifier, attention_weights, enc_output, risks, k
    
    def train_step(self, data):
        X, y = data

        with tf.GradientTape() as tape:
            y_pred, y_cls, _, _, _, _ = self(X, training=True)

            pred = tf.concat([y_pred, y_cls], 1)
            # print(pred.shape)

            loss = self.compiled_loss(y, pred)
        
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred[:, :1])
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        X, y = data
            
        y_pred, y_cls, _, _, _, _ = self(X, training=True)    
        pred = tf.concat([y_pred, y_cls], 1)
        loss = self.compiled_loss(y, pred)

        self.compiled_metrics.update_state(y, y_pred[:, :1])
        
        return {m.name: m.result() for m in self.metrics}

class MultiTaskingSurvivalTransformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, num_classes, num_features, embedding_size, max_features, rate=0.3, masking=False, with_prior=True, *args, **kwargs):

        super(MultiTaskingSurvivalTransformer, self).__init__()
        
        self.with_prior = with_prior
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, num_features, embedding_size, max_features, rate, with_prior=self.with_prior)
        
        self.final_regression = tf.keras.layers.Dense(1, activation='linear', name='value_output')
        self.final_layer= tf.keras.layers.Dense(num_features)
        self.out = tf.keras.layers.Softmax(name='token_output')
        self.survival_output = tf.keras.layers.Dense(1, use_bias=False, activation='linear', name='Survival_output')
        
        self.masking = masking
    
    def call(self, inp, training):
        
        if self.masking:
            mask = create_padding_mask(inp[1])
        else:
            mask = None
        
        enc_output, attention_weights, q, k = self.encoder(inp, training=training, mask=mask)  # (batch_size, inp_seq_len, d_model)
        
        # predict tokens
        final_output= self.out( self.final_layer( enc_output ), )  # (batch_size, tar_seq_len, target_vocab_size)

        # predict token values
        regression_output = self.final_regression( enc_output, )

        # predict survival
        survival_output = tf.math.divide(self.survival_output(enc_output), 32)

        return final_output, regression_output, survival_output, attention_weights, enc_output
    
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        X, y = data
        
        with tf.GradientTape() as tape:
            y_pred, v_pred, s_pred, _, _ = self(X, training=True)  # Forward pass
            pred = tf.concat([y_pred, v_pred], 2)
            pred = tf.concat([pred, s_pred], 2)
            
            loss = self.compiled_loss(y, pred)
            
        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y[:, 0, 4:], s_pred[:, 0, :])
        
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        X, y = data
            
        y_pred, v_pred, s_pred, _, _, = self(X, training=False)
        pred = tf.concat([y_pred, v_pred], 2)
        pred = tf.concat([pred, s_pred], 2)
        
        loss = self.compiled_loss(y, pred)
        self.compiled_metrics.update_state(y[:, 0, 4:], s_pred[:, 0, :])
        
        return {m.name: m.result() for m in self.metrics}
