import tensorflow as tf
from tensorflow.keras.losses import Loss


class ValueMaskLoss(Loss):
    """
    Mean-squared error that is evaluated only on masked positions.
    """

    def __init__(self, name="value_mask_loss"):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE)
        self.mse = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )

    def call(self, y_true, y_pred):
        """
        y_true: [..., 2] -> [:, :, 0] = mask indicator, [:, :, 1] = target value
        y_pred: [..., 1] value predictions
        """
        mask = tf.cast(y_true[:, :, 0], tf.float32)
        targets = y_true[:, :, 1:]

        loss = self.mse(targets, y_pred)
        weighted = loss * mask

        denom = tf.reduce_sum(mask, axis=1) + 1e-8
        return tf.reduce_sum(weighted, axis=1) / denom


loss = ValueMaskLoss()
