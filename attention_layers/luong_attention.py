import tensorflow as tf


class LuongAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        # For Eqn. (4), the  Luong attention
        self.W = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.Attention()

    def call(self, query, value):

        # From Eqn. (4), `W2@hs`.
        w_key = self.W(value)

        context_vector, attention_weights = self.attention(
            inputs=[query, value, w_key],
            return_attention_scores=True,
        )

        return context_vector, attention_weights
