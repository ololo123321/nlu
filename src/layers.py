import tensorflow as tf
from .utils import add_ones


class MLP(tf.keras.layers.Layer):
    def __init__(self, num_layers, hidden_dim, activation, dropout):
        super().__init__()
        self.dense_layers = [tf.keras.layers.Dense(hidden_dim, activation=activation) for _ in range(num_layers)]
        self.dropout_layers = [tf.keras.layers.Dropout(dropout) for _ in range(num_layers)]

    def call(self, x, training=False):
        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            x = dropout(x, training=training)
        return x


class BiLinear(tf.keras.layers.Layer):
    def __init__(self, left_dim, right_dim, output_dim):
        super().__init__()
        self.w = tf.get_variable("w", shape=(output_dim, left_dim + 1, right_dim + 1), dtype=tf.float32)
        self.u = tf.get_variable("u", shape=(left_dim, output_dim), dtype=tf.float32)
        self.v = tf.get_variable("v", shape=(right_dim, output_dim), dtype=tf.float32)

    def call(self, x_left, x_right):
        x_left_1 = add_ones(x_left)  # [N, T, left_dim + 1]
        x_right_1 = add_ones(x_right)  # [N, T, right_dim + 1]
        x_right_1_t = tf.transpose(x_right_1, [0, 2, 1])  # [N, right_dim + 1, T]
        x_left_u = tf.matmul(x_left, self.u)  # [N, T, output_dim]
        x_right_v = tf.matmul(x_right, self.v)  # [N, T, output_dim]
        scores = tf.expand_dims(x_left_1, [1]) @ self.w @ tf.expand_dims(x_right_1_t, [1])  # [N, output_dim, T, T]
        scores = tf.transpose(scores, [0, 2, 3, 1])  # [N, T, T, output_dim]
        scores += tf.expand_dims(x_left_u, [2])
        scores += tf.expand_dims(x_right_v, [2])
        return scores


class BiAffineAttention(tf.keras.layers.Layer):
    def __init__(self, left_dim, right_dim):
        super().__init__()
        self.w = tf.get_variable("W", shape=(left_dim + 1, right_dim + 1), dtype=tf.float32)

    def call(self, x_left, x_right):
        """
        x_left - tf.Tensor of shape [N, T, left_dim]
        x_right - tf.Tensor of shape [N, T, left_right]
        return: x - tf.Tensor of shape [N, T, T]
        """
        x_left_1 = add_ones(x_left)  # [N, T, left_dim + 1]
        x_right_1 = add_ones(x_right)  # [N, T, right_dim + 1]
        x_right_1_t = tf.transpose(x_right_1, [0, 2, 1])  # [N, right_dim + 1, T]
        x = x_left_1 @ self.w @ x_right_1_t  # [N, T, T]
        return x


class DotProductAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        d_model = kwargs["num_heads"] * kwargs["head_dim"]
        self.mha = MHA(**kwargs)
        self.dense_ff = tf.keras.layers.Dense(kwargs["dff"], activation=tf.nn.relu)
        self.dense_model = tf.keras.layers.Dense(d_model)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.dropout_rc1 = tf.keras.layers.Dropout(kwargs["dropout_rc"])
        self.dropout_rc2 = tf.keras.layers.Dropout(kwargs["dropout_rc"])
        self.dropout_ff = tf.keras.layers.Dropout(kwargs["dropout_ff"])

    def call(self, x, training=False, mask=None):
        x1 = self.mha(x, mask=mask)
        x1 = self.dropout_rc1(x1, training=training)
        x = self.ln1(x + x1)
        x1 = self.dense_ff(x)
        x1 = self.dropout_ff(x1, training=training)
        x1 = self.dense_model(x1)
        x1 = self.dropout_rc2(x1, training=training)
        x = self.ln2(x + x1)
        return x


class MHA(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_heads = kwargs["num_heads"]
        self.head_dim = kwargs["head_dim"]
        self.dense_input = tf.keras.layers.Dense(self.num_heads * self.head_dim * 3)

    def call(self, x, mask=None):
        """
        https://arxiv.org/abs/1706.03762
        :param x: tf.Tensor of shape [N, T, H * D]
        :param mask: tf.Tensor of shape [N, T]
        :return: tf.Tensor of shape [N, T, H * D]
        """
        batch_size = tf.shape(x)[0]
        qkv = self.dense_input(x)  # [N, T, H * D * 3]
        qkv = tf.reshape(qkv, [batch_size, -1, self.num_heads, self.head_dim, 3])  # [N, T, H, D, 3]
        qkv = tf.transpose(qkv, [4, 0, 2, 1, 3])  # [3, N, H, T, D]
        q, k, v = tf.unstack(qkv)  # 3 * [N, H, T, D]

        logits = tf.matmul(q, k, transpose_b=True)  # [N, H, T, T]
        logits /= self.head_dim ** 0.5  # [N, H, T, T]

        mask = mask[:, None, :, None]
        logits += (1. - mask) * -1e9

        w = tf.nn.softmax(logits, axis=-1)  # [N, H, T, T] (k-axis)
        x = tf.matmul(w, v)  # [N, H, T, D]
        x = tf.transpose(x, [0, 2, 1, 3])  # [N, T, H, D]
        x = tf.reshape(x, [batch_size, -1, self.num_heads * self.head_dim])  # [N, T, D * H]
        return x


class REHead(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()

        config_mlp = config["mlp"]
        config_type = config["type"]
        self.mlp_type_d = MLP(config_mlp["num_layers"], config_type["hidden_dim"], tf.nn.relu, config_mlp["dropout"])
        self.mlp_type_h = MLP(config_mlp["num_layers"], config_type["hidden_dim"], tf.nn.relu, config_mlp["dropout"])

        self.bilinear = BiLinear(config_type["hidden_dim"], config_type["hidden_dim"], config_type["num_labels"])

    def call(self, x, training=False):
        # head / dependent projections
        type_d = self.mlp_type_d(x, training)  # [N, T, type_dim], dependent
        type_h = self.mlp_type_h(x, training)  # [N, T, type_dim], head

        # type scores
        s_type = self.bilinear(type_d, type_h)  # [N, T, T, num_arc_labels]

        return s_type
