from collections import namedtuple
import tensorflow as tf


class MLP(tf.keras.layers.Layer):
    def __init__(self, num_layers, hidden_dim, activation, dropout):
        super().__init__()
        self.dense_layers = [tf.keras.layers.Dense(hidden_dim, activation=activation) for _ in range(num_layers)]
        self.dropout_layers = [tf.keras.layers.Dropout(dropout) for _ in range(num_layers)]

    def call(self, x: tf.Tensor, training: bool = False):
        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            x = dropout(x, training=training)
        return x


BiLinearInputs = namedtuple("BiLinearInputs", ["head", "dep"])


class BiLinear(tf.keras.layers.Layer):
    """
    Билинейная форма:
    x = a*w*b^T + a*u + b*v + bias, где
    tensor name     shape
    a               [N, T, D_a]
    b               [N, T, D_b]
    w               [D_out, D_a, D_b]
    u               [D_a, D_out]
    v               [D_b, D_out]
    bias            [D_out]
    """
    def __init__(self, head_dim, dep_dim, output_dim):
        super().__init__()
        self.w = tf.get_variable("w", shape=(output_dim, head_dim, dep_dim), dtype=tf.float32)
        self.u = tf.get_variable("u", shape=(head_dim, output_dim), dtype=tf.float32)
        self.v = tf.get_variable("v", shape=(dep_dim, output_dim), dtype=tf.float32)
        self.b = tf.get_variable("b", shape=(output_dim,), dtype=tf.float32, initializer=tf.initializers.zeros())

    def call(self, inputs: BiLinearInputs, training=None, mask=None) -> tf.Tensor:
        """
        head - tf.Tensor of shape [N, T, D_head] and type tf.float32
        bep - tf.Tensor as shape [N, T, D_dep] and type tf.float32
        :returns x - tf.Tensor of shape [N, T, T, output_dim] and type tf.float32
        """
        head = inputs.head
        dep = inputs.dep
        dep_t = tf.transpose(dep, [0, 2, 1])  # [N, right_dim, T]
        x = tf.expand_dims(head, 1) @ self.w @ tf.expand_dims(dep_t, 1)  # [N, output_dim, T, T]
        x = tf.transpose(x, [0, 2, 3, 1])  # [N, T, T, output_dim]

        head_u = tf.matmul(head, self.u)  # [N, T, output_dim]
        x += tf.expand_dims(head_u, 2)  # [N, T, T, output_dim]

        dep_v = tf.matmul(dep, self.v)  # [N, T, output_dim]
        x += tf.expand_dims(dep_v, 2)  # [N, T, T, output_dim]

        x += self.b[None, None, None, :]  # [N, T, T, output_dim]
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


GraphEncoderInputs = namedtuple("GraphEncoderInputs", ["head", "dep"])


class GraphEncoder(tf.keras.layers.Layer):
    """
    кодирование пар вершин
    """
    def __init__(
            self,
            num_mlp_layers: int,
            head_dim: int,
            dep_dim: int,
            output_dim: int,
            dropout: float = 0.2,
            activation: str = "relu"
    ):
        super().__init__()

        # рассмотрим ребро a -> b

        # векторное представление вершины a
        self.mlp_head = MLP(
            num_layers=num_mlp_layers,
            hidden_dim=head_dim,
            activation=activation,
            dropout=dropout
        )
        # векторное представление вершины b
        self.mlp_dep = MLP(
            num_layers=num_mlp_layers,
            hidden_dim=dep_dim,
            activation=activation,
            dropout=dropout
        )
        # кодирование рёбер a -> b
        self.bilinear = BiLinear(
            head_dim=head_dim,
            dep_dim=dep_dim,
            output_dim=output_dim
        )

    def call(self, inputs: GraphEncoderInputs, training: bool = False):
        head, dep = inputs  # чтоб не отходить от API
        head = self.mlp_head(head, training=training)  # [N, num_heads, type_dim]
        dep = self.mlp_dep(dep, training=training)  # [N, num_deps, type_dim]
        bilinear_inputs = BiLinearInputs(head=head, dep=dep)
        logits = self.bilinear(inputs=bilinear_inputs)  # [N, num_heads, num_deps, num_arc_labels]
        return logits
