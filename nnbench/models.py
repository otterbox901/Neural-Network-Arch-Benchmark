"""Architectures under comparison. Each builder returns an uncompiled Keras model.

All models end in a softmax over ``num_classes`` and take a normalised image
tensor of ``input_shape``. They are deliberately sized to train on a 4 GB GPU.
"""

from tensorflow import keras
from tensorflow.keras import layers

ARCHITECTURES = {}


def register(name):
    def wrap(fn):
        ARCHITECTURES[name] = fn
        return fn
    return wrap


def build(name, input_shape, num_classes):
    if name not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {name!r}; choose from {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[name](input_shape, num_classes)


def _conv_bn_relu(x, filters, kernel=3, strides=1):
    x = layers.Conv2D(filters, kernel, strides=strides, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


@register("mlp")
def mlp(input_shape, num_classes):
    """Fully connected baseline: no spatial inductive bias."""
    inp = keras.Input(input_shape)
    x = layers.Flatten()(inp)
    for units in (1024, 512, 256):
        x = layers.Dense(units, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="mlp")


@register("simple_cnn")
def simple_cnn(input_shape, num_classes):
    """Classic LeNet-style CNN: two conv/pool stages and a dense head."""
    inp = keras.Input(input_shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="simple_cnn")


@register("vgg_small")
def vgg_small(input_shape, num_classes):
    """VGG-style: three blocks of stacked 3x3 convs with BatchNorm and dropout."""
    inp = keras.Input(input_shape)
    x = inp
    for filters, drop in ((32, 0.2), (64, 0.3), (128, 0.4)):
        x = _conv_bn_relu(x, filters)
        x = _conv_bn_relu(x, filters)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(drop)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="vgg_small")


def _residual_block(x, filters, strides=1):
    shortcut = x
    y = _conv_bn_relu(x, filters, strides=strides)
    y = layers.Conv2D(filters, 3, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if strides != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    return layers.ReLU()(layers.Add()([y, shortcut]))


@register("resnet_small")
def resnet_small(input_shape, num_classes):
    """ResNet-14-ish: three stages of two residual blocks (He et al., 2016)."""
    inp = keras.Input(input_shape)
    x = _conv_bn_relu(inp, 32)
    for i, filters in enumerate((32, 64, 128)):
        x = _residual_block(x, filters, strides=1 if i == 0 else 2)
        x = _residual_block(x, filters)
    x = layers.GlobalAveragePooling2D()(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="resnet_small")


def _inverted_residual(x, expansion, filters, strides):
    in_ch = x.shape[-1]
    y = _conv_bn_relu(x, in_ch * expansion, kernel=1)
    y = layers.DepthwiseConv2D(3, strides=strides, padding="same", use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU(6.0)(y)
    y = layers.Conv2D(filters, 1, use_bias=False)(y)
    y = layers.BatchNormalization()(y)
    if strides == 1 and in_ch == filters:
        y = layers.Add()([x, y])
    return y


@register("mobilenet_small")
def mobilenet_small(input_shape, num_classes):
    """MobileNetV2-style inverted residuals with depthwise convs (Sandler et al., 2018)."""
    inp = keras.Input(input_shape)
    x = _conv_bn_relu(inp, 32)
    for expansion, filters, repeats, strides in ((1, 16, 1, 1), (6, 32, 2, 1),
                                                 (6, 64, 2, 2), (6, 128, 2, 2)):
        for r in range(repeats):
            x = _inverted_residual(x, expansion, filters, strides if r == 0 else 1)
    x = _conv_bn_relu(x, 256, kernel=1)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inp, out, name="mobilenet_small")
