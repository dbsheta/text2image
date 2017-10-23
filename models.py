import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn


class Embedding(gluon.Block):
    def __init__(self, vocab_size, embed_size, text_dims, **kwargs):
        super(Embedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        with self.name_scope():
            self.embed = nn.Embedding(input_dim=vocab_size, output_dim=embed_size)
            self.conv1d = nn.Conv1D(64, kernel_size=5, activation='relu')
            self.fc = nn.Dense(text_dims)

    def forward(self, x, *args, **kwargs):
        x = self.embed(x)
        x = self.conv1d(x)
        x = self.fc(x)
        return x


class Generator(gluon.Block):
    def __init__(self, n_dims=128, **kwargs):
        super(Generator, self).__init__(**kwargs)
        self.n_dims = n_dims
        with self.name_scope():
            # layers created in name_scope will inherit name space
            # from parent layer.
            self.fc1 = nn.Dense(n_dims * 8 * 4 * 4)
            self.fc1_bnorm = nn.BatchNorm()
            self.fc1_act = nn.Activation('relu')

            self.deconv1 = nn.Conv2DTranspose(n_dims * 4, 4, 2, 1)
            self.deconv2 = nn.Conv2DTranspose(n_dims * 2, 4, 2, 1)
            self.deconv3 = nn.Conv2DTranspose(n_dims, 4, 2, 1)
            self.deconv4 = nn.Conv2DTranspose(3, 4, 2, 1)

            self.deconv1_bnorm = nn.BatchNorm()
            self.deconv2_bnorm = nn.BatchNorm()
            self.deconv3_bnorm = nn.BatchNorm()
            self.deconv4_bnorm = nn.BatchNorm()

            self.deconv1_relu = nn.Activation('relu')
            self.deconv2_relu = nn.Activation('relu')
            self.deconv3_relu = nn.Activation('relu')
            self.deconv4_tanh = nn.Activation('tanh')

    def forward(self, x, *args, **kwargs):
        _noise = x[0]
        _text_embed = x[1]
        x = mx.nd.concat(_noise, _text_embed, dim=1)
        x = self.fc1(x)
        x = mx.nd.reshape(x, shape=[-1, self.n_dims * 8, 4, 4])
        x = self.fc1_act(self.fc1_bnorm(x))
        x = self.deconv1_relu(self.deconv1_bnorm(self.deconv1(x)))
        x = self.deconv2_relu(self.deconv2_bnorm(self.deconv2(x)))
        x = self.deconv3_relu(self.deconv3_bnorm(self.deconv3(x)))
        x = self.deconv4_tanh(self.deconv4_bnorm(self.deconv4(x)))
        return x


class Discriminator(gluon.Block):
    def __init__(self, n_dims=64, **kwargs):
        super(Discriminator, self).__init__(**kwargs)
        self.n_dims = n_dims
        with self.name_scope():
            self.conv1 = nn.Conv2D(n_dims, 4, 2, 1)
            self.conv2 = nn.Conv2D(n_dims * 2, 4, 2, 1)
            self.conv3 = nn.Conv2D(n_dims * 4, 4, 2, 1)
            self.conv4 = nn.Conv2D(n_dims * 8, 4, 2, 1)
            self.conv5 = nn.Conv2D(n_dims * 8, 4, 2, 1)

            self.conv2_bnorm = nn.BatchNorm()
            self.conv3_bnorm = nn.BatchNorm()
            self.conv4_bnorm = nn.BatchNorm()
            self.conv5_bnorm = nn.BatchNorm()

            self.conv1_lrelu = nn.LeakyReLU(0.2)
            self.conv2_lrelu = nn.LeakyReLU(0.2)
            self.conv3_lrelu = nn.LeakyReLU(0.2)
            self.conv4_lrelu = nn.LeakyReLU(0.2)
            self.conv5_lrelu = nn.LeakyReLU(0.2)
            self.flatten = nn.Flatten()
            self.fc = nn.Dense(1)

    def forward(self, x, *args, **kwargs):
        _image = x[0]
        _text_embed = x[1]
        x = self.conv1_lrelu(self.conv1(_image))
        x = self.conv2_lrelu(self.conv2_bnorm(self.conv2(x)))
        x = self.conv3_lrelu(self.conv3_bnorm(self.conv3(x)))
        x = self.conv4_lrelu(self.conv4_bnorm(self.conv4(x)))

        y = mx.nd.expand_dims(_text_embed, axis=2)
        y = mx.nd.expand_dims(y, axis=2)
        y = mx.nd.tile(y, reps=[1, 1, 4, 4])

        x = mx.nd.concat(x, y, dim=1)
        x = self.conv5_lrelu(self.conv5_bnorm(self.conv5(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x
