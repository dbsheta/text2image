import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet import gluon

from models import Generator, Discriminator, Embedding
from utils import time_since, facc
from data_utils import SmallDataset, vocab

import time

data_dir = "/Users/dhoomilbsheta/Development/datasets/flowers"

batch_size = 64
img_dims = 64
embed_size = 256
text_embed_dims = 128
max_seq_len = 50
z = 100
lr = 0.0002
beta1 = 0.5

ctx = mx.cpu()


def build_gan(vocab_size, embed_dims, text_dims):
    g = Generator()
    d = Discriminator()
    e = Embedding(vocab_size, embed_dims, text_dims)
    return g, d, e


def param_init(net, stddev=None):
    if stddev is None:
        net.initialize(mx.init.Normal(), ctx=ctx)
    else:
        net.initialize(mx.init.Normal(stddev), ctx=ctx)


def train(netG, netD, netE, batches, epochs=10, continued=False):
    gan_loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()

    # Initialize all networks
    if not continued:
        param_init(netG, 0.02)
        param_init(netD, 0.02)
        param_init(netE)
    else:
        netG.load_params("data/dcgan_g", ctx=ctx)
        netD.load_params("data/dcgan_d", ctx=ctx)
        netE.load_params("data/text_embed", ctx=ctx)

    trainerG = gluon.Trainer(netG.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerD = gluon.Trainer(netD.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})
    trainerE = gluon.Trainer(netE.collect_params(), 'adam', {'learning_rate': lr, 'beta1': beta1})

    metric = mx.metric.CustomMetric(facc)

    real_label = nd.ones((batch_size,), ctx=ctx)
    fake_label = nd.zeros((batch_size,), ctx=ctx)

    print(f'Training for {epochs}...')
    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch {epoch}----------")
        for n, batch in enumerate(batches):
            real_images, wrong_images, real_captions, noise = batch

            ############################
            # (1) Update D network: maximize log(D(x, y)) + log(1 - D(x, G(x, z)))
            ###########################
            with autograd.record():
                out_e = netE(real_captions)
                out_g = netG([noise, out_e])
                # update on real
                out_d_real = netD([real_images, out_e])
                errD_real = gan_loss(out_d_real, real_label)
                metric.update([real_label, ], [out_d_real, ])

                # update on wrong
                out_d_wrong = netD([wrong_images, out_e])
                errD_wrong = gan_loss(out_d_wrong, fake_label)
                metric.update([fake_label, ], [out_d_wrong, ])

                # update on fake
                out_d_fake = netD([out_g, out_e])
                errD_fake = gan_loss(out_d_fake, fake_label)
                metric.update([fake_label, ], [out_d_fake, ])

                errD = errD_real + 0.5 * errD_wrong + 0.5 * errD_fake
                errD.backward()

            trainerD.step(batch_size)
            # trainerE.step(batch_size)
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            with autograd.record():
                out_e = netE(real_captions)
                out_g = netG([noise, out_e])
                output = netD([out_g, out_e])
                errG = gan_loss(output, real_label)
                errG.backward()

            trainerG.step(batch_size)
            # trainerE.step(batch_size)

            if n % 10 == 0:
                name, acc = metric.get()
                print(
                    f"D_loss= {nd.mean(errD).asscalar()}, "
                    f"G_loss= {nd.mean(errG).asscalar()}, acc= {acc} at iteration {n}")

        print(f"Time spent: {time_since(start)}")
        if epoch % 5 == 0:
            netD.save_params("data/dcgan_d")
            netG.save_params("data/dcgan_g")
            netE.save_params("data/text_embed")

        metric.reset()

    netD.save_params("data/dcgan_d")
    netG.save_params("data/dcgan_g")
    netE.save_params("data/ext_embed")
    print(f"Total time spent: {time_since(start)}")


if __name__ == '__main__':
    print("1. Train\n2. Continue to Train")
    c = int(input())
    dataset = SmallDataset(batch_size)
    batches = dataset.load_all_batches(z, ctx)
    netG, netD, netE = build_gan(len(vocab), embed_size, text_embed_dims)
    if c == 1:
        train(netG, netD, netE, batches, epochs=50)
    elif c == 2:
        train(netG, netD, netE, batches, epochs=10, continued=False)
