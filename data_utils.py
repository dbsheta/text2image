import pickle
import mxnet as mx
import random

with open('data/encoded_captions.pkl', 'rb') as f:
    captions = pickle.load(f)

with open('data/processed_imgs.pkl', 'rb') as f:
    images = pickle.load(f)

with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)


class SmallDataset(object):
    def __init__(self, batch_size):
        self.keys = list(images.keys())
        random.shuffle(self.keys)
        self.keys_train = self.keys[:1024]
        self.train_data = []
        for key in self.keys_train:
            img = images[key]
            id1 = random.randint(0, 4)
            id2 = random.randint(0, 4)
            while id1 != id2:
                id2 = random.randint(0, 4)
            caption1 = captions[key][id1]
            caption2 = captions[key][id2]
            self.train_data.append((img, caption1))
            self.train_data.append((img, caption2))

        print(f"Length of train data: {len(self.train_data)}")
        random.shuffle(self.train_data)

        self.train_imgs, self.train_caps = zip(*self.train_data)

        self.num_batches = len(self.train_data) // batch_size
        print(f"Num of batches: {self.num_batches}")

    def get_batch(self, batch_size, batch_num, z, ctx=mx.cpu()):
        real_images = mx.nd.array(self.train_imgs[batch_num * batch_size:batch_num * batch_size + batch_size])
        real_captions = mx.nd.array(self.train_caps[batch_num * batch_size:batch_num * batch_size + batch_size])
        wrong_images = []
        range_ix = range(batch_num * batch_size, batch_num * batch_size + batch_size)
        for i in range(batch_size):
            ix = random.randint(0, len(self.train_imgs) - 1)
            while ix in range_ix:
                ix = random.randint(0, len(self.train_imgs) - 1)
            wrong_image = self.train_imgs[ix]
            wrong_images.append(wrong_image)
        wrong_images = mx.nd.array(wrong_images)
        noise = mx.nd.random_normal(0, 1, shape=(batch_size, z), ctx=ctx)
        return real_images, wrong_images, real_captions, noise
