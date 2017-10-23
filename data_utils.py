import pickle
import mxnet as mx
import random
import os

with open('data/encoded_captions.pkl', 'rb') as f:
    captions = pickle.load(f)

with open('data/processed_imgs.pkl', 'rb') as f:
    images = pickle.load(f)

with open('data/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)


class SmallDataset(object):
    def __init__(self, batch_size):
        self.train_imgs = []
        self.train_caps = []
        self.num_batches = 0
        self.batch_size = batch_size
        self.train_data = []
        if "train_keys.pkl" in os.listdir("data/"):
            print("Loading old dataset")
            with open("data/train_keys.pkl", "rb") as f:
                self.keys_train = pickle.load(f)
        else:
            self.keys = list(images.keys())
            random.shuffle(self.keys)
            self.keys_train = self.keys[:1024]
            with open("data/train_keys.pkl", "wb") as f:
                pickle.dump(self.keys_train, f)

    def get_batch(self, batch_num, z, ctx):
        real_images = mx.nd.array(
            self.train_imgs[batch_num * self.batch_size:batch_num * self.batch_size + self.batch_size])
        real_captions = mx.nd.array(
            self.train_caps[batch_num * self.batch_size:batch_num * self.batch_size + self.batch_size])
        wrong_images = []
        range_ix = range(batch_num * self.batch_size, batch_num * self.batch_size + self.batch_size)
        for i in range(self.batch_size):
            ix = random.randint(0, len(self.train_imgs) - 1)
            while ix in range_ix:
                ix = random.randint(0, len(self.train_imgs) - 1)
            wrong_image = self.train_imgs[ix]
            wrong_images.append(wrong_image)
        wrong_images = mx.nd.array(wrong_images)
        noise = mx.nd.random_normal(0, 1, shape=(self.batch_size, z), ctx=ctx)
        return real_images, wrong_images, real_captions, noise

    def save_all_batches(self, z, ctx=mx.cpu()):
        for key in self.keys_train:
            img = images[key]
            caption1 = captions[key][0]
            # caption2 = captions[key][id2]
            self.train_data.append((img, caption1))
            # self.train_data.append((img, caption2))

        print(f"Length of train data: {len(self.train_data)}")
        # random.shuffle(self.train_data)

        self.train_imgs, self.train_caps = zip(*self.train_data)

        self.num_batches = len(self.train_data) // self.batch_size
        print(f"Num of batches: {self.num_batches}")

        batches = []
        for i in range(self.num_batches):
            print(f"Getting batch {i+1}")
            batch = self.get_batch(i, z, ctx)
            batches.append(batch)
        with open("data/batches.pkl", "wb") as fi:
            pickle.dump(batches, fi)
        return batches

    def load_all_batches(self, z, ctx=mx.cpu()):
        if "batches.pkl" not in os.listdir("data/"):
            batches = self.save_all_batches(z, ctx)
        else:
            with open("data/batches.pkl", "rb") as fi:
                print("Loading ")
                batches = pickle.load(fi)
        return batches
