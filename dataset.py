import numpy as np
import os
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from PIL import Image


def load_image_array(img_dir, image_file, image_size):
    img = Image.open(os.path.join(img_dir, image_file))
    img_resized = img.resize((image_size, image_size), Image.ANTIALIAS)
    img_resized = np.array(img_resized, dtype="float32")
    # GRAYSCALE
    if len(img_resized.shape) == 2:
        img_new = np.ndarray((img.shape[0], img.shape[1], 3))
        img_new[:, :, 0] = img
        img_new[:, :, 1] = img
        img_new[:, :, 2] = img
        img_resized = img_new

    img_resized = np.transpose(img_resized, (2, 0, 1))
    img_resized = img_resized / 127.5 - 1
    return img_resized


def get_captions(img_dir, caption_dir):
    img_files = [f for f in os.listdir(img_dir) if 'jpg' in f]

    image_captions_dict = {img_file: [] for img_file in img_files}
    image_captions = []
    class_dirs = []
    for i in range(1, 103):
        class_dir_name = 'class_%.5d' % (i)
        class_dirs.append(os.path.join(caption_dir, class_dir_name))
    for class_dir in class_dirs:
        caption_files = [f for f in os.listdir(class_dir) if 'txt' in f]
        for cap_file in caption_files:
            with open(os.path.join(class_dir, cap_file)) as f:
                captions = f.read().split('\n')
            img_file = cap_file[0:11] + ".jpg"
            # 5 captions per image
            image_captions_dict[img_file] += [cap.lower() for cap in captions if len(cap) > 0][0:5]
            image_captions.extend(image_captions_dict[img_file])
    return image_captions_dict, image_captions


def preprocess_images(data_dir, img_dims):
    img_dir = os.path.join(data_dir, 'jpg')
    img_files = [f for f in os.listdir(img_dir) if 'jpg' in f]
    processed_imgs = {}
    for img_file in img_files:
        print(f"Processing {img_file}")
        img = load_image_array(img_dir, img_file, img_dims)
        processed_imgs[img_file] = img
    return processed_imgs


def build_dataset(max_seq_len, data_dir, img_dims):
    img_dir = os.path.join(data_dir, 'jpg')
    caption_dir = os.path.join(data_dir, 'text/text_c10')

    # imgs = preprocess_images(data_dir, img_dims)
    # with open('data/processed_imgs.pkl', 'wb') as f:
    #     pickle.dump(imgs, f)
    captions_dict, captions = get_captions(img_dir, caption_dir)
    print(f"Total Captions: {len(captions)}")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    word2ix = tokenizer.word_index
    print(f"Vocab len: {len(word2ix)}")

    encoded_captions = {}
    for img_file in captions_dict:
        sequences = tokenizer.texts_to_sequences(captions_dict[img_file])
        sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='post', truncating='post')
        encoded_captions[img_file] = sequences

    with open('encoded_captions.pkl', 'wb') as f:
        pickle.dump(encoded_captions, f)

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(word2ix, f)


# imgs = preprocess_images("D://Development/dataset/flower", 64)
# with open('processed_imgs.pkl', 'wb') as f:
#     pickle.dump(imgs, f)

build_dataset(50, "D:/Development/dataset/flower", 64)
