import matplotlib.image as mpimg
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

# VIDEO_DIR = 'dataset_small/frames'
# AUDIO_DIR = 'dataset_small/mfccs'

# MU_AU, STD_AU = None, None


def _intersection_filter(video_dirs, audio_paths):
    video_dirs_names = set([dir.stem for dir in video_dirs])
    audio_file_names = set([path.stem for path in audio_paths])
    intersected_names_set = video_dirs_names.intersection(audio_file_names)
    names = list(intersected_names_set)
    get_actual_indexes = lambda paths: [path.stem in names for path in paths]
    new_video_dirs = np.array(video_dirs)[get_actual_indexes(video_dirs)]
    new_audio_paths = np.array(audio_paths)[get_actual_indexes(audio_paths)]
    return new_video_dirs, new_audio_paths


def _sort_paths(video_dirs, audio_paths):
    new_video_dirs = sorted(video_dirs)
    new_audio_paths = sorted(audio_paths)
    sort_sheck = lambda X, Y: [x.stem == y.stem for x, y in zip(X, Y)]
    error_text = 'Ахтунг: ошибка сортировки'
    assert np.prod(sort_sheck(new_video_dirs, new_audio_paths)), error_text
    return new_video_dirs, new_audio_paths


def get_preprocessing_paths(im_path,
                            au_path):
    video_dirs = list(Path(im_path).glob('*'))
    audio_paths = list(Path(au_path).glob('*'))

    video_dirs, audio_paths = _intersection_filter(video_dirs, audio_paths)
    video_dirs, audio_paths = _sort_paths(video_dirs, audio_paths)
    return video_dirs, audio_paths


def get_audio_mu_std(audio_paths, count=100):
    audio_sample = np.genfromtxt(audio_paths[0]).T
    MU_AU = np.empty((0, audio_sample.shape[1]))
    STD_AU = np.empty((0, audio_sample.shape[1]))

    for audio_path in audio_paths[:count]:
        audio = np.genfromtxt(audio_path).T
        MU_AU = np.concatenate([MU_AU, [audio.mean(axis=0)]])
        STD_AU = np.concatenate([STD_AU, [audio.std(axis=0)]])
    MU_AU = MU_AU.mean(axis=0)
    STD_AU = STD_AU.mean(axis=0)
    return MU_AU, STD_AU


class CustomDataGen(tf.keras.utils.Sequence):
    def __init__(self,
                 video_dirs,
                 audio_paths,
                 image_shape=(50, 60, 3),
                 audio_shape=(43, 20),
                 frames=24,
                 batch_size=32,
                 shuffle=True,
                 MU=0.,
                 STD=255.,
                 MU_AU=MU_AU,
                 STD_AU=STD_AU,
                 augmentator=tr,
                 ):

        self.video_dirs = np.array(video_dirs)
        self.audio_paths = np.array(audio_paths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.video_dirs))
        self.image_shape = image_shape
        self.audio_shape = audio_shape
        self.frames = frames
        self.MU = MU
        self.STD = STD
        self.MU_AU = MU_AU
        self.STD_AU = STD_AU
        self.augmentator = augmentator
        self._len = len(self.video_dirs) // self.batch_size + \
                    int(bool(len(self.video_dirs) % self.batch_size))

        if self.shuffle:
            np.random.shuffle(self.indexes)

        sort_sheck = lambda X, Y: [x.stem == y.stem for x, y in zip(X, Y)]
        error_text = 'Ахтунг: ошибка сортировки'
        assert np.prod(sort_sheck(self.video_dirs, self.audio_paths)), error_text

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):

        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.video_dirs))
        batch_indexes = self.indexes[start:end]
        batch_vi_paths = self.video_dirs[batch_indexes]
        batch_au_paths = self.audio_paths[batch_indexes]
        video_batch = np.empty([0, self.frames, *self.image_shape])
        image_batch = np.empty([0, *self.image_shape])
        audio_batch = np.empty([0, *self.audio_shape])

        for vi_path, au_path in zip(batch_vi_paths, batch_au_paths):
            vi_paths = sorted(list(vi_path.glob('*')), key=lambda x: int(x.stem))
            video = np.array([mpimg.imread(path) for path in vi_paths])

            video = tf.convert_to_tensor(video, tf.float32)
            video = self.augmentator(video)

            image = video[0]
            image_batch = np.concatenate([image_batch, [image]])

            video = video[1:]

            video_batch = np.concatenate([video_batch, [video]])

            audio = np.genfromtxt(au_path).T
            audio_batch = np.concatenate([audio_batch, [audio]])

        video_batch -= self.MU
        video_batch /= self.STD

        image_batch -= self.MU
        image_batch /= self.STD

        audio_batch -= self.MU_AU
        audio_batch /= self.STD_AU

        X_batch = {
            'image': image_batch,
            'audio': audio_batch,
        }

        y_batch = video_batch

        return X_batch, y_batch

    def __len__(self):
        return self._len


def get_datasets(video_dir, audio_dir):
    video_dirs, audio_paths = get_preprocessing_paths(video_dir, audio_dir)
    _data = train_test_split(video_dirs, audio_paths, test_size=0.10)
    train_video_dirs, test_video_dirs, train_audio_paths, test_audio_paths = _data

    MU_AU, STD_AU = get_audio_mu_std(audio_paths, count=100)

    train_dataset = CustomDataGen(train_video_dirs, train_audio_paths, batch_size=8, MU_AU=MU_AU, STD_AU=STD_AU)
    test_dataset = CustomDataGen(test_video_dirs, test_audio_paths, batch_size=8, MU_AU=MU_AU, STD_AU=STD_AU)

    return train_dataset, test_dataset, MU_AU, STD_AU

