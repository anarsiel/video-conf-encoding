import os
import shutil
import warnings

from pre_processing.audio.audio_features import save_mfccs
from pre_processing.video.video_features import save_frames


def create_dataset(source_dir, dest_dir):
    warnings.filterwarnings("ignore")

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    mfccs_dest_dir = f"{dest_dir}/mfccs"
    if not os.path.exists(mfccs_dest_dir):
        os.makedirs(mfccs_dest_dir)

    frames_dest_dir = f"{dest_dir}/frames"
    if not os.path.exists(frames_dest_dir):
        os.makedirs(frames_dest_dir)

    split_dest_dir = "split"
    if not os.path.exists(split_dest_dir):
        os.makedirs(split_dest_dir)

    split_videos(source_dir, split_dest_dir)

    files = os.listdir(split_dest_dir)
    for file in files:
        if file == '.DS_Store':
            continue

        save_all_mfccs(f'{split_dest_dir}/{file}', mfccs_dest_dir)
        save_all_frames(f'{split_dest_dir}/{file}', frames_dest_dir)

        print(f"preprocessed: {file}")

    shutil.rmtree(split_dest_dir)


def split_videos(source_dir, dest_dir):
    files = os.listdir(source_dir)

    # source_dir = "../" + source_dir
    # dest_dir = "../" + dest_dir

    for file in files:
        filename = file.split(".")[0]
        os.system(f"ffmpeg -y -i {source_dir}/{file} -ss 00:00:00 -to 00:00:01 -c copy {dest_dir}/{filename}_01.mpg >/dev/null 2>&1")
        os.system(f"ffmpeg -y -i {source_dir}/{file} -ss 00:00:01 -to 00:00:02 -c copy {dest_dir}/{filename}_02.mpg >/dev/null 2>&1")
        os.system(f"ffmpeg -y -i {source_dir}/{file} -ss 00:00:02 -to 00:00:03 -c copy {dest_dir}/{filename}_03.mpg >/dev/null 2>&1")


def save_all_mfccs(file, mfccs_dest_dir):
    save_mfccs(file, dest_dir=mfccs_dest_dir)


def save_all_frames(file, frames_dest_dir):
    save_frames(file, frames_dest_dir, cut=True, apply_landmarks=False, save_landmarks=False)
