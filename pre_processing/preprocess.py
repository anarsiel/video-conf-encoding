import os
import shutil
import warnings
import numpy as np

from pre_processing.audio.audio_features import get_mfccs
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

    files = os.listdir(source_dir)
    for file in sorted(files):
        if file == '.DS_Store':
            continue

        save_mfccs(f'{source_dir}/{file}', mfccs_dest_dir)
        save_frames(f'{source_dir}/{file}', frames_dest_dir, cut=True, apply_landmarks=False)

        print(f"preprocessed: {file}")


def save_mfccs(file, mfccs_dest_dir):
    first, second, third = get_mfccs(file)

    filename = file.split('/')[-1].split('.')[0]

    np.savetxt(f"{mfccs_dest_dir}/{filename}_01.csv", first)
    np.savetxt(f"{mfccs_dest_dir}/{filename}_02.csv", second)
    # np.savetxt(f"{mfccs_dest_dir}/{filename}_03.csv", third)