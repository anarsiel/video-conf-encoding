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

    files = [file for file in os.listdir(source_dir) if check_video_file(file)]
    success = 0
    broken_files = []
    for file in sorted(files):
        filepath = f'{source_dir}/{file}'
        filename = file.split('.')[0]

        try:
            save_mfccs(filepath, mfccs_dest_dir)
        except:
            print(f"MFCC ERROR: {file}")
            broken_files.append(file)
            continue

        try:
            save_frames(filepath, frames_dest_dir, cut=True, apply_landmarks=False)

            print(f"preprocessed: {file}")
            success += 1
        except:
            os.remove(f"{mfccs_dest_dir}/{filename}_01.csv")
            os.remove(f"{mfccs_dest_dir}/{filename}_02.csv")
            # os.remove(f"{source_dir}/{filename}_03.csv")

            safe_delete_dir("fragmentated")
            safe_delete_dir(f"{frames_dest_dir}/{filename}_01")
            safe_delete_dir(f"{frames_dest_dir}/{filename}_02")
            safe_delete_dir(f"{frames_dest_dir}/{filename}_03")
            print(f"FRAMES ERROR: {file}")
            broken_files.append(file)

    print("--------------------------------------")
    print(f"Successfully preprocessed: {success} files.\nBroken: {len(broken_files)} files.\n")

    # TODO print broken_files_list to file




def save_mfccs(file, mfccs_dest_dir):
    first, second, third = get_mfccs(file)

    filename = file.split('/')[-1].split('.')[0]

    np.savetxt(f"{mfccs_dest_dir}/{filename}_01.csv", first)
    np.savetxt(f"{mfccs_dest_dir}/{filename}_02.csv", second)
    # np.savetxt(f"{mfccs_dest_dir}/{filename}_03.csv", third)


def check_video_file(file):
    elements = file.split('.')
    return len(elements) == 2 and elements[-1] == 'mpg'


def safe_delete_dir(path_to_dir):
    if os.path.exists(path_to_dir) and os.path.isdir(path_to_dir):
        shutil.rmtree(path_to_dir)
