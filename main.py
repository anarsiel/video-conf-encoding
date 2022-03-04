from pre_processing.audio.audio_features import *
from pre_processing.video.video_features import find_landmarks_on_video

frames_count = find_landmarks_on_video("resources/videos/sgwp8n.mpg", "landmarks")
get_mfccs_as_files("resources/videos/sgwp8n.mpg", frames_count=frames_count, dest_dir="mfccs")
