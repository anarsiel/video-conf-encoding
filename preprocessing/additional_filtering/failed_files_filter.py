import os


def safe_delete_dir(path_to_dir):
    if os.path.exists(path_to_dir) and os.path.isdir(path_to_dir):
        shutil.rmtree(path_to_dir)
        print(f"deleted: {path_to_dir}")


def safe_delete_file(path_to_file):
    if path.exists(path_to_file):
        os.remove(path_to_file)
        print(f"deleted: {path_to_file}")


filename = "broken_files.txt"
with open(filename) as file:
    lines = file.readlines()
    lines = sorted([line.rstrip().split('/')[-1] for line in lines])

print(f'Total broken files: {len(lines)}')
print(f'Unique: {len(set(lines))}')
print(*lines, sep='\n')

if len(set(lines)) != len(lines):
    path_to_frames = "grid_dataset/grid_dataset/frames/"
    path_to_mfccs = "grid_dataset/grid_dataset/mfccs/"

    for filename in lines:
        safe_delete_dir(path_to_frames + filename)
        safe_delete_file(path_to_mfccs + filename + ".csv")