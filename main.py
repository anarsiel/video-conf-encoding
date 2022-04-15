from preprocessing.preprocess import create_dataset

success, broken_files = 0, []
for person_id in range(1, 0):  # 35):
    if person_id == 21:
        continue

    # s, bf = create_dataset(f"GRID/s{person_id}", "grid_dataset")

    success += s
    broken_files += bf

    print(f"s{person_id} preprocessed")

print(f"\nTotal files preprocessed: {success}. \nBroken files: {len(broken_files)}.")

f = open("broken_files.txt", "w")
f.write('\n'.join(broken_files))
f.close()
