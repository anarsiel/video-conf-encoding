import matplotlib.pyplot as plt


def parse_bitrate(source):
    with open(source) as file:
        lines = file.readlines()
        bitrates = [float(line.split()[2]) for line in lines]
    return bitrates


def parse_psnr(source):
    with open(source) as file:
        lines = file.readlines()
        psnr = [float(line.split('=')[-1]) for line in lines]
        qp = [int(line.split('=')[1].split()[0]) for line in lines]
    return psnr, qp


path_to_black = "results/black"
path_to_generated = "results/generated"

bitrate_filename = "bitrate.txt"
psnr_filename = "results.txt"

count_last = 4
bitrates_black = parse_bitrate(f"{path_to_black}/{bitrate_filename}")
psnr_black, qp = parse_psnr(f"{path_to_black}/{psnr_filename}")

bitrates_black = bitrates_black[-count_last:]
psnr_black = psnr_black[-count_last:]
qp = qp[-count_last:]

print(bitrates_black, psnr_black)

bitrates = parse_bitrate(f"{path_to_generated}/{bitrate_filename}")
psnr, _ = parse_psnr(f"{path_to_generated}/{psnr_filename}")


bitrates = bitrates[-count_last:]
psnr = psnr[-count_last:]
print(bitrates, psnr)

plt.plot(bitrates_black, psnr_black, label="black")
plt.plot(bitrates, psnr, label=f"generated")

plt.scatter(bitrates_black, psnr_black)
plt.scatter(bitrates, psnr)

for i in range(len(psnr)):
    plt.text(bitrates_black[i], psnr_black[i], qp[i], fontsize=9)
    plt.text(bitrates[i], psnr[i], qp[i], fontsize=9)

# plt.title(f'Sasha')
plt.xlabel('Bitrate')
plt.ylabel('PSNR')
plt.legend()
plt.grid()
plt.show()
