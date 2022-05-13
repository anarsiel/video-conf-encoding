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
    return psnr


qp = [15, 20, 30, 40]

path_to_black = "results/black"

bitrate_filename = "bitrate.txt"
psnr_filename = "results.txt"

for i in range(len(qp)):
    bitrates_black = parse_bitrate(f"{path_to_black}/{bitrate_filename}")
    psnr_black = parse_psnr(f"{path_to_black}/{psnr_filename}")

    path_to_generated = f"results/generated_{qp[i]}"
    bitrates = parse_bitrate(f"{path_to_generated}/{bitrate_filename}")
    psnr = parse_psnr(f"{path_to_generated}/{psnr_filename}")

    # plot lines
    plt.plot(bitrates_black, psnr_black, label="black")
    plt.plot(bitrates, psnr, label=f"generated")
    plt.title(f'Sasha, qp = {qp[i]}')
    plt.xlabel('Bitrate')
    plt.ylabel('PSNR')
    plt.legend()
    plt.show()