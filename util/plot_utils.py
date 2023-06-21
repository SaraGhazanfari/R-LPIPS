import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt


def plot_histogram(lpips_list, r_lpips_list, save_path, y_bins_max=250, y_bins_slot=20):
    sns.set(style="darkgrid")
    bins = np.arange(0, 1.8, 0.2)
    ybins = np.arange(0, y_bins_max, y_bins_slot)
    # Creating histogram
    plt.rcParams['font.size'] = 2

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.histplot(data=lpips_list, color="#008080", label="LPIPS", kde=True, bins=100)
    sns.histplot(data=r_lpips_list, color="red",
                 label="R-LPIPS", kde=True, bins=100)
    plt.xlabel("")
    plt.ylabel("")
    plt.legend(fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks(bins)
    ax.set_yticks(ybins)
    ax.set(xlim=(0, 1.6), ylim=(0, y_bins_max))
    plt.savefig(save_path, format="pdf", dpi=300)
    plt.show()


def show_adversary_images(original_image_list, adversary_image_list,
                          lpips_distance_list, r_lpips_distance_list,
                          l2_distance_list, linf_distance_list,
                          threshold=0):
    for ii in range(lpips_distance_list.shape[0]):
        if lpips_distance_list[ii] >= 1.2:
            f, axarr = plt.subplots(1, 2, figsize=(5, 5))

            title = axarr[0].set_title("LPIPS: {:.2f}, R-LPIPS: {:.2f}, L2: {:.2f}, Linf: {:.2f}".format(
                lpips_distance_list[ii], r_lpips_distance_list[ii],
                l2_distance_list[ii], linf_distance_list[ii]))
            axarr[0].imshow(original_image_list[ii].cpu().numpy().transpose(1, 2, 0))
            plt.setp(title, color=('b'))
            axarr[1].imshow(adversary_image_list[ii].cpu().numpy().transpose(1, 2, 0))

            axarr[0].set_axis_off()
            axarr[1].set_axis_off()

            plt.subplots_adjust(right=2)
            plt.show()
