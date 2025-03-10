import torch
import matplotlib.pyplot as plt
import numpy as np
from snntorch import spikeplot as splt


# plotting
def plot_cur_mem_spk(
    state: torch.Tensor,
    spk: torch.Tensor,
    dt: float,
    target: torch.Tensor,
):
    # Generate Plots

    label = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    if state.ndim > 2:
        fig, ax = plt.subplots(
            3,
            figsize=(8, 30),
            sharex=False,
            gridspec_kw={"height_ratios": [1, 1, 2]},
            dpi=500,
        )

        # Plot V
        ax[0].plot(state[:, :, 0], label=label)
        ax[0].set_ylabel("Membrane Potential ($V$)")

        ax[0].legend(
            bbox_to_anchor=(0.0, 1.0, 1.0, 0.3),
            loc="lower left",
            ncol=len(label),
            mode="expand",
        )

        # Plot U
        ax[1].plot(state[:, :, 1], label=label)
        ax[1].set_ylabel("Membrane Potential ($U$)")

    else:
        fig, ax = plt.subplots(
            2,
            figsize=(8, 30),
            sharex=False,
            gridspec_kw={"height_ratios": [1, 1]},
            dpi=500,
        )

        # Plot V
        ax[0].plot(state[:, :], label=label)
        ax[0].set_ylabel("Membrane Potential ($V$)")

        ax[0].legend(
            bbox_to_anchor=(0.0, 1.0, 1.0, 0.3),
            loc="lower left",
            ncol=len(label),
            mode="expand",
        )

    # Plot output spike using spikeplot
    splt.raster(spk, ax[-1], s=100, c="black", marker="|")
    plt.xlabel(f"Time (ms)")
    plt.ylabel("Output spikes")

    ticks = np.arange(start=0, stop=state.shape[0], step=1 / dt)
    plt.xticks(ticks=ticks, color="w")

    fig.suptitle(f"Expected class: {label[target.argmax().item()]}")

    return fig


# post-train visualization
from torch.utils.data import DataLoader


def post_train_visual(
    net,
    data_test,
    device,
    dt,
    n_samples=1,
):

    uni_loader = DataLoader(data_test, batch_size=1, shuffle=True)
    load_iter = iter(uni_loader)

    figs = []
    for _ in range(n_samples):
        data, target = next(load_iter)
        uni_spk, uni_mem = net(data.flatten(2, -1).to(device))

        fig = plot_cur_mem_spk(
            state=uni_mem.squeeze().detach().cpu().numpy(),
            spk=uni_spk.squeeze().detach().cpu(),
            dt=dt,
            target=target,
        )
        figs.append(fig)
    return figs
