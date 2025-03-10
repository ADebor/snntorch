# imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2 as transforms
import wandb
from pathlib import Path
from datetime import datetime

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

from transforms import RateCoding
from utils import post_train_visual

import pprint
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(config_path="config/", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    # config
    pprint.pprint(OmegaConf.to_container(cfg, resolve=True))

    hypers_cfg = cfg.hypers
    neuron_cfg = cfg.neuron
    network_cfg = cfg.network
    task_cfg = cfg.task
    logger_cfg = cfg.logger

    # torch setup
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # data
    img_dim = task_cfg.img_dim
    data_path = "/tmp/data/mnist"
    transform = transforms.Compose(
        [
            transforms.Resize((img_dim, img_dim)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),
        ]
    )
    if task_cfg.name == "noise":
        transform = transforms.Compose(
            [
                transform,
                transforms.GaussianNoise(
                    mean=OmegaConf.select(task_cfg, "mean", default=0.0),
                    sigma=OmegaConf.select(task_cfg, "std", default=0.5),
                    clip=True,
                ),
            ]
        )
    transform = transforms.Compose(
        [
            transform,
            RateCoding(
                num_steps=hypers_cfg.n_steps,
                gain=hypers_cfg.spikegen_gain,
                num_silence_steps=OmegaConf.select(
                    task_cfg, "num_silence_steps", default=0
                ),
            ),
        ]
    )

    data_train = datasets.MNIST(
        data_path, train=True, download=True, transform=transform
    )
    data_test = datasets.MNIST(
        data_path, train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        data_train,
        batch_size=hypers_cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        data_test,
        batch_size=hypers_cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # net
    net = instantiate(
        network_cfg, n_inputs=img_dim**2, n_outputs=task_cfg.n_classes
    ).to(device)

    # loss
    loss = nn.CrossEntropyLoss()

    # optimizer
    opt = torch.optim.Adam(net.parameters(), lr=hypers_cfg.lr)

    # scheduler
    sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=opt, gamma=hypers_cfg.gamma_lr
    )

    # log
    run_name = logger_cfg.exp_name + datetime.now().strftime(
        "%Y_%m_%d-%I_%M_%S_%p"
    )
    if logger_cfg.backend == "wandb":
        wandb.init(
            project=logger_cfg.project_name,
            entity=logger_cfg.entity_name,
            name=run_name,
            mode=logger_cfg.mode,
            config=OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            ),
        )
        wandb.watch(net, log="all", log_freq=logger_cfg.watch_log_freq)
    log_interval = logger_cfg.log_interval

    # loop
    for epoch in range(hypers_cfg.n_epochs):
        # train
        net.train()
        for batch_id, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            # # sanity check
            # import matplotlib.pyplot as plt
            # import snntorch.spikeplot as splt
            # sample = data[0, :, 0, :, :]
            # fig, ax = plt.subplots()
            # anim = splt.animator(sample, fig, ax)
            # anim.save("sample_spk_data.gif")

            opt.zero_grad()

            # forward pass
            spk_rec, state_rec = net(data.flatten(2, -1))

            # compute firing rate
            fr = spk_rec.sum(dim=0)

            # compute loss
            xe_loss = loss(fr, target)
            loss_val = xe_loss

            # compute accuracy
            pred = fr.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = 100.0 * correct / hypers_cfg.batch_size

            # Gradient calculation + weight update
            loss_val.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)

            opt.step()

            if batch_id % log_interval == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_id * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_id / len(train_loader),
                        loss_val.item(),
                    )
                )
            if logger_cfg.backend:
                wandb.log(
                    {
                        "train/loss": loss_val.item(),
                        "train/lr": sched.get_last_lr()[0],
                        "train/accuracy": accuracy,
                    }
                )
        sched.step()

        # test
        with torch.no_grad():
            net.eval()
            test_loss = 0
            correct = 0
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                test_spk, _ = net(data.flatten(2, -1))
                test_fr = test_spk.sum(dim=0)
                tmp = loss(test_fr, target).item()
                test_loss += tmp
                pred = test_fr.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader)
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )
        if logger_cfg.backend:

            wandb.log(
                {
                    "test/loss": test_loss,
                    "test/accuracy": 100.0
                    * correct
                    / len(test_loader.dataset),
                }
            )

    # save model
    save_dir = Path(__file__).parent / "saved_models"
    torch.save(net.state_dict(), save_dir / "mnist_mqif.pt")

    # post-train visualization
    figs = post_train_visual(
        net=net,
        data_test=data_test,
        device=device,
        dt=OmegaConf.select(neuron_cfg.params, "dt", default=1.0),
        n_samples=logger_cfg.post_train_visual_n_samples,
    )
    if logger_cfg.backend:
        wandb.log(
            {
                f"post_train_visual_{i}": wandb.Image(fig)
                for i, fig in enumerate(figs)
            }
        )


if __name__ == "__main__":
    main()
