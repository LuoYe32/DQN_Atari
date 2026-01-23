import torch

from src.models.q_network import QNetwork, QNetworkConfig
from src.agents.target_update import hard_update, soft_update


def main():
    num_actions = 6
    online = QNetwork(QNetworkConfig(in_channels=4, num_actions=num_actions))
    target = QNetwork(QNetworkConfig(in_channels=4, num_actions=num_actions))

    x = torch.rand(32, 4, 84, 84)
    q = online(x)
    print("Q shape:", q.shape)

    hard_update(target, online)
    diff = 0.0
    for tp, op in zip(target.parameters(), online.parameters()):
        diff += (tp - op).abs().sum().item()
    print("Hard update param diff:", diff)

    with torch.no_grad():
        for p in online.parameters():
            p.add_(0.01 * torch.randn_like(p))

    soft_update(target, online, tau=0.1)
    print("Soft update applied OK")


if __name__ == "__main__":
    main()
