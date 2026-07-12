import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thdat

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def np_to_th(x):
    n_samples = len(x)
    return torch.from_numpy(x).to(torch.float).to(DEVICE).reshape(n_samples, -1)


class Net(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=1e-3,
        loss2=None,
        loss2_weight=0.1,
    ) -> None:
        super().__init__()

        self.epochs = epochs
        self.loss = loss
        self.loss2 = loss2
        self.loss2_weight = loss2_weight
        self.lr = lr
        self.n_units = n_units

        self.layers = nn.Sequential(
            nn.Linear(input_dim, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
            nn.Linear(self.n_units, self.n_units),
            nn.ReLU(),
        )
        self.out = nn.Linear(self.n_units, output_dim)

    def forward(self, x):
        h = self.layers(x)
        out = self.out(h)

        return out

    def fit(self, X, y):
        Xt = np_to_th(X)
        yt = np_to_th(y)

        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        for ep in range(self.epochs):
            optimiser.zero_grad()
            outputs = self.forward(Xt)
            loss = self.loss(yt, outputs)
            if self.loss2:
                loss += self.loss2_weight * self.loss2(self)
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            if ep % int(self.epochs / 10) == 0:
                print(f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.2f}")
        return losses

    def predict(self, X):
        self.eval()
        out = self.forward(np_to_th(X))
        return out.detach().cpu().numpy()


class NetDiscovery(Net):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=0.001,
        loss2=None,
        loss2_weight=0.1,
    ) -> None:
        super().__init__(
            input_dim, output_dim, n_units, epochs, loss, lr, loss2, loss2_weight
        )

        self.r = nn.Parameter(data=torch.tensor([0.]))


# --- Everything below this line is new: Net/NetDiscovery above are untouched. ---
#
# NetWeighted / NetDiscoveryWeighted let you explicitly pass a weight for the
# data loss (loss1_weight), not just the physics loss (loss2_weight), and
# require the two to sum to 1:
#     loss = loss1_weight * data_loss + loss2_weight * physics_loss
# loss1_weight defaults to (1 - loss2_weight) if you don't pass it.
#
# This is a separate class rather than a change to Net itself because Net's
# existing callers pass loss2_weight values like 2 (e.g. pinn_loss_weight = 2).
# Auto-deriving loss1_weight = 1 - loss2_weight for those would silently give
# loss1_weight = -1 (a negative data-loss weight) instead of raising an error.
# Keeping it separate means old Net/NetDiscovery calls keep working exactly as
# before, and the sum-to-1 constraint only applies where you opt into it.
class NetWeighted(Net):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=1e-3,
        loss2=None,
        loss2_weight=0.1,
        loss1_weight=None,
    ) -> None:
        super().__init__(
            input_dim, output_dim, n_units, epochs, loss, lr, loss2, loss2_weight
        )

        # loss1_weight (data) + loss2_weight (physics) must sum to 1
        if loss1_weight is None:
            loss1_weight = 1 - loss2_weight
        assert abs((loss1_weight + loss2_weight) - 1) < 1e-8, (
            f"loss1_weight + loss2_weight must equal 1, got {loss1_weight} + {loss2_weight}"
        )
        self.loss1_weight = loss1_weight

    def fit(self, X, y):
        Xt = np_to_th(X)
        yt = np_to_th(y)

        optimiser = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        losses = []
        for ep in range(self.epochs):
            optimiser.zero_grad()
            outputs = self.forward(Xt)
            # data loss and physics loss are computed and printed separately,
            # then combined as loss1_weight * data_loss + loss2_weight * physics_loss
            data_loss = self.loss(yt, outputs)
            loss = self.loss1_weight * data_loss
            physics_loss = None
            if self.loss2:
                physics_loss = self.loss2(self)
                loss = loss + self.loss2_weight * physics_loss
            loss.backward()
            optimiser.step()
            losses.append(loss.item())
            if ep % int(self.epochs / 10) == 0:
                msg = f"Epoch {ep}/{self.epochs}, loss: {losses[-1]:.4f}, data_loss: {data_loss.item():.4f}"
                if physics_loss is not None:
                    physics_loss_value = physics_loss.item() if torch.is_tensor(physics_loss) else physics_loss
                    msg += f", physics_loss: {physics_loss_value:.4f}"
                print(msg)
        return losses


class NetDiscoveryWeighted(NetWeighted):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_units=100,
        epochs=1000,
        loss=nn.MSELoss(),
        lr=0.001,
        loss2=None,
        loss2_weight=0.1,
        loss1_weight=None,
    ) -> None:
        super().__init__(
            input_dim,
            output_dim,
            n_units,
            epochs,
            loss,
            lr,
            loss2,
            loss2_weight,
            loss1_weight,
        )

        self.r = nn.Parameter(data=torch.tensor([0.]))
