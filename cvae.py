import torch

class CVAE(torch.nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc0  = torch.nn.Linear(feature_size + class_size, 256)
        self.fc1  = torch.nn.Linear(256, 256)

        self.fc21 = torch.nn.Linear(256, latent_size)
        self.fc22 = torch.nn.Linear(256, latent_size)

        # decode
        self.fc3 = torch.nn.Linear(latent_size + class_size, 256)
        self.fc4 = torch.nn.Linear(256, 256)
        self.fc5 = torch.nn.Linear(256, feature_size)

        self.elu = torch.nn.ELU()
        # self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        x = x.flatten(1)
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.elu(self.fc0(inputs))
        h2 = self.elu(self.fc1(h1))
        z_mu = self.fc21(h2)
        z_var = self.fc22(h2)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h3 = self.elu(self.fc3(inputs))
        h4 = self.elu(self.fc4(h3))
        # return self.sigmoid(self.fc4(h3))
        return self.fc5(h4)

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 12*2), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(reconstruction, original, mu, logvar):
    # BCE = torch.nn.functional.binary_cross_entropy(recon_x, x.flatten(1), reduction='sum')
    MSE = torch.nn.functional.mse_loss(input=reconstruction, target=original.flatten(1), reduction='mean')
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    return MSE + 0.01*KLD


def train(epoch_num, model, train_loader, optimizer, class_size, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        labels = one_hot(labels, class_size, device)
        recon_batch, mu, logvar = model(data, labels)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_num, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch_num, train_loss / len(train_loader.dataset)))

def test(model, test_loader, class_size, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for (data, labels) in test_loader:
            data, labels = data.to(device), labels.to(device)
            labels = one_hot(labels, class_size, device)
            recon_batch, mu, logvar = model(data, labels)
            test_loss += loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

def test_data_for_condition (test_data: torch.Tensor, condition: int):
    """Filer observations for a given condition."""
    for observation, label in test_data:
        if label == condition:
            yield observation

def one_hot(labels, class_size, device):
    """Convert label to one-hot vector."""
    targets = torch.zeros(labels.size(0), class_size)
    for i, label in enumerate(labels):
        targets[i, label-1] = 1
        # targets = torch.nn.functional.softmax(targets, dim=1)
    return targets.to(device)

def original_and_generated_for_condition(max_condition: int, test_data:torch.Tensor, model:torch.nn.Module, latent_size:int, device):
    """Yield original and generated data for each condition."""
    for condition in range(1, max_condition+1):
        # filter original data for the condition
        original_for_condition = torch.stack(list(test_data_for_condition(test_data=test_data, 
                                                condition=condition)))
        sample_len = len(original_for_condition)

        # generate data for the condition
        encoded_condition = one_hot(torch.tensor([condition]).repeat(sample_len, 1), class_size=max_condition, device=device)
        sample = torch.randn(sample_len, latent_size).to(device)
        sample = model.decode(sample, encoded_condition).detach().cpu().view(-1, 12, 2)

        yield (condition, original_for_condition, sample)
