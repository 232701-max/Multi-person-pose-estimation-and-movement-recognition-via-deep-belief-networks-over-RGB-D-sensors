import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# RBM Implementation
class RBM(nn.Module):
    def __init__(self, visible_units, hidden_units):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(visible_units, hidden_units) * 0.1)
        self.v_bias = nn.Parameter(torch.zeros(visible_units))
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))

    def sample_hidden(self, v):
        activation = torch.matmul(v, self.W) + self.h_bias
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_visible(self, h):
        activation = torch.matmul(h, self.W.t()) + self.v_bias
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def contrastive_divergence(self, v, k=1):
        v_k = v
        for _ in range(k):
            _, h_k = self.sample_hidden(v_k)
            p_v_given_h, v_k = self.sample_visible(h_k)
        return v, p_v_given_h

    def forward(self, v):
        _, h = self.sample_hidden(v)
        return h


# DBN Implementation
class DBN(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim):
        super(DBN, self).__init__()
        self.rbms = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            self.rbms.append(RBM(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.output_layer = nn.Linear(prev_dim, output_dim)

    def pretrain(self, data_loader, epochs=50, lr=0.01, batch_size=64, device='cpu'):
        for i, rbm in enumerate(self.rbms):
            print(f"Pretraining RBM layer {i + 1}")
            optimizer = optim.SGD(rbm.parameters(), lr=lr)
            for epoch in range(epochs):
                total_loss = 0
                for batch in data_loader:
                    batch = batch[0].to(device)  # Assuming data_loader yields (data, labels), but we ignore labels
                    if i > 0:
                        with torch.no_grad():
                            for prev_rbm in self.rbms[:i]:
                                batch = prev_rbm(batch)
                    v0, vk = rbm.contrastive_divergence(batch)
                    loss = torch.mean(torch.sum((v0 - vk) ** 2, dim=1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(data_loader)}")

    def forward(self, x):
        for rbm in self.rbms:
            x = rbm(x)
        return self.output_layer(x)


# Function to train and evaluate DBN (used as fitness in PSO)
def train_evaluate_dbn(X, y, selected_features, hidden_layers=[512, 256, 128], output_dim=10, epochs=50,
                       pretrain_epochs=50, lr=0.01, batch_size=64, test_size=0.2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Select features
    X_selected = X[:, selected_features]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=test_size, random_state=42)

    # Convert to tensors
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize DBN
    input_dim = X_selected.shape[1]
    dbn = DBN(input_dim, hidden_layers, output_dim).to(device)

    # Pretrain
    dbn.pretrain(train_loader, epochs=pretrain_epochs, lr=lr, batch_size=batch_size, device=device)

    # Fine-tune
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(dbn.parameters(), lr=lr)
    dbn.train()
    for epoch in range(epochs):
        total_loss = 0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = dbn(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Fine-tune Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

    # Evaluate
    dbn.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            outputs = dbn(data)
            _, predicted = torch.max(outputs.data, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


# PSO for Feature Selection
class PSOFeatureSelection:
    def __init__(self, num_particles=30, num_generations=50, c1=1.5, c2=1.5, w=0.7, num_features=66523, X=None, y=None):
        self.num_particles = num_particles
        self.num_generations = num_generations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.num_features = num_features
        self.X = X
        self.y = y
        # Initialize particles (positions as binary masks, velocities as floats)
        self.positions = np.random.randint(0, 2, size=(num_particles, num_features))  # Binary: 0 or 1
        self.velocities = np.random.uniform(-1, 1, size=(num_particles, num_features))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(num_particles, -np.inf)
        self.global_best_position = None
        self.global_best_score = -np.inf

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def update_velocity(self, particle_idx):
        r1 = np.random.uniform(0, 1, self.num_features)
        r2 = np.random.uniform(0, 1, self.num_features)
        cognitive = self.c1 * r1 * (self.personal_best_positions[particle_idx] - self.positions[particle_idx])
        social = self.c2 * r2 * (self.global_best_position - self.positions[particle_idx])
        self.velocities[particle_idx] = self.w * self.velocities[particle_idx] + cognitive + social

    def update_position(self, particle_idx):
        prob = self.sigmoid(self.velocities[particle_idx])
        self.positions[particle_idx] = np.where(np.random.uniform(0, 1, self.num_features) < prob, 1, 0)

    def fitness(self, position):
        selected_features = np.where(position == 1)[0]
        if len(selected_features) == 0:
            return 0.0
        return train_evaluate_dbn(self.X, self.y, selected_features)

    def optimize(self):
        for particle_idx in range(self.num_particles):
            score = self.fitness(self.positions[particle_idx])
            self.personal_best_scores[particle_idx] = score
            if score > self.global_best_score:
                self.global_best_score = score
                self.global_best_position = self.positions[particle_idx].copy()

        for generation in range(self.num_generations):
            print(f"Generation {generation + 1}/{self.num_generations}")
            for particle_idx in range(self.num_particles):
                self.update_velocity(particle_idx)
                self.update_position(particle_idx)
                score = self.fitness(self.positions[particle_idx])
                if score > self.personal_best_scores[particle_idx]:
                    self.personal_best_scores[particle_idx] = score
                    self.personal_best_positions[particle_idx] = self.positions[particle_idx].copy()
                if score > self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[particle_idx].copy()

        selected_features = np.where(self.global_best_position == 1)[0]
        print(f"Selected {len(selected_features)} features with accuracy: {self.global_best_score}")
        return selected_features