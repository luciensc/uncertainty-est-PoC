import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

from data_utils import *

from data_utils import SineData
class SimpleMLPClassifier:
    def __init__(self, 
                 hidden_layers=(128, 16),
                 batch_size=64, 
                 learning_rate=1e-3, 
                 max_epochs=100,
                 validation_split=0.1, 
                 shuffle=True, 
                 random_state=42,
                 dropout_rate=0.4, 
                 l2_reg=0.01, 
                 verbose=True):
        """
        A simple MLP classifier with dropout, L2 regularization, and standard cross-entropy loss.
        """
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.random_state = random_state
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.verbose = verbose

        self.model = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None

        if random_state is not None:
            torch.manual_seed(random_state)

    def _build_network(self, input_size, num_classes):
        layers = []
        in_features = input_size

        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            in_features = hidden_size

        # Output layer has 'num_classes' (no abstention)
        layers.append(nn.Linear(in_features, num_classes))

        self.model = nn.Sequential(*layers)

    def fit(self, X, y):
        """
        Fit the model to the training data.
        """
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes_ = self.label_encoder.classes_
        num_classes = len(self.classes_)

        # Split into train/validation if needed
        if self.validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_encoded, test_size=self.validation_split,
                shuffle=self.shuffle, random_state=self.random_state
            )
            X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        else:
            X_train = X
            y_train = y_encoded
            X_val_tensor = None
            y_val_tensor = None

        X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)

        input_size = X_train_tensor.shape[1]
        self._build_network(input_size, num_classes)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), 
                               lr=self.learning_rate, 
                               weight_decay=self.l2_reg)

        for epoch in range(self.max_epochs):
            total_loss = 0.0
            self.model.train()

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)

            # If using validation, evaluate
            if self.validation_split > 0:
                self._validate_model(X_val_tensor, y_val_tensor, epoch, avg_loss)
            else:
                if self.verbose:
                    print(f"Epoch [{epoch+1}/{self.max_epochs}], Loss: {avg_loss:.4f}")

        return self

    def _validate_model(self, X_val_tensor, y_val_tensor, epoch, avg_loss):
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        total_val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y).item()
                total_val_loss += loss

                _, predicted = torch.max(outputs, dim=1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        val_acc = 100.0 * correct / total
        avg_val_loss = total_val_loss / len(val_loader)
        if self.verbose:
            print(f"Epoch [{epoch+1}/{self.max_epochs}], "
                  f"Train Loss: {avg_loss:.4f}, "
                  f"Val Loss: {avg_val_loss:.4f}, "
                  f"Val Acc: {val_acc:.2f}%")

    def predict_proba(self, X):
        """
        Standard single forward pass (evaluation mode). No dropout at test time.
        """
        # Convert input to tensor
        X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32) if isinstance(X, pd.DataFrame) else X

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probabilities = nn.functional.softmax(logits, dim=1)
        return probabilities.cpu().numpy()

    def predict(self, X):
        """
        Predict class labels (hard labels) in standard eval mode.
        """
        probs = self.predict_proba(X)  # shape [N, num_classes]
        pred_indices = probs.argmax(axis=1)
        return self.label_encoder.inverse_transform(pred_indices)

    def predict_proba_mc(self, X, T=10):
        """
        Perform T forward passes with dropout *enabled*, 
        returning the stacked probabilities for each pass.
        """
        X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32) if isinstance(X, pd.DataFrame) else X
        
        # We'll store results here
        all_probs = []
        
        # Make sure dropout is active by calling train()
        self.model.train()
        
        for _ in range(T):
            # No grad but still in train mode => dropout is active
            with torch.no_grad():
                logits = self.model(X_tensor)
                probs = nn.functional.softmax(logits, dim=1)
                all_probs.append(probs.cpu().numpy())
        
        # all_probs is a list of length T, each [N, 10].
        # Stack: shape will be [T, N, num_classes].
        return np.stack(all_probs, axis=0)

    def predict_with_mc(self, X, T=10):
        """
        Predict class labels using the average probability across T MC dropout forward passes.
        """
        mc_probs = self.predict_proba_mc(X, T=T)  # shape [T, N, 10]
        mean_probs = mc_probs.mean(axis=0)        # shape [N, 10]
        pred_indices = mean_probs.argmax(axis=1)
        return self.label_encoder.inverse_transform(pred_indices)


import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SimpleMLPRegressor:
    def __init__(self,
                 hidden_layers=(128, 16),
                 batch_size=64,
                 learning_rate=1e-3,
                 max_epochs=100,
                 validation_split=0.1,
                 shuffle=True,
                 random_state=42,
                 dropout_rate=0.4,
                 l2_reg=0.01,
                 verbose=True):
        """
        A simple MLP regressor with dropout, L2 regularization, and MSE loss.
        """
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.random_state = random_state
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.verbose = verbose
        
        self.model = None

        # We will store the train and val losses for plotting later
        self.train_losses = []
        self.val_losses = []

        if random_state is not None:
            torch.manual_seed(random_state)
    
    def _build_network(self, input_size):
        """
        Build a feed-forward MLP with dropout.
        The output layer has a single neuron for regression.
        """
        layers = []
        in_features = input_size
        
        for i, hidden_size in enumerate(self.hidden_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0: #and i < len(self.hidden_layers) - 1:
                layers.append(nn.Dropout(self.dropout_rate))
            in_features = hidden_size
        
        # Output layer for regression: 1 neuron
        layers.append(nn.Linear(in_features, 1))
        
        self.model = nn.Sequential(*layers)
    
    def fit(self, X, y):
        """
        Fit the model to the training data (regression).
        X, y can be NumPy arrays or Pandas objects. We'll convert them to tensors.
        """
        # Convert to DataFrame/Series if not already
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
        
        # Split into train/validation if needed
        if self.validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=self.validation_split,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32).view(-1, 1)
        else:
            X_train = X
            y_train = y
            X_val_tensor = None
            y_val_tensor = None
        
        X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).view(-1, 1)
        
        input_size = X_train_tensor.shape[1]
        self._build_network(input_size)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.l2_reg)
        for epoch in range(self.max_epochs):
            total_loss = 0.0
            total_samples = 0  # Track total number of samples processed
            self.model.train()  # Enable dropout

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)

                # Accumulate loss scaled by batch size
                total_loss += loss.item() * batch_X.size(0)
                total_samples += batch_X.size(0)

                loss.backward()
                optimizer.step()

            # Compute the true average training loss for the epoch
            avg_loss = total_loss / total_samples
            self.train_losses.append(avg_loss)  # Store the average training loss

            # If using validation, evaluate
            if self.validation_split > 0 and X_val_tensor is not None:
                val_loss = self._validate_model(X_val_tensor, y_val_tensor)
                self.val_losses.append(val_loss)
                if self.verbose:
                    print(f"Epoch [{epoch+1}/{self.max_epochs}], "
                          f"Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                # if no val split, store a dummy None or repeat train loss if you want
                self.val_losses.append(None)
                if self.verbose:
                    print(f"Epoch [{epoch+1}/{self.max_epochs}], Loss: {avg_loss:.4f}")
        
        return self
    
    def _validate_model(self, X_val_tensor, y_val_tensor):
        """
        Evaluate the model on the validation set (MSE).
        """
        self.model.eval()
        criterion = nn.MSELoss()
        
        with torch.no_grad():
            outputs = self.model(X_val_tensor)
            val_loss = criterion(outputs, y_val_tensor).item()
        return val_loss
    
    def predict(self, X):
        """
        Single forward pass in eval mode (no dropout).
        Returns a NumPy array of shape (N,).
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor).cpu().numpy().flatten()
        return outputs
    
    def predict_mc(self, X, T=10):
        """
        Perform T forward passes with dropout *enabled*, 
        returning the stacked predictions for each pass.
        Shape of the returned array: (T, N).
        """
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X)
        X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32)
        
        self.model.train()  # Enable dropout
        all_preds = []
        
        for _ in range(T):
            with torch.no_grad():
                preds = self.model(X_tensor).cpu().numpy().flatten()
                all_preds.append(preds)
        
        return np.stack(all_preds, axis=0)  # Shape: (T, N)
    
    def predict_with_mc(self, X, T=10):
        """
        Predict using the average over T MC-dropout forward passes.
        Returns an array of shape (N,) with the mean prediction.
        """
        mc_preds = self.predict_mc(X, T=T)   # Shape: (T, N)
        mean_preds = mc_preds.mean(axis=0)   # Shape: (N,)
        return mean_preds

    def plot_losses(self):
        """
        Plot the training (and, if available, validation) loss over epochs.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label='Train Loss')
        
        # Only plot val losses if they are not None (i.e., if val_split > 0)
        # If you used self.val_losses.append(None) for no-val epochs, skip them
        val_loss_array = [v for v in self.val_losses if v is not None]
        if len(val_loss_array) == len(self.val_losses):  # means we have real val losses
            plt.plot(self.val_losses, label='Validation Loss')
        
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title("Training and Validation Loss Over Epochs")
        plt.legend()
        plt.show()



if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import pandas as pd

    CLASSIFICATION = False
    
    if CLASSIFICATION:
        # Load digits dataset
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Convert to DataFrame (optional)
        X_df = pd.DataFrame(X_scaled)
        y_df = pd.Series(y)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y_df, test_size=0.2, random_state=42
        )
        
        # Create classifier
        classifier = SimpleMLPClassifier(
            hidden_layers=(64, 8),
            dropout_rate=0.5,
            l2_reg=0.001,
            max_epochs=20,       # fewer epochs just for quick demo
            validation_split=0.2,
            random_state=42,
            verbose=True
        )
        
        # Fit
        classifier.fit(X_train, y_train)
        
        # Single-pass predictions (eval mode)
        probs_eval = classifier.predict_proba(X_test)
        preds_eval = classifier.predict(X_test)
        
        # MC Dropout predictions (multiple passes in train mode)
        T = 10
        mc_probs = classifier.predict_proba_mc(X_test, T=T)  # shape [T, N, 10]
        mc_mean_probs = mc_probs.mean(axis=0)  # average across T
        mc_preds = mc_mean_probs.argmax(axis=1)
        
        # Example: compute variance across T passes
        mc_var = mc_probs.var(axis=0)   # shape [N, 10]
        sample_variances = mc_var.sum(axis=1)  # shape [N], sum of variances across classes
        mean_sample_variance = sample_variances.mean()
        
        print("MC Dropout: sample variance shape:", sample_variances.shape)
        print("First 5 variance values:", sample_variances[:5])
        print("Mean sample variance:", mean_sample_variance)

    else: # regression
        sine_data = SineData(random_state=42)
        L = 30
        X_train, y_train = sine_data.get_data_epistemic_gap(L=L, gap_width=8, n_samples=500)

        # scale data
        scaler = StandardScaler()
        X_train = X_train.reshape(-1, 1)
        X_train = scaler.fit_transform(X_train)
        # visualise data
        # sine_data.visualise_data(X_train, y_train)

        # create model
        model = SimpleMLPRegressor(
            learning_rate=3e-4,
            hidden_layers=(1024, 1024, 1024, 1024),
            dropout_rate=0.2,
            l2_reg=0.000,
            max_epochs=300,
            validation_split=0.3,
            random_state=42,
            verbose=True, 
            batch_size=100
        )
        model.fit(X_train, y_train)

        # Plot train/val loss curves
        model.plot_losses()

        # generate test data: sinus data over linspace of full X range
        X_test = np.linspace(0, L, 500)
        y_test = np.sin(X_test)
        X_test = X_test.reshape(-1, 1)
        X_test = scaler.transform(X_test)

        # predict across multiple MC passes
        T = 200
        mc_preds = model.predict_mc(X_test, T=T)  # shape [T, N]
        # visualize multiple passes
        visualise_multiple_passes(
            X_test=X_test, 
            y_test=y_test, 
            mc_preds=mc_preds,
            X_train=X_train, 
            bins=50
        )