import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

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


if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    import pandas as pd

    CLASSIFICATION = True
    
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
        sine_data = SineData()
        X, y = sine_data.get_data_epistemic_gap(L=10, gap_width=2, n_samples=1000)
        sine_data.visualise_data(X, y)

        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

        # create model
        model = SimpleMLPRegressor(hidden_layers=(64, 8), dropout_rate=0.5, l2_reg=0.001, max_epochs=20, validation_split=0.2, random_state=42, verbose=True)
        model.fit(X_train, y_train)

        # predict across multiple MC passes
        T = 10
        mc_preds = model.predict_with_mc(X_test, T=T)

        # visualise: true distribution with overlay of model predictions across all passes
        # TODO: implement
