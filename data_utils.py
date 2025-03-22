import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle

class NoisyNumbers():
    """
    Class to load the sklearn digits dataset and apply noise to it.
    """
    def __init__(self):
        # load sklearn digits dataset as pandas dataframe
        digits_raw = load_digits()
        self.X = pd.DataFrame(digits_raw.data)
        self.y = pd.Series(digits_raw.target)

        # define train/test splitting mask
        np.random.seed(42)  # Ensures reproducibility
        self.train_mask = np.random.rand(len(self.X)) < 0.8
        self.test_mask = ~self.train_mask

        self.X_train = self.X[self.train_mask]
        self.y_train = self.y[self.train_mask]
        self.X_test = self.X[self.test_mask]
        self.y_test = self.y[self.test_mask]


    def apply_noise_to_digits(self, digit_indices, p_noise=0.1, random_state=None):
        # simulate sparse salt-and-pepper noise on multiple digits with probability p_noise
        if random_state is not None:
            np.random.seed(random_state)
        noisy_digits = self.X.iloc[digit_indices].copy()
        for idx in digit_indices:
            noise_mask = np.random.rand(self.X.shape[1]) < p_noise
            noisy_digits.loc[idx, noise_mask] = np.random.choice([0, 16], size=np.sum(noise_mask))
        return noisy_digits


    def visualise_digits(self, p_noise=None, random_state=42):
        # random selection of 10 digits
        digits = self.X.sample(n=10, random_state=random_state)
        if p_noise is not None:
            noisy_digits = digits.copy()
            for idx in digits.index:
                noise_mask = np.random.rand(digits.shape[1]) < p_noise
                noisy_digits.loc[idx, noise_mask] = np.random.choice([0, 16], size=np.sum(noise_mask))
        else:
            noisy_digits = digits

        # plot the digits
        fig, axs = plt.subplots(1, 10, figsize=(15, 2))
        for i, ax in enumerate(axs.flatten()):
            ax.imshow(noisy_digits.iloc[i].values.reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
            ax.set_title(f"Label: {self.y.iloc[digits.index[i]]}")
        plt.show()


    def get_noisy_data(self, p_dataset=0.1, p_image=0.1, random_state=None):
        """
        Generate noisy versions of the training and test datasets.

        Parameters:
        p_dataset (float): Proportion of samples in the dataset to have noise applied (0 <= p_dataset <= 1).
        p_image (float): Probability of noise being applied to individual pixels within a noisy sample (0 <= p_image <= 1).
        random_state (int or None): Seed for reproducibility of noise application. If None, the randomness is not seeded.

        Returns:
        tuple: A tuple containing noisy versions of the training dataset (noisy_train) and the test dataset (noisy_test), 
        as well as the indices of the corrupted samples.
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Select samples for noise in train and test datasets
        train_indices = self.X_train.sample(frac=p_dataset, random_state=random_state).index
        test_indices = self.X_test.sample(frac=p_dataset, random_state=random_state).index

        # Apply noise to selected samples
        noisy_train = self.X_train.copy()
        noisy_train.loc[train_indices] = self.apply_noise_to_digits(train_indices, p_noise=p_image, random_state=random_state)

        noisy_test = self.X_test.copy()
        noisy_test.loc[test_indices] = self.apply_noise_to_digits(test_indices, p_noise=p_image, random_state=random_state)

        return noisy_train, noisy_test, train_indices, test_indices

class SineData():
    """
    Class to generate a sine dataset.
    """
    def __init__(self, random_state=None):
        self.random_state = random_state

    def generate_data_pure(self, x_min, x_max, n_samples):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X = np.random.uniform(x_min, x_max, n_samples)
        y = np.sin(X)
        return X, y
    
    def get_data_epistemic_gap(self, L, gap_width=0.1, n_samples=1000):
        """
        create sinusoidal data with a gap in the middle
        """
        assert gap_width < L, "Gap width must be less than the length of the data"
        # range 1
        X_1, y_1 = self.generate_data_pure(0, L//2, n_samples//2)

        # range 2
        X_2, y_2 = self.generate_data_pure(L//2+gap_width, L, n_samples//2)

        # combine
        X = np.concatenate([X_1, X_2])
        y = np.concatenate([y_1, y_2])

        # shuffle
        X, y = shuffle(X, y, random_state=self.random_state)
        return X, y

    def get_data_variable_noise(self, L, X_noise_min, X_noise_max, noise_sd, n_samples=1000):
        """
        create sinusoidal data with a segment with noise
        """
        X, y = self.generate_data_pure(0, L, n_samples)

        # generate mask for samples on which noise is applied to y based on X_noise_min and X_noise_max
        noise_mask = (X >= X_noise_min) & (X <= X_noise_max)

        # apply noise to the samples
        y[noise_mask] = np.random.normal(y[noise_mask], noise_sd)

        # shuffle
        X, y = shuffle(X, y, random_state=self.random_state)

        return X, y

    def visualise_data(self, X, y):
        plt.scatter(X, y, marker='o')
        plt.show()

import matplotlib.pyplot as plt
import numpy as np

def visualise_multiple_passes(
    X_test, 
    y_test, 
    mc_preds,
    X_train=None, 
    bins=30
):
    """
    Plot:
      1) Top subplot: ground truth + T forward-pass predictions (all scatter)
      2) Middle subplot: standard deviation of T passes at each X (scatter)
      3) Optional bottom subplot: histogram of training X data distribution
    
    Args:
        X_test (array-like): shape (N,) or (N,1). The test input features.
        y_test (array-like): shape (N,). The true test target values.
        mc_preds (np.ndarray): shape (T, N). Predictions across T forward passes.
        X_train (array-like, optional): shape (M,) or (M,1). Training input features 
            for plotting a density/histogram. If None, we skip this subplot.
        bins (int): number of bins in the training data histogram.
    """
    # Convert test inputs to NumPy arrays and flatten if needed
    X_test = np.array(X_test).flatten()
    y_test = np.array(y_test).flatten()
    
    # Decide how many subplots
    # - 2 subplots if no training data is provided
    # - 3 subplots if training data is provided
    if X_train is not None:
        nrows = 3
    else:
        nrows = 2
    
    # Adjust figure size accordingly
    fig, axs = plt.subplots(nrows, 1, figsize=(8, 4 * nrows), sharex=True)
    if nrows == 2:
        ax_top, ax_mid = axs
        ax_bottom = None
    else:
        ax_top, ax_mid, ax_bottom = axs

    # Sort by X for a nicer left-to-right scatter (for test data)
    sorted_indices = np.argsort(X_test)
    X_sorted = X_test[sorted_indices]
    y_sorted = y_test[sorted_indices]

    # mc_preds should be shape (T, N). 
    # Sort each row by the same indices:
    mc_preds_sorted = mc_preds[:, sorted_indices]  # shape: (T, N)

    # -- Top Subplot: ground truth + T forward passes (scatter) --
    ax_top.scatter(X_sorted, y_sorted, color='red', s=10, label='Ground Truth')
    
    T = mc_preds.shape[0]
    for t in range(T):
        ax_top.scatter(
            X_sorted,
            mc_preds_sorted[t],
            color='blue',
            alpha=0.3,
            s=10,
            label='Predictions (MC pass)' if t == 0 else None
        )
    ax_top.set_ylabel('Value / Predictions')
    ax_top.set_title("MC Dropout Predictions vs Ground Truth")
    ax_top.legend()

    # -- Middle Subplot: standard deviation as scatter --
    # Compute std across T dimension -> shape [N]
    stds = mc_preds_sorted.std(axis=0)
    # max_min_diff = mc_preds_sorted.max(axis=0) - mc_preds_sorted.min(axis=0)
    ax_mid.scatter(X_sorted, stds, color='black', s=10, alpha=0.8, label='Std of MC predictions')
    ax_mid.set_ylabel('Standard Deviation')
    ax_mid.legend()

    # -- Optional Bottom Subplot: training data histogram (X_train) --
    if X_train is not None:
        # Convert and flatten training X if necessary
        X_train_flat = np.array(X_train).flatten()
        ax_bottom.hist(X_train_flat, bins=bins, color='gray', alpha=0.7)
        ax_bottom.set_xlabel('X')
        ax_bottom.set_ylabel('Count')
        ax_bottom.set_title('Training Data Distribution (X)')

        # We share the X-axis. If your X_test is outside your training range, 
        # the histogram might be squished. You can set the x-limits if you like:
        # ax_bottom.set_xlim([X_sorted.min(), X_sorted.max()])
    else:
        # If no training data is provided, label the bottom axis in the middle subplot
        ax_mid.set_xlabel('X')

    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    # noisy_numbers = NoisyNumbers()
    # print(noisy_numbers.X.shape)

    # # Example usage of apply_noise_to_digits
    # # noisy_digits = noisy_numbers.apply_noise_to_digits(digit_indices=[0, 1, 2], p_noise=0.1, random_state=42)
    # # print(noisy_digits)

    # noisy_numbers.visualise_digits(p_noise=0.1)

    sine_data = SineData()
    X, y = sine_data.get_data_epistemic_gap(L=10, gap_width=2, n_samples=1000)
    sine_data.visualise_data(X, y)

    X, y = sine_data.get_data_variable_noise(L=10, X_noise_min=5, X_noise_max=7, noise_sd=0.5, n_samples=1000)
    sine_data.visualise_data(X, y)
