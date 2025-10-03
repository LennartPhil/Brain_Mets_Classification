# TO-DO:
# - add option to switch between accuracy and auc for the k-fold plot function

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import re

# helper functions for the plots

def plot_single_history(path_to_train_history, adjust_scales = False, title = ""):

    # Create a figure for plotting all folds
    fig, axes = plt.subplots(3, 2, figsize=(15, 10)) # To visualize both loss and accuracy
    axes = axes.ravel()

    alpha_all = 0.3
    alpha_avg = 0.7

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

    # Load the training history
    history = np.load(path_to_train_history, allow_pickle=True).item()
    
    # Plot the loss and accuracy for the current fold
    history_df = pd.DataFrame(history)

    history_df["loss_diff"] = history_df["loss"] - history_df["val_loss"]
    history_df["accuracy_diff"] = history_df["accuracy"] - history_df["val_accuracy"]

    # Plot training and validation loss
    history_df[['loss']].plot(ax=axes[0], color=colors[0], legend=False, alpha=alpha_all)
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    # Plot training and validation accuracy
    history_df[['accuracy']].plot(ax=axes[1], color=colors[0], legend=False, alpha=alpha_all)
    axes[1].set_title('Training Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].grid(True)

    history_df[["val_loss"]].plot(ax=axes[2], color=colors[0], legend=False, alpha=alpha_all)
    axes[2].set_title('Validation Loss')
    axes[2].set_xlabel('Epochs')
    axes[2].set_ylabel('Val Loss')
    axes[2].grid(True)
    val_loss_y_max = abs(history_df[["val_loss"]].max().values[0])
    # if the maximum value of the val loss is larger than 10, adjust the y-axis
    if adjust_scales:
        if val_loss_y_max > 10:
            axes[2].set_ylim(0, 3)

    history_df[["val_accuracy"]].plot(ax=axes[3], color=colors[0], legend=False, alpha=alpha_all)
    axes[3].set_title('Validation Accuracy')
    axes[3].set_xlabel('Epochs')
    axes[3].set_ylabel('Val Accuracy')
    axes[3].grid(True)

    history_df[["loss_diff"]].plot(ax=axes[4], color=colors[0], legend=False, alpha=alpha_all)
    axes[4].set_title('Loss Difference between Training and Validation')
    axes[4].set_xlabel('Epochs')
    axes[4].set_ylabel('Loss Difference')
    axes[4].grid(True)
    loss_diff_y_max = abs(history_df[["loss_diff"]].max().values[0])
    loss_diff_y_min = abs(history_df[["loss_diff"]].min().values[0])
    if adjust_scales:
        if loss_diff_y_max > 10 or loss_diff_y_min > 10:
            axes[4].set_ylim(-2, 2)

    history_df[["accuracy_diff"]].plot(ax=axes[5], color=colors[0], legend=False, alpha=alpha_all)
    axes[5].set_title('Accuracy Difference between Training and Validation')
    axes[5].set_xlabel('Epochs')
    axes[5].set_ylabel('Accuracy Difference')
    axes[5].grid(True)

    plt.tight_layout()

    if title != "":
        plt.suptitle(title)

    plt.show()

def plot_single_history_one_graph(path_to_train_history, epochs = 600, title = ""):

    history = np.load(path_to_train_history, allow_pickle=True).item()

    print(history.keys())

    pd.DataFrame(history)[["loss","val_loss", "accuracy","val_accuracy"]].plot(figsize=(8,5),
                            xlim=(0,epochs),
                            ylim=(0,3),
                            grid=True,
                            xlabel="Epochs",
                            style=["r-","b-"],)
    
    if title != "":
        plt.title(title)

    plt.show()

def plot_training_history(path_to_train_history, title="", custom_loss_limit = None, compare_metric="accuracy"):
    """
    Loads and plots the training/validation loss and accuracy from a history file.

    This function automatically finds and annotates the epoch with the highest
    validation accuracy, which is often the best candidate for the final model.

    Args:
        path_to_train_history (str): The file path to the saved training history .npy file.
                                     The history is expected to be a dictionary with keys
                                     'loss', 'val_loss', 'accuracy', 'val_accuracy'.
        title (str, optional): A custom title for the plot. Defaults to "".
        custom_loss_limit (float, optional): A custom upper limit for the loss y-axis.
        compare_metric (str, optional): Which metric to compare on the right axis and for
                                        "best metric" annotations. Options: "accuracy" or "auc". Defaults to "accuracy".
    """
    try:
        history = np.load(path_to_train_history, allow_pickle=True).item()
    except FileNotFoundError:
        print(f"Error: The file '{path_to_train_history}' was not found.")
        return
    
    # --- Resolve which metric to use (accuracy or auc) ---
    compare_metric = str(compare_metric).lower().strip()
    if compare_metric not in {"accuracy", "auc"}:
        print(f"Warning: Unsupported compare_metric='{compare_metric}'. Falling back to 'accuracy'.")
        compare_metric = "accuracy"
    train_key = compare_metric
    val_key   = f"val_{compare_metric}"
    right_label = "AUC" if compare_metric == "auc" else "Accuracy"

    # Ensure all necessary keys are in the history file
    required_keys = ["loss", "val_loss", train_key, val_key]
    if not all(key in history for key in required_keys):
        print(f"Error: The history dictionary is missing one of the required keys: {required_keys}")
        print(f"Available keys: {list(history.keys())}")
        return

    # Convert to pandas DataFrame for easy handling
    history_clean = {key: [float(v) if v is not None else np.nan for v in values] for key, values in history.items()}
    df = pd.DataFrame(history_clean)
    epochs = len(df)

    # Robustness Check
    if df['val_loss'].isnull().all() or df[val_key].isnull().all():
        print("="*60)
        print(f"ERROR for '{title}':")
        print(f"The training history contains no valid data for 'val_loss' or '{val_key}'.")
        print("The entire column is NaN, indicating the training run may have failed immediately.")
        print("Aborting plot.")
        print("="*60)
        # You can optionally print the DataFrame to inspect it
        # print("DataFrame content:")
        # print(df.to_string())
        return

    # --- Find the best epoch based on validation accuracy ---
    # np.argmax returns the index of the maximum value

    #best_epoch_idx = df['val_accuracy'].idxmax()
    best_val_loss_epoch_idx = df['val_loss'].idxmin()
    best_accuracy_epoch_idc = df[val_key].idxmax()

    best_loss_metric = df.loc[best_val_loss_epoch_idx, val_key]
    best_loss_loss = df.loc[best_val_loss_epoch_idx, 'val_loss']
    
    # Retrieve the values at that best epoch
    best_metric_value = df.loc[best_accuracy_epoch_idc, val_key]
    best_metric_loss = df.loc[best_accuracy_epoch_idc, 'val_loss']
    
    # Add 1 to index because epochs are typically 1-based for humans
    best_val_loss_epoch_num = best_val_loss_epoch_idx + 1
    best_metric_epoch_num = best_accuracy_epoch_idc + 1

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot Loss on the primary y-axis (ax1)
    color = 'tab:red'
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    ax1.plot(df.index + 1, df['loss'], 'r-', label='Training Loss')
    ax1.plot(df.index + 1, df['val_loss'], 'm-', label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # Create a second y-axis for Accuracy that shares the same x-axis
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel(right_label, color=color, fontsize=12)
    ax2.plot(df.index + 1, df[train_key], 'b-', label=f'Training {right_label}')
    ax2.plot(df.index + 1, df[val_key], 'c-', label=f'Validation {right_label}')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    # --- Add annotation for the best validation accuracy ---
    plt.axvline(x=best_metric_epoch_num, color='k', linestyle='--', linewidth=2,
                label=f'Best Val Accuracy Epoch ({best_metric_epoch_num})')

    # Add an annotation box pointing to the best validation accuracy point
    annotation_text = f'Best {val_key}: {best_metric_value:.4f}\nEpoch: {best_metric_epoch_num}'
    ax2.annotate(annotation_text,
                xy=(best_metric_epoch_num, best_metric_value),
                xytext=(best_metric_epoch_num - (epochs*0.25), best_metric_value - 0.05), # Position text slightly away
                arrowprops=dict(facecolor='blue', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="k", lw=1, alpha=0.5))

    # --- Add annotation for the validation loss epoch ---
    # Draw a vertical line to mark the best epoch
    plt.axvline(x=best_val_loss_epoch_num, color='k', linestyle=':', linewidth=2,
                label=f'Best Epoch ({best_val_loss_epoch_num})')

    # Add an annotation box pointing to the best validation accuracy point
    annotation_text = f'Best val_loss: {best_loss_loss:.4f}\nWith {val_key}: {best_loss_metric:.4f}\nEpoch: {best_val_loss_epoch_num}'
    ax2.annotate(annotation_text,
                xy=(best_val_loss_epoch_num, best_loss_metric),
                xytext=(best_val_loss_epoch_num - (epochs*0.25), best_loss_metric - 0.05), # Position text slightly away
                arrowprops=dict(facecolor='green', shrink=0.05, width=1.5, headwidth=8),
                fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc="lightgreen", ec="k", lw=1, alpha=0.5))

    # --- Final plot styling ---
    plot_title = "Training & Validation History"
    if title:
        plot_title = f"{title} - {plot_title}"
    plt.title(plot_title, fontsize=16, fontweight='bold')
    
    # Set plot limits
    ax1.set_xlim(0, epochs + 1)
    if custom_loss_limit is not None:
        ax1.set_ylim(0, custom_loss_limit)
    else:
        ax1.set_ylim(0, max(df['loss'].max(), df['val_loss'].max()) * 1.1) # Dynamic y-limit for loss
    ax2.set_ylim(0, 1.05) # Accuracy/AUC is between 0 and 1

    fig.tight_layout() # Adjust plot to prevent labels from overlapping
    plt.show()

    # --- Print summary to the console ---
    print("="*50)
    print(f"Training History Analysis for: '{title}'")
    print(f"Total Epochs Trained: {epochs}")
    print("-"*50)
    print(f"Lowest Validation Loss of {best_loss_loss:.4f} achieved at Epoch {best_val_loss_epoch_num}.")
    print(f"Validation {right_label} at this epoch was {best_loss_metric:.4f}.")
    print("-"*50)
    print(f"Highest Validation {right_label} of {best_metric_value:.4f} achieved at Epoch {best_metric_epoch_num}.")
    print(f"Validation Loss at this epoch was {best_metric_loss:.4f}.")
    print("="*50)




# --- PLOT HISTORIES ---


def plot_fold_histories(path_to_train_history, title = ""):
    # Define the base path to your training history files
    base_path_to_train_history = path_to_train_history

    # Create a figure for plotting all folds
    fig, axes = plt.subplots(3, 2, figsize=(15, 10)) # To visualize both loss and accuracy
    axes = axes.ravel()

    alpha_all = 0.3
    alpha_avg = 0.7

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

    loss_list = []
    val_loss_list = []
    accuracy_list = []
    val_accuracy_list = []
    loss_diff_list = []
    accuracy_diff_list = []

    # Loop through each fold and load the training history
    for fold in range(10):
        # Construct the path for the current fold
        path_to_train_history = base_path_to_train_history + f"{fold}.npy"

        # Load the training history
        history = np.load(path_to_train_history, allow_pickle=True).item()
        
        # Plot the loss and accuracy for the current fold
        history_df = pd.DataFrame(history)

        history_df["loss_diff"] = history_df["loss"] - history_df["val_loss"]
        history_df["accuracy_diff"] = history_df["accuracy"] - history_df["val_accuracy"]

        loss_list.append(history_df["loss"])
        val_loss_list.append(history_df["val_loss"])
        accuracy_list.append(history_df["accuracy"])
        val_accuracy_list.append(history_df["val_accuracy"])
        loss_diff_list.append(history_df["loss_diff"])
        accuracy_diff_list.append(history_df["accuracy_diff"])

        # Plot training and validation loss
        history_df[['loss']].plot(ax=axes[0], color=colors[fold], legend=False, alpha=alpha_all)
        axes[0].set_title('Training Loss (All Folds & Average)')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].grid(True)

        # Plot training and validation accuracy
        history_df[['accuracy']].plot(ax=axes[1], color=colors[fold], legend=False, alpha=alpha_all)
        axes[1].set_title('Training Accuracy (All Folds & Average)')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Accuracy')
        axes[1].grid(True)

        history_df[["val_loss"]].plot(ax=axes[2], color=colors[fold], legend=False, alpha=alpha_all)
        axes[2].set_title('Validation Loss (All Folds & Average)')
        axes[2].set_xlabel('Epochs')
        axes[2].set_ylabel('Val Loss')
        axes[2].grid(True)

        history_df[["val_accuracy"]].plot(ax=axes[3], color=colors[fold], legend=False, alpha=alpha_all)
        axes[3].set_title('Validation Accuracy (All Folds & Average)')
        axes[3].set_xlabel('Epochs')
        axes[3].set_ylabel('Val Accuracy')
        axes[3].grid(True)

        history_df[["loss_diff"]].plot(ax=axes[4], color=colors[fold], legend=False, alpha=alpha_all)
        axes[4].set_title('Loss Difference between Training and Validation (All Folds & Average)')
        axes[4].set_xlabel('Epochs')
        axes[4].set_ylabel('Loss Difference')
        axes[4].grid(True)

        history_df[["accuracy_diff"]].plot(ax=axes[5], color=colors[fold], legend=False, alpha=alpha_all)
        axes[5].set_title('Accuracy Difference between Training and Validation (All Folds & Average)')
        axes[5].set_xlabel('Epochs')
        axes[5].set_ylabel('Accuracy Difference')
        axes[5].grid(True)

    # Add a legend for clarity (for different folds)
    for i in range(6):
        axes[i].legend([f'Fold {i+1}' for i in range(10)], loc='upper right', fontsize='small')

    # Calculate average metrics across all folds
    avg_loss = pd.concat(loss_list, axis=1).mean(axis=1)
    avg_val_loss = pd.concat(val_loss_list, axis=1).mean(axis=1)
    avg_accuracy = pd.concat(accuracy_list, axis=1).mean(axis=1)
    avg_val_accuracy = pd.concat(val_accuracy_list, axis=1).mean(axis=1)
    avg_loss_diff = pd.concat(loss_diff_list, axis=1).mean(axis=1)
    avg_accuracy_diff = pd.concat(accuracy_diff_list, axis=1).mean(axis=1)

    avg_loss.plot(ax=axes[0], color='black', linestyle='-', label='Average Loss')
    avg_val_loss.plot(ax=axes[2], color='black', linestyle='-', label='Average Val Loss')

    avg_accuracy.plot(ax=axes[1], color='black', linestyle='-', label='Average Accuracy')
    avg_val_accuracy.plot(ax=axes[3], color='black', linestyle='-', label='Average Val Accuracy')

    avg_loss_diff.plot(ax=axes[4], color='black', linestyle='-', label='Average Loss Diff')
    avg_accuracy_diff.plot(ax=axes[5], color='black', linestyle='-', label='Average Accuracy Diff')

    plt.tight_layout()

    if title != "":
        plt.suptitle(title)
        
    plt.show()

def plot_lr_history(path_to_train_history, title = ""):

    path_to_train_history = path_to_train_history

    history = np.load(path_to_train_history, allow_pickle=True).item()

    plt.semilogx(history["lr"], history["loss"])
    plt.title("loss vs. learning rate")
    plt.axis([1e-8, 1e-1, 0, 3])

    if title != "":
        title = title + " - loss vs. learning rate"
        plt.title(title)

    print("Epochs: ", len(history["loss"]))

def plot_lr_history(path_to_train_history, title: str = "", y_axis_max: int = 5) -> float:
    """Plot loss vs. learning-rate from a history npy file and highlight the best lr.

    Parameters
    ----------
    path_to_train_history : str
        Path to a saved history dictionary with keys "lr" and "loss".
    title : str, optional
        Optional prefix for the plot title.

    Returns
    -------
    float
        The learning rate that produced the lowest loss.
    """
    # --- load history --------------------------------------------------------
    try:
        # Load the history dictionary from the numpy file
        history = np.load(path_to_train_history, allow_pickle=True).item()
        lrs = history["lr"]
        losses = history["loss"]
    except FileNotFoundError:
        print(f"Error: The file '{path_to_train_history}' was not found.")
        return None, None
    except KeyError:
        print("Error: The history file must contain 'lr' and 'loss' keys.")
        return None, None
    
    min_loss_idx = np.argmin(losses)

    min_loss_value = losses[min_loss_idx]
    best_lr = lrs[min_loss_idx]

    # --- basic plot ----------------------------------------------------------
    plt.figure(figsize=(12, 6))
    plt.semilogx(lrs, losses)
    plt.xlabel("Learning Rate (log scale)", fontsize=12)
    plt.ylabel("loss", fontsize=12)
    plt.axis([1e-8, 1e-1, 0, y_axis_max])
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)


    # --- highlight the best point -------------------------------------------
    best_idx  = np.argmin(losses)
    best_lr   = lrs[best_idx]
    best_loss = losses[best_idx]

    # red dot on the curve
    plt.scatter(best_lr, best_loss, color="red", zorder=5)
    # annotate slightly to the upper right of the point
    plt.annotate(
        f"lowest loss = {best_loss:.4f}\nlr = {best_lr:.2e}\n",
        xy=(best_lr, best_loss),
        xytext=(0.03, 0.85),            # axes-fraction coordinates
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", lw=1),
        fontsize=9,
        horizontalalignment="left",
        verticalalignment="top"
    )

    # --- title & housekeeping -----------------------------------------------
    full_title = "loss vs. learning rate" if not title else f"{title} – loss vs. learning rate"
    plt.title(full_title)
    #plt.tight_layout()

    # --- console output ------------------------------------------------------
    print(f"Epochs: {len(losses)}")
    print(f"Lowest loss {best_loss:.4f} at learning rate {best_lr:.2e}")

    return best_lr

def plot_kfold_results_with_confidence_band(path_pattern, title="", custom_loss_limit=None):
    """
    Loads, aggregates, and plots k-fold cross-validation training results
    from files with complex names.

    This function finds all history files matching a given pattern, extracts the
    fold number, and visualizes the mean training/validation loss and accuracy
    across all folds, with a shaded region representing one standard deviation.

    Args:
        path_pattern (str): A path and file pattern using wildcards to find the
                            history files. The pattern should have a '*' in place
                            of variable parts like timestamps.
                            Example for a file named
                            'history_..._fold_1_run_..._57.npy':
                            'path/to/files/history_*_fold_*_run_*.npy'
        title (str, optional): A custom title for the plot. Defaults to "".
        custom_loss_limit (float, optional): A custom upper limit for the loss y-axis.
                                             Defaults to None (dynamic scaling).
    """
    all_histories = []
    min_epochs = float('inf')

    # --- 1. Find all history files using the glob pattern ---
    print(f"Searching for files with pattern: {path_pattern}")
    filepaths = glob.glob(path_pattern)

    if not filepaths:
        print("\n--- ERROR ---")
        print("No files found matching the specified pattern.")
        print("Please check your `path_pattern` for typos or incorrect wildcards.")
        print("Example pattern for a file named '..._fold_1_run_...npy':")
        print("'path/to/dir/*_fold_*_run_*.npy'")
        return

    print(f"Found {len(filepaths)} matching files.")

    # --- 2. Load files, parse fold number, and find min epochs ---
    for f_path in filepaths:
        # Extract fold number using a regular expression
        match = re.search(r'fold_(\d+)', f_path)
        if not match:
            print(f"Warning: Could not parse fold number from filename, skipping: {f_path}")
            continue
        
        fold_num = int(match.group(1)) # group(1) is the captured digits (\d+)

        try:
            history = np.load(f_path, allow_pickle=True).item()
            df = pd.DataFrame(history)
            
            required_cols = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: Skipping fold {fold_num} due to missing columns. Found: {df.columns.tolist()}")
                continue
            
            # Store dataframe with its fold number for context
            all_histories.append({'fold': fold_num, 'df': df})
            if len(df) < min_epochs:
                min_epochs = len(df)
        except Exception as e:
            print(f"Error loading or processing file for fold {fold_num} at {f_path}: {e}")

    if not all_histories:
        print("Error: No valid history files could be loaded and parsed. Aborting plot.")
        return

    # Sort by fold number for consistent coloring, if ever needed
    all_histories.sort(key=lambda x: x['fold'])
    
    # Inform the user about truncation
    print(f"All runs will be truncated to the shortest run length of {min_epochs} epochs for fair comparison.")

    # --- 3. Truncate all histories and collect metrics ---
    metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy']
    metric_data = {metric: [] for metric in metrics}

    for history_item in all_histories:
        df = history_item['df']
        truncated_df = df.iloc[:min_epochs]
        for metric in metrics:
            metric_data[metric].append(truncated_df[metric].reset_index(drop=True))

    # --- 4. Calculate mean and standard deviation for each metric ---
    mean_metrics = {}
    std_metrics = {}
    for metric in metrics:
        metric_df = pd.concat(metric_data[metric], axis=1)
        mean_metrics[metric] = metric_df.mean(axis=1)
        std_metrics[metric] = metric_df.std(axis=1)

    # --- 5. Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(16, 8))

    # Plot Loss (mean and confidence band) on the primary y-axis
    color = 'tab:red'
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Loss', color=color, fontsize=12)
    ax1.plot(mean_metrics['loss'], color='red', linestyle='-', label='Mean Training Loss')
    ax1.plot(mean_metrics['val_loss'], color='magenta', linestyle='-', label='Mean Validation Loss')
    ax1.fill_between(range(min_epochs), mean_metrics['loss'] - std_metrics['loss'], mean_metrics['loss'] + std_metrics['loss'], color='red', alpha=0.15)
    ax1.fill_between(range(min_epochs), mean_metrics['val_loss'] - std_metrics['val_loss'], mean_metrics['val_loss'] + std_metrics['val_loss'], color='magenta', alpha=0.15)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc='upper left')

    # Create a second y-axis for Accuracy
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color, fontsize=12)
    ax2.plot(mean_metrics['accuracy'], color='blue', linestyle='-', label='Mean Training Accuracy')
    ax2.plot(mean_metrics['val_accuracy'], color='cyan', linestyle='-', label='Mean Validation Accuracy')
    ax2.fill_between(range(min_epochs), mean_metrics['accuracy'] - std_metrics['accuracy'], mean_metrics['accuracy'] + std_metrics['accuracy'], color='blue', alpha=0.15)
    ax2.fill_between(range(min_epochs), mean_metrics['val_accuracy'] - std_metrics['val_accuracy'], mean_metrics['val_accuracy'] + std_metrics['val_accuracy'], color='cyan', alpha=0.15)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc='upper right')

    # --- 6. Annotations for best average performance ---
    best_val_loss_epoch = mean_metrics['val_loss'].idxmin()
    best_val_loss = mean_metrics['val_loss'].min()
    val_acc_at_best_loss = mean_metrics['val_accuracy'][best_val_loss_epoch]

    best_val_acc_epoch = mean_metrics['val_accuracy'].idxmax()
    best_val_acc = mean_metrics['val_accuracy'].max()
    
    ax1.axvline(x=best_val_loss_epoch, color='gray', linestyle=':', linewidth=2)
    annotation_text_loss = f'Lowest Avg. Val Loss: {best_val_loss:.4f}\n(Val Acc: {val_acc_at_best_loss:.4f})\nEpoch: {best_val_loss_epoch + 1}'
    ax1.annotate(annotation_text_loss, xy=(best_val_loss_epoch, best_val_loss),
                 xytext=(best_val_loss_epoch + 5, best_val_loss + 0.2),
                 arrowprops=dict(facecolor='magenta', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightpink", alpha=0.8))
    
    ax2.axvline(x=best_val_acc_epoch, color='black', linestyle='--', linewidth=2)
    annotation_text_acc = f'Highest Avg. Val Acc: {best_val_acc:.4f}\nEpoch: {best_val_acc_epoch + 1}'
    ax2.annotate(annotation_text_acc, xy=(best_val_acc_epoch, best_val_acc),
                 xytext=(best_val_acc_epoch - (min_epochs*0.4), best_val_acc - 0.1),
                 arrowprops=dict(facecolor='cyan', shrink=0.05, width=1.5, headwidth=8),
                 fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", alpha=0.8))

    # --- 7. Final plot styling ---
    plot_title = "K-Fold Cross-Validation Performance"
    if title:
        plot_title = f"{title} - {plot_title}"
    plt.title(plot_title, fontsize=16, fontweight='bold')
    
    ax1.set_xlim(0, min_epochs -1) # Adjust x-axis limit
    if custom_loss_limit is not None:
        ax1.set_ylim(0, custom_loss_limit)
    else:
        ylim_max = np.percentile(mean_metrics['val_loss'].dropna(), 98) * 1.2
        ax1.set_ylim(0, min(ylim_max, 5))
    ax2.set_ylim(0, 1.05)
    
    fig.tight_layout()
    plt.show()

    # --- 8. Print summary to the console ---
    print("="*60)
    print(f"K-Fold Cross-Validation Summary for: '{title}'")
    print(f"Analyzed {len(all_histories)} folds from pattern.")
    print(f"Metrics truncated to {min_epochs} epochs.")
    print("-"*60)
    print(f"Best Average Validation Loss: {best_val_loss:.4f} at epoch {best_val_loss_epoch + 1}")
    print(f"   - Validation Accuracy at this epoch: {val_acc_at_best_loss:.4f}")
    print(f"Highest Average Validation Accuracy: {best_val_acc:.4f} at epoch {best_val_acc_epoch + 1}")
    print("="*60)


def summarize_kfold_results(path_pattern, title=""):
    """
    Analyzes and summarizes k-fold cross-validation results without truncation.

    This function loads all history files matching a pattern, calculates key
    performance indicators for each fold at its best epoch (based on validation
    accuracy), and presents a detailed summary table and analysis.

    Args:
        path_pattern (str): A path and file pattern using wildcards to find the
                            history files.
        title (str, optional): A custom title for the summary output.
    """
    print("="*80)
    summary_title = "K-Fold Performance Summary"
    if title:
        summary_title = f"{title} - {summary_title}"
    print(summary_title)
    print("="*80)

    filepaths = glob.glob(path_pattern)
    if not filepaths:
        print("\n--- ERROR: No files found matching the specified pattern. ---\n")
        return

    results_list = []

    # --- 1. Loop through each fold's history file ---
    for f_path in filepaths:
        match = re.search(r'fold_(\d+)', f_path)
        if not match:
            continue
        fold_num = int(match.group(1))

        try:
            history = np.load(f_path, allow_pickle=True).item()
            df = pd.DataFrame(history)

            # --- 2. Find the best epoch for this fold ---
            # We define "best" as the epoch with the highest validation accuracy
            best_epoch_idx = df['val_accuracy'].idxmax()
            
            # --- 3. Extract all relevant metrics at that best epoch ---
            best_val_acc = df['val_accuracy'].iloc[best_epoch_idx]
            best_val_loss = df['val_loss'].iloc[best_epoch_idx]
            train_acc_at_best = df['accuracy'].iloc[best_epoch_idx]
            train_loss_at_best = df['loss'].iloc[best_epoch_idx]

            # --- 4. Calculate the "Overfitting Gap" ---
            # A large positive gap indicates overfitting.
            overfitting_gap = train_acc_at_best - best_val_acc

            results_list.append({
                'Fold': fold_num,
                'Total Epochs': len(df),
                'Best Epoch': best_epoch_idx + 1, # +1 for human-readable 1-based index
                'Val Accuracy': best_val_acc,
                'Val Loss': best_val_loss,
                'Train Accuracy': train_acc_at_best,
                'Overfitting Gap (Acc)': overfitting_gap
            })
        except Exception as e:
            print(f"Could not process file {f_path}: {e}")

    if not results_list:
        print("--- No valid results could be compiled. ---")
        return

    # --- 5. Create and display the summary DataFrame ---
    results_df = pd.DataFrame(results_list).sort_values(by='Fold').set_index('Fold')
    
    # Set display format for better readability
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print("\n--- Performance of Each Fold (at its best epoch) ---\n")
    print(results_df)

    # --- 6. Calculate and display aggregate statistics ---
    print("\n\n--- Aggregate Performance Across All Folds ---\n")
    summary_stats = results_df.describe().loc[['mean', 'std', 'min', 'max']]
    print(summary_stats)
    
    # --- 7. Provide analysis on Over/Underfitting ---
    print("\n\n--- Overfitting and Underfitting Analysis ---\n")
    avg_val_acc = results_df['Val Accuracy'].mean()
    avg_gap = results_df['Overfitting Gap (Acc)'].mean()

    # Underfitting Check
    print("1. Underfitting Check:")
    if avg_val_acc < 0.6: # This threshold is an example; adjust based on your problem's difficulty
        print(f"   - The average validation accuracy is low ({avg_val_acc:.4f}).")
        print("   - The model may be UNDERFITTING. It might be too simple, need more training time, or require better features.")
    else:
        print(f"   - Average validation accuracy ({avg_val_acc:.4f}) seems reasonable. Major underfitting is unlikely.")

    # Overfitting Check
    print("\n2. Overfitting Check:")
    if avg_gap > 0.15: # An average gap of >15% is a strong sign of overfitting
        print(f"   - The average overfitting gap is significant ({avg_gap:.4f}).")
        print("   - The model is likely OVERFITTING. It learns the training data well but doesn't generalize.")
        print("   - Consider adding more regularization (Dropout, L2), using more data augmentation, or reducing model complexity.")
    elif avg_gap > 0.05:
        print(f"   - A moderate overfitting gap was detected ({avg_gap:.4f}). Monitor closely.")
    else:
        print(f"   - The overfitting gap ({avg_gap:.4f}) is small. The model appears to generalize well.")
    
    print("\n" + "="*80)