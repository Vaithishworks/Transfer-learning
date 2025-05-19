#!/usr/bin/env python3
"""
Plot training and validation losses alongside mean average precision over epochs.
"""

import json
import matplotlib.pyplot as plt


def load_metrics(file_path):
    """
    Load training and validation loss metrics from a JSON file.

    Args:
        file_path (str): Path to the metrics JSON file.

    Returns:
        tuple[list[int], list[float], list[float]]: Epochs, training losses, validation losses.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    epochs = []
    train_losses = []
    val_losses = []
    for entry in data:
        epochs.append(entry["epoch"])
        train_losses.append(entry["train"]["loss"])
        val_losses.append(entry["validation"]["loss"])

    return epochs, train_losses, val_losses


def load_map(file_path):
    """
    Load mean average precision (mAP) values from a JSON file.

    Args:
        file_path (str): Path to the maps JSON file.

    Returns:
        tuple[list[int], list[float]]: Epochs, mAP values.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    # Convert string keys to int and sort
    epochs = sorted(int(k) for k in data.keys())
    map_values = [data[str(epoch)] for epoch in epochs]

    return epochs, map_values

def plot_metrics(
    loss_epochs, train_losses, val_losses, map_epochs, map_values
):
    """
    Plot loss curves and mAP on dual y-axes.
    """
    # 900×900 pixels at 100 dpi
    fig, ax1 = plt.subplots(figsize=(9, 9), dpi=100)
    ax2 = ax1.twinx()

    # Plot loss curves
    ax1.plot(loss_epochs, train_losses, label="Train Loss")
    ax1.plot(loss_epochs, val_losses, label="Validation Loss")
    ax1.set_xlabel("Epoch", fontsize=14)
    ax1.set_ylabel("Loss", fontsize=14)
    ax1.tick_params(axis="both", labelsize=12)
    ax1.set_xlim(0, 290)
    ax1.legend(loc="upper left")

    # Plot mAP and threshold line
    ax2.plot(map_epochs, map_values, label="mAP", color="tab:green")
    ax2.axhline(0.75, linestyle="--", color="red")
    ax2.set_ylabel("Mean Average Precision", fontsize=14)
    ax2.tick_params(axis="y", labelsize=12)
    ax2.legend(loc="upper right")

    plt.title("Loss and mAP over Epochs", fontsize=16)
    plt.tight_layout()
    plt.show()



def main():
    """
    Main entry point for the script.
    """
    metrics_path = "D:\module4_object_detection\deep_learning_transfer_exercise\graph\losses\metrics.json"
    maps_path = "D:\module4_object_detection\deep_learning_transfer_exercise\graph\maps\maps_combined.json"

    # Load data
    loss_epochs, train_losses, val_losses = load_metrics(metrics_path)
    map_epochs, map_values = load_map(maps_path)

    # Plot results
    plot_metrics(
        loss_epochs,
        train_losses,
        val_losses,
        map_epochs,
        map_values,
    )


if __name__ == "__main__":
    main()
