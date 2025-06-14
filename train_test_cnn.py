import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from model import CNNModel
from autoencoder import DenoisingAutoencoder
from dataset import PizzaDataset, NoisyPizzaDataset


def calculate_metrics(y_true, y_pred):
    metrics = {}
    metrics["precision"] = precision_score(y_true, y_pred, average="binary")
    metrics["recall"] = recall_score(y_true, y_pred, average="binary")
    metrics["f1"] = f1_score(y_true, y_pred, average="binary")
    metrics["cm"] = confusion_matrix(y_true, y_pred)

    return metrics


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA available...")
    else:
        device = torch.device("cpu")
        print("CUDA is not available...")

    # Датасет
    dataset = PizzaDataset("./data")
    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    val_len = int(total_len * 0.15)
    test_len = total_len - train_len - val_len
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_len, val_len, test_len], generator=generator
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Обучение автокодировщика
    print("Training denoising autoencoder...")

    noisy_dataset = NoisyPizzaDataset(dataset, noise_std=0.5)
    train_noisy, _ , _ = random_split(
        noisy_dataset, [train_len, val_len, test_len], generator=generator
    )
    train_loader_noisy = DataLoader(train_noisy, batch_size=64, shuffle=True)

    autoencoder = DenoisingAutoencoder().to(device)

    ae_epochs = 10
    ae_lr = 1e-3
    ae_criterion = nn.MSELoss()
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=ae_lr)

    for epoch in range(ae_epochs):
        print(f"Epoch {epoch + 1}/{ae_epochs}")
        autoencoder.train()

        total_loss = 0

        for noisy_images, images in tqdm(train_loader_noisy, desc="[AE Train]"):
            noisy_images, images = noisy_images.to(device), images.to(device)

            outputs = autoencoder(noisy_images)
            loss = ae_criterion(outputs, images)
            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[AE Train]: Loss - {avg_loss:.4f}")

    # Инициализацуия модели
    cnn = CNNModel().to(device)

    # Копируем сверточные слои
    cnn_layers = [m for m in cnn.model if isinstance(m, nn.Conv2d)]
    ae_layers = [m for m in autoencoder.encoder if isinstance(m, nn.Conv2d)]
    for cnn_l, ae_l in zip(cnn_layers, ae_layers):
        cnn_l.weight.data = ae_l.weight.data.clone()
        cnn_l.bias.data = ae_l.bias.data.clone()

    model = cnn

    # Гиперпараметры
    epochs = 20
    lr = 1e-3
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Обучение модели
    best_val_accuracy = 0.0
    best_model_path = "./saves/best_model.pth"

    print("Training CNNModel...")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()

        # Обучение
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in tqdm(train_loader, desc="[Train]"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (outputs > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += labels.numel()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Валидация
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="[Valid]"):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        print(f"[Train]: Loss - {avg_train_loss:.4f}, Accuracy - {train_accuracy:.4f}")
        print(f"[Valid]: Loss - {avg_val_loss:.4f}, Accuracy - {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)

    # Тестирование
    print("Testing model...")

    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="[Test]"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            preds = (outputs > 0.5).float()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    metrics = calculate_metrics(y_true, y_pred)
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics["cm"])


if __name__ == "__main__":
    main()
