import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm

from model import MLPModel
from autoencoder import MLPAutoencoder
from dataset import PizzaDataset


def calculate_metrics(labels_true, labels_pred):
    metrics = {}
    metrics["precision"] = precision_score(labels_true, labels_pred, average="binary")
    metrics["recall"] = recall_score(labels_true, labels_pred, average="binary")
    metrics["f1"] = f1_score(labels_true, labels_pred, average="binary")
    metrics["cm"] = confusion_matrix(labels_true, labels_pred)

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

    # Определяем архитектуру MLP
    input_size = 3 * 64 * 64
    hidden_layers = [2048, 1024, 512, 256]
    output_size = 1
    layer_sizes = [input_size] + hidden_layers

    # Предобучение автокодировщиков
    print("Pretraining autoencoders...")
    
    autoencoders = []
    prev_size = layer_sizes[0]
    data = []
    
    # Собираем все данные в один тензор для автокодировщиков
    for images, _ in train_loader:
        images = images.view(images.size(0), -1)
        data.append(images)
    data = torch.cat(data, dim=0).to(device)
    input_data = data

    for i, hidden_size in enumerate(layer_sizes[1:]):
        ae = MLPAutoencoder(prev_size, hidden_size).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(ae.parameters(), lr=1e-3)
        epochs = 10

        pbar = tqdm(range(epochs), desc=f"[AE {prev_size} -> {hidden_size}]")
        for epoch in pbar:
            ae.train()
            optimizer.zero_grad()
            output = ae(input_data)
            loss = criterion(output, input_data)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Получаем выходы энкодера для следующего слоя
        with torch.no_grad():
            input_data = ae.encoder(input_data)
        autoencoders.append(ae)
        prev_size = hidden_size

    # Инициализация и перенос весов в MLP
    mlp = MLPModel(input_size, hidden_layers, output_size).to(device)
    mlp_layers = [m for m in mlp.model if isinstance(m, nn.Linear)]
    for ae, mlp_layer in zip(autoencoders, mlp_layers[:-1]):
        ae_encoder = ae.encoder[0]
        mlp_layer.weight.data = ae_encoder.weight.data.clone()
        mlp_layer.bias.data = ae_encoder.bias.data.clone()

    # Обучение MLP
    criterion = nn.BCELoss()
    optimizer = optim.Adam(mlp.parameters(), lr=1e-4)
    epochs = 50
    best_val_accuraclabels = 0.0
    best_model_path = "./saves/best_mlp_model.pth"

    print("Training MLPModel...")
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        mlp.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for images, labels in tqdm(train_loader, desc="[Train]"):
            images, labels = images.view(images.size(0), -1).to(device), labels.to(device)
            
            outputs = mlp(images)
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
        mlp.eval()
        
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="[Valid]"):
                images, labels = images.view(images.size(0), -1).to(device), labels.to(device)
                
                outputs = mlp(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()
                
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        print(f"[Train]: Loss - {avg_train_loss:.4f}, Accuracy - {train_accuracy:.4f}")
        print(f"[Valid]: Loss - {avg_val_loss:.4f}, Accuracy - {val_accuracy:.4f}")

        if val_accuracy > best_val_accuraclabels:
            best_val_accuraclabels = val_accuracy
            torch.save(mlp.state_dict(), best_model_path)

    # Тестирование
    print("Testing model...")
    
    mlp.load_state_dict(torch.load(best_model_path))
    mlp.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="[Test]"):
            images, labels = images.view(images.size(0), -1).to(device), labels.to(device)
            
            outputs = mlp(images)
            
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
