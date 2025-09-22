# Pipeline för CNN som tränas när tillräckligt med data finns.

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from datetime import datetime
from tqdm import tqdm
import os
import sqlite3
import database_utils
import api

DB_FILE = database_utils.DB_FILE
MIN_IMAGES_FOR_TRAINING = 100
SAVE_DIR = api.SAVE_DIR
MODELS_DIR = "models"
BEST_MODEL_FILE = os.path.join(MODELS_DIR, "best_cnn_ice_model.pth")

os.makedirs(MODELS_DIR, exist_ok=True)

class IceMapDataset(Dataset):
    """
    Dataset för iskartbilder och deras tidsstämplar.
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")  # gråskala
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label
    
# CNN-modell
class CNN(nn.Module):
    """
    Enkel CNN för regression.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*32*32, 128)
        self.fc2 = nn.Linear(128, 1)  # regression

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# Hämtar bilderna från databasen.
def get_all_image_paths():
    """
    Hämtar alla bildvägar och deras labels (tidsstämpel) från databasen.
    Returnerar två listor: image_paths och labels.
    """
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT filepath, strftime('%s', last_modified) FROM icemap_metadata ORDER BY last_modified ASC")
    rows = cur.fetchall()
    conn.close()
    image_paths = [r[0] for r in rows]
    labels = [float(r[1]) / 1e9 for r in rows]  # Unix timestamp som numeriskt label, normaliserar då tidsstämpeln är i sekunder.
    return image_paths, labels

# Tränar modellen
def train_cnn_model():
    """
    Tränar en CNN-modell på bilderna i databasen. 70% träning, 15% validering, 15% test.
    Returnerar den tränade modellen och valideringsloss.
    """
    image_paths, labels = get_all_image_paths()

    # Behandlar bilder
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = IceMapDataset(image_paths, labels, transform=transform)
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val   = int(0.15 * n_total)
    n_test  = n_total - n_train - n_val
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=8, shuffle=False)
    test_loader  = DataLoader(test_set, batch_size=8, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 5
    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Tränar]", leave=False)
        for imgs, labels in train_bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})
        avg_loss = running_loss / len(train_loader)

        # Valideringsloss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():            
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        # Sammanfattning per epoch.
        tqdm.write(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Testar modellen på testdatan.
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testar", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs.squeeze(), labels)
            test_loss += loss.item()
    test_loss /= len(test_loader)
    tqdm.write(f"Test Loss: {test_loss:.4f}")
    return model, val_loss

# Tränar och sparar modellen om tillräckligt med bilder finns.
def train_and_save(force=False):
    """
    Tränar och sparar modellen om tillräckligt med bilder finns.
    Kontrollerar om det är den bästa modellen och sparar i så fall som best_cnn_ice_model.pth
    """
 
    image_paths, _ = get_all_image_paths()
    n_images = len(image_paths)

    if not force: # Villkor för att kunna träna utan korrekt antal bilder.
        if n_images < MIN_IMAGES_FOR_TRAINING or n_images % 100 != 0:
            print(f"[{datetime.now()}] Inte tillräckligt med bilder ({n_images}) för träning, väntar på fler...")
            return

    print(f"[{datetime.now()}] Tillräckligt med bilder ({n_images}) för träning")

    # Träna CNN och få valideringsloss
    model, val_loss = train_cnn_model()

    # Hitta modellnummer
    existing = [
            int(f.split('_')[0]) for f in os.listdir(MODELS_DIR) if f.endswith(".pth") and f[0].isdigit()   
        ]
    next_model_num = max(existing, default=0) + 1
    model_path = os.path.join(MODELS_DIR, f"{next_model_num}_cnn_ice_model.pth")

    # Spara modellen
    torch.save(model.state_dict(), model_path)
    print(f"[{datetime.now()}] Modell sparad som {model_path}")

    # Kontrollera om det är den bästa modellen
    best_val_loss = float('inf')
    if os.path.exists(BEST_MODEL_FILE):
        saved = torch.load(BEST_MODEL_FILE)
        best_val_loss = saved.get("val_loss", float('inf'))

        if val_loss < best_val_loss:
            print(f"[{datetime.now()}] Ny bästa modell! Val loss: {val_loss:.4f} (tidigare {best_val_loss:.4f})")
            torch.save({"model_state_dict": model.state_dict(), "val_loss": val_loss}, BEST_MODEL_FILE)
        else:
            print(f"[{datetime.now()}] Modell förbättrades inte. Val loss: {val_loss:.4f} ≥ {best_val_loss:.4f}")

# För testning, tvinga träning även om inte tillräckligt med bilder finns.
if __name__ == "__main__":
    train_and_save(force=True)