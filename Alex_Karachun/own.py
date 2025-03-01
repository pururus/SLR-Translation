import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import s3d, S3D_Weights
from tqdm import tqdm  # Импортируем tqdm для отображения прогресса

# Определяем класс датасета
class VideoDataset(Dataset):
    def __init__(self, annotations_file, videos_dir, num_frames=14, frame_step=1):
        self.annotations = pd.read_csv(annotations_file, sep='\t')
        self.videos_dir = videos_dir
        self.num_frames = num_frames
        self.frame_step = frame_step
        self.classes = sorted(self.annotations['text'].unique().tolist())
        self.label2idx = {label: idx for idx, label in enumerate(self.classes)}
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        attachment_id = row['attachment_id']
        label_text = row['text']
        label = self.label2idx[label_text]
        video_path = os.path.join(self.videos_dir, f"{attachment_id}.mp4")
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Видео не найдено: {video_path}")
        start_frame = int(row['begin'])
        width = int(row['width'])
        height = int(row['height'])
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames = []
        for i in range(self.num_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (width, height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            for _ in range(self.frame_step - 1):
                ret, _ = cap.read()
                if not ret:
                    break
        cap.release()
        if len(frames) < self.num_frames:
            if len(frames) == 0:
                raise ValueError(f"Видео {video_path} не содержит кадров, начиная с {start_frame}")
            while len(frames) < self.num_frames:
                frames.append(frames[-1])
        frames_np = np.stack(frames, axis=0)
        frames_np = frames_np.transpose(0, 3, 1, 2)
        video_tensor = torch.tensor(frames_np, dtype=torch.float32) / 255.0
        video_tensor = video_tensor.permute(1, 0, 2, 3)
        return video_tensor, label


if __name__ == '__main__':
    # Пути и параметры
    annotations_file = "Alex_Karachun/augmented/pupu.csv"
    videos_dir = "Alex_Karachun/augmented"
    num_frames = 14
    frame_step = 1
    batch_size = 4
    num_epochs = 3  # Задайте нужное количество эпох

    # Создаем датасет и DataLoader
    dataset = VideoDataset(annotations_file, videos_dir, num_frames, frame_step)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    num_classes = len(dataset.classes)
    print("Количество классов:", num_classes)
    print("Классы:", dataset.classes)

    # Настройка устройства и модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_weights = S3D_Weights.KINETICS400_V1.get_state_dict(progress=True)
    model = s3d(weights=None, num_classes=num_classes)
    model = model.to(device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict and "classifier" not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # Определяем оптимизатор и функцию потерь
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Цикл обучения по эпохам с отображением прогресса через tqdm
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        # Оборачиваем итерацию по DataLoader в tqdm для отображения прогресса
        for batch_idx, (videos, labels) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
            videos = videos.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            tqdm.write(f"Batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item():.4f}")
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    print("Обучение завершено.")
