import random
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import s3d, S3D_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
import json


# Параметры для выборки кадров
NUM_FRAMES = 14      # количество кадров в клипе
SAMPLING_STEP = 3    # между выбранными кадрами пропускаем 2 (то есть берем каждый третий)
EFFECTIVE_LENGTH = (NUM_FRAMES - 1) * SAMPLING_STEP + 1  # сколько кадров нужно, чтобы получить клип

def extract_clips_from_segment(frames, seg_start, seg_end, num_frames=NUM_FRAMES, step=SAMPLING_STEP):
    """
    Извлекает из последовательности кадров клипы из сегмента [seg_start, seg_end],
    распределяя кадры по циклическому принципу с шагом step.
    
    Для каждого смещения offset в диапазоне [0, step-1] создаётся клип:
    индексы: seg_start + offset, seg_start + offset + step, seg_start + offset + 2*step, ..., 
    seg_start + offset + (num_frames-1)*step.
    
    Если индекс выходит за пределы seg_end, последний кадр сегмента используется для паддинга.
    """
    clips = []
    for offset in range(step):
        start = seg_start + offset
        if start > seg_end:
            continue  # Если для данного offset нет кадров, пропускаем
        clip_frames = []
        for j in range(num_frames):
            idx = start + j * step
            if idx > seg_end:
                clip_frames.append(frames[seg_end])
            else:
                clip_frames.append(frames[idx])
        # Преобразуем список кадров (каждый кадр — numpy-массив [H, W, 3])
        # в массив [num_frames, H, W, 3], затем транспонируем в [3, num_frames, H, W]
        clip_np = np.stack(clip_frames, axis=0)
        clip_np = clip_np.transpose(3, 0, 1, 2)
        clip_tensor = torch.tensor(clip_np, dtype=torch.float32) / 255.0
        clips.append(clip_tensor)
    return clips


class VideoDataset(Dataset):
    def __init__(self, annotations_file, videos_dir, num_frames=NUM_FRAMES, frame_step=SAMPLING_STEP):
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
        
        # Считываем begin и end из CSV
        begin = int(row['begin'])
        end = int(row['end'])
        width = int(row['width'])
        height = int(row['height'])
        
        # Загружаем все кадры видео (это упрощает случай с произвольным доступом к кадрам)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_all = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (width, height))
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_all.append(frame_rgb)
        cap.release()
        if len(frames_all) == 0:
            raise ValueError(f"Видео {video_path} не содержит кадров")
        
        clips = []
        # Если есть жест (т.е. не "no_event"), извлекаем клипы из диапазона [begin, end]
        if label_text != "no_event":
            gesture_start = max(0, begin)
            gesture_end = min(end, len(frames_all) - 1)
            gesture_clips = extract_clips_from_segment(frames_all, gesture_start, gesture_end)
            clips.extend(gesture_clips)
            
            # Из не жестовых частей (до begin и после end) извлекаем примерно столько же клипов
            non_gesture_clips = []
            # Сегмент до жеста
            if gesture_start > 0:
                non_gesture_clips += extract_clips_from_segment(frames_all, 0, gesture_start - 1)
            # Сегмент после жеста
            if gesture_end < len(frames_all) - 1:
                non_gesture_clips += extract_clips_from_segment(frames_all, gesture_end + 1, len(frames_all) - 1)
            # Если получилось больше клипов, чем в жестовой части, можно ограничить их число (например, взять первые len(gesture_clips))
            if len(gesture_clips) > 0 and len(non_gesture_clips) > len(gesture_clips):
                random.shuffle(non_gesture_clips)
                non_gesture_clips = non_gesture_clips[:len(gesture_clips)]
            clips.extend(non_gesture_clips)
        else:
            # Для видео без жеста ("no_event") используем весь видеоряд
            clips = extract_clips_from_segment(frames_all, 0, len(frames_all) - 1)
        
        # # Если по какой-либо причине не получилось извлечь ни одного клипа, пробуем взять первые num_frames кадров с паддингом
        if len(clips) == 0:
            clips = extract_clips_from_segment(frames_all, 0, len(frames_all) - 1)
            print(f"WARNING: в {attachment_id} неудалось наскрести ни на один набор из {NUM_FRAMES} кадров")
        
    
        
        # Возвращаем словарь с клипами и меткой видео
        return {"clips": clips, "label": label}

def custom_collate(batch):
    """
    Собирает батч, выпрямляя список клипов из каждого видео.
    На выходе получаем тензор клипов размера [N, 3, num_frames, H, W] и тензор меток [N].
    """
    all_clips = []
    all_labels = []
    for sample in batch:
        for clip in sample["clips"]:
            all_clips.append(clip)
            all_labels.append(sample["label"])
    clips_tensor = torch.stack(all_clips, dim=0)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    return clips_tensor, labels_tensor

if __name__ == '__main__':
    # Пути и параметры
    annotations_file = "Alex_Karachun/augmented/pupu.csv"
    videos_dir = "Alex_Karachun/augmented"
    batch_size = 4
    num_epochs = 3  # Задайте нужное количество эпох

    # Создаем датасет и DataLoader с custom collate_fn
    dataset = VideoDataset(annotations_file, videos_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=custom_collate)

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

    loss_history = []  # для сохранения истории значений функции потерь

    # Цикл обучения по эпохам с отображением прогресса через tqdm
    model.train()
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        for batch_idx, (clips, labels) in enumerate(tqdm(dataloader, desc="Training", leave=False)):
            clips = clips.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_value = loss.item()
            epoch_loss += loss_value
            loss_history.append(loss_value)
            tqdm.write(f"Batch {batch_idx+1}: Loss = {loss_value:.4f}")
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
    
    print("Обучение завершено.")
    
    # Сохранение модели
    glob_path = 'Alex_Karachun/trained_models/1/'
    model_save_path = glob_path + "s3d_finetuned.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Модель сохранена в {model_save_path}")

    # Сохранение словаря label2idx
    label2idx_save_path = glob_path + "label2idx.json"
    with open(label2idx_save_path, "w", encoding="utf-8") as f:
        json.dump(dataset.label2idx, f, ensure_ascii=False, indent=4)
        
    with open(glob_path + 'losses.txt', 'w', encoding="utf-8") as f:
        print(loss_history, file=f)
    print(f"label2idx сохранен в {label2idx_save_path}")
    
    
    # Сохраним историю потерь в CSV для дальнейшего построения графика
    pd.DataFrame({"loss": loss_history}).to_csv("loss_history.csv", index=False)
    
    # Пример построения графика обучения
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss History")
    plt.savefig("loss_history.png")
    plt.show()
