import random
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.models.video import s3d
from tqdm import tqdm
import matplotlib.pyplot as plt
import json

NUM_FRAMES = 14
SAMPLING_STEP = 3
EFFECTIVE_LENGTH = (NUM_FRAMES - 1) * SAMPLING_STEP + 1

def extract_clips_from_segment(frames, seg_start, seg_end, num_frames=NUM_FRAMES, step=SAMPLING_STEP):
    clips = []
    for offset in range(step):
        start = seg_start + offset
        if start > seg_end:
            continue
        clip_frames = []
        for j in range(num_frames):
            idx = start + j * step
            if idx > seg_end:
                clip_frames.append(frames[seg_end])
            else:
                clip_frames.append(frames[idx])
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
        
        begin = int(row['begin'])
        end = int(row['end'])
        width = int(row['width'])
        height = int(row['height'])
        
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
        if label_text != "no_event":
            gesture_start = max(0, begin)
            gesture_end = min(end, len(frames_all) - 1)
            gesture_clips = extract_clips_from_segment(frames_all, gesture_start, gesture_end)
            clips.extend(gesture_clips)
            
            non_gesture_clips = []
            if gesture_start > 0:
                non_gesture_clips += extract_clips_from_segment(frames_all, 0, gesture_start - 1)
            if gesture_end < len(frames_all) - 1:
                non_gesture_clips += extract_clips_from_segment(frames_all, gesture_end + 1, len(frames_all) - 1)
            if len(gesture_clips) > 0 and len(non_gesture_clips) > len(gesture_clips):
                random.shuffle(non_gesture_clips)
                non_gesture_clips = non_gesture_clips[:len(gesture_clips)]
            clips.extend(non_gesture_clips)
        else:
            clips = extract_clips_from_segment(frames_all, 0, len(frames_all) - 1)
        
        if len(clips) == 0:
            clips = extract_clips_from_segment(frames_all, 0, len(frames_all) - 1)
            print(f"WARNING: в {attachment_id} неудалось наскрести ни на один набор из {NUM_FRAMES} кадров")
        
        return {"clips": clips, "label": label}

def custom_collate(batch):
    all_clips = []
    all_labels = []
    for sample in batch:
        for clip in sample["clips"]:
            all_clips.append(clip)
            all_labels.append(sample["label"])
    clips_tensor = torch.stack(all_clips, dim=0)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    return clips_tensor, labels_tensor

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for clips, labels in dataloader:
            clips = clips.to(device)
            labels = labels.to(device)
            outputs = model(clips)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    return accuracy

if __name__ == '__main__':
    # Пути к сохранённой модели и файлу label2idx
    model_save_path = 'Alex_Karachun/trained_models/1/s3d_finetuned.pth'
    label2idx_path = 'Alex_Karachun/trained_models/1/label2idx.json'

    with open(label2idx_path, 'r', encoding='utf-8') as f:
        label2idx = json.load(f)
    num_classes = len(label2idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = s3d(weights=None, num_classes=num_classes)
    model = model.to(device)
    state_dict = torch.load(model_save_path, map_location=device)
    model.load_state_dict(state_dict)
    print("Модель успешно загружена в переменную model.")

    test_annotations_file = "../slovo_full/augmented_annotations_test.csv"
    test_videos_dir = "../slovo_full/augmented_test"
    dataset_test = VideoDataset(test_annotations_file, test_videos_dir)
    test_dataloader = DataLoader(dataset_test, batch_size=2, shuffle=True, num_workers=4, collate_fn=custom_collate)
    
    accuracy = evaluate_model(model, test_dataloader, device)
    print(f"Точность распознавания жестов: {accuracy:.2f}%")
