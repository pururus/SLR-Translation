import torch
import cv2
import json
from torchvision.models.video import s3d
import numpy as np
from math import floor


SAMPLING_STEP = 3    # между соседними кадрами берём каждый третий


def prepare_and_load_video_frames(video_path):
    """
    Подготавливает видео перед нарезанием на кадры:
    - Если видео вертикальное (высота > ширины): сразу изменяет размер до 224x224.
    - Если видео горизонтальное:
         * сначала масштабирует кадр так, чтобы высота стала 1280,
         * затем центрированно обрезаем боковые части до ширины 720 
         * после чего изменяет размер до 224x224.
    - Переводит видео в ~24 fps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(floor(orig_fps / 24), 1)
    
    # Получаем исходные размеры видео
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_vertical = height >= width

    frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step == 0:
            # Преобразуем цвет из BGR в RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not is_vertical:
                # Для горизонтального видео:
                # 1. Масштабируем кадр, чтобы высота стала 1280
                scale = 1280 / frame.shape[0]
                new_width = int(frame.shape[1] * scale)
                frame = cv2.resize(frame, (new_width, 1280))
                # 2. Центрированно обрезаем боковые части до ширины 720
                if new_width > 720:
                    left = (new_width - 720) // 2
                    frame = frame[:, left:left+720]
    

            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        frame_idx += 1
    cap.release()
    return frames


def create_clips_sliding(frames, num_frames, step=SAMPLING_STEP):
    clips = []
    clip_indices = [] 
    for i in range(len(frames)):
        clip_frames = []
        last_index = i
        for j in range(num_frames):
            idx = i + j * step
            if idx < len(frames):
                clip_frames.append(frames[idx])
                last_index = idx
            else:
                clip_frames.append(frames[-1])
                last_index = len(frames) - 1
        # Преобразуем список кадров (каждый кадр имеет форму [H, W, 3])
        # в массив numpy [num_frames, H, W, 3] и транспонируем в [3, num_frames, H, W]
        clip_np = np.stack(clip_frames, axis=0)
        clip_np = clip_np.transpose(3, 0, 1, 2)
        clip_tensor = torch.tensor(clip_np, dtype=torch.float32) / 255.0
        clips.append(clip_tensor)
        clip_indices.append((i, last_index))
    return clips, clip_indices


def make_predictions_from_video(video_path, model_dir) -> list[str]:
    # предсказвыаем по окну на 14 кадров
    '''
    возвращает тензор, где в столбцы записаны векторы вероятностей принадлежности очередного окна видео к каждому жесту
    
    '''
    
    NUM_FRAMES = 14      # количество кадров в клипе

    frames = prepare_and_load_video_frames(video_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = f"{model_dir}/s3d_finetuned.pth"
    label2idx_path = f"{model_dir}/label2idx.json"
    
    
    with open(label2idx_path, "r", encoding="utf-8") as f:
        label2idx = json.load(f)
    num_classes = len(label2idx)
    idx2label = {v: k for k, v in label2idx.items()}

    model = s3d(weights=None, num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    
    clips, clip_indices = create_clips_sliding(frames, num_frames=NUM_FRAMES, step=SAMPLING_STEP)
    
    prob_list = []
    predictions = []
    for clip in clips:
        # Добавляем размер батча: [1, 3, NUM_FRAMES, H, W]
        clip = clip.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(clip)
            probabilities = torch.softmax(outputs, dim=1)  # shape: [1, num_classes]
            prob_vec = probabilities.squeeze(0)
            
            pred_idx = torch.argmax(prob_vec).item()
            pred_label = idx2label[pred_idx]
            predictions.append(pred_label)
            
        prob_list.append(prob_vec)
    
    # Стекуем векторы вероятностей в тензор shape: [num_clips, num_classes]
    probs_tensor = torch.stack(prob_list, dim=1)  # shape: [num_classes, num_clips]
    return probs_tensor
    # return probs_tensor, predictions



def make_predictions_from_video_7_frames(video_path, model_dir) -> list[str]:
    '''
    возвращает тензор, где в столбцы записаны векторы вероятностей принадлежности очередного окна видео к каждому жесту

    '''
    # предсказвыаем по окну на 14 кадров

    
    NUM_FRAMES = 7      # количество кадров в клипе

    frames = prepare_and_load_video_frames(video_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_path = f"{model_dir}/s3d_finetuned.pth"
    label2idx_path = f"{model_dir}/label2idx.json"
    
    
    with open(label2idx_path, "r", encoding="utf-8") as f:
        label2idx = json.load(f)
    num_classes = len(label2idx)
    idx2label = {v: k for k, v in label2idx.items()}

    model = s3d(weights=None, num_classes=num_classes)
    model.avgpool = torch.nn.AvgPool3d(kernel_size=(1, 7, 7), stride=1)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    
    clips, clip_indices = create_clips_sliding(frames, num_frames=NUM_FRAMES, step=SAMPLING_STEP)
    
    prob_list = []
    predictions = []

    for clip in clips:
        # Добавляем размер батча: [1, 3, NUM_FRAMES, H, W]
        clip = clip.unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(clip)
            probabilities = torch.softmax(outputs, dim=1)  # shape: [1, num_classes]
            prob_vec = probabilities.squeeze(0)
            
            pred_idx = torch.argmax(prob_vec).item()
            pred_label = idx2label[pred_idx]
            predictions.append(pred_label)
        prob_list.append(prob_vec)
    
    # Стекуем векторы вероятностей в тензор shape: [num_clips, num_classes]
    probs_tensor = torch.stack(prob_list, dim=1)  # shape: [num_classes, num_clips]
    return probs_tensor
    # return probs_tensor, predictions


# res = make_predictions_from_video_14_frames(
#     # video_path='../slovo_full/testing_videos/вы_хорошо_работать.mov', 
#     video_path='../slovo_full/testing_videos/я_дом_идти.mov', 
#     # video_path='../slovo_full/testing_videos/я_тебе_еда_делать.mov', 
#     # model_dir='Alex_Karachun/trained_models/s3d_1000_gestures_1000_videos_7_epochs_done/s3d_1000_gestures_1000_videos_5_epoch'
# )

res = make_predictions_from_video_7_frames(
    # video_path='../slovo_full/testing_videos/вы_хорошо_работать.mov', 
    # video_path='../slovo_full/testing_videos/я_дом_идти.mov', 
    video_path='../slovo_full/testing_videos/я_тебе_еда_делать.mov', 

    model_dir='Alex_Karachun/trained_models/s3d_1000_gestures_100_videos_10_epochs_done_7_frames/s3d_1000_gestures_100_videos_7_epoch'
    # model_dir='Alex_Karachun/trained_models/s3d_1000_gestures_1000_videos_7_epochs_done/s3d_1000_gestures_1000_videos_5_epoch'
)

print(res)

# print(clear_same_res(res))

    
'''
сделать предобработку видео
'''

'''
1ep: вы хорошо работать -> ['no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'тогда', 'тогда', 'тогда', 'no_event', 'получить', 'получить', 'недовольный', 'недовольный', 'ужасный', 'ужасный', 'важный', 'самовосприятие', 'самовосприятие', 'лить', 'дельфин', 'дельфин', 'лить', 'ранний', 'самообман', 'изможденный', 'страх', 'no_event', 'изможденный', 'самообман', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']
2ep: вы хорошо работать -> ['no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'хорошо', 'ожидающий', 'из-за', 'из-за', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'из-за', 'из-за', 'из-за', 'хорошо', 'хорошо', 'хорошо', 'из-за', 'лить', 'змея', 'лить', 'лить', 'важный', 'важный', 'Борода', 'женщина', 'женщина', 'лить', 'лить', 'лить', 'север', 'положить', 'положить', 'изможденный', 'изможденный', 'изможденный', 'изможденный', 'изможденный', 'изможденный', 'изможденный', 'изможденный', 'изможденный', 'изможденный', 'положить', 'положить', 'сожаление', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'пустой', 'пустой', 'пустой', 'пустой', 'работать', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'работать', 'работать', 'пустой', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']
3ep: вы хорошо работать -> ['no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'no_event', 'no_event', 'no_event', 'no_event', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'вопрос', 'ожидающий', 'лить', 'важный', 'важный', 'самообман', 'лить', 'положить', 'невиновный', 'положить', 'положить', 'положить', 'положить', 'положить', 'положить', 'положить', 'изможденный', 'положить', 'no_event', 'no_event', 'no_event', 'no_event', 'положить', 'сожаление', 'MakDonalds', 'MakDonalds', 'MakDonalds', 'MakDonalds', 'MakDonalds', 'работать', 'MakDonalds', 'MakDonalds', 'MakDonalds', 'работать', 'по часам', 'по часам', 'по часам', 'по часам', 'работать', 'работать', 'по часам', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'обнимать', 'обнимать', 'no_event', 'no_event', 'миллион', 'миллион', 'no_event', 'миллион', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']
4ep: вы хорошо работать -> ['no_event', 'переваривать', 'переваривать', 'переваривать', 'переваривать', 'переваривать', 'no_event', 'переваривать', 'переваривать', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'важный', 'хорошо', 'из-за', 'из-за', 'из-за', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'хорошо', 'хорошо', 'тогда', 'хорошо', 'хорошо', 'хорошо', 'из-за', 'из-за', 'важный', 'важный', 'важный', 'важный', 'важный', 'важный', 'важный', 'важный', 'важный', 'важный', 'лить', 'положить', 'положить', 'положить', 'положить', 'положить', 'положить', 'положить', 'положить', 'положить', 'сожаление', 'сожаление', 'сожаление', 'сожаление', 'сожаление', 'сожаление', 'сожаление', 'сожаление', 'доминантный', 'сожаление', 'доминантный', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'работать', 'по часам', 'по часам', 'работать', 'по часам', 'по часам', 'хотеть', 'по часам', 'по часам', 'работать', 'по часам', 'по часам', 'сожаление', 'сожаление', 'сожаление', 'сожаление', 'сожаление', 'сожаление', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']
5ep: вы хорошо работать -> ['no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'хорошо', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'змея', 'змея', 'важный', 'важный', 'остановиться', 'север', 'важный', 'важный', 'жёсткий', 'важный', 'лить', 'жёсткий', 'положить', 'положить', 'жёсткий', 'жёсткий', 'жёсткий', 'положить', 'no_event', 'no_event', 'no_event', 'no_event', 'те', 'сожаление', 'сожаление', 'сожаление', 'сожаление', 'работать', 'кусать', 'кусать', 'кусать', 'кусать', 'следовало', 'кусать', 'кусать', 'кусать', 'кусать', 'по часам', 'по часам', 'работать', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'назначенное время', 'работать', 'назначенное время', 'назначенное время', 'назначенное время', 'работать', 'назначенное время', 'работать', 'работать', 'работать', 'работать', 'работать', 'по часам', 'по часам', 'работать', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'работать', 'по часам', 'по часам', 'по часам', 'вечность', 'no_event', 'черный', 'черный', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event'] 
7ep: вы хорошо работать -> ['no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'хорошо', 'no_event', 'no_event', 'no_event', 'no_event', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'хорошо', 'ожидающий', 'агрессивный', 'агрессивный', 'гордость', 'вопрос', 'вопрос', 'вопрос', 'важный', 'лить', 'лить', 'лить', 'лить', 'лить', 'положить', 'положить', 'положить', 'положить', 'положить', 'положить', 'положить', 'положить', 'положить', 'положить', 'сожаление', 'сожаление', 'те', 'сожаление', 'сожаление', 'сожаление', 'сожаление', 'следовало', 'следовало', 'следовало', 'следовало', 'сожаление', 'сожаление', 'сожаление', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'по часам', 'хотеть', 'по часам', 'по часам', 'по часам', 'хотеть', 'хотеть', 'хотеть', 'хотеть', 'по часам', 'хотеть', 'хотеть', 'хотеть', 'хотеть', 'по часам', 'по часам', 'хотеть', 'хотеть', 'хотеть', 'хотеть', 'хотеть', 'хотеть', 'хотеть', 'хотеть', 'хотеть', 'сожаление', 'сожаление', 'черный', 'черный', 'вечность', 'вечность', 'вечность', 'вечность', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']



1ep: я дом идти -> ['no_event', 'я', 'я', 'no_event', 'я', 'мне', 'мне', 'мне', 'мне', 'мне', 'мне', 'я', 'я', 'я', 'я', 'я', 'меня', 'страх', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'сторона', 'сторона', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'поднос', 'различный', 'различный', 'поднос', 'поднос', 'прийти', 'Лопата', 'прийти', 'Лопата', 'поднос', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'поднос', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Футбол', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']
2ep: я дом идти -> ['no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'я', 'я', 'я', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'гордость', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'ребяческий', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'Лопата', 'прийти', 'прийти', 'прийти', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'слабый', 'слабый', 'слабый', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'слабый', 'слабый', 'Лопата', 'слабый', 'Лопата', 'слабый', 'Лопата', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']
3ep: я дом идти -> ['no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'лучше', 'no_event', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'поднос', 'поднос', 'поднос', 'поднос', 'петь', 'поднос', 'прийти', 'петь', 'прийти', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Футбол', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'поднос', 'поднос', 'поднос', 'Лопата', 'следовало', 'Лопата', 'no_event', 'no_event', 'no_event', 'no_event', 'нам', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']
4ep: я дом идти -> ['no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'мне', 'мне', 'мне', 'мне', 'мне', 'мне', 'мне', 'мне', 'мне', 'мне', 'no_event', 'мне', 'no_event', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'откусывание', 'дом', 'откусывание', 'откусывание', 'дом', 'откусывание', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'кроме', 'кроме', 'кроме', 'кроме', 'Забрать', 'Забрать', 'Забрать', 'Забрать', 'Забрать', 'петь', 'петь', 'Лопата', 'петь', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'идти', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'поднос', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'Лопата', 'слабый', 'Лопата', 'верх', 'Лопата', 'продать', 'продать', 'продать', 'верх', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']
5ep: я дом идти -> ['no_event', 'no_event', 'no_event', 'no_event', 'я', 'я', 'no_event', 'я', 'no_event', 'я', 'я', 'я', 'сорок', 'сорок', 'я', 'следовало', 'я', 'сорок', 'сорок', 'самопроверка', 'испуганный', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'откусывание', 'откусывание', 'откусывание', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'кроме', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'им', 'прийти', 'Лопата', 'Соль', 'Соль', 'Лопата', 'Соль', 'идти', 'поднос', 'Соль', 'Лопата', 'поднос', 'идти', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'Соль', 'следовало', 'Соль', 'верх', 'верх', 'верх', 'верх', 'верх', 'верх', 'верх', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']
7ep: я дом идти -> ['no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'мне', 'мне', 'меня', 'меня', 'мне', 'мне', 'меня', 'меня', 'напуганный', 'дать', 'самопроверка', 'жуткий', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'самопроверка', 'откусывание', 'откусывание', 'откусывание', 'дом', 'откусывание', 'дом', 'дом', 'откусывание', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'дом', 'Забрать', 'Забрать', 'Забрать', 'поднос', 'петь', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'идти', 'идти', 'идти', 'идти', 'идти', 'поднос', 'идти', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'поднос', 'сердитый', 'сердитый', 'Лопата', 'получить', 'верх', 'верх', 'верх', 'верх', 'верх', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']

1ep: я тебе еда делать -> ['no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'гордый', 'я', 'я', 'я', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'те', 'из-за', 'те', 'те', 'те', 'из-за', 'из-за', 'временный', 'временный', 'временный', 'временный', 'временный', 'временный', 'временный', 'временный', 'временный', 'Смех', 'Смех', 'Смех', 'Смех', 'временный', 'Смех', 'Смех', 'Смех', 'прием пищи', 'Смех', 'Смех', 'прием пищи', 'Смех', 'Смех', 'Смех', 'Смех', 'Смех', 'Смех', 'Смех', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'закуска перед едой', 'в то время как', 'в то время как', 'в то время как', 'подруга', 'в то время как', 'подруга', 'подруга', 'подруга', 'подруга', 'подруга', 'сестра', 'сестра', 'сестра', 'аллигатор', 'аниматор', 'аниматор', 'аниматор', 'аниматор', 'аниматор', 'Лопата', 'Лопата', 'Лопата', 'аниматор', 'аниматор', 'аниматор', 'Лопата', 'Лопата', 'скоро', 'пустой', 'пустой', 'пустой', 'пустой', 'пустой', 'скоро', 'скоро', 'no_event', 'дать', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']
2ep: я тебе еда делать -> ['no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'я', 'я', 'я', 'меня', 'меня', 'меня', 'меня', 'меня', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'из-за', 'тебя', 'тебя', 'тебя', 'Пока', 'те', 'временный', 'из-за', 'из-за', 'временный', 'временный', 'зелёный', 'зелёный', 'зелёный', 'зелёный', 'зелёный', 'Смех', 'вкусный', 'еда', 'еда', 'вкусный', 'еда', 'Смех', 'еда', 'еда', 'еда', 'еда', 'еда', 'еда', 'еда', 'потрясающий', 'потрясающий', 'потрясающий', 'еда', 'еда', 'потрясающий', 'потрясающий', 'еда', 'еда', 'еда', 'еда', 'еда', 'еда', 'еда', 'еда', 'еда', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'еда', 'еда', 'прием пищи', 'прием пищи', 'еда', 'еда', 'еда', 'еда', 'еда', 'еда', 'еда', 'еда', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'сухой', 'сухой', 'день и ночь', 'аниматор', 'кормить', 'кормить', 'мыть', 'мыть', 'сердитый', 'сердитый', 'зима', 'зима', 'сухой', 'сухой', 'зима', 'сухой', 'сухой', 'сухой', 'сухой', 'сухой', 'сухой', 'сухой', 'дать', 'строгий', 'строгий', 'вечный', 'вечный', 'вечный', 'no_event', 'тебя', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']
3ep: я тебе еда делать -> ['no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'я', 'сердитый', 'сердитый', 'сердитый', 'сердитый', 'сердитый', 'сердитый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'розовато-лиловый', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'золотой', 'золотой', 'зелёный', 'зелёный', 'вкусный', 'вкусный', 'вкусный', 'вкусный', 'вкусный', 'вкусный', 'вкусный', 'вкусный', 'вкусный', 'ленность', 'прием пищи', 'ленность', 'крыса', 'аппетит', 'аппетит', 'аппетит', 'потрясающий', 'аппетит', 'аппетит', 'аппетит', 'аппетит', 'аппетит', 'потрясающий', 'аппетит', 'аппетит', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'в то время как', 'в то время как', 'показать', 'показать', 'мыть', 'мыть', 'мыть', 'мыть', 'мыть', 'мыть', 'мыть', 'мыть', 'сердитый', 'мыть', 'мыть', 'мыть', 'мыть', 'мыть', 'сердитый', 'мыть', 'мыть', 'мыть', 'мыть', 'мыть', 'мыть', 'показать', 'доминантный', 'доминантный', 'доминантный', 'вечный', 'строгий', 'тебя', 'тебя', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']
4ep: я тебе еда делать -> ['no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'меня', 'меня', 'меня', 'меня', 'сердитый', 'сердитый', 'сердитый', 'сердитый', 'гордый', 'гордый', 'жуткий', 'гордый', 'жуткий', 'жуткий', 'гордый', 'жуткий', 'жуткий', 'гордый', 'жуткий', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'гордый', 'из-за', 'гордый', 'гордый', 'гордый', 'гордый', 'розовато-лиловый', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'временный', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'потрясающий', 'потрясающий', 'потрясающий', 'потрясающий', 'доставлять удовольствие', 'потрясающий', 'потрясающий', 'потрясающий', 'потрясающий', 'потрясающий', 'потрясающий', 'сюрприз', 'сюрприз', 'потрясающий', 'сюрприз', 'сюрприз', 'сюрприз', 'сюрприз', 'сюрприз', 'ежедневный', 'сюрприз', 'сюрприз', 'сюрприз', 'сюрприз', 'зима', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'ланч', 'ланч', 'ланч', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'зима', 'зима', 'зима', 'зима', 'зима', 'зима', 'зима', 'день и ночь', 'день и ночь', 'сердитый', 'зима', 'сердитый', 'зима', 'зима', 'зима', 'зима', 'зима', 'зима', 'зима', 'зима', 'доминантный', 'зима', 'зима', 'зима', 'зима', 'доминантный', 'доминантный', 'доминантный', 'вечный', 'дать', 'вечный', 'доминантный', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']
5ep: я тебе еда делать -> ['no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'я', 'лучше', 'гордый', 'гордый', 'гордый', 'друг', 'жуткий', 'гордый', 'жуткий', 'жуткий', 'лучше', 'одолжить', 'гордый', 'одолжить', 'гордый', 'одолжить', 'вверх', 'вверх', 'гордый', 'гордый', 'из-за', 'из-за', 'из-за', 'гордый', 'гордый', 'розовато-лиловый', 'розовато-лиловый', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'из-за', 'воскресенье', 'временный', 'из-за', 'бы', 'вкусный', 'вкусный', 'вкусный', 'вкусный', 'вкусный', 'потрясающий', 'доставлять удовольствие', 'ленивый', 'потрясающий', 'наблюдательность', 'крыса', 'крыса', 'крыса', 'крыса', 'глупый', 'глупый', 'глупый', 'потрясающий', 'потрясающий', 'ежедневный', 'глупый', 'ежедневный', 'ежедневный', 'зима', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'подруга', 'подруга', 'подруга', 'подруга', 'подруга', 'подруга', 'зима', 'зима', 'аниматор', 'аниматор', 'день и ночь', 'аниматор', 'зима', 'аниматор', 'аниматор', 'аниматор', 'аниматор', 'аниматор', 'аниматор', 'аниматор', 'аниматор', 'аниматор', 'зима', 'зима', 'зима', 'зима', 'аниматор', 'аниматор', 'зима', 'зима', 'зима', 'против', 'доминантный', 'строгий', 'вечный', 'строгий', 'вечный', 'строгий', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']
7ep: я тебе еда делать -> ['no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'я', 'я', 'я', 'меня', 'меня', 'меня', 'Память', 'сердитый', 'сердитый', 'сердитый', 'напуганный', 'жуткий', 'жуткий', 'одолжить', 'одолжить', 'жуткий', 'жуткий', 'жуткий', 'одолжить', 'вверх', 'одолжить', 'одолжить', 'одолжить', 'одолжить', 'одолжить', 'одолжить', 'вверх', 'вверх', 'одолжить', 'вверх', 'из-за', 'одолжить', 'розовато-лиловый', 'розовато-лиловый', 'одолжить', 'розовато-лиловый', 'розовато-лиловый', 'из-за', 'розовато-лиловый', 'из-за', 'возможный', 'возможный', 'возможный', 'из-за', 'З', 'из-за', 'воскресенье', 'воскресенье', 'воскресенье', 'воскресенье', 'воскресенье', 'З', 'прибывать', 'воскресенье', 'воскресенье', 'прибывать', 'прибывать', 'потрясающий', 'прибывать', 'прибывать', 'потрясающий', 'холодный', 'прибывать', 'холодный', 'потрясающий', 'глупый', 'глупый', 'глупый', 'холодный', 'холодный', 'холодный', 'холодный', 'холодный', 'холодный', 'холодный', 'холодный', 'холодный', 'холодный', 'холодный', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'прием пищи', 'есть', 'есть', 'есть', 'прием пищи', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'закуска перед едой', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'в то время как', 'закуска перед едой', 'подруга', 'зима', 'зима', 'зима', 'зима', 'зима', 'зима', 'зима', 'зима', 'зима', 'зима', 'зима', 'зима', 'зима', 'зима', 'зима', 'коричневый', 'зима', 'зима', 'зима', 'зима', 'зима', 'аниматор', 'зима', 'зима', 'зима', 'до', 'строгий', 'строгий', 'строгий', 'строгий', 'строгий', 'тебя', 'строгий', 'строгий', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event', 'no_event']


вывод
- 1, 2, 4, 7 эпохи булшит
- оставлять для перевода только слова, появившиеся n раз подряд (прикинуть эту n)

[('no_event', 0.97), ('no_event', 0.96), ('no_event', 0.97), ('no_event', 0.96), ('я', 0.62), ('я', 0.54), ('no_event', 0.57), ('я', 0.54), ('no_event', 0.46), ('я', 0.65), ('я', 0.58), ('я', 0.6), ('сорок', 0.84), ('сорок', 0.36), ('я', 0.36), ('следовало', 0.62), ('я', 0.31), ('сорок', 0.29), ('сорок', 0.55), ('самопроверка', 0.19), ('испуганный', 0.33), ('самопроверка', 0.84), ('самопроверка', 0.6), ('самопроверка', 0.96), ('самопроверка', 0.99), ('самопроверка', 0.99), ('самопроверка', 0.97), ('самопроверка', 0.98), ('самопроверка', 0.56), ('откусывание', 0.89), ('откусывание', 0.86), ('откусывание', 0.63), ('дом', 0.82), ('дом', 0.85), ('дом', 0.99), ('дом', 0.87), ('дом', 0.9), ('дом', 0.99), ('дом', 1.0), ('дом', 1.0), ('дом', 1.0), ('дом', 1.0), ('дом', 1.0), ('дом', 1.0), ('дом', 1.0), ('дом', 1.0), ('дом', 1.0), ('дом', 1.0), ('дом', 0.99), ('дом', 1.0), ('дом', 1.0), ('дом', 1.0), ('дом', 0.99), ('дом', 0.98), ('дом', 0.84), ('дом', 0.98), ('дом', 0.99), ('дом', 0.71), ('дом', 0.85), ('дом', 0.72), ('дом', 0.59), ('кроме', 0.48), ('поднос', 0.93), ('поднос', 0.92), ('поднос', 0.94), ('поднос', 0.74), ('поднос', 0.98), ('поднос', 0.98), ('поднос', 0.95), ('поднос', 0.96), ('поднос', 0.93), ('поднос', 0.5), ('поднос', 0.89), ('поднос', 0.92), ('поднос', 0.51), ('поднос', 0.73), ('им', 0.49), ('прийти', 0.4), ('Лопата', 0.51), ('Соль', 0.7), ('Соль', 0.58), ('Лопата', 0.36), ('Соль', 0.38), ('идти', 0.49), ('поднос', 0.48), ('Соль', 0.51), ('Лопата', 0.43), ('поднос', 0.71), ('идти', 0.37), ('поднос', 0.93), ('поднос', 0.87), ('поднос', 0.41), ('поднос', 0.87), ('поднос', 0.85), ('поднос', 0.38), ('поднос', 0.41), ('Соль', 0.44), ('следовало', 0.48), ('Соль', 0.35), ('верх', 0.51), ('верх', 0.58), ('верх', 0.69), ('верх', 0.85), ('верх', 0.81), ('верх', 0.93), ('верх', 0.65), ('no_event', 0.52), ('no_event', 0.59), ('no_event', 0.9), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0), ('no_event', 1.0)]


'''

