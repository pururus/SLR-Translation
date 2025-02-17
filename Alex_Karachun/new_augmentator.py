import moviepy as mp
from copy import deepcopy
from uuid import uuid4
import pandas as pd
import numpy as np
import random
import math
import cv2
import logging
import os

# Если хотите отключить вывод логов moviepy:
# logging.getLogger("moviepy").setLevel(logging.CRITICAL)

'''
ТЗ:
done - mirroring - зеркалить
done - zooming - приближать и возвращать к исходному размеру
done - (при сохранении менять bitrate) - качество видео
done - cropping - обрезать
done - (при сохранении добавлять ffmpeg_params со шумами)
done - respeeding - менять скорость
done - rebritning, x in [0.2, 1.7] - изменение яркости и контрастности

Новые задачи:
- Для каждого жеста (заданного диапазоном кадров begin-end) создать папку с кадрами, принадлежащими ему.
  Кадры, не попадающие в этот диапазон, сохраняются в папку no_event.
  Видео без жестов (text = "no_event") – все кадры в папку no_event.
- Оставить в датасете только вертикальные видео.
- Обрабатывать только те видео, у которых (height, width) ∈ [(1920, 1080), (1280, 720)].
'''

def rebritning(clip: mp.VideoFileClip, x: float) -> mp.VideoFileClip:
    # Изменение яркости/контрастности
    respeeder = mp.video.fx.MultiplyColor(factor=x)
    clip = respeeder.apply(clip)
    return clip

def respeeding(clip: mp.VideoFileClip, x: float) -> mp.VideoFileClip:
    respeeder = mp.video.fx.MultiplySpeed()
    respeeder.factor = x
    clip = respeeder.apply(clip)
    return clip

def cropping(clip: mp.VideoFileClip, left_down: list, right_upper: list) -> mp.VideoFileClip:
    original_size = clip.size
    cropper = mp.video.fx.Crop()
    cropper.x1, cropper.y1 = left_down
    cropper.x2, cropper.y2 = right_upper
    clip = cropper.apply(clip)
    clip = resizing(clip, original_size)
    return clip

def resizing(clip: mp.VideoFileClip, needed_size: list) -> mp.VideoFileClip:
    # Используем встроенный метод resizing
    clip = clip.resized(new_size=needed_size)
    return clip

def zooming(clip: mp.VideoFileClip, k=1.5) -> mp.VideoFileClip:
    original_size = clip.size
    cropper = mp.video.fx.Crop()
    cropper.x_center = clip.size[0] / 2
    cropper.y_center = clip.size[1] / 2
    cropper.width = clip.size[0] / k
    cropper.height = clip.size[1] / k
    clip = cropper.apply(clip)
    clip = resizing(clip, original_size)
    return clip

def mirroring(clip: mp.VideoFileClip) -> mp.VideoFileClip:
    clip = mp.video.fx.MirrorX().apply(clip)
    return clip

# def process_video(video_path: str, result_dir: str, multiplyer: int, expected_size: list[int, int]) -> None:
def pricess_video(video_path, result_dir, frames_dir, original_annotations_data, result_annotations_file_path, multiplyer, expected_size) -> None:
    '''
    Берёт видео из video_path, делает multiplyer версий. к каждой:
    - применяет случайный набор преобразований.
    - видео сохраняются в result_dir с рандомными именами.
    - выделяет кадры с жестом и без и раскладывает их по нужным папочкам
    - добавляет запись в result_annotations_file_path
    '''
    original_clip = mp.VideoFileClip(video_path)
    clips = [deepcopy(original_clip) for _ in range(multiplyer)]
    video_names = [str(uuid4()) for _ in range(multiplyer)]
    
    for i, clip in enumerate(clips):
        # Случайные параметры для аугментации:
        will_mirror = random.choice([True, False])
        k_for_zooming = random.uniform(1, 1.3)
        cropping_left_down = [int(random.uniform(0, clip.size[0] * 0.1)), int(random.uniform(0, clip.size[1] * 0.1))]
        cropping_right_upper = [clip.size[0] - int(random.uniform(0, clip.size[0] * 0.1)),
                                clip.size[1] - int(random.uniform(0, clip.size[1] * 0.1))]
        new_speed = random.uniform(0.8, 1.5)
        new_britness = random.uniform(0.5, 1.7)
        bitrate = str(random.choice(range(700, 3100, 100))) + 'k'
        noize_k = random.uniform(0, 20)
        
        if will_mirror:
            clip = mirroring(clip)
        clip = zooming(clip, k_for_zooming)
        clip = cropping(clip, cropping_left_down, cropping_right_upper)
        clip = respeeding(clip, new_speed)
        clip = rebritning(clip, new_britness)
        
        # Ресайз до ожидаемого размера (здесь используем высоту из expected_size[0])
        clip = clip.resized(height=expected_size[0])
        
        output_path = os.path.join(result_dir, video_names[i] + '.mp4')
        clip.write_videofile(output_path, bitrate=bitrate, 
                               ffmpeg_params=['-vf', f'noise=alls={noize_k}:allf=t+u'])
        ############    
        for new_name in dupped_names:
            row['attachment_id'] = new_name
            row['height'] = expected_size[1]
            row['width'] = expected_size[0]
            new_data.loc[new_video_id] = row
            new_video_id += 1

        new_data.to_csv(result_annotations_file_path, sep='\t', index=False)
        
        os.makedirs(frames_dir, exist_ok=True)
        extract_frames_for_dataset(result_annotations_file_path, result_dir, frames_dir)
        #############
        
        print(f"{'*' * 100}\nРазмер видео: {clip.size}\nПараметры: mirror={will_mirror}, zoom k={k_for_zooming}, "
              f"cropping_left_down={cropping_left_down}, cropping_right_upper={cropping_right_upper}, "
              f"speed={new_speed}, brightness={new_britness}, bitrate={bitrate}, noise={noize_k}")
    
    return video_names

def extract_frames(video_path: str, annotation_row: pd.Series, frames_output_dir: str) -> None:
    '''
    Извлекает кадры из видео. Если жест присутствует (text != "no_event"),
    кадры с индекса begin по end сохраняются в папку с названием жеста,
    остальные — в папку "no_event". Если жест отсутствует, то все кадры в "no_event".
    '''
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Не удалось открыть видео {video_path}")
        return

    gesture_label = str(annotation_row['text'])
    begin_frame = int(annotation_row['begin'])
    end_frame = int(annotation_row['end'])
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Если видео содержит жест (text != "no_event") и кадр входит в диапазон,
        # то сохраняем в папку с названием жеста, иначе – в папку no_event.
        if gesture_label != "no_event" and begin_frame <= frame_idx <= end_frame:
            folder_name = gesture_label
        else:
            folder_name = "no_event"
        target_dir = os.path.join(frames_output_dir, folder_name)
        os.makedirs(target_dir, exist_ok=True)
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        frame_filename = f"{video_id}_{frame_idx:04d}.jpg"
        frame_filepath = os.path.join(target_dir, frame_filename)
        cv2.imwrite(frame_filepath, frame)
        frame_idx += 1

    cap.release()

def extract_frames_for_dataset(annotations_csv: str, videos_dir: str, frames_output_dir: str) -> None:
    '''
    По CSV с аннотациями (результирующий датасет) проходит по каждому видео и извлекает кадры.
    '''
    annotations = pd.read_csv(annotations_csv, sep='\t')
    for i, row in annotations.iterrows():
        video_path = os.path.join(videos_dir, row['attachment_id'] + '.mp4')
        print(f"Извлечение кадров из {video_path}...")
        extract_frames(video_path, row, frames_output_dir)

def duper(dataset_dir_path: str, result_dir: str, original_annotations_file_path: str, 
          result_annotations_file_path: str, multiplyer: int, expected_size=[256, 144],
          frames_dir: str = None) -> None:
    '''
    Для каждого видео из датасета:
      - Фильтрует по разрешению и оставляет только вертикальные видео с разрешениями (1920, 1080), (1280, 720), (1920, 960),
      - Создаёт multiplyer аугментированных копий со случайными преобразованиями
      - С каждой копии видео, разкладывает кадры с жестом и без по папочкам
      - Сохраняет новые видео в result_dir
      - Обновляет annotations.csv
    
    '''
    data = pd.read_csv(original_annotations_file_path, sep='\t')
    
    # Фильтруем по разрешению, чтобы оставить вертикальные видео:
    allowed_res = [(1920, 1080), (1280, 720), (1920, 960)]
    data = data[data.apply(lambda row: (row['height'], row['width']) in allowed_res, axis=1)]
    
    
    for i, row in data.iterrows():
        video_file = os.path.join(dataset_dir_path, str(row['attachment_id']) + '.mp4')
        if not os.path.exists(video_file):
            logging.error(f"Видео не найдено: {video_file}")
            continue

        process_video(
            
            video_path=video_file,
            result_dir=result_dir,
            frames_dir=frames_dir,
            original_annotations_data=row,
            result_annotations_file_path=result_annotations_file_path,
            multiplyer=multiplyer,
            expected_size=expected_size,            
        )
        


duper(
    dataset_dir_path='Alex_Karachun/to_augment/',
    result_dir='Alex_Karachun/augmented/',
    original_annotations_file_path='Alex_Karachun/to_augment/annotations.csv',
    result_annotations_file_path='Alex_Karachun/augmented/pupu.csv',
    multiplyer=2,
    expected_size=[256, 144],  # высота, ширина
    frames_dir='Alex_Karachun/augmented/frames/'
)
