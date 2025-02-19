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
import csv
from proglog import MuteProgressBarLogger



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
    return respeeder.apply(clip)

def respeeding(clip: mp.VideoFileClip, x: float) -> mp.VideoFileClip:
    # respeeder = mp.video.fx.MultiplySpeed()
    # respeeder.factor = x
    # return respeeder.apply(clip)
    return clip.with_speed_scaled(x)

def cropping(clip: mp.VideoFileClip, left_down: list, right_upper: list) -> mp.VideoFileClip:
    original_size = clip.size
    cropper = mp.video.fx.Crop()
    cropper.x1, cropper.y1 = left_down
    cropper.x2, cropper.y2 = right_upper 
    return resizing(cropper.apply(clip), original_size)

def resizing(clip: mp.VideoFileClip, needed_size: list) -> mp.VideoFileClip:
    # Используем встроенный метод resizing
    return clip.resized(new_size=needed_size)

def zooming(clip: mp.VideoFileClip, k=1.5) -> mp.VideoFileClip:
    original_size = clip.size
    cropper = mp.video.fx.Crop()
    cropper.x_center = clip.size[0] / 2
    cropper.y_center = clip.size[1] / 2
    cropper.width = clip.size[0] / k
    cropper.height = clip.size[1] / k
    return resizing(cropper.apply(clip), original_size)

def mirroring(clip: mp.VideoFileClip) -> mp.VideoFileClip:
    return mp.video.fx.MirrorX().apply(clip)


# def process_video(video_path: str, result_dir: str, multiplyer: int, expected_size: list[int, int]) -> None:
def process_video(video_path: str,
                  result_dir: str,
                  frames_dir: str,
                  original_annotation: pd.Series,
                  result_annotations_file_path: str,
                  multiplyer: int,
                  expected_size: list[int, int]) -> pd.DataFrame:
    '''
    Берёт видео из video_path, делает multiplyer версий. к каждой:
    - применяет случайный набор преобразований.
    - видео сохраняются в result_dir с рандомными именами.
    - добавляет запись в cumulitive_annotation
    - выделяет кадры с жестом и без и раскладывает их по нужным папочкам
    
    
    '''
    
    annotations = pd.DataFrame()
    
    original_clip = mp.VideoFileClip(video_path)
    original_clip = original_clip.resized(height=expected_size[0], width=expected_size[1])

    # clips = [deepcopy(original_clip) for _ in range(multiplyer)]
    # video_names = [str(uuid4()) for _ in range(multiplyer)]
    
    for i in range(multiplyer):
        clip = original_clip.copy()
        video_name = str(uuid4())
        
        new_fps = 24
        clip = clip.with_fps(new_fps)
        
        
        # Случайные параметры для аугментации:
        will_mirror = random.choice([True, False])
        k_for_zooming = random.uniform(1, 1.2)
        cropping_left_down = [int(random.uniform(0, clip.size[0] * 0.05)),
                              int(random.uniform(0, clip.size[1] * 0.05))]
        cropping_right_upper = [clip.size[0] - int(random.uniform(0, clip.size[0] * 0.05)),
                                clip.size[1] - int(random.uniform(0, clip.size[1] * 0.05))]
        new_speed = random.uniform(0.8, 1.5)
        new_britness = random.uniform(0.5, 1.7)
        bitrate = str(random.choice(range(700, 5000, 100))) + 'k'
        noize_k = random.uniform(0, 20)
        
        

        # применяет случайный набор преобразований.

        # НЕ ЗАБЫТЬ РАЗКОМЕНТИРОВАТЬ
        if will_mirror:
            clip = mirroring(clip)

        clip = zooming(clip, k_for_zooming)
        clip = cropping(clip, cropping_left_down, cropping_right_upper)
        clip = respeeding(clip, new_speed)
        clip = rebritning(clip, new_britness)
        
        # # Ресайз до ожидаемого размера (здесь используем высоту из expected_size[0])
        
        
        
        
        output_path = os.path.join(result_dir, video_name + '.mp4')
        
        # видео сохраняются в result_dir с рандомными именами.
        original_clip_length = int(original_clip.fps * original_clip.duration)
        begin_gest_part = original_annotation['begin'] / original_clip_length
        end_gest_part = original_annotation['end'] / original_clip_length
        
        
    

        clip.write_videofile(output_path,
                             bitrate=bitrate, 
                             ffmpeg_params=['-vf', f'noise=alls={noize_k}:allf=t+u', "-loglevel", "quiet"],
                            #  write_logfile=False,
                             logger=None)
       
        
        # добавляет запись в result_annotations_file_path
        
        new_clip_lenght = int(clip.fps * clip.duration)


        annotation = original_annotation.copy()
        annotation['attachment_id'] = video_name
        annotation['height'] = expected_size[0]
        annotation['width'] = expected_size[1]
        annotation['lenght'] = new_clip_lenght
        annotation['begin'] = int(original_clip_length * begin_gest_part)
        annotation['end'] = int(original_clip_length * end_gest_part)
        
        annotations = pd.concat([annotations, annotation.to_frame().T], ignore_index=True)
        
        clip.close()
    
    original_clip.close()
    
    return annotations
            


def duper(dataset_dir_path: str, result_dir: str, original_annotations_file_path: str, 
          result_annotations_file_path: str, multiplyer: int, expected_size=[256, 144],
          frames_dir: str = None) -> None:
    '''
    Для каждого видео из датасета:
      - Фильтрует по разрешению и оставляет только вертикальные видео с разрешениями (1920, 1080), (1280, 720), (1920, 960)
      - Создаёт multiplyer аугментированных копий со случайными преобразованиями
      - С каждой копии видео, разкладывает кадры с жестом и без по папочкам
      - Сохраняет новые видео в result_dir
      - Обновляет annotations.csv
    
    '''
    data = pd.read_csv(original_annotations_file_path, sep='\t')
    
    # Фильтрует по разрешению и оставляет только вертикальные видео с разрешениями (1920, 1080), (1280, 720), (1920, 960),
    allowed_res = [(1920, 1080), (1280, 720), (1920, 960)]
    data = data[data.apply(lambda row: (row['height'], row['width']) in allowed_res, axis=1)]
    
    new_data = pd.DataFrame(columns=data.columns)
    
    for i, row in data.iterrows():
        video_file = os.path.join(dataset_dir_path, str(row['attachment_id']) + '.mp4')
        if not os.path.exists(video_file):
            logging.error(f"Видео не найдено: {video_file}")
            continue

        curr_data = process_video(
            video_path=video_file,
            result_dir=result_dir,
            frames_dir=frames_dir,
            original_annotation=row,
            result_annotations_file_path=result_annotations_file_path,
            multiplyer=multiplyer,
            expected_size=expected_size,  
        )
        
        new_data = pd.concat([new_data, curr_data], ignore_index=True)

    print(new_data)
    new_data.to_csv(result_annotations_file_path, sep="\t", index=False, mode="w")

        

# for l in ['tqdm.cli', 'tqdm', 'dotenv.main', 'dotenv', 'imageio', 'imageio_ffmpeg', 'imageio.plugins.ffmpeg', 'imageio.plugins', 'PIL.Image', 'PIL', 'stack_data.serializing', 'stack_data', 'parso', 'concurrent.futures', 'concurrent', 'asyncio', 'prompt_toolkit.buffer', 'prompt_toolkit', 'parso.python.diff', 'parso.python', 'parso.cache']:
#     logging.getLogger(l).setLevel(logging.ERROR)



duper(
    dataset_dir_path='Alex_Karachun/to_augment/',
    result_dir='Alex_Karachun/augmented/',
    original_annotations_file_path='Alex_Karachun/to_augment/annotations.csv',
    result_annotations_file_path='Alex_Karachun/augmented/pupu.csv',
    multiplyer=2,
    expected_size=[256, 144],  # высота, ширина
    frames_dir='Alex_Karachun/augmented/frames/'
)

