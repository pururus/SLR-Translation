import moviepy as mov
from copy import deepcopy
from uuid import uuid4
import pandas as pd
import random
import logging
import os
import imageio

from tqdm.contrib.concurrent import process_map


import time

'''
ТЗ:
done - mirroring - зеркалить
done - zooming - приближать и возвращать к исходному размеру
done - (при сохранении менять bitrate) - качество видео
done - cropping - обрезать
done - (при сохранении добавлять ffmpeg_params со шумами)
done - respeeding - менять скорость
done - rebritning, x in [0.2, 1.7] - изменение яркости и контрастности
'''

def rebritning(clip: mov.VideoFileClip, x: float) -> mov.VideoFileClip:
    # Изменение яркости/контрастности
    rebriter = mov.video.fx.MultiplyColor(factor=x)
    return rebriter.apply(clip)


def respeeding(clip: mov.VideoFileClip, x: float) -> mov.VideoFileClip:
    # respeeder = mov.video.fx.MultiplySpeed()
    # respeeder.factor = x
    # return respeeder.apply(clip)
    return clip.with_speed_scaled(x)

def cropping(clip: mov.VideoFileClip, left_down: list, right_upper: list) -> mov.VideoFileClip:
    original_size = clip.size
    cropper = mov.video.fx.Crop()
    cropper.x1, cropper.y1 = left_down
    cropper.x2, cropper.y2 = right_upper 
    return resizing(cropper.apply(clip), original_size)

def resizing(clip: mov.VideoFileClip, needed_size: list) -> mov.VideoFileClip:
    # Используем встроенный метод resizing
    return clip.resized(new_size=needed_size)

def zooming(clip: mov.VideoFileClip, k=1.5) -> mov.VideoFileClip:
    original_size = clip.size
    cropper = mov.video.fx.Crop()
    cropper.x_center = clip.size[0] / 2
    cropper.y_center = clip.size[1] / 2
    cropper.width = clip.size[0] / k
    cropper.height = clip.size[1] / k
    return resizing(cropper.apply(clip), original_size)

def mirroring(clip: mov.VideoFileClip) -> mov.VideoFileClip:
    return mov.video.fx.MirrorX().apply(clip)


def save_clip_frames(clip: mov.VideoFileClip,
                     path_to_save_dir,
                     clip_name,
                     gest_name,
                     no_gest_name,
                     start_gest_frame_ind,
                     end_gest_frame_ind) -> None:
    
    gest_path = os.path.join(path_to_save_dir, gest_name)
    no_gest_path = os.path.join(path_to_save_dir, no_gest_name)

    os.makedirs(path_to_save_dir, exist_ok=True)
    os.makedirs(gest_path, exist_ok=True)
    os.makedirs(no_gest_path, exist_ok=True)
    
    named_gest_file = os.path.join(gest_path, "gestname.txt")
    not_named_gest_file = os.path.join(no_gest_path, "gestname.txt")
    
    with open(named_gest_file, 'w') as f:
        print(gest_name, file=f)
    
    with open(not_named_gest_file, 'w') as f:
        print(gest_name, file=f)

    
    
    for i, frame in enumerate(clip.iter_frames()):
        if i >= start_gest_frame_ind and i <= end_gest_frame_ind:
            filename = os.path.join(gest_path, f'{clip_name}_{i:04d}.png')
        else:
            filename = os.path.join(no_gest_path, f'{clip_name}_{i:04d}.png')
    
        imageio.imwrite(filename, frame)
            


def process_video(args) -> pd.DataFrame:
    video_path, result_dir, original_annotation, multiplyer, expected_size = args

    '''
    Берёт видео из video_path, делает multiplyer версий. к каждой:
    - применяет случайный набор преобразований.
    - видео сохраняются в result_dir с рандомными именами.
    - добавляет запись в cumulitive_annotation
    - выделяет кадры с жестом и без и раскладывает их по нужным папочкам
    
    
    '''
        
    
    annotations = pd.DataFrame()
    
    original_clip = mov.VideoFileClip(video_path)
    original_clip = resizing(original_clip, needed_size=expected_size)
    original_clip_length = int(original_clip.fps * original_clip.duration)

    
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
        new_britness = random.uniform(0.5, 1.4)
        bitrate = str(random.choice(range(700, 5000, 100))) + 'k'
        noize_k = random.uniform(0, 20)
        
        

        # применяет случайный набор преобразований.

        if will_mirror:
            clip = mirroring(clip)

        clip = zooming(clip, k_for_zooming)
        clip = cropping(clip, cropping_left_down, cropping_right_upper)
        clip = respeeding(clip, new_speed)
        clip = rebritning(clip, new_britness)
        
        
        
        output_path = os.path.join(result_dir, video_name + '.mp4')
        
        # видео сохраняются в result_dir с рандомными именами.
        begin_gest_part = original_annotation['begin'] / original_clip_length
        end_gest_part = original_annotation['end'] / original_clip_length
        
        
    

        clip.write_videofile(output_path,
                            bitrate=bitrate, 
                            ffmpeg_params=['-vf', f'noise=alls={noize_k}:allf=t+u', "-loglevel", "quiet"],
                            #  write_logfile=False,
                            logger=None)
    
    
        
        # добавляет запись в result_annotations_file_path
        new_clip_length = int(clip.fps * clip.duration)
        
        annotation = original_annotation.copy()
        annotation['attachment_id'] = video_name
        annotation['height'] = clip.size[1]
        annotation['width'] = clip.size[0]
        annotation['length'] = new_clip_length
        annotation['begin'] = int(new_clip_length * begin_gest_part)
        annotation['end'] = int(new_clip_length * end_gest_part)
        
        
        annotations = pd.concat([annotations, annotation.to_frame().T], ignore_index=True)
        
        

        save_clip_frames(clip=clip,
                        path_to_save_dir=result_dir,
                        clip_name=video_name, 
                        gest_name=annotation['text'], 
                        no_gest_name='no_event', 
                        start_gest_frame_ind=annotation['begin'], 
                        end_gest_frame_ind=annotation['end'])
        clip.close()
    
    original_clip.close()
    
    return annotations
            

def duper(dataset_dir_path: str,
          result_dir: str,
          original_annotations_file_path: str,
          result_annotations_file_path: str,
          multiplyer: int,
          expected_size=[256, 144],
          n_processes: int = 4) -> None:
    '''
    Для каждого видео из датасета:
      - Фильтрует по разрешению (оставляет только видео с нужными размерами)
      - Создаёт multiplyer аугментированных копий со случайными преобразованиями
      - С каждой копии видео разбивает кадры по папкам
      - Сохраняет новые видео в result_dir
      - Обновляет annotations.csv, содержащий все метаданные по созданным видео
    '''
    data = pd.read_csv(original_annotations_file_path, sep='\t')
    
    # Фильтрация по разрешению:
    allowed_res = [(1920, 1080), (1280, 720), (1920, 960)]
    data = data[data.apply(lambda row: (row['height'], row['width']) in allowed_res, axis=1)]
    
    tasks = []
    for i, row in data.iterrows():
        video_file = os.path.join(dataset_dir_path, str(row['attachment_id']) + '.mp4')
        if not os.path.exists(video_file):
            logging.error(f"Видео не найдено: {video_file}")
            continue
        tasks.append((
            video_file,
            result_dir,
            row,
            multiplyer,
            expected_size
        ))
    
    # with mp.Pool(n_processes) as pool:
    #     results = pool.map(process_video, tasks)
    results = process_map(process_video, tasks, max_workers=n_processes)
    
    new_data = pd.concat(results, ignore_index=True)
    # new_data.to_csv(result_annotations_file_path, sep="\t", index=False, mode="w")
    if os.path.exists(result_annotations_file_path) and os.stat(result_annotations_file_path).st_size > 0:
        new_data.to_csv(result_annotations_file_path, sep="\t", index=False, mode="a", header=False)
    else:
        new_data.to_csv(result_annotations_file_path, sep="\t", index=False, mode="w", header=True)

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    mp.freeze_support()
    
    t1 = time.time()
    workers_mult = 1/3
    multiplyer = 1
    duper(
        dataset_dir_path='../slovo_full/original/',
        result_dir='../slovo_full/augmented/',
        original_annotations_file_path='../slovo_full/annotations_full.csv',
        result_annotations_file_path='../slovo_full/augmented_annotations.csv',
        multiplyer=multiplyer,
        expected_size=[224, 224],  # высота, ширина
        n_processes=int(mp.cpu_count() * workers_mult)
        
        
        # dataset_dir_path='Alex_Karachun/to_augment/',
        # result_dir='Alex_Karachun/augmented/',
        # original_annotations_file_path='Alex_Karachun/to_augment/annotations.csv',
        # result_annotations_file_path='Alex_Karachun/augmented/pupu.csv',
        # multiplyer=3,
        # expected_size=[224, 224],  # высота, ширина
        # n_processes=mp.cpu_count()
    )
    t2 = time.time()
    
    print(f'{int(t2 - t1)} секунд работало на {workers_mult = } для {multiplyer = }, 100 ориг видео')
    
'''
16.5 мб/1 итоговое видео

223 секунд работало на workers_mult = 0.5 для multiplyer = 2, 100 ориг видео
195 секунд работало на workers_mult = 1 для multiplyer = 2, 100 ориг видео
196 секунд работало на workers_mult = 1.5 для multiplyer = 2, 100 ориг видео
192 секунд работало на workers_mult = 2 для multiplyer = 2, 100 ориг видео
221 секунд работало на workers_mult = 5 для multiplyer = 2, 100 ориг видео

96 секунд работало на workers_mult = 2 для multiplyer = 1, 100 ориг видео

лучше всего использовать workers_mult
тогда 1 итог видео делается 2 сек
тогда увеличить весь датасет в два раза занимает 11.5 часов
'''