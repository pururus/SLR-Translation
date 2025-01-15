import moviepy as mp
from copy import deepcopy
from uuid import uuid4
import pandas as pd
import numpy as np
import random
import math
import cv2
import logging

logging.getLogger("moviepy").setLevel(logging.CRITICAL)


'''
тз
done - mirroring - зеркалить
done - zooming - приближать и возвращать к исходному размеру
done - (надо просто при сохранении менять bitrate) - качество видео
done - cropping - обрезать
done - (при сохранении добавлять ffmpeg_params=['-vf', f'noise=alls={n}:allf=t+u'], где n in [0, 100])шумы
done - respeeding - менять скорость
done - rebritning, x in [0.2, 1.7] - изменение яркости и контрастности

fuck - крутить
fuck - дрожь
fuck - цветофильтры
fuck - Гауссово размытие
'''



def rebritning(clip: mp.VideoFileClip, x: float) -> mp.VideoFileClip:
    respeeder = mp.video.fx.MultiplyColor(factor = x)
    # respeeder.factor = x
    clip = respeeder.apply(clip)
    return clip




# def bluring(clip: mp.VideoFileClip, blur_strength: float) -> mp.VideoFileClip:
#     bluerer = mp.video.fx.HeadBlur(
#         fx = lambda x: clip.size[0] / 2,
#         fy = lambda x: clip.size[1] / 2,
#         radius = max(clip.size),
#     )
#     bluerer.intensity = blur_strength
    
#     clip = bluerer.apply(clip)
    
#     return clip




def respeeding(clip: mp.VideoFileClip, x: float) -> mp.VideoFileClip:
    respeeder = mp.video.fx.MultiplySpeed()
    respeeder.factor = x
    clip = respeeder.apply(clip)
    return clip


# clip = mp.VideoFileClip('Alex_Karachun/to_augment/2.mp4')
# processed_clip = respeeding(clip, x=0.8) 
# print('i' * 100)
# print(clip.size)
# print(processed_clip.size)
# processed_clip.write_videofile('Alex_Karachun/augmented/done.mp4')


# clip = mp.VideoFileClip('Alex_Karachun/to_augment/1.mp4')
# clip.write_videofile('Alex_Karachun/augmented/done.mp4', bitrate='500k')


def cropping(clip: mp.VideoFileClip, left_down: list[int, int], right_upper: list[int, int]) -> mp.VideoFileClip:
    original_size = clip.size
    
    cropper = mp.video.fx.Crop()
    cropper.x1, cropper.y1 = left_down
    cropper.x2, cropper.y2 = right_upper

    clip = cropper.apply(clip)
    clip = resizing(clip, original_size)
    
    return clip




def resizing(clip: mp.VideoFileClip, needed_size: list[int, int]) -> mp.VideoFileClip:
    resizer = mp.video.fx.Resize()
    resizer.new_size = needed_size
    clip = resizer.apply(clip)
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




# clip = mp.VideoFileClip('Alex_Karachun/to_augment/2.mp4')
# clip.write_videofile('Alex_Karachun/augmented/done.mp4', bitrate='700k')  # 100k - 500k


# def rotating(clip: mp.VideoFileClip, angle=10) -> mp.VideoFileClip:
#     rotator = mp.video.fx.Rotate(angle)

#     # clip = rotator.apply(clip)
#     rotated_clip = rotator.apply(clip)


#     width, height = rotated_clip.size
#     print('1' * 100)
#     print(clip.size)
#     print(rotated_clip.size)
#     diagonal_before = math.sqrt(clip.size[0]**2 + clip.size[1]**2)
#     diagonal_after = math.sqrt(width**2 + height**2)
    
#     # Коэффициент зума — это отношение диагоналей до и после поворота
#     # zoom_factor = diagonal_after / diagonal_before
    
#     # Применим зум с использованием вашей функции zooming
#     # zoomed_clip = zooming(rotated_clip, k=zoom_factor * 1.42)
#     angle_rad = math.radians(angle)
#     # k = abs(0.5 * width * math.cos(2 * angle_rad) / (0.5 * width * math.cos(angle_rad) - 0.5 * height * math.sin(angle_rad)))
#     k = height / width * math.sin(angle_rad)
#     print('k' * 100)
#     print(k)
#     zoomed_clip = zooming(rotated_clip, k=k)
#     # zoomed_clip = zooming(rotated_clip, k=1)
    
#     return zoomed_clip
    
#     # return clip






def mirroring(clip: mp.VideoFileClip) -> mp.VideoFileClip:
    clip = mp.video.fx.MirrorX().apply(clip)
    return clip



def process_video(video_path: str, result_dir: str, multiplyer: int) -> list[str]:
    '''
    берем видео из video_path
    стакаем его multiplyer раз
    
    к каждой из версий применяем случайный набор 
    преобразований со случайными коэффицентами
         
    записываем все видео в result_dir со случайными именами
    
    возвращаем список имен созданных видео без расширения
    (далее они будут использоваться для занесения в табличку)
    
    
    <можно еще добавить настройку формы видео (ширина x высота) в пикселях для модели>
    
    '''
    
    
    original_clip = mp.VideoFileClip(video_path)
    
    clips = [deepcopy(original_clip) for _ in range(multiplyer)]
    
    

    
    video_names = [str(uuid4()) for _ in range(multiplyer)]
    for i, clip in enumerate(clips):
        # кручу верчу запутать хочу - надо попортить видео
        '''
        mirroring - will_mirror: bool
        zooming - k: float in [1, 1.3]
        (надо просто при сохранении менять bitrate in ['300k', ...]) - качество видео
        cropping - left_down = [x_1 in [0; 10%], y_1 in [0; 10%]], right_upper = [x_2 in [-10%; 100%], y_2 in [-10%; 100%]]
        noize: (при сохранении добавлять ffmpeg_params=['-vf', f'noise=alls={n}:allf=t+u'], где n in [0, 100]) - bool
        respeeding x in [0.8, 1.5]
        rebritning, x in [0.2, 1.7] - изменение яркости и контрастности
        '''
        will_mirror = random.choice([True, False])
        
        k_for_zooming = random.uniform(1, 1.3)
        
        
        cropping_left_down = [int(random.uniform(0, clip.size[0] * 0.1)), int(random.uniform(0, clip.size[1] * 0.1))]
        cropping_right_upper = [clip.size[0] - int(random.uniform(0, clip.size[0] * 0.1)), clip.size[1] - int(random.uniform(0, clip.size[1] * 0.1))]
        
        
        new_speed = random.uniform(0.8, 1.5)
        
        new_britness = random.uniform(0.2, 1.7)
        
        # bitrate = random.choice(['100k', '200k', '300k', '400k', '500k', '600k', '700k'])
        # bitrate = random.choice(['300k', '400k', '500k', '600k', '700k, '])
        bitrate = str(random.choice(range(500, 3100, 100))) + 'k'

        # will_noize = random.choice([True, False])
        noize_k = random.uniform(0, 50)



        
        if will_mirror:
            clip = mirroring(clip)
            
        clip = zooming(clip, k_for_zooming)
        
        clip = cropping(clip, cropping_left_down, cropping_right_upper)
        
        clip = respeeding(clip, new_speed)
        
        clip = rebritning(clip, new_britness)
        
        
        clip.write_videofile(result_dir + video_names[i] + '.mp4', bitrate=bitrate, ffmpeg_params=['-vf', f'noise=alls={noize_k}:allf=t+u'])
        
        print(f"*" * 100, clip.size, f'will_mirror, {k_for_zooming=}, {cropping_left_down=}, {cropping_right_upper=}, {new_speed=}, {new_britness=}, {bitrate=}, {noize_k=}')
    
    return video_names

    
# process_video('main/to_augment/1.mp4', 'main/augmentated/', 3)

def duper(dataset_dir_path: str, result_dir: str, original_annotations_file_path: str, result_annotations_file_path: str, multiplyer: int) -> None:
    '''
    перебираем все видео таким вот так: dataset_dir_path + annotations['attachment_id'] + '.mp4'
    
    создаем multiplyer измененных копий каждого видео
    сохраняем все новые видео со случайными названиями в result_dir
    
    записываем информацию о новых видео в новую табличку в формате аналогичном slovo/annotations.csv
    данные о видео берем из original_annotations_file_path
    сохраняем новую табличку в result_annotations_file_path
    
    '''
    
    data = pd.read_csv(original_annotations_file_path, sep='\t')
    
    new_data = pd.DataFrame(columns=data.columns)
    
    
    new_video_id = 0
    for i, row in data.iterrows():
        
        dupped_names = process_video(
            video_path=dataset_dir_path + str(row['attachment_id']) + '.mp4',
            result_dir=result_dir,
            multiplyer=multiplyer
            )
        
        for new_name in dupped_names:
            row['attachment_id'] = new_name
            new_data.loc[new_video_id] = row
            new_video_id += 1
            

    
    new_data.to_csv(result_annotations_file_path, '\t')
    
    

duper(
    dataset_dir_path='Alex_Karachun/to_augment/',
    result_dir='Alex_Karachun/augmented/',
    original_annotations_file_path='Alex_Karachun/to_augment/annotations.csv',
    result_annotations_file_path='Alex_Karachun/augmented/pupu.csv',
    multiplyer=10
)
    
