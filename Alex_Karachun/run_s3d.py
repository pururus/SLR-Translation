import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.video import s3d, S3D_Weights


def load_video_frames(
    video_path: str,
    start_frame: int,
    num_frames: int,
    frame_step: int,
    frame_size: tuple = (224, 224),
) -> np.ndarray:
    """
    Загружает num_frames кадров из видео, начиная с кадра start_frame,
    выбирая каждый frame_step-й кадр. Кадры изменяются до размера frame_size
    и конвертируются из BGR в RGB.

    Args:
        video_path (str): путь к видеофайлу.
        start_frame (int): номер начального кадра.
        num_frames (int): количество кадров для выборки.
        frame_step (int): шаг выборки кадров.
        frame_size (tuple): желаемый размер кадра (width, height).

    Returns:
        np.ndarray: массив кадров размерности (num_frames, height, width, 3).
    """
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Переходим к указанному стартовому кадру
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            raise ValueError(
                f"Не удалось прочитать кадр {start_frame + i * frame_step}. "
                f"Видео может быть слишком коротким."
            )
        # Изменяем размер кадра
        frame = cv2.resize(frame, frame_size)
        # Конвертируем из BGR в RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        # Пропускаем frame_step-1 кадров
        for _ in range(frame_step - 1):
            ret, _ = cap.read()
            if not ret:
                break

    cap.release()

    if len(frames) < num_frames:
        raise ValueError(
            f"Видео содержит только {len(frames)} кадров, а требуется {num_frames}."
        )
    return np.stack(frames, axis=0)


def preprocess_frames(frames_np: np.ndarray) -> torch.Tensor:
    """
    Преобразует numpy-массив кадров в тензор с нужным форматом:
    (batch, channel, time, height, width) и нормализует значения пикселей.

    Args:
        frames_np (np.ndarray): массив кадров размерности (T, H, W, C).

    Returns:
        torch.Tensor: тензор размерности (1, 3, T, H, W).
    """
    # Переставляем оси: (T, H, W, C) -> (T, C, H, W)
    frames_np = frames_np.transpose(0, 3, 1, 2)
    frames_tensor = torch.tensor(frames_np, dtype=torch.float32) / 255.0
    # Добавляем размерность батча и переставляем оси:
    # из (T, C, H, W) делаем (1, C, T, H, W)
    frames_tensor = frames_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)
    return frames_tensor


def classify_video(
    video_path: str, start_frame: int = 0, num_frames: int = 14, frame_step: int = 1
):
    """
    Загружает видео, выбирает кадры начиная с start_frame с указанным шагом,
    запускает модель S3D и выводит предсказанный класс.

    Args:
        video_path (str): путь к видеофайлу.
        start_frame (int): номер стартового кадра.
        num_frames (int): количество кадров для выборки.
        frame_step (int): шаг между выбранными кадрами.
    """
    # Загружаем предобученную модель S3D
    weights = S3D_Weights.KINETICS400_V1
    model = s3d(weights=weights)
    model.eval()

    # Загружаем и подготавливаем кадры
    frames_np = load_video_frames(video_path, start_frame, num_frames, frame_step)
    frames_tensor = preprocess_frames(frames_np)

    # Пропускаем данные через модель
    with torch.no_grad():
        output = model(frames_tensor)

    # Преобразуем логиты в вероятности
    probabilities = F.softmax(output, dim=1)
    predicted_index = probabilities.argmax(dim=1).item()
    predicted_label = weights.meta["categories"][predicted_index]

    print("Индекс предсказанного класса:", predicted_index)
    print("Название предсказанного класса:", predicted_label)


if __name__ == "__main__":
    video_path = "Alex_Karachun/video.mp4"
    start_frame = 10   # Начинаем с первого кадра (можно задать другое значение)
    frame_step = 2    # Например, брать каждый 2-й кадр
    num_frames = 14   # Выбираем 14 кадров

    classify_video(video_path, start_frame, num_frames, frame_step)
