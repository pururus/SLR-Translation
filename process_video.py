from Alex_Karachun.run_s3d_new import make_predictions_from_video, make_predictions_from_video_7_frames
model_dir='Alex_Karachun/trained_models/s3d_1000_gestures_1000_videos_7_epochs_done/s3d_1000_gestures_1000_videos_5_epoch'
model_dir2="Alex_Karachun/trained_models/s3d_1000_gestures_100_videos_10_epochs_done_7_frames/s3d_1000_gestures_100_videos_7_epoch"
from project.gigachat.gigachat import GigaChat
from project.postprocess.postprocess_uniq import PreprocessUniq
import json
import numpy as np
import torch
import asyncio

async def process_video(video_path, model_dir=model_dir2, model=None, chat=None, preprocessor=None, labels=None):
    if not chat:
        chat = GigaChat()
        await chat.check_token()
    
    if not preprocessor:
        preprocessor = PreprocessUniq()
        
    if not labels:
        label2idx_path = f"{model_dir}/label2idx.json"
        
        
        with open(label2idx_path, "r", encoding="utf-8") as f:
            label2idx = json.load(f)
        num_classes = len(label2idx)
        idx2label = {v: k for k, v in label2idx.items()}
    
    # res = make_predictions_from_video(video_path, model_dir, model)
    res=make_predictions_from_video_7_frames(video_path, model_dir=model_dir2)
    res = preprocessor.preprocess(res)
    
    text = []
    # print(idx2label[721])
    for i in range(res.shape[0]):
        label = idx2label[res[i].item()]
        print(res[i].item(), label)
        if label != 'no_event':
            text.append(label)
    # print(' '.join(text))
    if len(text):
        return await chat.parse_translation(' '.join(text))

    return "Не удалось распознать предложение. Убедитесь, что слова хороо видно в кадре и попробуйте еще раз)"

if __name__ == "__main__":
    async def main():
        print(await process_video('/Users/svatoslavpolonskiy/Documents/Deep_python/SLR-Translation/IMG_0208.MOV'))

    asyncio.run(main())