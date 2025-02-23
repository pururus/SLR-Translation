import mediapipe as mp
import torch
import numpy as np

class MediapipeTransformer():
    def __init__(self):
        self.hands = mp.solutions.hands.Hands()

    def both_hand(self, multi_handedness):
        '''
        Checks that has two hands landmarks
        '''
        return len(multi_handedness) == 2

    def no_hands(self, multi_handedness):
        '''
        Checks that has no hands landmarks
        '''
        return len(multi_handedness) == 0

    def left_hand_only(self, multi_handedness):
        '''
        Checks that has left hand landmarks
        '''
        return not self.both_hand(multi_handedness) and not self.no_hands(multi_handedness) and multi_handedness[0].classification[0].label  == "Left"

    def right_hand_only(self, multi_handedness):
        '''
        Checks that has right hand landmarks
        '''
        return not self.both_hand(multi_handedness) and not self.no_hands(multi_handedness) and multi_handedness[0].classification[0].label  == "Right"

    def landmarcks_tovec(self, multi_hands_landmarks, multi_handedness):
        '''
        Transforms landmarks to tensor with size 1 x 126
        '''
        res = []
        if multi_handedness == None:
            res = [0.] * 126
            return torch.tensor(res)
        
        for landmarks in multi_hands_landmarks:
            for landmark in landmarks.landmark:
                res.extend([landmark.x, landmark.y, landmark.z])
                
        if self.no_hands(multi_handedness):
            res = [0.] * 126
        elif self.left_hand_only(multi_handedness):
            res += [0.] * 63
        elif self.right_hand_only(multi_handedness):
            res = [0.] * 63 + res
            
        return torch.tensor(res)
    
    def process_img(self, img: torch.tensor):
        '''
        Returns tensor of landmarks for hands in photo
        '''
        img_np = (img.permute(0, 2, 3, 1) * 255).to(dtype=torch.uint8)

        processed_img = self.hands.process(img_np[0])
        return self.landmarcks_tovec(processed_img.multi_hand_landmarks, processed_img.multi_handedness).unsqueeze(0)

    
    def __del__(self):
        self.hands.close()