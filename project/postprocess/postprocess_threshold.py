from torch import tensor
import torch

class PreprocessThreshold():
    def __init__(self, threshold):
        self.threshold = threshold
        
    def preprocess(self, glosses: tensor):
        '''
        Function between V2G and G2T. 
        Shortens the gloss sequence, leaves in the sequence only the glosses, with probability higher than threshold
        :param: glosses' tensor with shape m x n
        '''
        processed_glosses = []
        
        for i in range(glosses.shape[0]):
            if torch.max(glosses[i]) > self.threshold:
                processed_glosses.append(torch.argmax(glosses[i]))
            
            if len(processed_glosses) >= 2 and processed_glosses[-1] == processed_glosses[-2]:
                processed_glosses.pop()
            
        return tensor(processed_glosses)