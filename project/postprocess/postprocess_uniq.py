from torch import tensor
import torch

class PreprocessUniq():
    @classmethod
    def is_new(self, index: int, glosses: tensor, processed_glosses: list):
        '''
        Checks that the gloss at index in glosses is different from the last gloss in processed_glosses
        '''
        return len(processed_glosses) == 0 or glosses[index].item() != processed_glosses[-1]

    @classmethod
    def is_not_mistake(self, index: int, glosses: tensor):
        '''
        Checks that the gloss at index in glosses is similar to the next one or the one after the next
        '''
        return ((index + 1 < glosses.shape[0] and glosses[index].item() == glosses[index + 1].item()) or
                (index + 2 < glosses.shape[0] and glosses[index].item() == glosses[index + 2].item()))

    def preprocess(self, glosses: tensor):
        '''
        Function between V2G and G2T. 
        Shortens the gloss sequence, leaves in the sequence only the glosses, which are long enough
        :param: glosses' tensor with shape m x n
        '''
        glosses = torch.argmax(glosses, dim=0)
        processed_glosses = []
        for i in range(glosses.shape[0]):
            if PreprocessUniq.is_new(i, glosses, processed_glosses) and PreprocessUniq.is_not_mistake(i, glosses):
                processed_glosses.append(glosses[i])
        
        return tensor(processed_glosses)
