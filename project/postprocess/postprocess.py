from torch import tensor

def is_new(index: int, glosses: tensor, processed_glosses: list):
    return len(processed_glosses) == 0 or glosses[index].item() != processed_glosses[-1]

def is_not_mistake(index: int, glosses: tensor):
    return ((index + 1 < glosses.shape[0] and glosses[index].item() == glosses[index + 1].item()) or
            (index + 2 < glosses.shape[0] and glosses[index].item() == glosses[index + 2].item()))

def preprocess(glosses: tensor):
    '''
    Function between V2G and G2T. 
    Shortens the gloss sequence, leaves in the sequence only the glosses, which are long enough
    :param: glosses' tensor with shape 1 x n
    '''
    processed_glosses = []
    for i in range(glosses.shape[0]):
        if is_new(i, glosses, processed_glosses) and is_not_mistake(i, glosses):
            processed_glosses.append(glosses[i])
    
    return tensor(processed_glosses)
