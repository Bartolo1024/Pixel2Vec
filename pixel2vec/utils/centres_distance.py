import torch


def choose_best_idxes_with_cosine_similarity(centres, features):
    _, img_height, img_width = features.shape
    centres = torch.stack([el.rename(None) for el in centres])
    features = features.flatten(('H', 'W'), 'N').align_to('N', 'C')
    similarities = [(features @ centre).rename(None) for centre in centres]
    similarities = torch.stack(similarities).rename('D', 'N')
    _, best_idxes = similarities.max('D')
    best_idxes = best_idxes.unflatten('N', (('H', img_height), ('W', img_width)))
    return best_idxes


def choose_best_idxes_with_euclidean_distances(centres, features):
    _, img_height, img_width = features.shape
    centres = torch.stack([el.rename(None) for el in centres])
    features = features.flatten(('H', 'W'), 'N')
    diffs = [((features - centre)**2).sum('C').rename(None) for centre in centres]
    diffs = torch.stack(diffs).rename('D', 'N')
    _, best_idxes = diffs.min('D')
    best_idxes = best_idxes.unflatten('N', (('H', img_height), ('W', img_width)))
    return best_idxes
