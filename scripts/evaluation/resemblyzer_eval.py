"""
This script uses adapted code from resemblyzer's demo02_diarization.py script only for evaluation purpose.
https://github.com/resemble-ai/Resemblyzer
"""

from typing import Tuple, List
import numpy as np

def get_best_speaker(similarity_dict, wav_splits) -> Tuple[List, List]:
    x_crop = 5
    names = []
    messages = []
    times = [((s.start + s.stop) / 2) / 16000 for s in wav_splits]
    rate = 1 / (times[1] - times[0])
    crop_range = int(np.round(x_crop * rate))

    for i in range(len(wav_splits)):
        crop = (max(i - crop_range, 0), i + crop_range)
        similarities = [s[i] for s in similarity_dict.values()]
        # print(similarities)
        best = np.argmax(similarities)
        name, similarity = list(similarity_dict.keys())[best], similarities[best]
        if similarity > 0.75:
            message = "Speaker: %s (confident)" % name
            names.append(name)
            messages.append(message)
        elif similarity > 0.65:
            message = "Speaker: %s (uncertain)" % name
            names.append(name)
            messages.append(message)
        else:
            name = 'Unknown'
            message = "Unknown/No speaker"
            names.append(name)
            messages.append(message)

    return times, names
