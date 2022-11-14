import sys
import os

import_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'external'))
sys.path.insert(0, import_root_path)
print(import_root_path)

from Resemblyzer.demo_utils import *
from Resemblyzer.resemblyzer import preprocess_wav, VoiceEncoder

sampling_rate = 16000

wav_fpath = os.path.join('/home/tandon/SpeakerDiarizationProject/data/SyntheticVerbmobil/dialog_MEM_MBS_2.wav')
wav = preprocess_wav(wav_fpath)

# Cut some segments from single speakers as reference audio
segments = [[0, 8.0], [9.9, 19.0]]
speaker_names = ["MEM", "MBS"]
speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1] * sampling_rate)] for s in segments]

# Compare speaker embeds to the continuous embedding of the interview
# Derive a continuous embedding of the interview. We put a rate of 16, meaning that an
# embedding is generated every 0.0625 seconds. It is good to have a higher rate for speaker
# diarization, but it is not so useful for when you only need a summary embedding of the
# entire utterance. A rate of 2 would have been enough, but 16 is nice for the sake of the
# demonstration.
# We'll exceptionally force to run this on CPU, because it uses a lot of RAM and most GPUs
# won't have enough. There's a speed drawback, but it remains reasonable.

encoder = VoiceEncoder("cpu")
print("Running the continuous embedding on cpu, this might take a while...")
_, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True)

# Get the continuous similarity for every speaker. It amounts to a dot product between the
# embedding of the speaker and the continuous embedding of the interview
speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in
                   zip(speaker_names, speaker_embeds)}

# Run the interactive demo

interactive_diarization(similarity_dict, wav, wav_splits)
