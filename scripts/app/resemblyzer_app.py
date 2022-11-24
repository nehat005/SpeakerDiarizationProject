#!/usr/bin/python
"""
This script uses adapted code from resemblyzer's demo02_diarization.py script only for evaluation purpose.
https://github.com/resemble-ai/Resemblyzer
"""
import argparse
import sys
import os
import glob
import time

import_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'external'))
eval_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'evaluation'))

sys.path.insert(0, import_root_path)
sys.path.insert(1, eval_root_path)

import resemblyzer_eval
from Resemblyzer.demo_utils import *
from Resemblyzer.resemblyzer import preprocess_wav, VoiceEncoder
import der_evaluation

sampling_rate = 16000
encoder = VoiceEncoder("cpu")


def diarize(src_path, tgt_path):
    """

    Parameters
    ----------
    src_path
    tgt_path

    Returns
    -------

    """
    data_root_path = src_path
    save_path=tgt_path
    inference_start_time = time.time()
    for file in glob.glob(data_root_path + '/*.wav'):
        filename = os.path.basename(file).split('.')[0]
        wav_fpath = file
        wav = preprocess_wav(wav_fpath)
        segments = []
        speaker_names = []
        reference_rttm_file_path = os.path.join(data_root_path, filename + '.rttm')

        # Cut some segments from single speakers as reference audio
        n_speakers = len(filename.split('_')[1:-1])
        with open(reference_rttm_file_path, 'r') as in_file:
            data = in_file.readlines()
        for line in data:
            x = line.split(' ')[3:5]
            speaker = line.split(' ')[7]
            if len(speaker_names) >= n_speakers:
                break
            else:
                if speaker not in speaker_names:
                    speaker_names.append(speaker)
                    segments.append([float(x[0]), float(x[0]) + float(x[1]) - 0.1])

        speaker_wavs = [wav[int(s[0] * sampling_rate):int(s[1] * sampling_rate)] for s in segments]

        # Compare speaker embeds to the continuous embedding of the interview
        # Derive a continuous embedding of the interview. We put a rate of 16, meaning that an
        # embedding is generated every 0.0625 seconds. It is good to have a higher rate for speaker
        # diarization, but it is not so useful for when you only need a summary embedding of the
        # entire utterance. A rate of 2 would have been enough, but 16 is nice for the sake of the
        # demonstration.
        # We'll exceptionally force to run this on CPU, because it uses a lot of RAM and most GPUs
        # won't have enough. There's a speed drawback, but it remains reasonable.

        print("Running the continuous embedding on cpu, this might take a while...")
        _, cont_embeds, wav_splits = encoder.embed_utterance(wav, return_partials=True)

        # Get the continuous similarity for every speaker. It amounts to a dot product between the
        # embedding of the speaker and the continuous embedding of the interview
        speaker_embeds = [encoder.embed_utterance(speaker_wav) for speaker_wav in speaker_wavs]
        similarity_dict = {name: cont_embeds @ speaker_embed for name, speaker_embed in
                           zip(speaker_names, speaker_embeds)}

        # Run the interactive demo
        start_times = [0.0]
        times, names = resemblyzer_eval.get_best_speaker(similarity_dict, wav_splits)
        end_times = [times[0]]
        for i in range(1, len(times)):
            end_times.append(times[i])
            start_times.append(times[i - 1])

        result_rttm_file_path = os.path.join(save_path, filename + '_hyp.rttm')
        with open(result_rttm_file_path, 'w') as rttm_file:
            for i in range(len(start_times)):
                rttm_file.write('{} {} {} {:.3f} {:.3f} {} {} {} {} {}\n'.format('SPEAKER', filename, 1, start_times[i],
                                                                                 end_times[i] - start_times[i],
                                                                                 '<NA>', '<NA>', names[i], '<NA>',
                                                                                 '<NA>'))

        if not os.path.exists(os.path.join(save_path, 'hyp_rttm.scpf')):
            with open(os.path.join(save_path, 'hyp_rttm.scpf'), 'w') as out_scp_file:
                out_scp_file.write(f'{filename} {result_rttm_file_path}\n')
        else:
            with open(os.path.join(save_path, 'hyp_rttm.scpf'), 'a') as out_scp_file:
                out_scp_file.write(f'{filename} {result_rttm_file_path}\n')

    der_evaluation.compute_der(reference_rttm_scpf=os.path.join(data_root_path, 'ref_rttm.scpf'),
                               hypothesis_rttm_scpf=os.path.join(save_path,
                                                                 'hyp_rttm.scpf'),
                               save_file_path=os.path.join(save_path, 'absolute_evaluation.txt'))
    end_time = time.time()
    print('Inference Time: ', (end_time - inference_start_time) / 60)


def main(args):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--src', help='Source directory path containing wav files', required=False)
    parser.add_argument('--tgt', help='Target directory path to store experiment results')

    args = parser.parse_args(args)
    diarize(args.src, args.tgt)


if __name__ == '__main__':
    main(sys.argv[1:])
