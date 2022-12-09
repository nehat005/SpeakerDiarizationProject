import os
import sys
import glob
import time

file_tool_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'file_utils'))
eval_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'scripts', 'evaluation'))
sys.path.insert(0, eval_root_path)
sys.path.insert(1, file_tool_path)

import utils
from pyAudioAnalysis import audioAnalysis as analysis
import der_evaluation


def main(args):
    source_data_path, save_path = args

    os.makedirs(save_path, exist_ok=True)
    start_of_inference_time = time.time()
    for file in glob.glob(source_data_path + '/*.wav'):
        filename = os.path.basename(file).split('.')[0]
        num_speakers = len(filename.split('_')[1:-1])
        cls, purity_cluster_m, purity_speaker_m = analysis.speakerDiarizationWrapper(
            inputFile=os.path.join(file),
            numSpeakers=num_speakers, useLDA=False)

        start_times, end_times, speaker_labels = utils.labels_to_segments(cls)

        hypothesis_rttm_file_path = os.path.join(save_path, filename + '_hyp.rttm')
        with open(hypothesis_rttm_file_path, 'w') as out_file:
            # cls stores frame-wise speaker labels (each frame is classified into a class)
            for i in range(len(start_times)):
                duration = end_times[i] - start_times[i]
                out_file.write(
                    f'SPEAKER {filename} 1 {start_times[i]:.3f} {duration:.3f} <NA> <NA> SPEAKER_{speaker_labels[i]} <NA> <NA>\n')

        if os.path.exists(os.path.join(save_path, 'hyp_rttm.scpf')):
            with open(os.path.join(save_path, 'hyp_rttm.scpf'), 'a') as scpf_file:
                scpf_file.write(
                    f'{filename} {hypothesis_rttm_file_path}\n')
        else:
            with open(os.path.join(save_path, 'hyp_rttm.scpf'), 'w') as scpf_file:
                scpf_file.write(
                    f'{filename} {hypothesis_rttm_file_path}\n')

    der_evaluation.compute_der(
        reference_rttm_scpf=os.path.join(source_data_path, 'ref_rttm.scpf'),
        hypothesis_rttm_scpf=os.path.join(save_path, 'hyp_rttm.scpf'),
        save_file_path=os.path.join(save_path, 'relative_evaluation.txt'))
    print('Inference Time: ', time.time() - start_of_inference_time)


if __name__ == '__main__':
    main(sys.argv[1:])
