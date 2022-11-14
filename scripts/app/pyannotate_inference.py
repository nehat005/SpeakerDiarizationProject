import os
import sys
import torch
import glob

import_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, import_path)

from scripts.evaluation import der_evaluation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

pipeline = torch.hub.load('pyannote/pyannote-audio', 'dia')

if __name__ == '__main__':
    data_root_path = '/data/SyntheticVerbmobilData/Train/entire_train'
    save_path = os.path.join(import_path, 'experiments/pyannotate/SyntheticVerbmobil/Train')
    for file in glob.glob(data_root_path + '/*.wav'):
        filename = os.path.basename(file).split('.')[0]
        diarization = pipeline({'audio': file})

        # print the result
        result_rttm_file_path = os.path.join(save_path, filename + '_hyp.rttm')
        with open(result_rttm_file_path, 'w') as out_file:
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
                duration = turn.end - turn.start
                out_file.write(
                    f'SPEAKER {filename} 1 {turn.start:.3f} {duration:.3f} <NA> <NA> SPEAKER_{speaker} <NA> <NA>\n')

        if not os.path.exists(os.path.join(save_path, 'hyp_rttm.scpf')):
            with open(os.path.join(save_path, 'hyp_rttm.scpf'), 'w') as out_scp_file:
                out_scp_file.write(f'{filename} {result_rttm_file_path}\n')
        else:
            with open(os.path.join(save_path, 'hyp_rttm.scpf'), 'a') as out_scp_file:
                out_scp_file.write(f'{filename} {result_rttm_file_path}\n')

    der_evaluation.compute_der(reference_rttm_scpf=os.path.join(data_root_path, 'ref_rttm.scpf'),
                               hypothesis_rttm_scpf=os.path.join(save_path,
                                                                 'hyp_rttm.scpf'),
                               save_file_path=os.path.join(save_path, 'relative_evaluation.txt'))
