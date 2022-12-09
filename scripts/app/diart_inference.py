import argparse
import os
import wave
import sys
import numpy as np
from diart import OnlineSpeakerDiarization
from diart.models import SegmentationModel
from diart.inference import Benchmark
from diart.blocks.diarization import PipelineConfig
import_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, import_root)
from scripts.file_utils import rttm_file
from scripts.evaluation import der_evaluation


class AudioFileInput:
    """
    audio file input module
    """

    def __init__(self, audio_file_path: str, chunk_size: int = 4096):
        """
        initialize audio file module
        :param audio_file_path: audio file path
        :param chunk_size: audio chunk size to read
        """
        self.wavefile = wave.open(audio_file_path, 'rb')
        self.chunk_size = chunk_size

    def __del__(self):
        """
        destructor
        :return: None
        """
        self.wavefile.close()

    def read(self) -> np.array:
        """
        read chunk from audio file
        :return: audio data chunk as int16 value array
        """
        data_bytes = self.wavefile.readframes(self.chunk_size)
        audio_samples_int16 = np.frombuffer(data_bytes, dtype='<i2').reshape(-1, )
        return audio_samples_int16


def diarize(speech_dir, rttm_dir, output_path):
    """
    Load configuration and diarize using diart Benchmark
    Parameters
    ----------
    speech_dir
    rttm_dir
    output_path

    Returns
    -------

    """
    config = PipelineConfig(
        # Set the model used in the paper
        segmentation=SegmentationModel.from_pyannote("pyannote/segmentation@Interspeech2021"),
        step=0.5,
        latency=0.5,
        tau_active=0.555,
        rho_update=0.422,
        delta_new=1.517
    )
    pipeline = OnlineSpeakerDiarization(config)

    benchmark = Benchmark(speech_path=speech_dir,
                          reference_path=rttm_dir,
                          output_path=output_path)

    benchmark(pipeline)


def evaluate(hyp_rttm_path, ref_rttm_scpf_path):
    """
    Compute DER for given diart diarization hypothesis RTTM files.
    Parameters
    ----------
    hyp_rttm_path
    ref_rttm_scpf_path

    Returns
    -------

    """
    save_path = hyp_rttm_path
    hyp_scpf_path = os.path.join(hyp_rttm_path, 'hyp_rttm.scpf')
    rttm_file.write_scpf_file(hyp_rttm_path, hyp_scpf_path)
    der_evaluation.compute_der(
        reference_rttm_scpf=ref_rttm_scpf_path,
        hypothesis_rttm_scpf=hyp_scpf_path,
        save_file_path=os.path.join(save_path, 'relative_evaluation.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-path', required=True, help='Source directory path to speech data')
    parser.add_argument('--ref-rttm-path', required=True, help='Source Directory path to reference rttm files')
    parser.add_argument('--save-path', required=True, help='Save results directory path')
    parser.add_argument('--dscore-evaluation', required=False, default=False, help='Use dscore module for evaluation')
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)

    diarize(speech_dir=args.source_path, rttm_dir=args.ref_rttm_path, output_path=args.save_path)

    if args.dscore_evaluation:
        if not os.path.exists(os.path.join(args.ref_rttm_path, 'ref_rttm.scpf')):
            raise ValueError('SCP file for reference RTTM files does not exist')
        else:
            evaluate(hyp_rttm_path=args.save_path,
                     ref_rttm_scpf_path=os.path.join(args.ref_rttm_path, 'ref_rttm.scpf'))
