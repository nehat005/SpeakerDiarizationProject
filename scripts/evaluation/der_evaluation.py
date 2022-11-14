import os
import sys
from tabulate import tabulate

module_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
module_path = os.path.join(module_root_path, 'external', 'dscore')

if module_path not in sys.path:
    sys.path.insert(0, module_path)

import score
from scorelib import utils, turn

sys.path.remove(module_path)


def compute_der(reference_rttm_scpf: str, hypothesis_rttm_scpf: str, save_file_path: str, step=0.010):
    """
    This function computes der and other metrics with the score function from '/dscore/score.py' module.
    This code is in lines with the score.py main function

    :param reference_rttm_scpf: path to the scpf storing all reference rttm file paths
    :param hypothesis_rttm_scpf: path to the scpf storing all hypothesis rttm file paths
    :param save_file_path: path to the file to store all evaluation metrics
    :param step: default value = 0.10 (refer /dscore/score.py)
    :return: None
    """
    ref_turns = []
    hyp_turns = []
    hyp_rttm_fns = []
    ref_rttm_fns = []

    if reference_rttm_scpf is not None:
        ref_rttm_fns = [item.split()[1] for item in score.load_script_file(reference_rttm_scpf)]
        ref_turns, _ = score.load_rttms(ref_rttm_fns)

    if hypothesis_rttm_scpf is not None:
        hyp_rttm_fns = [item.split()[1] for item in score.load_script_file(hypothesis_rttm_scpf)]
        hyp_turns, _ = score.load_rttms(hyp_rttm_fns)

    if not ref_rttm_fns:
        utils.error('No reference RTTMs specified.')
        sys.exit(1)
    if not hyp_rttm_fns:
        utils.error('No system RTTMs specified.')
        sys.exit(1)

    uem = score.gen_uem(ref_turns, hyp_turns)
    # Trim turns to UEM scoring regions
    ref_turns = turn.trim_turns(ref_turns, uem)
    hyp_turns = turn.trim_turns(hyp_turns, uem)

    ref_turns = turn.merge_turns(ref_turns)
    hyp_turns = turn.merge_turns(hyp_turns)

    score.check_for_empty_files(ref_turns, hyp_turns, uem)
    # Score
    file_scores, global_scores = score.score(
        ref_turns, hyp_turns, uem, step=step,
        jer_min_ref_dur=0.0, collar=0.0,
        ignore_overlaps=False)

    tbl = score.print_table(
        file_scores, global_scores, 2, 'simple')

    with open(save_file_path, 'w') as file_id:
        file_id.write(tbl)

