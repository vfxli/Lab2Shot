"""SATA official semantic graph encoder/decoder, including its real T5 features.

Calls follow the official evaluation path (src/eval/retarget.py): per-window
`model(src_window, tgt_window, window_frames)` (the VAE returns (z, hatD), the
RVQ a 4-tuple), the official keep-after-overlap sliding assembly, then one
`hatD_recon_motion` over the whole sequence. The released legacy
`motion_2_graph`/test wrappers predate the temporal model signature (missing tf
/ consq_n); the states are assembled here as the official loader does.
"""
from __future__ import annotations

import torch
import re
from lab2shot_worker import load_job, serve
from lab2shot_worker.run import Run
from lab2shot_worker import rig_retarget as bridge

WINDOW_OVERLAP = 16  # the official eval/retarget.py default (window 64, overlap 16)


def main(job_path):
    run = Run(load_job(job_path), "SATA")
    data = bridge.read_job(run.job.inputs["retarget"])
    original = data
    p = run.params
    # The 「重定向预处理」 node decided which joints go in (its ignored list,
    # checked against this node's rules); the graph carries the rest.
    data, projection = bridge.graph_inputs(data)
    # Upstream test.py imports random training-skeleton generation. Defer its
    # training-data reads until that function is called, never fabricate a
    # character or write training files into the upstream checkout.
    import importlib.util
    import sys
    code = (run.job.repo_dir / "src/sata/utils/skel_gen_utils.py").read_text()
    begin = code.index("char_skel_list = parse_all_char(")
    end = code.index('\n\ndef create_random_skel', begin)
    initialization = code[begin:end]
    code = code[:begin] + 'char_skel_list = jointScaleStat = None\n' + code[end:]
    signature = 'def create_random_skel(mode="data", rnd_hierarchy=True):\n'
    code = code.replace(signature, signature + '    global char_skel_list, jointScaleStat\n'
        + '    if char_skel_list is None:\n'
        + '\n'.join('        ' + line for line in initialization.splitlines()) + '\n')
    local = run.job.dir / "inference_skel_gen.py"
    local.write_text(code)
    module_name = "sata.utils.skel_gen_utils"
    spec = importlib.util.spec_from_file_location(module_name, local)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    from sata.utils.model_loading import load_model_by_type
    from sata.utils.bvh2joint import JointTextProcessor
    from sata.utils.motion_utils import motion_normalize_h2s
    from sata.conversions.motion_to_graph import motion_2_states, skel_2_graph
    from sata.conversions.graph_to_motion import hatD_recon_motion
    from sata.mydataset import npz_2_data
    from sata.skel_pose_graph import SkelPoseGraph
    from torch_geometric.data import Batch
    from transformers import T5EncoderModel, T5Tokenizer

    name = p["model"]
    model_type = "rvq" if name.startswith("rvq") else "vae"
    model, cfg, stats = run.model("load_model", load_model_by_type, model_type, name, "cuda:0",
                                  stage_params={"model": f"SATA {name}"})
    run.stage("encode_joint_text")
    t5_path = run.job.weights_dir / "t5-base"
    tokenizer = T5Tokenizer.from_pretrained(t5_path, local_files_only=True)
    # Joint text has hundreds of tokens at most; keep this auxiliary encoder on
    # CPU so the shared GPU residency accounts only for the motion model.
    text_encoder = T5EncoderModel.from_pretrained(t5_path, local_files_only=True).eval()
    processor = JointTextProcessor()
    features = {}
    for side in ("src", "dst"):
        values = []
        roles = {j: (part, k) for part, joints in data[f'{side}_parts'].items() for k, j in enumerate(joints)}
        for j, name_i in enumerate(data[f"{side}_names"]):
            # The author's table lists only Spine..Spine4 and Neck..Neck1.
            # Extra segments have the SAME anatomical description, regardless
            # of a rig's chain length; never encode an opaque custom label.
            part, ordinal = roles.get(j, ('', 0))
            key = name_i
            parent_part = roles.get(int(data[f'{side}_parents'][j]), ('', 0))[0]
            toe_part = part if part.endswith('.toe') else parent_part if parent_part.endswith('.toe') else ''
            if toe_part:
                prefix = 'Left' if toe_part.startswith('l.') else 'Right'
                mode = p.get(('source' if side == 'src' else 'target') + '_toe_semantics', 'auto')
                native_idx = int(projection[side]['idx'][j])
                original_name = original[f'{side}_names'][native_idx].rsplit('/', 1)[-1].rsplit(':', 1)[-1] if native_idx >= 0 else ''
                known = {prefix + stem + suffix for stem in ('Toe', 'ToeBase') for suffix in ('', '_End')}
                if mode == 'auto' and original_name in known:
                    key = original_name
                else:
                    if mode == 'auto' and not part:
                        parent_idx = int(projection[side]['idx'][int(data[f'{side}_parents'][j])])
                        parent_name = original[f'{side}_names'][parent_idx].rsplit('/', 1)[-1].rsplit(':', 1)[-1] if parent_idx >= 0 else ''
                        mode = 'big_toe' if parent_name == prefix + 'Toe' else 'base'
                    key = prefix + ('Toe' if mode == 'big_toe' else 'ToeBase') + ('_End' if not part else '')
            if key not in processor.map_descriptive:
                if part in ('spine', 'chest'):
                    key = 'Spine'
                elif part == 'neck':
                    key = 'Neck'
                elif part.split('.')[-1] in ('thumb', 'index', 'middle', 'ring', 'pinky'):
                    prefix = 'Left' if part.startswith('l.') else 'Right'
                    key = prefix + 'Hand' + part.split('.')[-1].capitalize() + str(min(ordinal + 1, 4))
                else:
                    base_name = re.sub(r'\d+(?=_End$|$)', '', name_i)
                    if base_name in processor.map_descriptive:
                        key = base_name
            try:
                text = processor.get_text(key, mode="descriptive")
            except KeyError:  # a joint name outside the official map: describe it by its name
                text = f"The {name_i}"
            inputs = tokenizer(text, return_tensors="pt")
            with torch.inference_mode():
                values.append(text_encoder(**inputs).last_hidden_state.mean(dim=1))
        features[side] = torch.cat(values).numpy()
    del text_encoder
    source, _ = motion_normalize_h2s(bridge.fairmotion_input(data, "src", fps=20), False)
    target, _ = motion_normalize_h2s(bridge.fairmotion_input(data, "dst", fps=20), False)
    (lo, go, qb, edges), (q, pos, root, pv, qv, previous, contact) = motion_2_states(source)
    skeleton, poses = npz_2_data(lo, go, qb, edges, q, pos, qv, pv, previous, contact, root, features["src"])
    graphs = [SkelPoseGraph(skeleton, pose) for pose in poses]
    target_graph = skel_2_graph(target.skel, features["dst"])
    targets = [target_graph] * len(graphs)
    torch.manual_seed(int(p["seed"]))
    run.stage("retarget")
    window = int(p["window"])
    overlap = min(WINDOW_OVERLAP, window // 4)
    stride = window - overlap
    total = len(graphs)
    num_nodes = len(data["dst_names"])
    windows = 1 if total <= window else (total - window + stride - 1) // stride + 1
    pieces = []
    with torch.inference_mode():
        for w in range(windows):
            start = w * stride
            if start >= total:
                break
            end = min(start + window, total)
            src = Batch.from_data_list(graphs[start:end]).to("cuda:0")
            dst = Batch.from_data_list(targets[start:end]).to("cuda:0")
            out = model(src, dst, end - start)
            hatD = out[1].detach()                        # [T*nodes, out_dim]
            keep = slice(0, None) if not pieces else slice(overlap, None)
            pieces.append(hatD.view(end - start, num_nodes, -1)[keep].reshape(-1, hatD.shape[-1]))
    dst = Batch.from_data_list(targets).to("cuda:0")
    out, _ = hatD_recon_motion(torch.cat(pieces), dst, cfg["representation"]["out"], stats, len(graphs))
    facing = source.poses[1].get_root_facing_transform_byRoot(use_height=False)
    facing[:3, 3] = 0
    predicted = bridge.fairmotion_output(out[0], data, fps=20, normalized=True, facing=facing)
    world = bridge.fixed_output(predicted, bridge.graph_driven(projection["dst"]), original, fps=None)
    bridge.write_result(run, world, **bridge.sent(projection["src"], projection["dst"], src_names=data["src_names"],
                                                  dst_names=data["dst_names"]),
                        method="SATA", model=p["model"], semantic_encoder="T5-base",
                        window=window, overlap=overlap, model_fps=20)


if __name__ == "__main__":
    serve(main)
