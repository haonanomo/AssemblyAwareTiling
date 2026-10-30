"""Convert a PyTorch sequence stability checkpoint to JAX/Haiku parameters."""

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import torch
import jax
import haiku as hk

from tiling_rl.stability.stability_checker_jax_seq import build_checker_model


logger = logging.getLogger(__name__)


def conv_w_pt_to_jax(w):
    # PyTorch: [out_c, in_c, kh, kw]
    # Haiku:   [kh, kw, in_c, out_c]
    return np.transpose(w, (2, 3, 1, 0))


def linear_w_pt_to_jax(w):
    # PyTorch: [out_dim, in_dim]
    # Haiku:   [in_dim, out_dim]
    return np.transpose(w, (1, 0))


def load_pt_checkpoint(pt_path: str):
    payload = torch.load(pt_path, map_location="cpu")
    if "model_state_dict" in payload:
        sd = payload["model_state_dict"]
    else:
        sd = payload

    out = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.detach().cpu().numpy()
        else:
            out[k] = v
    return out, payload


def set_hk_param(params_mutable, mod_name, p_name, value):
    if mod_name not in params_mutable:
        raise KeyError(f"Missing Haiku module: {mod_name}")
    if p_name not in params_mutable[mod_name]:
        raise KeyError(f"Missing Haiku param: {mod_name}/{p_name}")
    params_mutable[mod_name][p_name] = value


def convert_pt_state_to_hk_params(pt_state, hk_params):
    params_mutable = hk.data_structures.to_mutable_dict(hk_params)

    HK = "stability_u_net2_djax"

    # stem
    set_hk_param(params_mutable, f"{HK}/input_proj", "w",
                 conv_w_pt_to_jax(pt_state["model.input_proj.weight"]))
    set_hk_param(params_mutable, f"{HK}/input_proj", "b",
                 pt_state["model.input_proj.bias"])

    set_hk_param(params_mutable, f"{HK}", "stem_norm_scale", pt_state["model.stem_norm.weight"])
    set_hk_param(params_mutable, f"{HK}", "stem_norm_offset", pt_state["model.stem_norm.bias"])

    def copy_res(pt_prefix, hk_prefix):
        set_hk_param(params_mutable, f"{hk_prefix}/conv1", "w",
                     conv_w_pt_to_jax(pt_state[f"{pt_prefix}.conv1.weight"]))
        set_hk_param(params_mutable, f"{hk_prefix}/conv2", "w",
                     conv_w_pt_to_jax(pt_state[f"{pt_prefix}.conv2.weight"]))
        set_hk_param(params_mutable, f"{hk_prefix}/conv3", "w",
                     conv_w_pt_to_jax(pt_state[f"{pt_prefix}.conv3.weight"]))

        set_hk_param(params_mutable, hk_prefix, "norm1_scale", pt_state[f"{pt_prefix}.norm1.weight"])
        set_hk_param(params_mutable, hk_prefix, "norm1_offset", pt_state[f"{pt_prefix}.norm1.bias"])

        set_hk_param(params_mutable, hk_prefix, "norm2_scale", pt_state[f"{pt_prefix}.norm2.weight"])
        set_hk_param(params_mutable, hk_prefix, "norm2_offset", pt_state[f"{pt_prefix}.norm2.bias"])

        set_hk_param(params_mutable, hk_prefix, "norm3_scale", pt_state[f"{pt_prefix}.norm3.weight"])
        set_hk_param(params_mutable, hk_prefix, "norm3_offset", pt_state[f"{pt_prefix}.norm3.bias"])

        shortcut_key = f"{pt_prefix}.shortcut.weight"
        if shortcut_key in pt_state:
            set_hk_param(params_mutable, f"{hk_prefix}/shortcut", "w",
                         conv_w_pt_to_jax(pt_state[shortcut_key]))

    def copy_down(i):
        pt_prefix = f"model.down.{i}.down.conv"
        hk_prefix = f"{HK}/down_{i}_down/conv"
        set_hk_param(params_mutable, hk_prefix, "w",
                     conv_w_pt_to_jax(pt_state[f"{pt_prefix}.weight"]))
        set_hk_param(params_mutable, hk_prefix, "b",
                     pt_state[f"{pt_prefix}.bias"])

        j = 0
        while f"model.down.{i}.res.{j}.conv1.weight" in pt_state:
            copy_res(
                pt_prefix=f"model.down.{i}.res.{j}",
                hk_prefix=f"{HK}/down_{i}_res_{j}",
            )
            j += 1

    def copy_up(i):
        copy_res(
            pt_prefix=f"model.up.{i}.merge",
            hk_prefix=f"{HK}/up_{i}_merge",
        )

        j = 0
        while f"model.up.{i}.res.{j}.conv1.weight" in pt_state:
            copy_res(
                pt_prefix=f"model.up.{i}.res.{j}",
                hk_prefix=f"{HK}/up_{i}_res_{j}",
            )
            j += 1

        pt_prefix = f"model.up.{i}.up.conv"
        hk_prefix = f"{HK}/up_{i}_up/conv"
        set_hk_param(params_mutable, hk_prefix, "w",
                     conv_w_pt_to_jax(pt_state[f"{pt_prefix}.weight"]))
        set_hk_param(params_mutable, hk_prefix, "b",
                     pt_state[f"{pt_prefix}.bias"])

    # down stack
    i = 0
    while f"model.down.{i}.down.conv.weight" in pt_state:
        copy_down(i)
        i += 1

    # mid
    copy_res("model.mid.block1", f"{HK}/mid_block1")
    copy_res("model.mid.block2", f"{HK}/mid_block2")

    # up stack
    i = 0
    while f"model.up.{i}.merge.conv1.weight" in pt_state:
        copy_up(i)
        i += 1

    # out feat
    set_hk_param(params_mutable, f"{HK}/conv_out_feat", "w",
                 conv_w_pt_to_jax(pt_state["model.conv_out_feat.weight"]))
    set_hk_param(params_mutable, f"{HK}/conv_out_feat", "b",
                 pt_state["model.conv_out_feat.bias"])

    # head
    set_hk_param(params_mutable, f"{HK}/head_linear1", "w",
                 linear_w_pt_to_jax(pt_state["model.head.0.weight"]))
    set_hk_param(params_mutable, f"{HK}/head_linear1", "b",
                 pt_state["model.head.0.bias"])

    set_hk_param(params_mutable, f"{HK}/head_linear2", "w",
                 linear_w_pt_to_jax(pt_state["model.head.2.weight"]))
    set_hk_param(params_mutable, f"{HK}/head_linear2", "b",
                 pt_state["model.head.2.bias"])

    return hk.data_structures.to_immutable_dict(params_mutable)


def convert_checkpoint(pt_ckpt_path, jax_out_path):
    """Convert explicit input/output files and return the output path.

    Accepts a PyTorch training payload or a plain model state dictionary.
    The output is a pickle with params and source-checkpoint metadata.
    """
    pt_ckpt_path = Path(pt_ckpt_path)
    jax_out_path = Path(jax_out_path)
    if pt_ckpt_path.resolve() == jax_out_path.resolve():
        raise ValueError("Input and output checkpoint paths must differ")
    pt_state, _ = load_pt_checkpoint(str(pt_ckpt_path))

    model = build_checker_model()
    rng = jax.random.PRNGKey(0)
    in_channels = int(pt_state["model.input_proj.weight"].shape[1])
    dummy = np.zeros((1, 16, 16, in_channels), dtype=np.float32)
    hk_params = model.init(rng, dummy, is_training=False)
    converted_params = convert_pt_state_to_hk_params(pt_state, hk_params)

    save_payload = {
        "params": converted_params,
        "extra": {"source_pt_ckpt": str(pt_ckpt_path)},
    }
    jax_out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jax_out_path, "wb") as f:
        pickle.dump(save_payload, f)
    logger.info("Saved converted JAX checker: %s", jax_out_path)
    return jax_out_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt_ckpt_path", type=Path, required=True,
                        help='required')
    parser.add_argument("--jax_out_path", type=Path, required=True,
                        help='required')
    args = parser.parse_args(argv)
    if not args.pt_ckpt_path.is_file():
        parser.error(f"Checkpoint does not exist: {args.pt_ckpt_path}")
    if args.pt_ckpt_path.resolve() == args.jax_out_path.resolve():
        parser.error("Input and output checkpoint paths must differ")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    return convert_checkpoint(args.pt_ckpt_path, args.jax_out_path)


if __name__ == "__main__":
    main()
