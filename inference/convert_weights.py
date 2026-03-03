#!/usr/bin/env python3
"""Convert Qwen2.5-0.5B-Instruct safetensors → flat binary for ANE inference.

Output format (F32): config header (8 ints) + all weights in f32
Output format (F16): config header (8 ints) + embeddings f32 + projection weights f16
Output format (Q8):  config header (8 ints) + embeddings f32 + projection weights q8_0
Output format (Q4):  config header (8 ints) + embeddings f32 + projection weights q4_0

The 8th config int is the format flag: 0 = F32, 1 = F16, 2 = Q8, 3 = Q4.
Q8_0 format: blocks of 32 values, each block = 1 f16 scale + 32 int8 values (34 bytes).
Q4_0 format: blocks of 32 values, each block = 1 f16 scale + 1 f16 zero + 16 uint8 packed pairs (20 bytes).

Usage:
    python3 convert_weights.py <model_dir> <output.bin> [--f16|--q8|--q4]
"""

import struct
import sys
import numpy as np
from pathlib import Path
from safetensors import safe_open

Q8_BLOCK_SIZE = 32
Q4_BLOCK_SIZE = 32

def quantize_q4_0(weights_f32):
    """Quantize a 2D weight matrix to Q4_0 block format.
    Returns bytes: for each row, blocks of (f16_scale + f16_zero + 16*uint8 packed pairs).
    Each uint8 stores two 4-bit values: low nibble = even index, high nibble = odd index."""
    out_dim, in_dim = weights_f32.shape
    assert in_dim % Q4_BLOCK_SIZE == 0, f"in_dim {in_dim} not divisible by {Q4_BLOCK_SIZE}"

    n_blocks_per_row = in_dim // Q4_BLOCK_SIZE
    result = bytearray()

    for r in range(out_dim):
        row = weights_f32[r]
        for b in range(n_blocks_per_row):
            block = row[b * Q4_BLOCK_SIZE : (b + 1) * Q4_BLOCK_SIZE]
            bmin = np.min(block)
            bmax = np.max(block)
            if bmax == bmin:
                scale = np.float16(0.0)
                zero = np.float16(0.0)
                packed = bytes(Q4_BLOCK_SIZE // 2)
            else:
                scale_f = (bmax - bmin) / 15.0
                zero_f = bmin
                scale = np.float16(scale_f)
                zero = np.float16(zero_f)
                scale_f = float(scale) if float(scale) != 0.0 else 1e-10
                quant = np.clip(np.round((block - float(zero)) / scale_f), 0, 15).astype(np.uint8)
                packed = bytearray(Q4_BLOCK_SIZE // 2)
                for i in range(0, Q4_BLOCK_SIZE, 2):
                    packed[i // 2] = quant[i] | (quant[i + 1] << 4)
            result += scale.tobytes()
            result += zero.tobytes()
            result += bytes(packed)

    return bytes(result)


def quantize_q8_0(weights_f32):
    """Quantize a 2D weight matrix to Q8_0 block format.
    Returns bytes: for each row, blocks of (f16_scale + 32*int8)."""
    out_dim, in_dim = weights_f32.shape
    assert in_dim % Q8_BLOCK_SIZE == 0, f"in_dim {in_dim} not divisible by {Q8_BLOCK_SIZE}"

    n_blocks_per_row = in_dim // Q8_BLOCK_SIZE
    result = bytearray()

    for r in range(out_dim):
        row = weights_f32[r]
        for b in range(n_blocks_per_row):
            block = row[b * Q8_BLOCK_SIZE : (b + 1) * Q8_BLOCK_SIZE]
            amax = np.max(np.abs(block))
            scale = amax / 127.0 if amax > 0 else 0.0
            if scale > 0:
                quant = np.round(block / scale).astype(np.int8)
            else:
                quant = np.zeros(Q8_BLOCK_SIZE, dtype=np.int8)
            result += np.float16(scale).tobytes()
            result += quant.tobytes()

    return bytes(result)


def convert(model_dir: str, output_path: str, fmt: str = "f32"):
    model_dir = Path(model_dir)

    st_files = list(model_dir.glob("*.safetensors"))
    if not st_files:
        print(f"No safetensors files in {model_dir}")
        sys.exit(1)

    tensors = {}
    for f in st_files:
        with safe_open(str(f), framework="pt") as sf:
            for key in sf.keys():
                tensors[key] = sf.get_tensor(key).float().numpy()

    print(f"Loaded {len(tensors)} tensors from {len(st_files)} files")
    print(f"Mode: {fmt.upper()} projections (embeddings + norms + biases stay F32)")

    dim = 896
    hidden = 4864
    n_layers = 24
    n_heads = 14
    n_kv_heads = 2
    vocab_size = 151936
    max_seq = 512
    fmt_flag = {"f32": 0, "f16": 1, "q8": 2, "q4": 3}[fmt]

    def write_proj(f_out, tensor_f32):
        if fmt == "q4":
            f_out.write(quantize_q4_0(tensor_f32))
        elif fmt == "q8":
            f_out.write(quantize_q8_0(tensor_f32))
        elif fmt == "f16":
            f_out.write(tensor_f32.astype(np.float16).tobytes())
        else:
            f_out.write(tensor_f32.astype(np.float32).tobytes())

    with open(output_path, "wb") as f:
        f.write(struct.pack("iiiiiiii",
            dim, hidden, n_layers, n_heads, n_kv_heads, vocab_size, max_seq, fmt_flag))

        emb = tensors["model.embed_tokens.weight"].astype(np.float32)
        print(f"embed: {emb.shape} (f32)")
        f.write(emb.tobytes())

        for l in range(n_layers):
            prefix = f"model.layers.{l}"

            rms_att = tensors[f"{prefix}.input_layernorm.weight"].astype(np.float32)
            f.write(rms_att.tobytes())

            wq = tensors[f"{prefix}.self_attn.q_proj.weight"].astype(np.float32)
            wk = tensors[f"{prefix}.self_attn.k_proj.weight"].astype(np.float32)
            wv = tensors[f"{prefix}.self_attn.v_proj.weight"].astype(np.float32)
            wo = tensors[f"{prefix}.self_attn.o_proj.weight"].astype(np.float32)
            write_proj(f, wq)
            write_proj(f, wk)
            write_proj(f, wv)
            write_proj(f, wo)

            qb = tensors.get(f"{prefix}.self_attn.q_proj.bias")
            kb = tensors.get(f"{prefix}.self_attn.k_proj.bias")
            vb = tensors.get(f"{prefix}.self_attn.v_proj.bias")
            f.write((qb if qb is not None else np.zeros(wq.shape[0])).astype(np.float32).tobytes())
            f.write((kb if kb is not None else np.zeros(wk.shape[0])).astype(np.float32).tobytes())
            f.write((vb if vb is not None else np.zeros(wv.shape[0])).astype(np.float32).tobytes())

            rms_ffn = tensors[f"{prefix}.post_attention_layernorm.weight"].astype(np.float32)
            f.write(rms_ffn.tobytes())

            w_gate = tensors[f"{prefix}.mlp.gate_proj.weight"].astype(np.float32)
            w_up = tensors[f"{prefix}.mlp.up_proj.weight"].astype(np.float32)
            w_down = tensors[f"{prefix}.mlp.down_proj.weight"].astype(np.float32)
            write_proj(f, w_gate)
            write_proj(f, w_up)
            write_proj(f, w_down)

            print(f"  Layer {l}: Q{wq.shape} K{wk.shape} V{wv.shape} O{wo.shape} "
                  f"gate{w_gate.shape} up{w_up.shape} down{w_down.shape} [{fmt}]")

        rms_final = tensors["model.norm.weight"].astype(np.float32)
        f.write(rms_final.tobytes())

    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"\nWritten: {output_path} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 convert_weights.py <model_dir> <output.bin> [--f16|--q8|--q4]")
        sys.exit(1)
    fmt = "f32"
    if "--f16" in sys.argv:
        fmt = "f16"
    elif "--q8" in sys.argv:
        fmt = "q8"
    elif "--q4" in sys.argv:
        fmt = "q4"
    convert(sys.argv[1], sys.argv[2], fmt)
