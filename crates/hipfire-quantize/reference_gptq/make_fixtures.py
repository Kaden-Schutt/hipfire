# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import numpy as np
from formats import write_hfhs, write_hblk, accumulate_block_diagonal_xxT, hessian_key

def main() -> None:
    out = Path(__file__).resolve().parent / "fixtures"
    out.mkdir(exist_ok=True)
    h1 = np.array([[1.0, 0.25], [0.25, 2.0]], dtype=np.float64)
    h2 = np.array([[3.0, -0.5, 0.1], [-0.5, 4.0, 0.2], [0.1, 0.2, 5.0]], dtype=np.float64)
    write_hfhs(out / "smoke.hfhs", [
        ("model.layers.0.q_proj", 0, h1),
        ("model.layers.1.mlp.down_proj", 3, h2),
    ], dtype="f32")
    write_hfhs(out / "smoke_f64.hfhs", [("tB", 0, h2)], dtype="f64")
    rng = np.random.default_rng(0)
    k = 512
    acts = rng.standard_normal((10, k)).astype(np.float64)
    blocks = accumulate_block_diagonal_xxT(acts)
    name = "layers.0.mlp.experts.3.gate_up_proj.weight"
    key = hessian_key(name)
    write_hblk(out / f"{key}.hblk", k, blocks)
    (out / "e8h1_meta.txt").write_text(
        f"name={name}\nkey={key}\nk={k}\nn_blocks={k//256}\nentry_b1_3_5={blocks[1,3,5]:.17e}\n"
    )
    print("ok", sorted(p.name for p in out.iterdir()))

if __name__ == "__main__":
    main()
