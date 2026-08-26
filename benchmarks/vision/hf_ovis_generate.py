#!/usr/bin/env python3
"""Generate an OvisOCR2 reference, optionally after Hipfire Q8 round-trip.

The fake-Q8 mode changes only decoder matmul weights and implements the same
32-value symmetric quantization with an F16 scale as `hipfire-quantize`.
It distinguishes an inherently quantization-sensitive checkpoint from a GPU
Q8 execution bug without involving Hipfire's decoder implementation.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageTextToText, AutoTokenizer

from dump_hf_decoder_reference import fake_q8f16_


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--prompt", default="Read the page.")
    parser.add_argument("--fake-q8", action="store_true")
    parser.add_argument("--fake-f16", action="store_true")
    parser.add_argument(
        "--keep-f32",
        action="append",
        default=[],
        metavar="SUBSTRING",
        help="with --fake-q8, leave matching decoder parameter names untouched",
    )
    args = parser.parse_args()
    if args.fake_q8 and args.fake_f16:
        parser.error("--fake-q8 and --fake-f16 are mutually exclusive")

    image_processor = AutoImageProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.float32,
        device_map="cpu",
    )
    model.eval()

    if args.fake_q8:
        seen = set()
        quantized = 0
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                if (
                    name.startswith("model.visual.")
                    or not name.endswith("weight")
                    or "norm" in name
                    or any(pattern in name for pattern in args.keep_f32)
                ):
                    continue
                fake_q8f16_(parameter)
                quantized += parameter.numel()
        print(f"fake-q8: round-tripped {quantized} decoder parameters", flush=True)
    elif args.fake_f16:
        converted = 0
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.startswith("model.visual."):
                    continue
                parameter.copy_(parameter.to(torch.float16).float())
                converted += parameter.numel()
        print(f"fake-f16: round-tripped {converted} decoder parameters", flush=True)

    image = Image.open(args.image).convert("RGB")
    encoded_image = image_processor(images=image, return_tensors="pt")
    pixel_values = encoded_image["pixel_values"]
    grid_thw = encoded_image["image_grid_thw"]
    merge = model.config.vision_config.spatial_merge_size
    n_visual = int((grid_thw[0, 1] // merge) * (grid_thw[0, 2] // merge))

    image_pad = "<|image_pad|>" * n_visual
    messages = [
        {
            "role": "user",
            "content": (
                f"<|vision_start|>{image_pad}<|vision_end|>\n{args.prompt}"
            ),
        }
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    tokens = tokenizer(rendered, return_tensors="pt")
    input_ids = tokens["input_ids"]

    print(
        f"grid={grid_thw.tolist()} visual={n_visual} prompt={input_ids.shape[1]}",
        flush=True,
    )
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=tokens["attention_mask"],
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            max_new_tokens=args.max_tokens,
            do_sample=False,
        )
    generated = output[0, input_ids.shape[1] :]
    text = tokenizer.decode(generated, skip_special_tokens=False)
    Path(args.out).write_text(text)
    print(f"generated={generated.numel()}\n{text}", flush=True)


if __name__ == "__main__":
    main()
