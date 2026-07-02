# Synthetic medical bring-up prompts

Hand-authored test prompts for hipfire-steer driver bring-up. All benign and
medical-domain only.

- `good_prompts.txt` — factual/educational medical questions a model answers
  normally (the "answered" contrast).
- `bad_prompts.txt` — legitimate *personal-clinical* questions (symptoms,
  dosing, "should I worry") that trigger MedGemma's over-cautious
  disclaimer/refusal mode. These are the over-refusals the steering aims to
  suppress — not harmful content.

10 each, enough to validate the capture→derive→apply→score loop end to end.
Swap in a larger curated medical refusal +/- set when one exists.
