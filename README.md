# ComfyUI HR Nodes

Custom ComfyUI nodes by **Hashu786**.

Currently includes:

## 🎛️ LTX Multi IC-LoRA

Run **two LTX-2 IC-LoRAs at the same time** — e.g. **pose/structural control + an
image/appearance reference** — in a single generation, instead of being limited
to one IC-LoRA at a time.

| | |
|---|---|
| Category | `HRNodes/IC-LoRA` |
| Node | **LTX Multi IC-LoRA** |
| Models | LTX-2 / LTX-2.3 (the IC-LoRA distilled pipeline) |

### Why this exists

Stock IC-LoRA loading merges a LoRA into the model weights **globally**, so it
affects every token. Load two and their effects simply add together — the model
gets the *average* of "follow this pose" and "match this reference," which muddies
both. LTX already supports multiple **guides** (each `Add Video IC-LoRA Guide`
appends its conditioning tokens to the sequence); the missing piece was applying
each LoRA only where it belongs.

### Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/<your-github-username>/comfyui-hrnodes
# restart ComfyUI
```