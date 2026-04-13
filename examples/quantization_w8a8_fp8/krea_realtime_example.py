import torch
from diffusers import WanTransformer3DModel

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# select a Mixture of Experts model for quantization
MODEL_ID = "/data/wangzhe/models/krea-realtime-video-diffusers/"

model = WanTransformer3DModel.from_pretrained(
    MODEL_ID,
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)

recipe = QuantizationModifier(
    # targets=[
    #     "re:.*to_q$",
    #     "re:.*to_k$",
    #     "re:.*to_v$",
    #     "re:.*to_out.0$",
    #     ], 
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "re:.*condition_embedder.*",
        "re:.*patch_embedding.*",
        "re:.*proj_out.*",
    ],
)

# Apply quantization.
oneshot(model=model, recipe=recipe)

# Save to disk in compressed-tensors format.
model.name_or_path = MODEL_ID
SAVE_DIR = "transformer"
model.save_pretrained(SAVE_DIR, save_compressed=True)

print("========== SAVED COMPRESSED MODEL TO DISK ==============")
