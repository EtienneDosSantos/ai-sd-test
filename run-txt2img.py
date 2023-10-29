import torch
from diffusers import StableDiffusionPipeline

# ref
# https://huggingface.co/docs/diffusers/api/pipelines/overview

# model
sd_model_path = "Lykon/dreamshaper-8"

pipe = StableDiffusionPipeline.from_pretrained(
    sd_model_path,
    use_safetensors=True,
    # torch_dtype=torch.float16,
)

# torch
torch_device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# pipe
pipe = pipe.to(torch_device)

# prompt
prompt = "a woman with red hair, realistic"
negative_prompt = "tattooing, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, braid hair"

output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=20,
    guidance_scale=7.5,
)

out_image = output.images[0]
out_image.save("output-txt2img.png")
