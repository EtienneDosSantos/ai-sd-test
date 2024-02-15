import torch
from diffusers import StableDiffusionUpscalePipeline
from PIL import Image

# ref
# https://huggingface.co/docs/diffusers/api/pipelines/stable_diffusion/upscale

# model
sd_model_path = "stabilityai/stable-diffusion-x4-upscaler"

pipe = StableDiffusionUpscalePipeline.from_pretrained(
    sd_model_path,
    revision="fp16",
    # torch_dtype=torch.float16,
)

# torch
torch_device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

# pipe
pipe = pipe.to(torch_device)

# prompt
prompt = "a white cat"

# generate
low_res_img = Image.open("extras/images/low-res-model.png").convert("RGB")

output = pipe(
    prompt=prompt,
    image=low_res_img,
)

# save image
out_image = output.images[0]
out_image.save("output-upscale.png")
