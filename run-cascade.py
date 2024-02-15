import torch
from diffusers import StableCascadeDecoderPipeline, StableCascadePriorPipeline

# ref
# https://huggingface.co/stabilityai/stable-cascade

# torch
torch_device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
)

dtype = torch.bfloat16

# model
prior = StableCascadePriorPipeline.from_pretrained(
    "stabilityai/stable-cascade-prior",
    torch_dtype=dtype,
    low_cpu_mem_usage=False,
    ignore_mismatched_sizes=True,
).to(torch_device)

decoder = StableCascadeDecoderPipeline.from_pretrained(
    "stabilityai/stable-cascade",
    torch_dtype=dtype,
).to(torch_device)

# prompt
prompt = "a woman with red hair, realistic"
negative_prompt = "tattooing, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry, braid hair"

# generate
num_images_per_prompt = 2

prior_output = prior(
    prompt=prompt,
    height=1024,
    width=1024,
    negative_prompt=negative_prompt,
    guidance_scale=4.0,
    num_images_per_prompt=num_images_per_prompt,
)

decoder_output = decoder(
    image_embeddings=prior_output.image_embeddings,
    prompt=prompt,
    negative_prompt=negative_prompt,
    guidance_scale=0.0,
    output_type="pil",
).images

# save image
out_image = decoder_output[0]
out_image.save("output-cascade.png")
