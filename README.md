## Exploring StableDiffusion and ControlNet
This repository stores the code related to Assignment 1. I have not worked with SD1.5+ControlNet inference before, and had actually declined to participate in this hiring process, but I was sent the assignment anyway. I decided to complete it as the tasks seemed interesting and I had wanted to try out SD1.5+ControlNet for a while, and here are the results.

The overarching goal of this assignment is to critique the various guidance and conditioning controls used in image generation using [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) (SD1.5) with [ControlNet](https://github.com/lllyasviel/ControlNet). The main tasks for this assignment ([`.pdf`](./Assignment__HB1_2.pdf)) were as follows:
1. Generate the "best" possible output images given the prompts and depth-maps
    1. Verify that the generated output depths are the same as the input depths.
    2. Ensure that the pipeline/heuristics remain constant across all inputs.
2. Check if it is possible to generate images of different aspect ratios (test using `2_no_crop.png`)?
    1. Compare generation quality against the 1:1 aspect ratio input for the same image
3. Profile generation latency
    1. Are there any quick fixes to reduce it?
    2. Check how much reduction is possible.
    3. Compare generation quality vs latency.

The following sections describe the steps taken to complete these tasks.

## Overall Methodology: A Brief Overview

### Directory Structure / Table of Contents
* [`./0_standardize.ipynb`](./0_standardize.ipynb): Contains the code used to standardize the depth-maps and a study of its impact on generation.
* [`./1_experiments.ipynb`](./1_experiments.ipynb): Contains the image generation experiments and results.
* [`./2_aspect_ratio.ipynb`](./2_aspect_ratio.ipynb): Contains the experiments related to processing and generating images with non-1:1 aspect ratios.
* [`./3_gen_latency.ipynb`](./3_gen_latency.ipynb): Contains the profiling results for the various workflows and the optimizations applied.
* [`./utils.py`](./utils.py): Contains the utility functions used in the experiments.
* `./workflows/`: Contains the ComfyUI workflows (`.json` files) (and their screenshots) used for the experiments.
* `./metadata/`: Contains the provided images & prompts. `./metadata/grayscale/` contains the standardized depth-maps, used in the experiments.
* `./results/`: Contains the results of the experiments. This includes generated images, depth-maps, preprocessor outputs, and metrics (`.json`), alongside plots.

> [!IMPORTANT]
> Each notebook above describes the strategies, experiments, takeaways, and limitations for the corresponding task. Please check the notebooks for more details.

### Initial Setup
Given the time constraints and my unfamiliarity with the existing SD1.5+ControlNet pipelines, I decided to take a visual approach<sup><a href="#1-comfyui-woes" style="color: crimson;">#1</a></sup> to complete the assignment. I had some idea about WebUIs from using HuggingFace Spaces for casual image generation. So, I looked up what free-and-open-source UI to use for the project, not only to make it simple to quickly iterate on the project via visual feedback, but also to ensure that I can make modifications to the code, if needed<sup><a href="#2-code-modification-limitations" style="color: crimson;">#2</a></sup>. I found three good options - [ComfyUI](https://github.com/comfyanonymous/ComfyUI), [Automatic111](https://github.com/AUTOMATIC1111/stable-diffusion-webui), and [InvokeAI](https://github.com/invoke-ai/InvokeAI). I decided to go with ComfyUI as it seemed to be the most user-friendly due to its Node Graph interface that balanced granularity with ease-of-use and support for bleeding edge features, such as `ControlNet-Union`, that I was planning on using (unfortunately, it only supports `SDXL` at the moment). I set up a manual installation of ComfyUI following their [Manual Install](https://docs.comfy.org/get_started/manual_install) guide. The list below provides some related technical details:
* Models used:
  * **`SD1.5` Checkpoints**: https://huggingface.co/runwayml/stable-diffusion-v1-5
  * **`ControlNet`s used**: https://huggingface.co/lllyasviel/ControlNet | https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/tree/main
* Inference details:
  * **Inference UI**: ComfyUI (**with [`ComfyUI Manager`](https://github.com/ltdrdata/ComfyUI-Manager)** & several custom nodes). Please install `ComfyUI Manager`, load the workflows, and then, install the missing nodes through the manager to run the workflows.
  * **Inference Seed**: `12345`
  * **Inference Steps**: `20` (Modified in the latency experiments)
  * **Classifer-Free Guidance Scale (CFG)**: `8`
  * **Sampler & Scheduler**: `DPM++ 2M SDE (GPU)` & `Karras`
* Metrics used: `NRMSE`, `SSIM`, `PSNR` for depth-map fidelity or disparity, and `MANIQA` to assess generated image quality.
* Relevant system details:
  * **OS**: Ubuntu 22.04 LTS
  * **Python**: 3.12.5 (under `miniconda`)
  * **GPU**: NVIDIA GeForce RTX 3090 (24GB)
  * **Conda Environment File**: [`avataar-hb1.env.yml`](./avataar-hb1.env.yml)

To quantitatively analyze and compare the depth-maps of the generated images with the input depth-maps, I used the Normalized Root Mean Square Error (NRMSE), Structural Similarity Index Measure (SSIM), and Peak Signal-to-Noise Ratio (PSNR) metrics from `scikit-image`.

### Problems faced

Some issues arose during the experiments - some of which were due to the limitations of the tools used, while others were due to the complexity of the task.

#### #1: ComfyUI woes

While ComfyUI's interface and extension ecosystem are excellent for serial iterations to improve the output for a small set of inputs, it is not well-suited to running parameter sweeps or batched experiments. This limitation made automating the experiments and collating the results across multiple tests difficult. I had to manually adjust the parameters for each image and then save and analyze the results. This was a time-consuming process, which made it difficult to obtain any statistics from the experiments. I tried the `ComfyUI-to-Python-Extension` to get some raw code, I could modify, but it failed for complex workflows. Facing these issues, I attempted to switch to the `Diffusers` library, but it seemingly does not support all combinations of models (particularly, the LoRA models), that I have tested here. Moving forward, I would like to use `Diffusers` to redo (some of) these tests and, more importantly, report parameter sweeps.

#### #2: Code modification limitations
The changes mentioned here were limited to a bug/regression in `ComfyUI-to-Python-Extension`. My idea was to use an extension that has nodes that internally make use of `Diffusers` and then to convert the workflow into Python code, which I could then modify. However, in my testing, these nodes not only support only a miniscule number of the models tested here, they also do not support being exported to Python code (it could be a `ComfyUI-to-Python-Extension` issue). Additionally, to profile the generation latency, my plan was to attach profilers to this exported code, but this too was not possible, leading me to resort to a `ComfyUI Profiler` extension, that is highly unstable.

In hindsight, some of the planned experiments were over-ambitious and the decision to use ComfyUI was partially a mistake. Problems related to the extensions and an excruciatingly slow VPN did not help either. Even though I was able to complete the tasks, the results are not as comprehensive as I would have liked. I have collected some ideas for future work below, whose results I will slowly add to this repository.

### Remaining tasks
- [ ] Use `Diffusers` to redo these tests, and more importantly, report parameter sweeps.
- [ ] Generation Quality Experiments
  - [ ] Create controls from generated images and use them to generate new images (multi-round generation) (NOTE: This was cut form the assignment due to tool constraints).
  - [ ] Test [`IPAdapter`](https://github.com/tencent-ailab/IP-Adapter/blob/main/ip_adapter_demo.ipynb) and relighting to match the individual scenes better, and potentially improve quality.
  - [ ] Test `SDXL + ControlNet-Union`
  - [ ] Use prompt enhancer / explainer language models to improve the prompts and check if it improves the generation quality.
- [ ] Generation Latency Reduction, 
  - [ ] Use [Latent Consistency Models (LCMs)](https://huggingface.co/docs/diffusers/en/using-diffusers/inference_with_lcm) that bypass iterative sampling.
  - [ ] Parallelize node execution for the various pre-processors.

### References
* [StableDiffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
* ControlNet: [HF 🤗 Blog](https://huggingface.co/blog/controlnet) | [Base Checkpoints (v1-1)](https://huggingface.co/lllyasviel/ControlNet-v1-1/tree/main) | [`fp16` + LoRA CNs](https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/tree/main)
* ComfyUI
  * [PromptingPixels ComfyUI Workflows](https://promptingpixels.com/comfyui-workflows/)
  * [CivitAI Guide to ControlNet](https://education.civitai.com/civitai-guide-to-controlnet/)
  * [ComfyUI `Diffusers` Nodes](https://github.com/Jannchie/ComfyUI-J)
  * Primary Extensions:
    * [ComfyUI Manager](https://github.com/ltdrdata/ComfyUI-Manager)
    * [ComfyUI-to-Python-Extension](https://github.com/pydn/ComfyUI-to-Python-Extension)
    * [ComfyUI Profiler](https://github.com/tzwm/comfyui-profiler)
