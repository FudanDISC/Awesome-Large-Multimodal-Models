
<div align="center">
  <h1 style="display: inline-block; font-size: 32px;">Awesome-Large-Multimodal-Models</h1>
</div>
This repo summarizes the construction of current LMMs from the perspective of 

<font size=7><div align='center' >***input-output representation space extension***</div></font>

* Based on the structure of input-output spaces, we systematically review the existing models, including main-stream models based on discrete-continuous hybrid spaces and models with unified multi-modal discrete representations. 
* Readers can refer to our [[📖 Preprint Paper](https://www.preprints.org/manuscript/202411.0685)] for detailed explanations.


<p align="center">
    <img src="./assets/introduction.png" width="80%" height="80%">
</p>


- [Preliminary](#preliminary)
- [Awesome Models 🤗](#awesome-models)
  - [Large Vision-Language Models 🤗](#large-vision-language-models-)
    - [Output Modality: 📝](#mode-text)
    - [Output Modality: 📝🖼️](#mode-image)
  - [Large Audio-Language Models 🤗](#large-audio-language-models-)
  - [Any Modality Models 🤗](#any-modality-models-)


# Preliminary

As presented in Figure below, the evolution of multi-modal research paradigms could be divided into three stages. 

<p align="center">
    <img src="./assets/research_evolution.png" width="80%" height="80%">
</p>

For readers to have a general picture about the development, we provide a tutorial here. The contents are summarized as follows:


* Part 1: [Vision-Language Pre-Training](https://github.com/FudanDISC/Awesome-Large-Multimodal-Models/blob/main/assets/tutorial-part1.pdf)
* Part 2: [Architectures and Traning of LMMs](https://github.com/FudanDISC/Awesome-Large-Multimodal-Models/blob/main/assets/tutorial-part2.pdf)
* Part 3: [Evaluation of LMMs](https://github.com/FudanDISC/Awesome-Large-Multimodal-Models/blob/main/assets/tutorial-part3.pdf)
* Part 4: [Further Capability of LMMs](https://github.com/FudanDISC/Awesome-Large-Multimodal-Models/blob/main/assets/tutorial-part4.pdf)
* Part 5: [Extension to Embodied Agents](https://github.com/FudanDISC/Awesome-Large-Multimodal-Models/blob/main/assets/tutorial-part5.pdf)




# Awesome Models (Sort by Time of Release) 📄

- 🗂️ **Original Table:** [Google Sheets Link ⭐](https://docs.google.com/spreadsheets/d/1KkaqTO9c5eJQdVdpDWPrb4SmO8mJ55NEk159epA9X4o/edit?usp=sharing)
- ⚙️ **Model Training Settings Table:** [Google Sheets Link ⭐](https://docs.google.com/spreadsheets/d/1hMlhT_MzItdgiYt1XWUB3GoWEmK5jAr_eYzcFx2lM9w/edit?usp=sharing)

**Input Type**
> Type A: Discrete Text Token + Continuous 🖼️🔊🧊 Feature

> Type B: Discrete Text Token + Discrete 🖼️🔊🧊 Token

**Output Type**
> Type 1: Discrete Text Token Only

> Type 2: Discrete Text Token + Continuous 🖼️🔊🧊 Feature

> Type 3: Discrete Text Token + Discrete 🖼️🔊🧊 Token

**Modality**
> Text: 📝
> Vision: 🖼️
> Audio: 🔊
> 3D: 🧊

## Large Vision-Language Models 🤗

### <a id="mode-text"></a>Output Modality: 📝

| **Model** | **Code** | **Input** | **Output** | **Architecture (LLM & Encoder & Conn.)** | **Res.** | **Date** |
| :-- | :--: | :--: | :--: | :-- | :--: | :--: |
| [Flamingo](https://arxiv.org/abs/2204.14198) | [🔗](https://github.com/lucidrains/flamingo-pytorch) | A | 1 | Chinchilla & NFNet & Perceiver | 480 | 2022/04 |
| [BLIP-2](https://arxiv.org/abs/2301.12597) | [🔗](https://github.com/salesforce/LAVIS/tree/main/projects/blip2) | A | 1 | Flan-T5 / OPT & CLIP ViT-L / Eva-CLIP & Q-Former | 224 | 2023/01 |
| [LLaMA-Adapter](https://arxiv.org/abs/2303.16199) | [🔗](https://github.com/OpenGVLab/LLaMA-Adapter) | A | 1 | LLaMA & CLIP-ViT-L/14 & MLP | 224 | 2023/03 |
| [MiniGPT-4](https://arxiv.org/abs/2304.10592) | [🔗](https://github.com/Vision-CAIR/MiniGPT-4) | A | 1 | Vicuna & Eva-CLIP ViT-G/14 & Q-Former | 224 | 2023/04 |
| [LLaVA](https://arxiv.org/abs/2304.08485) | [🔗](https://github.com/haotian-liu/LLaVA) | A | 1 | Vicuna & CLIP ViT-L/14 & Linear | 224 | 2023/04 |
| [mPLUG-Owl](https://arxiv.org/abs/2304.14178) | [🔗](https://github.com/X-PLUG/mPLUG-Owl) | A | 1 | LLaMA & CLIP ViT-L/14 & Abstractor | 224 | 2023/04 |
| [LLaMA-Adapter V2](https://arxiv.org/abs/2304.15010) | [🔗](https://github.com/OpenGVLab/LLaMA-Adapter) | A | 1 | LLaMA & CLIP-ViT-L/14 & MLP | 224 | 2023/04 |
| [InstructBLIP](https://arxiv.org/abs/2305.06500) | [🔗](https://github.com/salesforce/LAVIS/tree/main/projects/instructblip) | A | 1 | Flan-T5 / Vicuna & Eva-CLIP ViT-G/14 & Q-Former | 224 | 2023/05 |
| [Otter](https://arxiv.org/abs/2305.03726) | [🔗](https://github.com/Luodian/otter) | A | 1 | LLaMA & CLIP ViT-L/14 & Perceiver | 224 | 2023/05 |
| [LaVIN](https://arxiv.org/pdf/2305.15023) | [🔗](https://github.com/luogen1996/LaVIN) | A | 1 | LLaMA & CLIP ViT-L/14 & MLP | 224 | 2023/05 |
| [MultiModal-GPT](https://arxiv.org/abs/2305.04790) | [🔗](https://github.com/open-mmlab/Multimodal-GPT) | A | 1 | LLaMA & CLIP ViT-L/14 & Perceiver | 224 | 2023/05 |
| [Shikra](https://arxiv.org/abs/2306.15195) | [🔗](https://github.com/shikras/shikra) | A | 1 | Vicuna & CLIP ViT-L/14 & Linear | 224 | 2023/06 |
| [Video-ChatGPT](https://arxiv.org/abs/2306.05424) | [🔗](https://github.com/mbzuai-oryx/Video-ChatGPT) | A | 1 | Vicuna & CLIP ViT-L/14 & Linear | 224 | 2023/06 |
| [Valley](https://arxiv.org/pdf/2306.07207) | [🔗](https://github.com/RupertLuo/Valley) | A | 1 | Stable-Vicuna & CLIP ViT-L/14 & Temporal + Linear | 224 | 2023/06 |
| [Lynx](https://arxiv.org/abs/2307.02469) | [🔗](https://github.com/bytedance/lynx-llm) | A | 1 | Vicuna & EVA-1B & Resampler | 420 | 2023/07 |
| [Qwen-VL](https://arxiv.org/abs/2308.12966) | [🔗](https://github.com/QwenLM/Qwen-VL) | A | 1 | Qwen & OpenCLIP ViT-bigG & Cross-Attention | 448 | 2023/08 |
| [BLIVA](https://arxiv.org/abs/2308.09936) | [🔗](https://github.com/mlpc-ucsd/BLIVA) | A | 1 | Flan-T5 / Vicuna & Eva-CLIP ViT-G/14 & Q-Former + MLP | 224 | 2023/08 |
| [IDEFICS](https://huggingface.co/blog/idefics) | [🔗](https://huggingface.co/blog/idefics) | A | 1 | LLaMA & OpenCLIP ViT-H/14 & Perceiver | 224 | 2023/08 |
| [OpenFlamingo](https://arxiv.org/abs/2308.01390) | [🔗](https://github.com/mlfoundations/open_flamingo) | A | 1 | LLaMA / MPT & CLIP ViT-L/14 & Perceiver | 224 | 2023/08 |
| [InternLM-XComposer](https://arxiv.org/pdf/2309.15112) | [🔗](https://github.com/InternLM/InternLM-XComposer/tree/main/InternLM-XComposer-1.0) | A | 1 | InternLM & Eva-CLIP ViT-G/14 & Perceiver | 224 | 2023/09 |
| [LLaVA-1.5](https://arxiv.org/abs/2310.03744) | [🔗](https://github.com/haotian-liu/LLaVA) | A | 1 | Vicuna 1.5 & CLIP ViT-L/14 & MLP | 336 | 2023/10 |
| [MiniGPT-v2](https://arxiv.org/abs/2310.09478) | [🔗](https://github.com/Vision-CAIR/MiniGPT-4) | A | 1 | LLaMA-2 & EVA & Linear | 448 | 2023/10 |
| [Fuyu-8B](https://www.adept.ai/blog/fuyu-8b) | [🔗](https://huggingface.co/adept/fuyu-8b) | A | 1 | Persimmon & - & Linear | ∞ | 2023/10 |
| [UReader](https://arxiv.org/abs/2310.05126) | [🔗](https://github.com/LukeForeverYoung/UReader) | A | 1 | LLaMA & CLIP ViT-L/14 & Abstractor | 224*20 | 2023/10 |
| [CogVLM](https://arxiv.org/abs/2311.03079) | [🔗](https://github.com/THUDM/CogVLM) | A | 1 | Vicuna 1.5 & EVA2-CLIP-E & MLP | 490 | 2023/11 |
| [Monkey](https://arxiv.org/abs/2311.06607) | [🔗](https://github.com/Yuliang-Liu/Monkey) | A | 1 | Qwen & OpenCLIP ViT-bigG & Cross-Attention | 896 | 2023/11 |
| [ShareGPT4V](https://arxiv.org/pdf/2311.12793) | [🔗](https://github.com/ShareGPT4Omni/ShareGPT4V) | A | 1 | Vicuna-1.5 & CLIP ViT-L/14 & MLP | 336 | 2023/11 |
| [mPLUG-Owl2](https://arxiv.org/abs/2311.04257) | [🔗](https://github.com/X-PLUG/mPLUG-Owl/tree/main/mPLUG-Owl2) | A | 1 | LLaMA-2 & CLIP ViT-L/14 & Abstractor | 448 | 2023/11 |
| [SPHINX](https://arxiv.org/abs/2311.07575) | [🔗](https://github.com/Alpha-VLLM/LLaMA2-Accessory) | A | 1 | LLaMA-2 & CLIP/DINOv2 & Linear + Q-Former | 672 | 2023/11 |
| [InternVL](https://arxiv.org/abs/2312.14238) | [🔗](https://github.com/OpenGVLab/InternVL) | A | 1 | Vicuna & InternViT & QLLaMA / MLP | 336 | 2023/12 |
| [MobileVLM](https://arxiv.org/abs/2312.16886) | [🔗](https://github.com/Meituan-AutoML/MobileVLM) | A | 1 | MobileLLaMA & CLIP ViT-L/14 & LDP (conv) | 336 | 2023/12 |
| [VILA](https://arxiv.org/abs/2312.07533) | [🔗](https://github.com/NVlabs/VILA) | A | 1 | LLaMA-2 & CLIP ViT-L & Linear | 336 | 2023/12 |
| [Osprey](https://arxiv.org/pdf/2312.10032) | [🔗](https://github.com/CircleRadon/Osprey) | A | 1 | Vicuna & CLIP ConvNeXt-L & MLP | 512 | 2023/12 |
| [Honeybee](https://arxiv.org/abs/2312.06742) | [🔗](https://github.com/khanrc/honeybee) | A | 1 | Vicuna-1.5 & CLIP ViT-L/14 & C/D-Abstractor | 336 | 2023/12 |
| [Omni-SMoLA](https://arxiv.org/abs/2312.00968) | - | A | 1 | UL2 & Siglip ViT-G/14 & Linear | 1064 | 2023/12 |
| [LLaVA-NeXT](https://llava-vl.github.io/blog/2024-01-30-llava-next/) | [🔗](https://github.com/haotian-liu/LLaVA) | A | 1 | Vicuna/Mistral/Yi & CLIP ViT-L/14 & MLP | 672 | 2024/01 |
| [InternLM-XComposer2](https://arxiv.org/abs/2401.16420) | [🔗](https://github.com/InternLM/InternLM-XComposer) | A | 1 | InternLM-2 & CLIP ViT-L/14 & MLP | 490 | 2024/01 |
| [MouSi](https://arxiv.org/abs/2401.17221) | [🔗](https://github.com/FudanNLPLAB/MouSi) | A | 1 | Vicuna-1.5 & Multi-Encoders & Poly-Expert Fusion | 1024 | 2024/01 |
| [LLaVA-MoLE](https://arxiv.org/abs/2401.16160) | [🔗](https://github.com/forwchen/LLaVA-MoLE) | A | 1 | Vicuna1.5 & CLIP ViT-L/14 & MLP | 336 | 2024/01 |
| [MoE-LLaVA](https://arxiv.org/abs/2401.15947) | [🔗](https://github.com/PKU-YuanGroup/MoE-LLaVA) | A | 1 | StableL / Qwen / Phi-2 & CLIP ViT-L/14 & MLP | 336 | 2024/01 |
| [MobileVLM V2](https://arxiv.org/abs/2402.03766) | [🔗](https://github.com/Meituan-AutoML/MobileVLM) | A | 1 | MobileLLaMA & CLIP ViT-L/14 & LDP v2 | 336 | 2024/02 |
| [Bunny](https://arxiv.org/abs/2402.11530) | [🔗](https://github.com/BAAI-DCAI/Bunny) | A | 1 | Phi/LLaMA/StableLM & SigLIP/EVA-CLIP & MLP | 1152 | 2024/02 |
| [TinyLLaVA](https://arxiv.org/abs/2402.14289) | [🔗](https://github.com/DLCV-BUAA/TinyLLaVABench) | A | 1 | TinyLLaMA/Phi-2/StableLM & SigLIP/CLIP & MLP | 336/384 | 2024/02 |
| [SPHINX-X](https://arxiv.org/abs/2402.05935) | [🔗](https://github.com/Alpha-VLLM/LLaMA2-Accessory) | A | 1 | Multi-LLM & CLIP/DINOv2 & Linear | 672 | 2024/02 |
| [Mini-Gemini](https://arxiv.org/abs/2403.18814) | [🔗](https://github.com/dvlab-research/MGM) | A | 1 | Gemma/Vicuna/Mixtral & CLIP+ConvNext & Cross-Attn+MLP | 1536 | 2024/03 |
| [DeepSeek-VL](https://arxiv.org/abs/2403.05525) | [🔗](https://github.com/deepseek-ai/DeepSeek-VL) | A | 1 | Deepseek LLM & SigLIP-L / SAM-B & MLP | 1024 | 2024/03 |
| [LLaVA-UHD](https://arxiv.org/abs/2403.11703) | [🔗](https://github.com/thunlp/LLaVA-UHD) | A | 1 | Vicuna & CLIP ViT-L/14 & Perceiver | 336*6 | 2024/03 |
| [Yi-VL](https://arxiv.org/html/2403.04652v1) | [🔗](https://github.com/01-ai/Yi) | A | 1 | Yi & CLIP ViT-H/14 & MLP | 448 | 2024/03 |
| [MM1](https://arxiv.org/abs/2403.09611) | [🔗](https://github.com/kyegomez/MM1) | A | 1 | in-house LLM & CLIP ViT-H* & C-Abstractor | 1792 | 2024/03 |
| [VL-Mamba](https://arxiv.org/abs/2403.13600) | [🔗](https://github.com/ZhengYu518/VL-Mamba) | A | 1 | Mamba LLM & CLIP/SigLIP & VSS + MLP | 384 | 2024/03 |
| [Cobra](https://arxiv.org/abs/2403.14520) | [🔗](https://github.com/h-zhao1997/cobra) | A | 1 | Mamba-Zephyr & DINOv2 + SigLIP & MLP | 384 | 2024/03 |
| [InternVL 1.5](https://arxiv.org/abs/2404.16821) | [🔗](https://github.com/OpenGVLab/InternVL) | A | 1 | InternLM2 & InternViT-6B & MLP | 448*40 | 2024/04 |
| [Phi-3-Vision](https://arxiv.org/abs/2404.14219) | [🔗](https://github.com/microsoft/Phi-3CookBook) | A | 1 | Phi-3 & CLIP ViT-L/14 & MLP | 336*16 | 2024/04 |
| [PLLaVA](https://arxiv.org/abs/2404.16994) | [🔗](https://github.com/magic-research/PLLaVA) | A | 1 | Vicuna/Mistral/Yi & CLIP ViT-L/14 & MLP + Pooling | 336 | 2024/04 |
| [TextHawk](https://arxiv.org/abs/2404.09204) | [🔗](https://github.com/yuyq96/TextHawk) | A | 1 | InternLM-1 & SigLIP-SO400M/14 & Resampler + MLP | ∞ | 2024/04 |
| [Imp](https://arxiv.org/abs/2405.12107) | [🔗](https://github.com/MILVLG/imp) | A | 1 | Phi-2 & SigLIP & MLP | 384 | 2024/05 |
| [IDEFICS2](https://arxiv.org/abs/2405.02246) | [🔗](https://huggingface.co/docs/transformers/en/model_doc/idefics2) | A | 1 | Mistral-v0.1 & SigLIP-SO400M/14 & Perceiver + MLP | 384*4 | 2024/05 |
| [ConvLLaVA](https://arxiv.org/abs/2405.15738) | [🔗](https://github.com/alibaba/conv-llava) | A | 1 | Vicuna- & CLIP-ConvNeXt-L* & MLP | 1536 | 2024/05 |
| [Ovis](https://arxiv.org/abs/2405.20797) | [🔗](https://github.com/AIDC-AI/Ovis) | A | 1 | LLaMA3 / Qwen1.5 & CLIP + Visual Emb. & - | 336 | 2024/05 |
| [DeCo](https://arxiv.org/abs/2405.20985) | [🔗](https://github.com/yaolinli/DeCo) | A | 1 | Vicuna-1.5 & CLIP ViT-L/14 & MLP + Pooling | 336 | 2024/05 |
| [CuMo](https://arxiv.org/abs/2405.05949) | [🔗](https://github.com/SHI-Labs/CuMo) | A | 1 | Mistral / Mixtral & CLIP ViT-L/14 & MLP | 336 | 2024/05 |
| [Cambrian-1](https://arxiv.org/abs/2406.16860) | [🔗](https://github.com/cambrian-mllm/cambrian) | A | 1 | Vicuna/LLaMA/Yi & 4x Vision Encoders & Spatial Aggregator | 1024 | 2024/06 |
| [GLM-4v](https://arxiv.org/abs/2408.16500) | [🔗](https://github.com/THUDM/GLM-4/blob/main/README_en.md) | A | 1 | GLM4 & EVA-CLIP-E & Conv + SwiGLU | 1120 | 2024/06 |
| [InternLM-XComposer-2.5](https://arxiv.org/abs/2407.03320) | [🔗](https://github.com/InternLM/InternLM-XComposer) | A | 1 | InternLM-2 & CLIP ViT-L/14 & MLP | 560*24 | 2024/07 |
| [IDEFICS3](https://arxiv.org/abs/2408.12637) | [🔗](https://huggingface.co/docs/transformers/main/en/model_doc/idefics3) | A | 1 | LLaMA 3.1 & SigLIP-SO400M/14 & Perceiver + MLP | 1820 | 2024/08 |
| [mPLUG-Owl3](https://arxiv.org/abs/2408.04840) | [🔗](https://github.com/X-PLUG/mPLUG-Owl) | A | 1 | Qwen2 & SigLIP-SO400M/14 & Linear | 384*6 | 2024/08 |
| [CogVLM2](https://arxiv.org/abs/2408.16500) | [🔗](https://github.com/THUDM/CogVLM2) | A | 1 | LLaMA3 & EVA-CLIP-E & Conv + SwiGLU | 1344 | 2024/08 |
| [CogVLM2-video](https://arxiv.org/abs/2408.16500) | [🔗](https://github.com/THUDM/CogVLM2) | A | 1 | LLaMA3 & EVA-CLIP-E & Conv + SwiGLU | 224 | 2024/08 |
| [LLaVA-OneVision](https://arxiv.org/abs/2408.03326) | [🔗](https://github.com/LLaVA-VL/LLaVA-NeXT/blob/main/docs/LLaVA_OneVision.md) | A | 1 | Qwen-2 & SigLIP-SO400M/14 & MLP | 384*36 | 2024/09 |
| [Qwen2-VL](https://arxiv.org/abs/2409.12191) | [🔗](https://github.com/QwenLM/Qwen2-VL) | A | 1 | Qwen-2 & ViT-675M & MLP | ∞ | 2024/09 |
| [Aria](https://arxiv.org/abs/2410.05993) | [🔗](https://github.com/rhymes-ai/Aria) | A | 1 | Aria-MoE & SigLIP-SO400M & Cross-Attention + Linear | 980*n | 2024/10 |
| [DeepSeek-VL2](https://arxiv.org/abs/2412.10302) | [🔗](https://github.com/deepseek-ai/DeepSeek-VL2) | A | 1 | DeepSeekMoE & SigLIP & MLP | 384*9 | 2024/12 |
| [InternVL-2.5](https://arxiv.org/abs/2412.05271) | [🔗](https://github.com/OpenGVLab/InternVL) | A | 1 | InternLM2 / Qwen2... & InternViT & MLP | 448*48 | 2024/12 |
| [SAIL-VL](https://arxiv.org/abs/2501.05952) | [🔗](https://huggingface.co/BytedanceDouyinContent) | A | 1 | Qwen2.5 & InternViT & MLP | 448*10 | 2025/01 |
| [Qwen2.5-VL](https://arxiv.org/abs/2502.13923) | [🔗](https://github.com/QwenLM/Qwen2-VL) | A | 1 | Qwen2.5 & Modified ViT & MLP | ∞ | 2025/02 |
| [Granite-Vision](https://arxiv.org/abs/2502.09927) | [🔗](https://github.com/ibm-granite/granite-vision-models) | A | 1 | Granite & SigLIP & MLP | 384*10 | 2025/02 |
| [Gemma 3](https://arxiv.org/abs/2503.19786) | [🔗](https://huggingface.co/collections/google/gemma-3-release) | A | 1 | Gemma 3 & SigLIP & MLP | 896*n | 2025/03 |
| [InternVL3](https://arxiv.org/abs/2504.10479) | [🔗](https://github.com/OpenGVLab/InternVL) | A | 1 | InternLM3 / Qwen2.5 & InternViT & MLP | 448*48 | 2025/04 |
| [Kimi-VL](https://arxiv.org/abs/2504.07491) | [🔗](https://github.com/MoonshotAI/Kimi-VL) | A | 1 | MoonViT & Moolight & MLP | ∞ | 2025/06 |
| [Mimo-VL](https://arxiv.org/abs/2506.03569) | [🔗](https://github.com/XiaomiMiMo/MiMo-VL) | A | 1 | Qwen2.5-ViT & MiMo-Base & MLP | ∞ | 2025/06 |
| [GLM-4.5V](https://arxiv.org/abs/2507.01006) | [🔗](https://github.com/zai-org/GLM-V) | A | 1 | GLM-4 & AIMv2 & MLP | ∞ | 2025/07 |
| [InternVL3.5](https://arxiv.org/abs/2508.18265) | [🔗](https://github.com/OpenGVLab/InternVL) | A | 1 | Qwen3 / GPT-OSS & InternViT & VRR + MLP | 448*48 | 2025/08 |
| [Ovis2.5](https://arxiv.org/abs/2508.11737) | [🔗](https://github.com/AIDC-AI/Ovis) | B | 1 | Qwen3 & SigLIP2 + Visual Embedding & - | ∞ | 2025/08 |
| [VarCo-Vision-2](https://arxiv.org/abs/2509.10105) | [🔗](https://huggingface.co/collections/NCSOFT/varco-vision-20) | A | 1 | Qwen3 & SigLIP-2 & MLP | 448*9 | 2025/09 |
| [MiniCPM-V4.5](https://arxiv.org/abs/2509.18154) | [🔗](https://github.com/OpenBMB/MiniCPM-V) | A | 1 | Qwen3 & SigLIP & 3D Resampler | 448*9 | 2025/09 |
| [Qwen3-VL](https://arxiv.org/abs/2511.21631) | [🔗](https://github.com/QwenLM/Qwen-VL) | A | 1 | Qwen3 & SigLIP-2 & MLP | ∞ | 2025/11 |


### <a id="mode-image"></a>Output Modality: 📝🖼️

| **Model** | **Code** | **Input** | **Output** | **Architecture (LLM & Encoder & Decoder)** | **Date** |
| :-- | :-- | :--: | :--: | :-- | :-- |
| [GILL](https://arxiv.org/abs/2305.17216) | [🔗](https://github.com/kohjingyu/gill) | A | 2 | OPT & CLIP ViT-L & SD | 2023/05 |
| [Emu](https://arxiv.org/abs/2307.05222) | [🔗](https://github.com/baaivision/Emu) | A | 2 | LLaMA & EVA-02-CLIP-1B & SD | 2023/07 |
| [LaVIT](https://arxiv.org/abs/2309.04669) | [🔗](https://github.com/jy0205/LaVIT) | A | 3 | LLaMA & Eva-CLIP ViT-G/14 + LaVIT Tokenizer & LaVIT De-Tokenizer | 2023/09 |
| [CM3Leon](https://arxiv.org/abs/2309.02591) | [🔗](https://github.com/kyegomez/CM3Leon) | B | 3 | CM3Leon & Make-A-Scene & Make-A-Scene | 2023/09 |
| [DreamLLM](https://arxiv.org/abs/2309.11499) | [🔗](https://github.com/RunpeiDong/DreamLLM) | A | 2 | Vicuna & CLIP ViT-L/14 & SD | 2023/09 |
| [Kosmos-G](https://arxiv.org/abs/2310.02992) | [🔗](https://github.com/xichenpan/Kosmos-G) | A | 2 | MAGNETO & CLIP ViT-L/14 & SD | 2023/10 |
| [SEED-LLaMA](https://arxiv.org/abs/2310.01218) | [🔗](https://github.com/AILab-CVC/SEED) | B | 3 | Vicuna / LLaMA-2 & SEED Tokenizer & SEED D e-Tokenizer | 2023/10 |
| [MiniGPT-5](https://arxiv.org/abs/2310.02239) | [🔗](https://github.com/eric-ai-lab/MiniGPT-5) | A | 2 | Vicuna & Eva-CLIP ViT-G/14 & SD | 2023/10 |
| [Emu-2](https://arxiv.org/abs/2312.13286) | [🔗](https://github.com/baaivision/Emu) | A | 2 | LLaMA & EVA-02-CLIP-E-plus & SDXL | 2023/12 |
| [Chameleon](https://arxiv.org/abs/2405.09818) | [🔗](https://github.com/facebookresearch/chameleon) | B | 3 | Chameleon & Make-A-Scene & Make-A-Scene | 2024/05 |
| [MoMa](https://arxiv.org/abs/2407.21770) | - | B | 3 | Chameleon & Make-A-Scene & Make-A-Scene | 2024/07 |
| [Show-o](https://arxiv.org/abs/2408.12528) | [🔗](https://github.com/showlab/Show-o) | B | 3 | Phi-1.5 & MAGVIT-v2 & MAGVIT-v2 | 2024/08 |
| [Transfusion](https://arxiv.org/abs/2408.11039) | [🔗](https://github.com/lucidrains/transfusion-pytorch) | A | 2 | from scratch & VAE & VAE | 2024/08 |
| [VILA-U](https://arxiv.org/abs/2409.04429) | [🔗](https://github.com/mit-han-lab/vila-u) | B | 3 | LLaMA-2 & SigLIP + RQ-VAE & RQ-VAE | 2024/09 |
| [MoMA](https://arxiv.org/abs/2407.21770) | [🔗](https://github.com/bytedance/MoMA?tab=readme-ov-file) | B | 3 | Chameleon & Make-A-Scene & Make-A-Scene | 2024/09 |
| [Janus](https://arxiv.org/abs/2410.13848) | [🔗](https://github.com/deepseek-ai/Janus) | A | 3 | DeepSeek-LLM & LlamaGen Tokenizer & LlamaGen Tokenizer | 2024/10 |
| [Metaqueries](https://arxiv.org/abs/2504.06256) | [🔗](https://github.com/facebookresearch/metaquery) | A | 2 | Qwen2.5 VL & QwenViT + VAE & SANA DiT | 2025/04 |
| [BLIP3o](https://arxiv.org/abs/2505.09568) | [🔗](https://github.com/JiuhaiChen/BLIP3o) | A | 2 | Qwen2.5 VL & QwenViT + VAE & SDXL | 2025/05 |
| [Bagel](https://arxiv.org/abs/2505.14683) | [🔗](https://github.com/bytedance-seed/BAGEL) | A | 2 | Qwen2.5 & QwenViT + VAE & VAE | 2025/05 |
| [MMaDA](https://arxiv.org/abs/2505.15809) | [🔗](https://github.com/Gen-Verse/MMaDA) | B | 3 | LLaDA & MAGVIT-v2 & MAGVIT-v2 | 2025/05 |
| [Qwen-Image](https://arxiv.org/abs/2508.02324) | [🔗](https://github.com/QwenLM/Qwen-Image) | A | 2 | Qwen2.5 VL & QwenViT + VAE & MMDiT | 2025/08 |
| [BLIP3o-Next](https://arxiv.org/abs/2510.15857) | [🔗](https://github.com/JiuhaiChen/BLIP3o) | A | 2 | Qwen2.5 VL & VQ-Siglip2 + VAE & SANA DiT | 2025/10 |


## Large Audio-Language Models 🤗

| **Model** | **Code** | **Input** | **Output** | **Output Modality** | **Architecture (LLM & Encoder & Decoder)** | **Date** |
| :-- | :-- | :--: | :-: | :-- | :-- | :-- |
| [SpeechGPT](https://arxiv.org/abs/2305.11000) | [🔗](https://github.com/0nutation/SpeechGPT) | B | 3 | 📝🔊 | LLaMA & HuBERT & Unit Vocoder | 2023/05 |
| [Speech-LLaMA](https://arxiv.org/abs/2307.03917) | - | A | 1 | 📝 | LLaMA & CTC compressor & - | 2023/07 |
| [SALMONN](https://arxiv.org/abs/2310.13289) | [🔗](https://github.com/bytedance/SALMONN) | A | 1 | 📝 | Vicuna & Whisper-Large-v2 + BEATs & - | 2023/10 |
| [Qwen-Audio](https://arxiv.org/abs/2311.07919) | [🔗](https://github.com/QwenLM/Qwen-Audio) | A | 1 | 📝 | Qwen & Whisper-Large-v2 & - | 2023/11 |
| [SpeechGPT-Gen](https://arxiv.org/abs/2401.13527) | [🔗](https://github.com/0nutation/SpeechGPT) | B | 3 | 📝🔊 | LLaMA-2 & SpeechTokenizer & SpeechTokenizer | 2024/01 |
| [SLAM-ASR](https://arxiv.org/abs/2402.08846) | [🔗](https://github.com/X-LANCE/SLAM-LLM/blob/main/examples/asr_librispeech/README.md) | A | 1 | 📝 | LLaMA-2 & HuBERT & - | 2024/02 |
| [WavLLM](https://arxiv.org/abs/2404.00656) | [🔗](https://github.com/microsoft/SpeechT5/tree/main/WavLLM) | A | 1 | 📝 | LLaMA-2 & Whisper-Large-v2 + WavLM-Base & - | 2024/04 |
| [SpeechVerse](https://arxiv.org/abs/2405.08295) | - | A | 1 | 📝 | Flan-T5-XL & WavLM-Large / Best-RQ & - | 2024/05 |
| [Qwen2-Audio](https://arxiv.org/abs/2407.10759) | [🔗](https://github.com/QwenLM/Qwen2-Audio) | A | 1 | 📝 | Qwen & Whisper-Large-v3 & - | 2024/07 |
| [LLaMA-Omni](https://arxiv.org/abs/2409.06666) | [🔗](https://github.com/ictnlp/LLaMA-Omni) | A | 2 | 📝🔊 | LLaMA-3.1 & Whisper-Large-v3 & Unit Vocoder | 2024/09 |
| [SpeechGPT-Gen](https://arxiv.org/abs/2401.13527) | [🔗](https://github.com/0nutation/SpeechGPT) | B | 3 | 📝🔊 | LLaMA-2 & SpeechTokenizer & SpeechTokenizer | 2024/09 |
| [Moshi](https://arxiv.org/abs/2410.00037) | [🔗](https://github.com/kyutai-labs/moshi) | B | 3 | 📝🔊 | Helium & Mimi & Mimi | 2024/10 |
| [GLM-4-Voice](https://arxiv.org/abs/2412.02612) | [🔗](https://github.com/THUDM/GLM-4-Voice) | B | 3 | 📝🔊 | GLM-4 & Whisper-Large-v3 + VQ & CosyVoice | 2024/12 |
| [Slam-Omni](https://arxiv.org/abs/2412.15649) | [🔗](https://github.com/X-LANCE/SLAM-LLM/tree/main/examples/s2s) | A | 3 | 📝🔊 | Qwen2 & Whisper-small & CosyVoice | 2024/12 |
| [Step-Audio](https://arxiv.org/abs/2502.11946) | [🔗](https://github.com/stepfun-ai/Step-Audio) | B | 3 | 📝🔊 | Step-1 & Paraformer + CosyVoice & Step-A-TTS-3B | 2025/02 |
| [Baichuan-Audio](https://arxiv.org/abs/2502.17239) | [🔗](https://github.com/baichuan-inc/Baichuan-Audio) | B | 3 | 📝🔊 | Qwen2.5 (Specialized) & Baichuan-A Tokenizer & Baichuan-A Decoder | 2025/02 |
| [Kimi-Audio](https://arxiv.org/abs/2504.18425) | [🔗](https://github.com/MoonshotAI/Kimi-Audio) | B | 3 | 📝🔊 | Qwen2.5 & GLM-4-Voice Tokenizer + Whisper-v3 & MoonCast | 2025/04 |
| [LLaMA-Omni2](https://arxiv.org/abs/2505.02625) | [🔗](https://github.com/ictnlp/LLaMA-Omni2) | A | 2 | 📝🔊 | Qwen2.5 & Whisper-Large-v3 & CosyVoice 2 | 2025/05 |
| [Audio Flamingo 3](https://arxiv.org/abs/2507.08128) | [🔗](https://github.com/NVIDIA/audio-flamingo/tree/audio_flamingo_3) | A | 1 | 📝🔊 | Qwen2.5 & AF-Whisper & Streaming TTS | 2025/07 |
| [MiMo-Audio](https://arxiv.org/abs/2512.23808) | [🔗](https://github.com/XiaomiMiMo/MiMo-Audio) | B | 3 | 📝🔊 | MiMo-7B-Base & MiMo-A-Tokenizer & MiMo-A-Tokenizer | 2025/09 |


## Any-to-Any Modality Models 🤗

| **Model** | **Code** | **Input** | **Input Modality** | **Output** | **Output Modality** | **Architecture (LLM & Encoder & Decoder)** | **Date** |
| :--- | :---: | :---: | :--- | :---: | :--- | :--- | :--- |
| [PandaGPT](https://arxiv.org/abs/2305.16355) | [🔗](https://github.com/yxuansu/PandaGPT) | A | 📝🖼️🔊... | 1 | 📝 | Vicuna & ImageBind & - | 2023/05 |
| [ImageBind-LLM](https://arxiv.org/abs/2309.03905) | [🔗](https://github.com/OpenGVLab/LLaMA-Adapter/tree/main/imagebind_LLM) | A | 📝🖼️🔊🧊 | 1 | 📝 | Chinese-LLaMA & ImageBind + PointBind & - | 2023/09 |
| [NExT-GPT](https://arxiv.org/abs/2309.05519) | [🔗](https://github.com/NExT-GPT/NExT-GPT) | A | 📝🖼️🔊 | 2 | 📝🖼️🔊 | Vicuna & ImageBind & SD + AudioLDM + Zeriscope | 2023/09 |
| [CoDi-2](https://arxiv.org/abs/2311.18775) | [🔗](https://github.com/microsoft/i-Code/tree/main/CoDi-2) | A | 📝🖼️🔊 | 2 | 📝🖼️🔊 | LLaMA-2 & ImageBind & SD + AudioLDM2 + zeroscope v2 | 2023/11 |
| [Unified-IO 2](https://arxiv.org/abs/2312.17172) | [🔗](https://github.com/allenai/unified-io-2) | A | 📝🖼️🔊 | 3 | 📝🖼️🔊 | UnifiedIO2 & OpenCLIP ViT-B + AST & VQ-GAN + ViT-VQGAN | 2023/12 |
| [AnyGPT](https://arxiv.org/abs/2402.12226) | [🔗](https://github.com/OpenMOSS/AnyGPT) | B | 📝🖼️🔊 | 3 | 📝🖼️🔊 | LLaMA-2 & SEED + Encodec + SpeechTokenizer & SEED + Encodec + SpeechTokenizer | 2024/02 |
| [Uni-MoE](https://arxiv.org/abs/2405.11273) | [🔗](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs) | A | 📝🖼️🔊 | 1 | 📝 | LLaMA & CLIP ViT-L/14 + Whisper-small + BEATs & - | 2024/05 |
| [Mini-Omni2](https://arxiv.org/abs/2410.11190) | [🔗](https://github.com/gpt-omni/mini-omni2) | A | 📝🖼️🔊 | 3 | 📝🔊 | Qwen2 & CLIP ViT-B/32 + Whisper-small & SNAC | 2024/10 |
| [Baichuan-Omni-1.5](https://github.com/baichuan-inc/Baichuan-Omni-1.5) | [🔗](https://github.com/baichuan-inc/Baichuan-Omni-1.5) | A | 📝🖼️🔊 | 1 | 📝🔊 | Baichuan LLM & QwenViT + Baichuan-A-Tokenizer & Baichuan-A-Tokenizer | 2025/01 |
| [MiniCPM-o 2.6](https://openbmb.notion.site/MiniCPM-o-2-6-A-GPT-4o-Level-MLLM-for-Vision-Speech-and-Multimodal-Live-Streaming-on-Your-Phone-185ede1b7a558042b5d5e45e6b237da9) | [🔗](https://github.com/IsiSinclair/MiniCPM-o-2.6) | A | 📝🖼️🔊 | 2 | 📝🔊 | Qwen2.5 & SigLip-400M + Whisper-medium-300M & ChatTTS-200M | 2025/01 |
| [Ola](https://arxiv.org/abs/2502.04328) | [🔗](https://github.com/Ola-Omni/Ola) | A | 📝🖼️🔊 | 1 | 📝🔊 | Qwen2.5 & OryxViT + Whisper-V3 + BEATs & CosyVoice | 2025/02 |
| [Phi-4-Multimodal](https://arxiv.org/abs/2503.01743) | [🔗](https://github.com/microsoft/PhiCookBook) | A | 📝🖼️🔊 | 1 | 📝 | Phi-4 & SigLIP + ConFormer & - | 2025/03 |
| [Qwen2.5-Omni](https://arxiv.org/abs/2503.20215) | [🔗](https://github.com/QwenLM/Qwen2.5-Omni) | A | 📝🖼️🔊 | 2 | 📝🔊 | Qwen2.5 & QwenViT + Whisper-large-v3 & Codec Decoder | 2025/05 |
| [ShapeLLM-Omni](https://arxiv.org/abs/2506.01853) | [🔗](https://github.com/JAMESYJL/ShapeLLM-Omni) | A | 📝🖼️🧊 | 3 | 📝🧊 | Qwen2.5 VL & QwenViT + 3D VQVAE & 3D VQVAE | 2025/06 |
| [UniUGG](https://arxiv.org/abs/2508.11952) | [🔗](https://github.com/fudan-zvg/UniUGG) | A | 📝🖼️ | 2 | 📝🧊 | Qwen2.5 & RADIOv2.5-L + Spatial-VAE & Unet + Spatial-VAE | 2025/08 |
| [Qwen3-Omni](https://arxiv.org/abs/2509.17765) | [🔗](https://github.com/QwenLM/Qwen3-Omni) | A | 📝🖼️🔊 | 2 | 📝🔊 | Qwen3 & QwenViT + AuT & Codec Decoder | 2025/09 |
| [Next-Omni](https://arxiv.org/abs/2510.13721) | - | B | 📝🖼️🔊 | 3 | 📝🖼️🔊 | Qwen2.5 & VQ-Whisper-Turbo + VQ-CLIP & VQ-Whisper-Turbo + VQ-CLIP | 2025/10 |
| [Omni-View](https://arxiv.org/abs/2511.07222) | [🔗](https://github.com/AIDC-AI/Omni-View) | A | 📝🖼️ | 2 | 📝🖼️🧊 | Qwen2.5 VL & SigLIP + VAE & VAE + VGGT | 2025/11 |
| [UniMoE-2.0-Omni](https://arxiv.org/abs/2511.12609) | [🔗](https://github.com/HITsz-TMG/Uni-MoE) | A | 📝🖼️🔊 | 3 | 📝🖼️🔊 | Qwen2.5-7B & SigLIP + Whisper-Large-v3 + VAE & Codec Decoder + DiT | 2025/11 |


## Contributing

If you find missing/incorrect entries, feel free to open a PR:
- Add the paper link
- Add code link (if available)
- Keep the table style consistent

---

## License

This repository is intended for research and educational purposes. Please follow the license of each referenced project/paper.
