# EECS6694 Project Proposal

## DynaPrompt – Dynamic Prompt Guidance for Text-to-Image Diffusion Models

**Team Members:**
- Charles Chaoyu Hou <ch3889>
- Max Zishock Kim <zk2295>
- Swapnil Banerjee <sb5041>

---

## 1. Motivation

Text-to-image diffusion models such as Stable Diffusion, Imagen, and DALL·E 2 can generate highly realistic images from text prompts but remain sensitive to how prompts are phrased. They lack a mechanism to verify ongoing semantic alignment during the generation process. For instance, given the prompt "A golden retriever playing with a red ball in a snowy park," the model might omit the ball or misrepresent colors. This happens because the prompt conditioning remains fixed once sampling begins, causing the image to gradually diverge from the input prompt. To address this, we propose a feedback-driven guidance system that dynamically updates the prompt embedding during generation, ensuring continuous semantic fidelity.

## 2. Objective

We propose **Dynaprompt**, a Dynamic Prompt Guidance method that incorporates an external semantic feedback loop to maintain alignment between the generated image and its textual description throughout every stage of sampling, a limitation in current text-to-image diffusion models.

Specifically, we propose the following operating procedure for Dynaprompt:

1. At intermediate denoising steps, the partially generated image is evaluated.
2. A pretrained vision-language model (e.g., CLIP) computes the semantic similarity between the intermediate image and the text prompt.
3. Underrepresented or missing concepts (e.g., "golden retriever," "snow park") within the prompt are detected via this external assessment.
4. The prompt conditioning vector is adaptively modified or re-weighted to emphasize these missing concepts.
5. Sampling continues with the updated prompt embedding, enabling iterative realignment.

This adaptive, feedback-controlled guidance loop enables the model to self-correct and maintain prompt adherence in real-time. Think of it as a human artist revisiting the original description during the creation process.

## 3. Related Work

We have identified related works and recent advances in text-to-image diffusion that focus on greater semantic controllability by adjusting internal model parameters.

- **Prompt-to-Prompt (P2P)** and **Cross-Attention Control (CAC)** manipulate attention maps for local semantic edits. [1]
- **Attend-and-Excite** dynamically re-weights attention for neglected prompt tokens to ensure comprehensive object inclusion. [2]
- **Dynamic Classifier-Free Guidance (CFG)** varies guidance strength across timesteps to enhance the fidelity of the prompt. [3]
- **Composable Diffusion** and **GLIGEN** introduce spatial or compositional conditioning but still rely on static, internally embedded prompts. [4][5]

Despite these improvements, existing methods remain limited. They only adjust intra-network parameters and assume the initial text embedding will suffice for full semantic alignment. Generally, they lack an external, model-agnostic semantic feedback mechanism to monitor and correct alignment between the evolving image and the original prompt.

Dynaprompt establishes its novelty by incorporating the following:

- **Feedback-Driven**: It transforms diffusion sampling into a closed-loop system using real-time, external evaluation for semantic correction.
- **Model-Agnostic**: The approach functions entirely outside the network, enabling easy integration with any pre-trained diffusion model without retraining or architectural modification.
- **Semantic Generalization**: By leveraging external feedback, Dynaprompt dynamically aligns both global and fine-grained prompt semantics, overcoming the scope and limitations of prior attention-based or token-centric techniques.

## 4. Plan to do

We plan to build a lightweight controller around Stable Diffusion v1.5 using the CompVis implementation. The system will decode partial images every few denoising steps, measure their semantic similarity to the input text using CLIP, and determine which words are underrepresented. The controller will then re-weight the corresponding token embeddings, normalize the new prompt representation, and continue the denoising process with the updated conditioning. This adaptive loop will be repeated several times until the final image is generated.

All components will be implemented in PyTorch and run on a single GPU (e.g., Colab Pro T4). The project requires no model retraining and adds minimal overhead as only CLIP forward passes for intermediate images. The dataset will consist of roughly 200 multi-object COCO-style prompts designed to test compositional understanding and attribute binding. We will compare Dynaprompt with several strong baselines: a static prompt baseline (no feedback), a dynamic CFG schedule (adapting guidance strength), and a simple between-run prompt rewrite approach (re-sampling when CLIP similarity is low). Together, these comparisons will isolate whether content-level prompt adaptation yields measurable improvements beyond existing guidance schemes.

## 5. Experiments need to run

We plan to first plot alignment curves showing how CLIP similarity evolves through denoising steps for static, dynamic-CFG, and Dynaprompt methods. This visualizes whether the adaptive feedback genuinely prevents semantic drift. Next, we will test compositional accuracy using BLIP-2 captions to extract mentioned objects and attributes from generated images, measuring recall against the prompt tokens. To assess robustness, we will run ablation studies varying feedback frequency (every 5, 10, or 20 steps), step size of the updates, and type of feedback model (CLIP ViT-B/32 vs L/14). We will also analyze efficiency vs quality, reporting CLIP, FID, and runtime per image to demonstrate that Dynaprompt improves alignment with minimal cost. Finally, a small human evaluation will ask participants to choose which of two images better matches the text, confirming perceptual gains.

## 6. Outcomes we need to achieve

To have a complete inference of our experiments, we plan to calculate outcomes in a two-phase approach:

### Quantitative metrics-based outcome:

We will conduct analysis using the following quantitative methods to ensure maximum confidence.

- **Latency evaluation**: We will take into account the Resolution, steps, and blocks, and then get the seconds per image that we use to compare inference speed across different model configurations. [6]
- **FID score**: Calculating the FID score is another commonly used method, which will help us quantify how close the distribution of the generated images is to that of real images. We can randomly select a set of generated and reference images, extract their inception-v3 feature embeddings, and compute the mean and covariance statistics to obtain their final FID value. A low FID score indicates a better quality of the image. [7]

### Qualitative metrics-based outcome:

- **CLIPScore**: We also plan to use CLIPScore to evaluate text-image semantic alignment. This score, based on CLIP embeddings by measuring how accurately the generated image reflects the input prompt. A high CLIPScore indicates better alignment between text and image. [7]
- **ImageReward**: We will also be using ImageReward as a learned reward model to reflect how humans judge text-to-image pairs. We plan to collect data based on human comparisons between images. Then we train a model to predict a scalar reward that approximates the human preference. [6]

By combining both the quantitative and qualitative metrics, we aim to achieve a well-rounded evaluation of the model's speed, accuracy, and text-to-image consistency. This will give us a balanced understanding of the overall performance of the model.

## 7. References

[1] https://arxiv.org/abs/2208.01626

[2] https://arxiv.org/abs/2301.13826

[3] https://arxiv.org/abs/2509.16131

[4] https://codi-gen.github.io/

[5] https://gligen.github.io/

[6] https://arxiv.org/pdf/2501.13107

[7] https://huggingface.co/docs/diffusers/en/conceptual/evaluation

---

## Base Model

We are using the CompVis Stable Diffusion implementation as our base model:
https://github.com/CompVis/stable-diffusion/tree/main

## Datasets

**Primary Dataset**: COCO 2017 Validation Set
- Clean, structured captions
- Multi-object scenes with attributes
- 200 curated prompts for compositional understanding

**Reference**:
- https://cocodataset.org/
- ~~LAION-5B~~ (not using - too large, designed for training)
