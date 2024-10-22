<div align="center">

<h1>Concept Conductor</h1>
<h3>Concept Conductor: Orchestrating Multiple Personalized Concepts in Text-to-Image Synthesis</h3>

Zebin Yao, &nbsp; Fangxiang Feng, &nbsp; Ruifan Li, &nbsp; Xiaojie Wang

Beijing University of Posts and Telecommunications

[![Project Website](https://img.shields.io/badge/Project-Website-orange)](https://nihukat.github.io/Concept-Conductor/)
[![arXiv](https://img.shields.io/badge/arXiv-<2408.03632>-<COLOR>.svg)](https://arxiv.org/abs/2408.03632)

</div>

<img src="assets/teaser.png" width=100%>

## 🔍 Results

### Combination of 2 Concepts:

<img src="assets/results_of_2_concepts.png" width=100%>

### Combination of More Than 2 Concepts:

<img src="assets/more_than_2_concepts.png" width=100%>


## 🛠️ Installation

```bash
git clone https://github.com/Nihukat/Concept-Conductor.git
cd Concept-Conductor
pip install -r requirements.txt
```

## 📝 Preparation

### 1. Download Pretrained Text-to-Image Models.

We implemented our method on both Stable Diffusion 1.5 and SDXL 1.0 respectively. 

For Stable Diffusion 1.5, we adopt [ChilloutMix](https://civitai.com/models/6424/chilloutmix) for real-world concepts and [Anything-v4](https://huggingface.co/xyn-ai/anything-v4.0) for anime concepts.

```bash
cd experiments/pretrained_models

# Diffusers-version ChilloutMix
git-lfs clone https://huggingface.co/windwhinny/chilloutmix.git

# Diffusers-version Anything-v4
git-lfs clone https://huggingface.co/xyn-ai/anything-v4.0.git
```

For SDXL 1.0, we adopt [RealVisXL V5.0](https://civitai.com/models/139562?modelVersionId=789646) for real-world concepts and [Anything-XL](https://civitai.com/models/9409/or-anything-xl) for anime concepts.

```bash
cd experiments/pretrained_models

# Diffusers-version RealVisXL V5.0
git-lfs clone https://huggingface.co/SG161222/RealVisXL_V5.0.git

# Diffusers-version Anything-XL
git-lfs clone https://huggingface.co/eienmojiki/Anything-XL.git
```

### 2. (Optional) Train ED-LoRAs.

We adopt ED-LoRAs (proposed in [Mix-of-Show](https://github.com/TencentARC/Mix-of-Show)) as single-concept customization models.
If you want to train ED-LoRAs yourself, you can download the training data we used in our paper on [Google Drive](https://drive.google.com/drive/folders/1roYyOL7e5Ivx3lvLAXz8XKY00sDLC377?usp=drive_link).

You can also construct personalized concept datasets with your own custom images and corresponding text captions, referring to the structure of our dataset directory.

We provide training scripts for both Stable Diffusion 1.5 and SDXL 1.0. 

**For Stable Diffusion 1.5 :**

```bash
# Train ED-LoRAs for real-world concepts
python train_edlora.py -opt configs/edlora/train/chow_dog.yml

# Train ED-LoRAs for anime concepts
python train_edlora.py -opt configs/edlora/train/mitsuha_girl.yml
```

**For SDXL 1.0 :**

```bash
# Train ED-LoRAs for real-world concepts
python train_edlora_sdxl.py -opt configs/edlora/train_sdxl/chow_dog.yml

# Train ED-LoRAs for anime concepts
python train_edlora_sdxl.py -opt configs/edlora/train_sdxl/mitsuha_girl.yml
```

### 3. (Optional) Download our trained ED-LoRAs. 

To quickly reimplement our method, you can download our trained ED-LoRAs from [Google Drive](https://drive.google.com/drive/folders/1roYyOL7e5Ivx3lvLAXz8XKY00sDLC377?usp=drive_link).

## 🚀 Usage

### Generate multiple personalized concepts in an image

**For Stable Diffusion 1.5 :**

```bash
python sample.py \
--ref_prompt "A dog and a cat in the street." \
--base_prompt "A dog and a cat on the beach." \
--custom_prompts "A <chow_dog_1> <chow_dog_2> on the beach." "A <siberian_cat_1> <siberian_cat_2> on the beach."\
--ref_image_path "examples/a dog and a cat in the street.png" \
--ref_mask_paths "examples/a dog and a cat in the street_mask1.png" "examples/a dog and a cat in the street_mask2.png" \
--edlora_paths "experiments/ED-LoRAs/real/chow_dog.pth" "experiments/ED-LoRAs/real/siberian_cat.pth" \
--start_seed 0 \
--batch_size 4 \
--n_batches 1

```

<img src="assets/A dog and a cat on the beach._seed0-3.png" width=100%>

You can also pass parameters using a configuration file (like ./configs/sample_config.yaml) :

```bash
python sample.py --config_file "path/to/your/config.yaml"
```

**For SDXL 1.0 :**

```bash
python sample_sdxl.py \
--ref_prompt "A cat on a stool and a dog on the floor." \
--base_prompt "A cat on a stool and a dog on the floor." \
--custom_prompts "A <siberian_cat_1> <siberian_cat_2> on a stool and a <siberian_cat_1> <siberian_cat_2> on the floor." "A <chow_dog_1> <chow_dog_2> on a stool and a <chow_dog_1> <chow_dog_2> on the floor."\
--ref_image_path "examples/a cat on a stool and a dog on the floor.png" \
--ref_mask_paths "examples/a cat on a stool and a dog on the floor_mask1.png" "examples/a cat on a stool and a dog on the floor_mask2.png" \
--edlora_paths "experiments/SDXL_ED-LoRAs/real/siberian_cat.pth" "experiments/SDXL_ED-LoRAs/real/chow_dog.pth" \
--start_seed 0 \
--batch_size 1 \
--n_batches 4

```

<img src="assets/A cat on a stool and a dog on the floor._seed0-3.png" width=100%>

You can also pass parameters using a configuration file (like ./configs/sample_config_sdxl.yaml) :

```bash
python sample_sdxl.py --config_file "path/to/your/config.yaml"
```

## ✅ To-Do List

- [ ] Create a gradio demo.
- [ ] Add more usage and applications.
- [x] Add support for SDXL.
- [x] Release the training data and trained models.
- [x] Release the source code.


## 📚 Citation

If you find this code useful for your research, please consider citing:

```
@article{yao2024concept,
  title={Concept Conductor: Orchestrating Multiple Personalized Concepts in Text-to-Image Synthesis},
  author={Yao, Zebin and Feng, Fangxiang and Li, Ruifan and Wang, Xiaojie},
  journal={arXiv preprint arXiv:2408.03632},
  year={2024}
}
```
