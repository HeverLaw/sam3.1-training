# SAM 3.1 Tracking Finetuning

This guide provides instructions for finetuning the SAM 3.1 tracking (VOS) component. We have extended the training infrastructure to support flexible resolutions and seamless integration with the SAM 3.1 multiplex architecture.

## Key Modifications for Tracking Finetuning

To support efficient and effective finetuning of SAM 3.1 on tracking tasks (like VOS), several key changes were implemented:

### 1. Training Adapter: `SAM3MultiplexTrain`
We use the `SAM3MultiplexTrain` class (located in `training/model/sam3_multiplex_train.py`) for finetuning, which directly inherits from **`VideoTrackingDynamicMultiplex`**.
- **Why not the Demo class?**: While `Sam3VideoTrackingMultiplexDemo` is used for **Inference** and provides high-level APIs for video handling and caching, it is not suitable for training. `VideoTrackingDynamicMultiplex` provides the core logic for gradient backpropagation and multi-frame state transitions required for effective finetuning.
- **Infrastructure Compatibility:** It overrides the `forward()` method to accept `BatchedVideoDatapoint` from the SAM2-style training pipeline, allowing us to reuse existing data loaders, loss functions, and trainers without modification.
- **State Dict Compatibility:** By using inheritance rather than wrapping, the saved state dict remains 1:1 compatible with the official SAM 3.1 checkpoints (i.e., no `model.` prefix), making deployment straightforward.

### 2. Multi-Resolution Support
The training system now supports finetuning at various resolutions (e.g., 672x672, 1008x1008) instead of being locked to a fixed size.
- **Resolution-Aware Components:** During initialization, `SAM3MultiplexTrain` dynamically calculates feature sizes and interpolation targets based on the provided `image_size`.
- **Configurable Training:** Users can specify the target resolution in the Hydra configuration (e.g., `scratch.resolution: 672`).
- **Adaptive Transforms:** The data pipeline in `training/dataset/transforms.py` handles resizing and padding consistently across the video sequence to match the chosen resolution.

### 3. Flexible Freezing Strategy
To preserve the open-vocabulary detection capabilities while adapting the tracker:
- **`freeze_patterns`:** Supports regex-based freezing of specific model components. Typically, we freeze the heavy image encoder and parts of the detector to focus the training on the memory attention and mask decoder layers.

### 4. Automatic Checkpoint Merging
After training finishes, the `training/train.py` script automatically merges the finetuned tracker weights back into the official SAM 3.1 release checkpoint.
- **Result**: This produces a single `sam3.1_merged.pt` file that combines your **custom-tuned tracking precision** with the original **Open-Vocabulary detection capabilities** (Detector) of SAM 3.1.
- **Seamless Deployment**: The merged checkpoint is 1:1 compatible with the existing SAM 3.1 inference pipeline, including the Demo classes.

### 5. Dynamic SAM 3.1 & Multiplex Variants
We support the **Dynamic SAM 3.1** architecture, which allows for a flexible number of objects to be tracked and updated throughout a video sequence.

#### Understanding the Multiplex Classes:
- **`VideoTrackingMultiplex`**: The base class implementing the core multiplex (bucketized) tracking logic.
- **`VideoTrackingDynamicMultiplex`**: Extends the base class to support "dynamic" object counts (objects entering/leaving the scene). This is the core class used for training.
- **`Sam3VideoTrackingMultiplexDemo`**: A high-level wrapper used for **Inference** and interactive use-cases. It handles video loading and feature caching but is not used during training. Note that interactive correction (Interactions) is currently in development for the multiplex mode.

## Prerequisites

Before running the finetuning script, ensure you have:
1. **Installed SAM 3.1**: Follow the main project installation guide to set up the base environment.
2. **Additional Training Dependencies**: The training pipeline requires `tensordict`. Install it via:
   ```bash
   pip install tensordict
   ```

## How to Run

### 1. Preparation
Ensure your dataset is in the expected VOS format (images and masks). 

- **Data Acquisition**: You can obtain the **EndoVis17** dataset from [Surgical-SAM-2](https://github.com/jinlab-imvr/Surgical-SAM-2).
- **Supported Formats**: Currently, the training pipeline is tested and confirmed to support datasets using **PNG masks**, such as **MOSE** and **DAVIS**. These datasets follow the same structure as **EndoVis17** and can be run directly.
- **Untested Datasets**: Support for **SA-V** is currently **untested**. We recommend using PNG-mask-based datasets for the most stable experience.
- **Preprocessing**: You can use the scripts in `data_preprocess/` to convert your dataset if necessary.

### 2. Configuration
Select or modify a config file in `training/configs/sam3_multiplex/`. For example, `sam3_multiplex_ev17_finetune_672.yaml` is pre-configured for EndoVis17 at 672 resolution.

### 3. Start Training
Run the training script using torchrun for distributed training:

```bash
python training/train.py --config-name sam3_multiplex/sam3_multiplex_ev17_finetune_672
```

## Acknowledgments
We would like to thank the authors of [SAM 3](https://github.com/facebookresearch/sam3) for their groundbreaking work and for providing the base architecture upon which this training pipeline is built.
