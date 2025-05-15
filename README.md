---
title: Autoforge
emoji: 🏢
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 5.29.1
app_file: app.py
pinned: false
license: other
short_description: Generating 3D printed layered models from an input image
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
# AutoForge

AutoForge is a Python tool for generating 3D printed layered models from an input image. Using a learned optimization strategy with a Gumbel softmax formulation, AutoForge assigns materials per layer and produces both a discretized composite image and a 3D-printable STL file. It also generates swap instructions to guide the printer through material changes during a multi-material print. \

**TLDR:** It uses a picture to generate a 3D layer image that you can print with a 3d printer. Similar to [Hueforge](https://shop.thehueforge.com/), but without the manual work (and without the artistic control).

## Example
All examples use only the 13 BambuLab Basic filaments, currently available in Hueforge, the background color is set to black.
The pruning is set to a maximum of 8 color and 20 swaps, so each image uses at most 8 different colors and swaps the filament at most 20 times. 
<div style="display: flex; justify-content: center; gap: 20px;">
  <div style="text-align: center;">
    <h3>Input Image</h3>
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/lofi.jpg" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/nature.jpg" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/cat.jpg" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/chameleon.jpg" width="200" />
  </div>
  <div style="text-align: center;">
    <h3>Autoforge Output</h3>
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/lofi_discretized.png" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/nature_discretized.png" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/cat_discretized.png" width="200" />
    <img src="https://github.com/hvoss-techfak/AutoForge/blob/main/images/chameleon_discretized.png" width="200" />
  </div>
</div>

## Features

- **Image-to-Model Conversion**: Converts an input image into a layered model suitable for 3D printing.
- **Learned Optimization**: Optimizes per-pixel height and per-layer material assignments using PyTorch.
- **Learned Heightmap**: Optimizes the height of the layered model to create more detailed prints.
- **Gumbel Softmax Sampling**: Leverages the Gumbel softmax method to decide material assignments for each layer.
- **STL File Generation**: Exports an ASCII STL file based on the optimized height map.
- **Swap Instructions**: Generates clear swap instructions for changing materials during printing.
- **Live Visualization**: (Optional) Displays live composite images during the optimization process.
- **Hueforge export**: Outputs a project file that can be opened with hueforge.