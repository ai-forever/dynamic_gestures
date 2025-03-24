# HaGRID Dynamic Gestures
[HaGRID main repo](https://github.com/hukenovs/hagrid)

## Overview
This repository contains using HaGRID dataset for dynamic gesture recognition. The dataset is available [here](https://github.com/hukenovs/hagrid).

### Project

```
├── ocsort/ # source code for Observation-Centric Sorting
│   ├── kalmanfilter.py # Kalman filter
│   ├── kalmanboxtracker.py # Kalman box tracker
│   ├── association.py # Association of boxes with trackers
├── utils/ # useful utils
│   ├── action_controller.py # Action controller for dynamic gestures
│   ├── box_utils_numpy.py # Box utils for numpy
│   ├── enums.py # Enums for dynamic gestures and actions
│   ├── hand.py # Hand class for dynamic gestures recognition
│   ├── drawer.py # Debug drawer
├── onnx_models.py # ONNX models for gesture recognition
├── main_controller.py # Main controller for dynamic gestures recognition, uses ONNX models, ocsort and utils
├── run_demo.py # Demo script for dynamic gestures recognition
```

## Installation
Clone and install required python packages:
```bash
git clone https://github.com/ai-forever/dynamic_gestures.git
# or mirror link:
cd dynamic_gestures
# Create virtual env by conda or venv
conda create -n dynamic_gestures python=3.9 -y
conda activate dynamic_gestures
# Install requirements
pip install -r requirements.txt
```

## Demo
To run demo, you just need to run `run_demo.py` script.

```bash
python run_demo.py --detector <path_to_detector> --classifier <path_to_classifier> --debug
```
`--detector   (optional)`  Path to the hand detector model.
                         **Default:** `models/hand_detector.onnx`

`--classifier (optional)`  Path to the crops classifier model.
                         **Default:** `models/crops_classifier.onnx`

`--debug      (optional)`  Enables debug mode to see bounding boxes and class labels.



## Dynamic gestures
Next, we will show dynamic gestures in user mode and debug mode. In user mode, we show only the final result of dynamic gesture recognition. In debug mode, we show the result of each step of dynamic gesture recognition:
1. hand detection
2. hand tracking
3. gesture recognition
4. action recognition

At the moment the code supports 6 groups of dynamic gestures:

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <td valign="top" style="padding: 10px; text-align: left; min-height: 600px;">
      <h4 style="text-align: left;">ZOOM</h4>
      <div style="text-align: left;">
        <img src="images/zoom_in_out.gif" height="200" alt="Zoom In/Out">
        <img src="images/zoom.gif" height="200" alt="Zoom">
      </div>
      <h4 style="text-align: left;">DRAG AND DROP</h4>
      <div style="text-align: left;">
        <img src="images/dndv1.gif" height="200" alt="Drag and Drop 1">
        <img src="images/dndv2.gif" height="200" alt="Drag and Drop 2">
        <img src="images/dndv3.gif" height="200" alt="Drag and Drop 3">
      </div>
    </td>
    <td valign="top" style="padding: 10px; text-align: left; min-height: 600px;">
      <h4 style="text-align: left;">FAST SWIPE UP / DOWN</h4>
      <div style="text-align: left;">
        <img src="images/fast_up_down.gif" height="200" alt="Fast Swipe Up/Down">
      </div>
      <h4 style="text-align: left;">CLICK</h4>
      <div style="text-align: left;">
        <img src="images/clicks.gif" height="200" alt="Clicks">
      </div>
      <h4 style="text-align: left;">SWIPES LEFT / RIGHT</h4>
      <div style="text-align: left;">
        <img src="images/swipe_left_right.gif" height="200" alt="Swipe Left/Right">
        <img src="images/swipe2_left_right.gif" height="200" alt="Swipe 2 Left/Right">
        <img src="images/swipe3_left_right.gif" height="200" alt="Swipe 3 Left/Right">
      </div>
    </td>
  </tr>
</table>

<h4 style="text-align: left;">SWIPES UP / DOWN</h4>
<div style="text-align: left; min-height: 200px;">
  <img src="images/swipe_up_down.gif" height="200" alt="Swipe Up/Down">
  <img src="images/swipe2_up_down.gif" height="200" alt="Swipe 2 Up/Down">
  <img src="images/swipe3_up_down.gif" height="200" alt="Swipe 3 Up/Down">
</div>

### License
This work is licensed under a variant of <a rel="license" href="https://www.apache.org/licenses/LICENSE-2.0">Apache License, Version 2.0</a>.

Please see the specific [license](./license/LICENSE-2.0.txt).

### Citation
You can cite the paper using the following BibTeX entry:

    @misc{nuzhdin2024hagridv21mimagesstatic,
          title={HaGRIDv2: 1M Images for Static and Dynamic Hand Gesture Recognition},
          author={Anton Nuzhdin and Alexander Nagaev and Alexander Sautin and Alexander Kapitanov and Karina Kvanchiani},
          year={2024},
          eprint={2412.01508},
          archivePrefix={arXiv},
          primaryClass={cs.CV},
          url={https://arxiv.org/abs/2412.01508},
    }
