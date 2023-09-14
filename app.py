import gradio as gr
import pil
from PIL import Image
from pathlib import Path
import os

from PIL import Image
from torchvision.io.image import read_image
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
import torch

weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
#path = Path("object_detection_model.pth")
#model = torch.load(path)

def predict(inp1):
  model.eval()

  # Step 2: Initialize the inference transforms
  preprocess = weights.transforms()

  # Step 3: Apply inference preprocessing transforms
  batch = [preprocess(inp1)]

  # Step 4: Use the model and visualize the prediction
  prediction = model(batch)[0]
  labels = [weights.meta["categories"][i] for i in prediction["labels"]]
  box = draw_bounding_boxes(img, boxes=prediction["boxes"],
                            labels=labels,
                            colors="red",
                            width=4, font_size=30)
  im = to_pil_image(box.detach())
  return labels[0],im


# Get example filepaths in a list of lists
examples_path = Path("examples/")
example_list = [["examples/" + example] for example in os.listdir(examples_path)]

title = "Object Detection Model😀😀"
description = "A Object Detection model based on pretrained resnet neural network"
article = "Created by [Rounak Gera](https://github.com/rounak890)"

demo = gr.Interface(fn=predict, inputs=[gr.components.Image(type="pil", label="Image 1")],
                    outputs= [gr.components.Label(num_top_classes = 4, label = "Predictions"),
                              gr.components.Image(type = "pil",label = "predicted")],
                    title = title,
                    description = description,
                    article = article,
                    examples = example_list)
demo.launch()