# SOLO (Small Only Look Once)
A Small CNN Model (15M parameters) similar to the **YOLO v1** model structure.

## How does it work
During the Preprocess, each image from the training dataset gets a custom grid-like label as a numpy array. This makes the model and the loss functions able to correctly subdivide the image in the requested grid, and for each grid the model will predict 5 things:
- norm x
- norm y
- width
- height
- confidence

And along with this, the model will also predict 20 classes in One Hot Encoding style.

The training process is straightforward, I use a cosine decay schedule for the LR warmup, then the model runs through a minimum of 100 epochs, writing in a log file all the losses.

During Training the model uses 3.4Gb of VRAM on a RTX 3060 12Gb GPU, thanks to a few optimization techniques used in the model's architecture, like gradient checkpointing.
