# Flower image classifier

An image classifier for flower species, built with PyTorch transfer learning as a course project from a starter template. A pretrained network (VGG16 by default) gets a new classifier head that is trained on the flower images. `cat_to_name.json` maps class labels to flower names.

## Files

- `Image Classifier Project.ipynb`: development notebook (also as `.html`)
- `train.py`: train a model from the command line and save a checkpoint (`--arch` picks the network)
- `predict.py`: load a checkpoint and predict the top classes for an image
- `model_req.py`, `data_req.py`: model building and data loading helpers
- `modeltest.ipynb`: checks of a trained model
