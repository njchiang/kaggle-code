import matplotlib.pyplot as plt
import torchvision
import torch
from torch.autograd import Variable

from tqdm import tqdm

use_gpu = torch.cuda.is_available()

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    # plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def show_databatch(inputs, classes, class_names):
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])

def visualize_model(model, ds, mode="val", num_images=6):
    was_training = model.training
    dataloader = ds.get_dataloaders()[mode]
    class_names = ds.get_class_names()
    # Set model for evaluation
    model.train(False)
    model.eval()
    
    images_so_far = 0

    # for i, data in enumerate(dataloaders[sel]):
    for i, data in tqdm(enumerate(dataloader)):
        inputs, labels = data
        size = inputs.size()[0]
        
        with torch.no_grad():
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
        
        outputs = model(inputs)
        
        _, preds = torch.max(outputs.data, 1)
        predicted_labels = [preds[j] for j in range(inputs.size()[0])]
        
        print("Ground truth:")
        show_databatch(inputs.data.cpu(), labels.data.cpu(), class_names)
        print("Prediction:")
        show_databatch(inputs.data.cpu(), predicted_labels, class_names)
        
        del inputs, labels, outputs, preds, predicted_labels
        torch.cuda.empty_cache()
        images_so_far += size
        if images_so_far >= num_images:
            break
        
    model.train(mode=was_training) # Revert model back to original training state