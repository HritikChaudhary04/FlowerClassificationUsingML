import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from data_req import process_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_network(architecture, hidden_units):
    print("Building network ... architectur: {}, hidden_units: {}".format(architecture, hidden_units))
    
    if architecture =='vgg16':
        model = models.vgg16(pretrained = True)
        input_units = 25088
    elif architecture =='vgg13':
        model = models.vgg13(pretrained = True)
        input_units = 25088
    elif architecture =='alexnet':
        model = models.alexnet(pretrained = True)
        input_units = 9216
        
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(
              nn.Linear(input_units, hidden_units),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(hidden_units, 256),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(256, 102),
              nn.LogSoftmax(dim = 1)
            )

    model.classifier = classifier
    
    print("Finished building network.")
    
    return model

def train_network(model, epochs, learning_rate, trainloader, validloader, gpu):
    print("Training network ... epochs: {}, learning_rate: {}, gpu used for training: {}".format(epochs, learning_rate, gpu))
    
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

    model.to(device)
    
    steps = 0
    running_loss = 0
    print_every = 20

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            logps = model(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in validloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"valid loss: {test_loss/len(validloader):.3f}.. "
                    f"valid accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
                
    print('Training completed')
    
    return model,criterion

def evaluate_model(model, testloader, criterion, gpu):
    print("Testing network ... gpu used for testing: {}".format(gpu))
    
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    # Validation on the test set
    test_loss = 0
    test_accuracy = 0
    model.eval() # We just want to evaluate and not train the model
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            # Calculate accuracy of test set
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Test loss: {test_loss/len(testloader):.3f}, "
        f"Test accuracy: {test_accuracy/len(testloader):.3f}")
    running_loss = 0
    print('Testing completed')
    
    
def save_model(model, architecture, hidden_units, epochs, learning_rate, save_dir):
    print("Saving model ... epochs: {}, learning_rate: {}, save_dir: {}".format(epochs, learning_rate, save_dir))
    checkpoint = {
        'architecture': architecture,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    
    checkpoint_path = save_dir + "checkpoint.pth"

    torch.save(checkpoint, checkpoint_path)
    
    print("Model saved to {}".format(checkpoint_path))
    
    
def load_model(filepath):
    print("Loading and building model from {}".format(filepath))

    checkpoint = torch.load(filepath)
    model = build_network(checkpoint['architecture'], checkpoint['hidden_units'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    
    return model

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model. '''
    
    model.eval()
    image = process_image(image_path)
    image = torch.from_numpy(image).type(torch.FloatTensor)
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)

        # Convert indices to classes
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_class = [idx_to_class[each] for each in top_class.cpu().numpy()[0]]
        top_p = top_p.cpu().numpy()[0]

    return top_p, top_class