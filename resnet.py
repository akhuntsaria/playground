import torch
from torchvision import transforms
from PIL import Image
import urllib

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
model.eval()

while True:
    filename = 'tmp/img.jpg'
    url = str(input('\nImage URL: ') or 'https://cdn.britannica.com/60/8160-050-08CCEABC/German-shepherd.jpg')
    urllib.request.urlretrieve(url, filename)

    print(filename)

    input_image = Image.open(filename)

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    print(input_tensor.size())

    # transform_to_pil = transforms.ToPILImage()
    # pil_image = transform_to_pil(input_tensor)
    # pil_image.save("tmp/dog_preprocessed.jpg")

    input_batch = input_tensor.unsqueeze(0)
    print(input_batch.size())

    if torch.backends.mps.is_available():
        input_batch = input_batch.to('mps')
        model.to('mps')
    else:
        print ('MPS device not found.')

    with torch.no_grad():
        output = model(input_batch)

    print('Confidence scores', output[0].size(), output[0][0])

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    print('Probabilities', probabilities.size(), probabilities[0])

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    print('Top 5', top5_prob, top5_catid)

    urllib.request.urlretrieve('https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt', 
                            'tmp/imagenet_classes.txt')

    with open('tmp/imagenet_classes.txt', 'r') as f:
        categories = [s.strip() for s in f.readlines()]
    print('Categories', len(categories), categories[:3])

    print('\nTop 5 categories')
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())