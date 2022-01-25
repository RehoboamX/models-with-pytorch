import os
import torch
from torchvision import transforms
from PIL import Image
from alex import Alexnet

if __name__ == '__main__':
    text_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    normlize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    img_transforms = transforms.Compose([transforms.Resize((227, 227)),
                                         transforms.ToTensor(),
                                         normlize])

    root_path = os.getcwd()
    model_name = 'alexnet.pth'
    model_path = os.path.join(root_path, 'model', model_name)
    path_img = os.path.join(root_path, "data", "ship.jpg")    # 改变这里文件名来查看不同图片的预测结果
    img = Image.open(path_img).convert('RGB')
    img_tensor = img_transforms(img)
    img_tensor.unsqueeze_(0)  # 添加batchsize维度

    alexnet = Alexnet(num_classes=10)
    log_dir = model_path
    # checkpoint = torch.load(log_dir)
    checkpoint = torch.load(log_dir, map_location=torch.device('cpu'))
    alexnet.load_state_dict(checkpoint['model'])
    preds = text_labels[alexnet(img_tensor).argmax(axis=1)]
    print(f'the prediction is: {preds}')
