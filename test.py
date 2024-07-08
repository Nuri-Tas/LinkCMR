import os
import argparse
import random
import sys

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import utils as ut
from dataset import ACDC_Dataset, CVCClinicDB_Dataset
from linknet import LinkNet
from dsc import DiceScoreCoefficient


### save images ###
def Save_image(img, seg, ano, path):
    seg = np.argmax(seg, axis=1)
    img = img[0]
    img = np.transpose(img, (1,2,0))
    seg = seg[0]
    ano = ano[0]
    dst1 = np.zeros((seg.shape[0], seg.shape[1],3))
    dst2 = np.zeros((seg.shape[0], seg.shape[1],3))

    #class1 : background
    #class0 : polyp

    dst1[seg==0] = [0.0, 0.0, 0.0]
    dst1[seg==1] = [0.0, 0.0, 255.0]
    dst1[seg==2] = [0.0, 255.0, 0.0]
    dst1[seg==3] = [255.0, 0.0, 0.0]

    dst2[ano==0] = [0.0, 0.0, 0.0]
    dst2[ano==1] = [0.0, 0.0, 255.0]
    dst2[ano==2] = [0.0, 255.0, 0.0]
    dst2[ano==3] = [255.0, 0.0, 0.0]

    img = Image.fromarray(np.uint8(img*255.0))
    dst1 = Image.fromarray(np.uint8(dst1))
    dst2 = Image.fromarray(np.uint8(dst2))

    img.save("{}/Image/Inputs/{}.png".format(args.out, path), quality=95)
    dst1.save("{}/Image/Seg/{}.png".format(args.out, path), quality=95)
    dst2.save("{}/Image/Ano/{}.png".format(args.out, path), quality=95)

def class_pixel_accuracy(gt_masks, pred_masks, num_classes):
    """
    Calculate class-wise pixel accuracy for segmentation masks.

    Parameters:
        gt_masks (numpy array): Ground truth segmentation masks with shape (N, H, W).
        pred_masks (numpy array): Predicted segmentation masks with shape (N, H, W).
        num_classes (int): Number of classes.

    Returns:
        class_acc (list): List of pixel accuracy for each class.
    """
    class_acc = []
    for class_id in range(num_classes):
        # Extract masks for the current class
        pred_masks_ = np.argmax(pred_masks, axis=1)
        class_gt_masks = (gt_masks == class_id)
        class_pred_masks = (pred_masks_ == class_id)

        # Count the number of pixels for the current class in both ground truth and prediction
        total_pixels = np.sum(class_gt_masks)
        correct_pixels = np.sum(np.logical_and(class_gt_masks, class_pred_masks))

        # Calculate pixel accuracy for the current class
        if total_pixels > 0:
            class_accuracy = correct_pixels / total_pixels
        else:
            class_accuracy = 0.0

        class_acc.append(class_accuracy)

    return class_acc

### test ###
def test(n_classes):
    model_path = "{}/model/model_bestdsc.pth".format(args.out)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    predict = []
    answer = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.cuda(device)
            targets = targets.cuda(device)
            targets = targets.long()
    
            output = model(inputs)
            print(output.shape)
            output = F.softmax(output, dim=1)
            inputs = inputs.cpu().numpy()
            output = output.cpu().numpy()
            targets = targets.cpu().numpy()

            for j in range(args.batchsize):
                predict.append(output[j])
                answer.append(targets[j])

            Save_image(inputs, output, targets, batch_idx+1)

        accuracy = class_pixel_accuracy(np.array(answer), np.array(predict), args.classes)
        print(accuracy)

        dsc = DiceScoreCoefficient(n_classes=args.classes)(predict, answer)


        print("Dice")
        for i in range(n_classes):
            print(f"class {i}  = {dsc[i]}")
            
        print("mDice     = %f" % (np.mean(dsc)))
        
        with open(PATH, mode = 'a') as f:
            f.write("%f\t%f\t%f\n" % ((dsc[i] for i in range(n_classes)), np.mean(dsc)))


###### main ######
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AtvMF Dice loss')
    parser.add_argument('--classes', '-c', type=int, default=3)
    parser.add_argument('--batchsize', '-b', type=int, default=24)
    parser.add_argument('--dataset', '-i', default='./Dataset')
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--gpu', '-g', type=str, default=0)
    parser.add_argument('--seed', '-s', type=int, default=0)
    args = parser.parse_args()
    gpu_flag = args.gpu

    # device #
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    # save #
    if not os.path.exists("{}".format(args.out)):
        os.mkdir("{}".format(args.out))
    if not os.path.exists(os.path.join("{}".format(args.out), "model")):
        os.mkdir(os.path.join("{}".format(args.out), "model"))
    if not os.path.exists(os.path.join("{}".format(args.out), "Image")):
        os.mkdir(os.path.join("{}".format(args.out), "Image"))
    if not os.path.exists(os.path.join("{}".format(args.out), "Image", "Inputs")):
        os.mkdir(os.path.join("{}".format(args.out), "Image", "Inputs"))
    if not os.path.exists(os.path.join("{}".format(args.out), "Image", "Seg")):
        os.mkdir(os.path.join("{}".format(args.out), "Image", "Seg"))
    if not os.path.exists(os.path.join("{}".format(args.out), "Image", "Ano")):
        os.mkdir(os.path.join("{}".format(args.out), "Image", "Ano"))

    PATH = "{}/predict.txt".format(args.out)

    with open(PATH, mode = 'a') as f:
        f.write(f"dataset: {args.dataset}, seed: {args.seed}, results: ")


    # seed #
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # preprocceing #
    test_transform = ut.ExtCompose([ut.ExtResize((224,224)),
                                    ut.ExtToTensor(),
                                   ])
    # data loader #
    if args.dataset == "ACDC":
        assert args.classes == 3
        data_test = ACDC_Dataset(dataset_type='test', transform=test_transform)
    elif args.dataset == "CVC_Clinical":
        assert args.classes == 2
        data_test = CVCClinicDB_Dataset(dataset_type='test', transform=test_transform)
        
    test_loader = torch.utils.data.DataLoader(data_test, batch_size=args.batchsize, shuffle=False, drop_last=True, num_workers=2)

    # model #
    model = LinkNet(num_classes=args.classes).cuda(device)

    # print model param size
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total_params : %d" % pytorch_total_params)

    ### test ###
    test(args.classes)
