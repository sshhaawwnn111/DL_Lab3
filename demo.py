from models import *
from torch.autograd import Variable
from dataloader import read_bci_data
import pandas as pd
import torch
import numpy 
import os

def testing(x_test,y_test,device,model):

    model.eval()
    with torch.no_grad():
        model.to(device)
        n = x_test.shape[0]

        x_test = x_test.astype("float32")
        y_test = y_test.astype("float32").reshape(y_test.shape[0],)

        x_test, y_test = Variable(torch.from_numpy(x_test)),Variable(torch.from_numpy(y_test))
        x_test,y_test = x_test.to(device),y_test.to(device)
        y_pred_test = model(x_test)
        correct = (torch.max(y_pred_test,1)[1]==y_test).sum().item()
        # print("testing accuracy:",correct/n)
        return correct/n

if __name__ == "__main__":

    all_results = []
    model_list=[EEGNetELU, EEGNetReLU, EEGNetLeakyReLU, DeepConvNetELU, DeepConvNetReLU, DeepConvNetLeakyReLU]
    model_file_path=["EEGNetELU.bin", "EEGNetReLU.bin", "EEGNetLeakyReLU.bin", "DeepConvNetELU.bin", "DeepConvNetReLU.bin", "DeepConvNetLeakyReLU.bin"]
    
    device = torch.device("cuda:0")
    for i in range(6):
        model = model_list[i]()
        model.load_state_dict(torch.load(model_file_path[i]))
        train_data, train_label, test_data, test_label = read_bci_data()
        testing_accuracy = testing(test_data,test_label,device,model)
        all_results.append(testing_accuracy)
        # print(testing_accuracy)

    print(f"{'EEGNet with ELU' : <27}{'Max test Acc' : <14}{'= '}{all_results[0]}")
    print(f"{'EEGNet with ReLU' : <27}{'Max test Acc' : <14}{'= '}{all_results[1]}")
    print(f"{'EEGNet with LeakyReLU' : <27}{'Max test Acc' : <14}{'= '}{all_results[2]}")

    print(f"{'DeepConvNet with ELU' : <27}{'Max test Acc' : <14}{'= '}{all_results[3]}")
    print(f"{'DeepConvNet with ReLU' : <27}{'Max test Acc' : <14}{'= '}{all_results[4]}")
    print(f"{'DeepConvNet with LeakyReLU' : <27}{'Max test Acc' : <14}{'= '}{all_results[5]}")