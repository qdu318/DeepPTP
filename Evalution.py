# encoding utf-8

import torch
import json
from DataLoad import get_loader
import utils
import torch.nn.functional as F


config = json.load(open('./config.json', 'r'))


def Evalution(model, batchsize, max_accuracy, max_precise, max_recall, max_f1_score, log):
    model.eval()
    eval_loss = 0
    accuracy = 0
    confusion_matrix = torch.zeros(4, 4)

    if torch.cuda.is_available():
        confusion_matrix.cuda()
    with torch.no_grad():
        for file in config['eval_set']:
            dataset = get_loader(file, batchsize)
            for idx, parameters in enumerate(dataset):
                parameters = utils.to_var(parameters)
                out = model(parameters)
                loss = F.nll_loss(out, parameters['direction'], size_average=False).item()
                eval_loss += loss
                pred = out.data.max(1, keepdim=True)[1]
                pred = torch.squeeze(pred)
                for i in range(len(pred)):
                    confusion_matrix[parameters['direction'][i]][pred[i]] += 1

                accuracy += pred.eq(parameters['direction'].data.view_as(pred)).cpu().sum()

            precise, recall, f1_score = utils.CalConfusionMatrix(confusion_matrix)

            accuracy_value = accuracy.item() / len(dataset.dataset)
            if accuracy_value > max_accuracy:
                max_accuracy = accuracy_value
            if precise > max_precise:
                max_precise = precise
            if recall > max_recall:
                max_recall = recall
            if f1_score > max_f1_score:
                max_f1_score = f1_score

            eval_loss /= len(dataset.dataset)
            print('Evalution: Average loss: {:.4f}'.format(eval_loss))
            print(
                'Current Evalution: Accuracy: {}/{} ({:.4f}), Precise: {:.4f}, ReCall: {:.4f}, F1 Score: {:.4f}'.format(
                    accuracy, len(dataset.dataset), accuracy_value, precise, recall, f1_score))
            print(
                'Max Evalution: Accuracy: {:.4f}, Precise: {:.4f}, ReCall: {:.4f}, F1 Score: {:.4f}'.format(
                    max_accuracy, max_precise, max_recall, max_f1_score))

            log.info('Accuracy:{:.4f}, Precise:{:.4f}, Recall:{:.4f}, F1 Score:{:.4f}'.format(accuracy_value, precise, recall, f1_score))

    return max_accuracy, max_precise, max_recall, max_f1_score