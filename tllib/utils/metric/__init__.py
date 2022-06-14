import torch
import prettytable
import numpy as np

__all__ = ['keypoint_detection']

def binary_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        batch_size = target.size(0)
        pred = (output >= 0.5).float().t().view(-1)
        correct = pred.eq(target.view(-1)).float().sum()
        correct.mul_(100. / batch_size)
        return correct


def accuracy(output, target, topk=(1,)):
    r"""
    Computes the accuracy over the k top predictions for the specified values of k

    Args:
        output (tensor): Classification outputs, :math:`(N, C)` where `C = number of classes`
        target (tensor): :math:`(N)` where each value is :math:`0 \leq \text{targets}[i] \leq C-1`
        topk (sequence[int]): A list of top-N number.

    Returns:
        Top-N accuracies (N :math:`\in` topK).
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res

def compute_confusionmatrix(gt_ls, pred_class_ls, pred_score_ls, class_names, score_thresholds=[0]):
    res = {}
    for score_threshold in score_thresholds:
        # pred_class_ls[pred_score_ls < score_threshold] = len(class_names)
        pred_class_ls = [pred_class_ls[idx] if pred_score_ls[idx] >= score_threshold \
        else len(class_names) - 1 for idx in range(len(pred_class_ls))]
        mat = np.zeros((len(class_names), len(class_names)))
        for i, j in zip(pred_class_ls, gt_ls):
            mat[i, j] = mat[i, j] + 1
        acc = np.diag(mat) / mat.sum(1)
        recall = np.diag(mat) / mat.sum(0)
        res[str(score_threshold)] = {
                'score_thresholds': score_threshold,
                'confusion_matrix': mat,
                'acc': acc,
                'recall': recall
            }
    return res

class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, target, output):
        """
        Update confusion matrix.

        Args:
            target: ground truth
            output: predictions of models

        Shape:
            - target: :math:`(minibatch, C)` where C means the number of classes.
            - output: :math:`(minibatch, C)` where C means the number of classes.
        """
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=target.device)
        with torch.no_grad():
            k = (target >= 0) & (target < n)
            inds = n * target[k].to(torch.int64) + output[k]
            self.mat += torch.bincount(inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        """compute global accuracy, per-class accuracy and per-class IoU"""
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        # acc = torch.diag(h) / h.sum(1)
        # recall = torch.diag(h) / h.sum(0)
        recall = torch.diag(h) / h.sum(1)
        acc = torch.diag(h) / h.sum(0)
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        return acc_global, acc, recall, iu

    # def reduce_from_all_processes(self):
    #     if not torch.distributed.is_available():
    #         return
    #     if not torch.distributed.is_initialized():
    #         return
    #     torch.distributed.barrier()
    #     torch.distributed.all_reduce(self.mat)

    def __str__(self):
        acc_global, acc, iu = self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
                acc_global.item() * 100,
                ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
                ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
                iu.mean().item() * 100)

    def compute_mat_percent(self):
        if self.mat_percent is None:
            n = self.num_classes + 1
            self.mat_percent = torch.zeros((n, n), dtype=torch.int64, device=self.mat.device)
            self.mat_percent[:-1, :-1] = self.mat.copy()
            

    def format(self, classes: list):
        """Get the accuracy and IoU for each class in the table format"""
        acc_global, acc, recall, iu = self.compute()
        fg_acc = acc[:-1].mean().item() * 100
        bg_acc = acc[-1].item() * 100
        fg_recall = recall[:-1].mean().item() * 100
        bg_recall = recall[-1].item() * 100
        fg_iu = iu[:-1].mean().item() * 100
        bg_iu = iu[-1].mean().item() * 100

        h = self.mat.float()
        acc_global_fg = torch.diag(h)[:-1].sum() / h[:-1, :-1].sum()
        recall_global_fg = torch.diag(h)[:-1].sum() / h.sum(1)[:-1].sum()
        iu_global_fg = torch.diag(h)[:-1].sum() / (h.sum(1)[:-1].sum() + h.sum(0)[:-1].sum() - torch.diag(h)[:-1].sum())

        table = prettytable.PrettyTable(["class", "acc", "recall", "iou"])
        for i, class_name, per_acc, per_recall, per_iu in zip(range(len(classes)), classes, (acc * 100).tolist(), (recall * 100).tolist(), (iu * 100).tolist()):
            table.add_row([class_name, per_acc, per_recall, per_iu])

        return 'global correct: {:.1f}\nglobal correct fg:{:.1f} global recall fg:{:.1f} global IoU fb: {:.1f}\nmean correct:{:.1f} mean recall:{:.1f} mean IoU: {:.1f}\nfg_acc: {:.1f} fg_recall: {:.1f} fg_iou: {:.1f}\nbg_acc: {:.1f} bg_recall: {:.1f} bg_iou: {:.1f}\n{}\n'. \
                format(acc_global.item() * 100, acc_global_fg.mean().item() * 100, recall_global_fg.mean().item() * 100, iu_global_fg.mean().item() * 100, 
                acc.mean().item() * 100, recall.mean().item() * 100, iu.mean().item() * 100, 
                fg_acc, fg_recall, fg_iu, bg_acc, bg_recall, bg_iu, table.get_string())

