from utils.paddle_utils import get_one_hot
from tqdm import tqdm
import torch
import time
from copy import deepcopy
import numpy as np
from utils.utils import majority_or_original

class BASE(object):

    def __init__(self, device, args):
        self.device = device
        self.iter = args['iter']
        self.lambd = int(args['num_classes_test'] / args['k_eff']) * args['n_query']
        self.init_info_lists()
        self.args = args
        self.eps = 1e-15
        self.iter_mm = args['iter_mm']

    def init_info_lists(self):
        self.timestamps = []
        self.criterions = []
        self.test_acc = []
        self.preds_q = []

    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]
        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        l1 = torch.lgamma(self.alpha.sum(-1)).unsqueeze(1)
        l2 = - torch.lgamma(self.alpha).sum(-1).unsqueeze(1)
        print(self.alpha.shape)
        print(samples.shape)
        l3 = ((self.alpha.unsqueeze(1) - 1) * samples.unsqueeze(2)).sum(-1)
        logits = l1 + l2 + l3
        return logits  # N x n x K

    def record_convergence(self, new_time, criterions):
        """
        inputs:
            new_time : scalar
            criterions : torch.Tensor of shape [n_task]
        """
        self.criterions.append(criterions)
        self.timestamps.append(new_time)

    def compute_acc(self, y_q):
        """
        inputs:
            y_q : torch.Tensor of shape [n_task, n_query] :
        """

        preds_q = self.u.argmax(2).to(torch.device('cpu'))
        y_q = y_q.to(torch.device('cpu'))

        if self.maj_vote:
            
            preds_q_maj = majority_or_original(preds_q)
            accuracy = (preds_q_maj == y_q).float().mean(1, keepdim=True)
            self.preds_q.append(preds_q_maj)

        else:
            accuracy = (preds_q == y_q).float().mean(1, keepdim=True)
            
        self.test_acc.append(accuracy)
        

    def get_logs(self):
        self.criterions = torch.stack(self.criterions, dim=0).cpu().numpy()
        self.test_acc = torch.cat(self.test_acc, dim=1).cpu().numpy()
        return {'timestamps': np.array(self.timestamps).mean(), 'criterions': self.criterions,
                'acc': self.test_acc,'preds_q':self.preds_q}

    def run_task(self, task_dic, shot=10):
        """
        inputs:
            task_dic : dictionnary with n_task few-shot tasks
            shot : scalar, number of shots
        """

        # Extract support and query
        y_s = task_dic['y_s']               # [n_task, shot]
        y_q = task_dic['y_q']               # [n_task, n_query]
        support = task_dic['x_s']           # [n_task, shot, feature_dim]
        query = task_dic['x_q']             # [n_task, n_query, feature_dim]

        # Transfer tensors to GPU if needed
        support = support.to(self.device)
        query = query.to(self.device)
        y_s = y_s.long().squeeze(2).to(self.device)
        y_q = y_q.long().squeeze(2).to(self.device)
        del task_dic

        # Run adaptation
        self.run_method(support=support, query=query, y_s=y_s, y_q=y_q)

        # Extract adaptation logs
        logs = self.get_logs()
        return logs


class HARD_EM_DIRICHLET(BASE):

    def __init__(self, device, args):
        super().__init__(device=device, args=args)

    def u_update(self, query):
        """
        inputs:
            query : torch.Tensor of shape [n_task, n_query, feature_dim]

        updates:
            self.u : torch.Tensor of shape [n_task, n_query, num_class]
        """
        __, n_query = query.size(-1), query.size(1)
        logits = self.get_logits(query)
        self.u = (logits + self.lambd *
                  self.v.unsqueeze(1) / n_query).softmax(2)

    def v_update(self):
        """
        updates:
            self.v : torch.Tensor of shape [n_task, num_class]
            --> corresponds to the log of the class proportions
        """
        self.v = torch.log(self.u.sum(1) / self.u.size(1) + self.eps) + 1

    def curvature(self, alpha):
        digam = torch.polygamma(0, alpha + 1)
        return torch.where(alpha > 1e-11, abs(2 * (self.log_gamma_1 - torch.lgamma(alpha + 1) + digam * alpha) / alpha**2), self.zero_value), digam

    def update_alpha(self, alpha_0, y_cst):
        alpha = deepcopy(alpha_0)

        for l in range(self.iter_mm):
            curv, digam = self.curvature(alpha)
            b = digam - \
                torch.polygamma(0, alpha.sum(-1)).unsqueeze(-1) - curv * alpha
            print(b.shape)
            b = b - y_cst
            a = curv
            delta = b**2 + 4 * a
            alpha_new = (- b + torch.sqrt(delta)) / (2 * a)

            if l > 0 and l % 50 == 0:
                criterion = torch.norm(
                    alpha_new - alpha)**2 / torch.norm(alpha)**2
                if l % 1000 == 0:
                    print('iter', l, 'criterion', criterion)
                if criterion < 1e-11:
                    break
            alpha = deepcopy(alpha_new)
        self.alpha = deepcopy(alpha_new)

    def objective_function(self, support, query, y_s_one_hot):

        l1 = torch.lgamma(self.alpha.sum(-1)).unsqueeze(1)
        l2 = - torch.lgamma(self.alpha).sum(-1).unsqueeze(1)
        l3 = ((self.alpha.unsqueeze(1) - 1) *
              torch.log(query + self.eps).unsqueeze(2)).sum(-1)
        datafit_query = -(self.u * (l1 + l2 + l3)).sum(-1).sum(1)
        l1 = torch.lgamma(self.alpha.sum(-1)).unsqueeze(1)
        l2 = - torch.lgamma(self.alpha).sum(-1).unsqueeze(1)
        l3 = ((self.alpha.unsqueeze(1) - 1) *
              torch.log(support + self.eps).unsqueeze(2)).sum(-1)
        datafit_support = -(y_s_one_hot * (l1 + l2 + l3)).sum(-1).sum(1)
        datafit = 1 / 2 * (datafit_query + datafit_support)

        reg_ent = (self.u * torch.log(self.u + self.eps)).sum(-1).sum(1)

        props = self.u.mean(1)
        part_complexity = - self.lambd * \
            (props * torch.log(props + self.eps)).sum(-1)

        return datafit + reg_ent + part_complexity

    def run_method(self, support, query, y_s, y_q):
        """
        Corresponds to the Hard EM DIRICHLET inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :from copy import deepcopy
            self.u : torch.Tensor of shape [n_task, n_query, num_class]         (soft labels)
            self.v : torch.Tensor of shape [n_task, num_class]                  (dual variable)
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]     (centroids)
        """
        
        self.zero_value = torch.polygamma(
            1, torch.Tensor([1]).to(self.device)).float()
        self.log_gamma_1 = torch.lgamma(
            torch.Tensor([1]).to(self.device)).float()
        n_task, n_class = query.shape[0], self.args['num_classes_test']

        # Initialization
        self.v = torch.zeros(n_task, n_class).to(
            self.device)        # dual variable set to zero
        if self.args['use_softmax_feature']:
            self.u = deepcopy(query)
        else:
            raise ValueError(
                "The selected method is unable to handle query features that are not in the unit simplex")
        self.alpha = torch.ones((n_task, n_class, n_class)).to(self.device)
        y_s_one_hot = get_one_hot(y_s)
        alpha_old = deepcopy(self.alpha)
        t0 = time.time()

        # inplace operations to save memory
        support.add_(self.eps)
        query.add_(self.eps)
        support.log_()
        query.log_()

        
        pbar = tqdm(range(self.iter))
        for i in pbar:

            # update of dirichlet parameter alpha
            y_s_sum = y_s_one_hot.sum(dim=1)  # Shape [n_task, num_class]
            u_sum = self.u.sum(dim=-1)
            print(y_s_sum.shape)
            print(u_sum.shape)
            y_cst = (1 / (y_s_sum + u_sum)).unsqueeze(-1).unsqueeze(1)
            print(y_cst.shape)
            print(support.shape)
            print(y_s_one_hot.shape)
            #y_cst = y_cst * 
            print('---')
            print((self.u.unsqueeze(2) * query.unsqueeze(1)).sum(dim=-1).shape)
            print(((y_s_one_hot.unsqueeze(-1) * support.unsqueeze(2))).sum(dim=1).shape)

            y_cst = y_cst * (((y_s_one_hot.unsqueeze(-1) * support.unsqueeze(2))).sum(
                dim=1).sum(dim=-1).unsqueeze(-1) + (self.u.unsqueeze(1) * query.unsqueeze(2)).sum(dim=1).sum(dim=-1).unsqueeze(-1))
            print(y_cst.shape)
            self.update_alpha(self.alpha, y_cst)

            # update on dual variable v
            self.v_update()

            # update hard assignment variable u
            self.u_update(query)
            labels = torch.argmax(self.u, dim=-1)
            self.u.zero_()
            self.u.scatter_(2, labels.unsqueeze(-1), 1.0)

            u_old = deepcopy(self.u)
            alpha_diff = ((alpha_old - self.alpha).norm(dim=(1, 2)
                                                        ) / alpha_old.norm(dim=(1, 2))).mean(0)
            criterions = alpha_diff
            alpha_old = deepcopy(self.alpha)
            t1 = time.time()

            # compute criterion
            alpha_diff = ((alpha_old - self.alpha).norm(dim=(1, 2)
                                                        ) / alpha_old.norm(dim=(1, 2))).mean(0)
            criterions = alpha_diff
            alpha_old = deepcopy(self.alpha)

            pbar.set_description(f"Criterion: {criterions}")
            t1 = time.time()
            self.record_convergence(
                new_time=(t1-t0) / n_task, criterions=criterions)

        self.compute_acc(y_q=y_q)