import torch.nn.functional as F
from utils.paddle_utils import get_one_hot, Logger, most_common_value, top_k_most_common
from utils.utils import majority_or_original
from tqdm import tqdm
import torch
import time
import numpy as np


class KM(object):

    def __init__(self, device,args):#, log_file, args):
        self.device = device
        self.iter = args['iter']
        self.alpha = args['alpha']
        if 'maj_vote' in args.keys():
            self.maj_vote = args['maj_vote']
        else:
            self.maj_vote = False
        
        #self.log_file = log_file
        #self.logger = Logger(__name__, self.log_file)
        self.init_info_lists()

    def init_info_lists(self):
        self.timestamps = []
        self.criterions = []
        self.test_acc = []
        self.test_acc_top5 = []
        self.preds_q = []

    
    def get_logits(self, samples):
        """
        inputs:
            samples : torch.Tensor of shape [n_task, shot, feature_dim]
        returns :
            logits : torch.Tensor of shape [n_task, shot, num_class]
        """
        diff = self.w.unsqueeze(1) - samples.unsqueeze(2)  # N x n x K x C
        logits = (diff.square_()).sum(dim=-1)

        return - 1 / 2 * logits  # N x n x K

    def init_w(self, support, y_s):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]

        updates :
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]
        """
        n_tasks = support.size(0)
        one_hot = get_one_hot(y_s)

        counts = one_hot.sum(1).view(n_tasks, -1, 1)
        weights = one_hot.transpose(1, 2).matmul(support)
        self.w = weights / counts
        self.w = self.w / self.w.norm(dim=-1, keepdim=True)

    def record_convergence(self, new_time, criterions):
        """
        inputs:
            new_time : scalar
            criterions : torch.Tensor of shape [n_task]
        """
        self.criterions.append(criterions)
        self.timestamps.append(new_time)

    
    def record_info(self, y_q):
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
        #self.test_acc_top5 = torch.cat(self.test_acc_top5, dim=1).cpu().numpy()
        return {'timestamps': self.timestamps, 'criterions':self.criterions,
                'acc': self.test_acc,'preds_q':self.preds_q}#,'acc_top5': self.test_acc_top5}

    def run_task(self, task_dic):
        """
        inputs:
            task_dic : dictionnary with n_tasks few-shot tasks
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

        # Run adaptation
        self.run_method(support=support, query=query, y_s=y_s, y_q=y_q)

        # Extract adaptation logs
        logs = self.get_logs()

        return logs


class PADDLE(KM):

    def __init__(self, device,args):#, log_file, args):
        super().__init__(device=device, args=args)#log_file=log_file, args=args)

    #def __del__(self):
    #    self.logger.del_logger()

    def A(self, p):
        """
        inputs:
            p : torch.tensor of shape [n_tasks, n_query, num_class]

        returns:
            v : torch.Tensor of shape [n_task, n_query, num_class]
        """

        n_query = p.size(1)
        v = p.sum(1) / n_query
        return v

    def A_adj(self, v, n_query):
        """
        inputs:
            V : torch.tensor of shape [n_tasks, num_class]
            n_query : int

        returns:
            p : torch.Tensor of shape [n_task, n_query, num_class]
        """

        p = v.unsqueeze(1).repeat(1, n_query, 1) / n_query
        return p

    def u_update(self, query):
        """
        inputs:
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
         
        updates:
            self.u : torch.Tensor of shape [n_task, n_query, num_class]
        """
        
        n_query = query.size(1)
        logits = self.get_logits(query).detach()
        self.u = (logits + self.alpha * self.A_adj(self.v, n_query)).softmax(2)


    def v_update(self):
        """
        updates:
            self.v : torch.Tensor of shape [n_task, num_class]
        """

        self.v = torch.log(self.A(self.u) + 1e-6) + 1

    def w_update(self, support, query, y_s_one_hot):
        """
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s_one_hot : torch.Tensor of shape [n_task, shot, n_ways]

        updates :
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]
        """

        num = torch.einsum('bkq,bqd->bkd',torch.transpose(self.u, 1, 2), query) \
                   + torch.einsum('bkq,bqd->bkd',torch.transpose(y_s_one_hot, 1, 2), support)
        den  = self.u.sum(1) + y_s_one_hot.sum(1)
        self.w = torch.div(num, den.unsqueeze(2))
        self.w = self.w / self.w.norm(dim=-1, keepdim=True)

    def run_method(self, support, query, y_s, y_q):
        """
        Corresponds to the PADDLE inference
        inputs:
            support : torch.Tensor of shape [n_task, shot, feature_dim]
            query : torch.Tensor of shape [n_task, n_query, feature_dim]
            y_s : torch.Tensor of shape [n_task, shot]
            y_q : torch.Tensor of shape [n_task, n_query]

        updates :
            self.u : torch.Tensor of shape [n_task, n_query, num_class]         (soft labels)
            self.v : torch.Tensor of shape [n_task, num_class]                  (dual variable)
            self.w : torch.Tensor of shape [n_task, num_class, feature_dim]     (centroids)
        """

        #self.logger.info(" ==> Executing PADDLE with LAMBDA = {}".format(self.alpha))
        
        y_s_one_hot = get_one_hot(y_s)
        n_task, n_ways = y_s_one_hot.size(0), y_s_one_hot.size(2)

        self.init_w(support=support, y_s=y_s)                           # initialize basic prototypes
        self.v = torch.zeros(n_task, n_ways).to(self.device)            # initialize v to vector of zeros

        for i in range(self.iter):#tqdm(range(self.iter)):

            w_old = self.w
            t0 = time.time()

            self.u_update(query)
            self.v_update()
            self.w_update(support, query, y_s_one_hot)

            t1 = time.time()
            weight_diff = (w_old - self.w).norm(dim=-1).mean(-1)
            criterions = weight_diff
            self.record_convergence(new_time=t1-t0, criterions=criterions)
            
        self.record_info(y_q=y_q)
