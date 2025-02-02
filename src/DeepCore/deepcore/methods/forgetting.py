from .earlytrain import EarlyTrain
import torch, time
from torch import nn
import numpy as np

# Acknowledgement to
# https://github.com/mtoneva/example_forgetting

class Forgetting(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None, balance=True,
                 dst_test=None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model=specific_model,
                         dst_test=dst_test)

        self.balance = balance
        self.initialized = False

    def get_hms(self, seconds):
        # Format time for printing purposes

        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        return h, m, s

    def before_train(self):
        if not self.initialized:
            self.train_loss = 0.
            self.correct = 0.
            self.total = 0.
            self.initialized = True

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        with torch.no_grad():
            for i in range(len(batch_inds)):
                batch_inds[i] = batch_inds[i].cpu()
            _, predicted = torch.max(outputs.data, 1)

            cur_acc = (predicted == targets).clone().detach().requires_grad_(False).type(torch.float32)
            self.forgetting_events[torch.tensor(batch_inds)[(self.last_acc[batch_inds]-cur_acc)>0.01]]+=1.
            self.last_acc[batch_inds] = cur_acc

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        self.train_loss += loss.item()
        self.total += targets.size(0)
        _, predicted = torch.max(outputs.data, 1)
        self.correct += predicted.eq(targets.data).cpu().sum()

        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' % (
            epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item(),
            100. * self.correct.item() / self.total))

    def before_epoch(self):
        self.start_time = time.time()

    def after_epoch(self):
        epoch_time = time.time() - self.start_time
        self.elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (self.get_hms(self.elapsed_time)))

    def before_run(self):
        if not self.initialized:
            self.elapsed_time = 0

            self.forgetting_events = torch.zeros(self.n_train, requires_grad=False).to(self.args.device)
            self.last_acc = torch.zeros(self.n_train, requires_grad=False).to(self.args.device)

    def finish_run(self):
        pass

    def select(self, **kwargs):
        if self.args.per_epoch:
            self.train_indx = np.arange(self.n_train)
            # model is needs to be set before
            self.criterion = nn.CrossEntropyLoss().to(self.args.device)
            self.criterion.__init__()

            # Setup optimizer
            if self.args.selection_optimizer == "SGD":
                self.model_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.selection_lr,
                                                       momentum=self.args.selection_momentum,
                                                       weight_decay=self.args.selection_weight_decay,
                                                       nesterov=self.args.selection_nesterov)
            elif self.args.selection_optimizer == "Adam":
                self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.selection_lr,
                                                        weight_decay=self.args.selection_weight_decay)
            else:
                self.model_optimizer = torch.optim.__dict__[self.args.selection_optimizer](self.model.parameters(),
                                                                                           lr=self.args.selection_lr,
                                                                                           momentum=self.args.selection_momentum,
                                                                                           weight_decay=self.args.selection_weight_decay,
                                                                                           nesterov=self.args.selection_nesterov)
            self.before_run()
            warmup_test_acc = 0.0
            list_of_train_idx = np.random.choice(np.arange(self.n_pretrain if self.if_dst_pretrain else self.n_train),
                                                 self.n_pretrain_size, replace=False)
            self.before_epoch()
            self.train_no_backward_pass(0, list_of_train_idx)
            self.after_epoch()

            top_examples = self.train_indx[np.argsort(self.forgetting_events.cpu().numpy())][::-1][:self.coreset_size]
            return {"indices": top_examples, "scores": self.forgetting_events}, warmup_test_acc
        else:

            selection_result, warmup_test_acc = self.run()

            if not self.balance:
                top_examples = self.train_indx[np.argsort(self.forgetting_events.cpu().numpy())][::-1][:self.coreset_size]
            else: #default
                top_examples = np.array([], dtype=np.int64)
                for c in range(self.num_classes):
                    c_indx = self.train_indx[self.dst_train.targets == c]
                    budget = round(self.fraction * len(c_indx))
                    top_examples = np.append(top_examples,
                                        c_indx[np.argsort(self.forgetting_events[c_indx].cpu().numpy())[::-1][:budget]])

            return {"indices": top_examples, "scores": self.forgetting_events}, warmup_test_acc
