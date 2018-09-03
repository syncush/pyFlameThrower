import Trainers.Trainer as Trainer
from pkg_resources import parse_version
import torch.__version__ as installed_torch_version
import torch
from Utils.Global_Vars import GlobalVars



class AbstractTrainer(Trainer):
    def __init__(self, net=None, listeners=[], result_printers=[],
                 num_epochs=10, optimizer=None, crit=None, lr_scheduler=None, device=None):
        global_vars = GlobalVars()
        self.listeners_messages = global_vars["LISTENERS_MESSAGES"]
        self.model = net
        self.listeners = listeners
        self.result_printers = result_printers
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.criterion = crit
        if parse_version(global_vars["TORCH_VISION_WITH_DEVICE"]) <= parse_version(installed_torch_version):
            if device is None:
                self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            else:
                self.device = device
        else:
            self.device = None

    def register_listener(self, listener):
        self.listeners.append(listener)

    def remove_listener(self, listener):
        self.listeners.remove(listener)

    def _notify_listeners(self, message):
        for listener in self.listeners:
            listener.notify(message, self)

    def get_listeners(self):
        return self.listeners

    def add_result_printer(self, printer):
        self.result_printers.append(printer)

    def remove_result_printer(self, printer):
        self.result_printers.remove(printer)

    def _run_epoch(self):
        self._notify_listeners(self.listeners_messages["EPOCH_STARTED"])
        self.model.train()
        for i, (data, target) in enumerate(self.train_loader):
            self._notify_listeners(self.listeners_messages["BATCH_STARTED"])
            if self.device is not None:
                data, target = data.to(self.device), target.to(self.device)
            # Clear gradients w.r.t. parameters
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            # Getting gradients w.r.t. parameters
            loss.backward()
            # Updating parameters
            self.optimizer.step()
            self._notify_listeners(self.listeners_messages["BATCH_ENDED"])
        self._notify_listeners(self.listeners_messages["EPOCH_ENDED"])

    def train(self):
        self._notify_listeners(self.listeners_messages["LEARNING_STARTED"])
        for x in range(1, self.num_epochs + 1):
            self._run_epoch()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
        self._notify_listeners(self.listeners_messages["LEARNING_ENDED"])

    def set_num_epochs(self, new_num_epochs):
        self.num_epochs = new_num_epochs

    def set_optimizer(self, new_optimizer):
        self.optimizer = new_optimizer

    def set_lr_scheduler(self, new_lr_scheduler):
        self.lr_scheduler = new_lr_scheduler

    def get_NN(self):
        return self.model
