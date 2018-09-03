
class Trainer(object):
    def register_listener(self, listener):
        raise NotImplementedError

    def remove_listener(self, listener):
        raise NotImplementedError

    def get_listeners(self):
        raise NotImplementedError

    def add_result_printer(self, printer):
        raise NotImplementedError

    def remove_result_printer(self, printer):
        raise NotImplementedError

    def run_epoch(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def set_num_epochs(self, new_num_epochs):
        raise NotImplementedError

    def set_optimizer(self, new_optimizer):
        raise NotImplementedError

    def set_lr_scheduler(self, new_lr_scheduler):
        raise NotImplementedError

    def get_NN(self):
        raise NotImplementedError
