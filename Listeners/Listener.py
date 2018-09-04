class Listener(object):
    def __init__(self, message, action):
        self.message = message
        self.action = action

    def notify(self, message, trainer):
        if self.message == message:
            self.action(trainer)
