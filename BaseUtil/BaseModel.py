import abc

class BaseModel(object):
    __metaclsaa__ = abc.ABCMeta

    @abc.abstractmethod
    def instantiate_weight(self):
        return


    @abc.abstractmethod
    def inference(self):
        return


    @abc.abstractmethod
    def loss(self):
        return


    @abc.abstractmethod
    def train(self):
        return

