import sys


class DnnModel(object):
    def __init__(self, model_name, intance_name, model_layer, input_w, input_h, num_channels, num_classes, batch_size, optimizer, learning_rate, activation):

        self.modelName = model_name
        self.instanceName = model_name + intance_name
        self.modelLayer = model_layer
        self.inputWidth = input_w
        self.inputHeight = input_h
        self.numChannels = num_channels
        self.numClasses = num_classes
        self.batchSize = batch_size
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.activation = activation
        import_model = __import__(self.modelName)
        
        #the imported module and the regarding class name have to be the same
        clazz = getattr(import_model, self.modelName)
        self.modelEntity = clazz(net_name=self.instanceName, model_layer=self.modelLayer, input_h=self.inputHeight, 
                                 input_w=self.inputWidth, channel_num=self.numChannels, num_classes=self.numClasses, batch_size=self.batchSize, 
                                 opt=self.optimizer, learning_rate=self.learning_rate, activation=self.activation)
        
    def getModelLayer(self):
        return self.modelLayer

    def getInstanceName(self):
        return self.instanceName

    def getModelName(self):
        return self.modelName

    def getModelEntity(self):
        return self.modelEntity

    def getBatchSize(self):
        return self.batchSize

    def getDesiredAccuracy(self):
        return self.desiredAccuracy
