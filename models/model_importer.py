
class DnnModel(object):
    def __init__(self, model_type, model_instance_name, num_layer, input_h, input_w, num_channels,
                 num_classes, batch_size, optimizer, learning_rate, activation, batch_padding):
        self.modelType = model_type
        self.modelInstanceName = model_instance_name
        self.numLayer = num_layer
        self.inputWidth = input_w
        self.inputHeight = input_h
        self.numChannels = num_channels
        self.numClasses = num_classes
        self.batchSize = batch_size
        self.optimizer = optimizer
        self.learningRate = learning_rate
        self.activation = activation
        self.batch_padding = batch_padding

        import_model = __import__(self.modelType)
        clazz = getattr(import_model, self.modelType)
        self.modelEntity = clazz(net_name=self.modelInstanceName, num_layer=self.numLayer,
                                 input_h=self.inputHeight, input_w=self.inputWidth, num_channel=self.numChannels,
                                 num_classes=self.numClasses, batch_size=self.batchSize, opt=self.optimizer,
                                 learning_rate=self.learningRate, activation=self.activation, batch_padding=self.batch_padding)

    def getModelEntity(self):
        return self.modelEntity