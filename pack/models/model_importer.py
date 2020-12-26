import importlib


class ModelImporter(object):
    def __init__(self, model_type, model_instance_name, num_layer, input_h, input_w, num_channels,
                 num_classes, batch_size, optimizer, learning_rate, activation, batch_padding):
        self.model_type = model_type
        self.model_instance_name = model_instance_name
        self.num_layer = num_layer
        self.input_width = input_w
        self.input_height = input_h
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.learn_rate = learning_rate
        self.activation = activation
        self.batch_padding = batch_padding

        import_model = importlib.import_module('models.model_' + self.model_type)
        clazz = getattr(import_model, self.model_type)
        self.model_entity = clazz(net_name=self.model_instance_name, num_layer=self.num_layer,
                                  input_h=self.input_height, input_w=self.input_width, num_channel=self.num_channels,
                                  num_classes=self.num_classes, batch_size=self.batch_size, opt=self.optimizer,
                                  learning_rate=self.learn_rate, activation=self.activation, batch_padding=self.batch_padding)

    def get_model_entity(self):
        return self.model_entity
