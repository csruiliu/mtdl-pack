import importlib


class ModelImporter:
    def __init__(self,
                 model_type,
                 model_instance_name,
                 num_layer,
                 input_h,
                 input_w,
                 num_channels,
                 num_classes,
                 batch_size,
                 optimizer,
                 learning_rate,
                 activation,
                 batch_padding):

        import_model = importlib.import_module('pack.models.model_' + model_type.lower())
        clazz = getattr(import_model, model_type)
        self.model_entity = clazz(net_name=model_instance_name,
                                  num_layer=num_layer,
                                  input_h=input_h,
                                  input_w=input_w,
                                  num_channel=num_channels,
                                  num_classes=num_classes,
                                  batch_size=batch_size,
                                  opt=optimizer,
                                  learning_rate=learning_rate,
                                  activation=activation,
                                  batch_padding=batch_padding)

    def get_model_entity(self):
        return self.model_entity
