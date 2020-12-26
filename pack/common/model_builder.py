import numpy as np
from pack.models.model_importer import ModelImporter


def build_model_single(rand_seed,
                       model_type,
                       feature_ph,
                       label_ph,
                       layer_num=1,
                       img_height=224,
                       img_width=224,
                       num_channel=3,
                       num_class=1000,
                       batch_size=32,
                       optimizer='Adam',
                       learning_rate=0.000001,
                       train_activation='relu',
                       is_pack=False):

    model_name_abbr = np.random.choice(rand_seed, 1, replace=False).tolist()
    dm = ModelImporter(model_type, str(model_name_abbr.pop()), layer_num, img_height, img_width, num_channel,
                       num_class, batch_size, optimizer, learning_rate, train_activation, is_pack)

    model_entity = dm.get_model_entity()
    model_logit = model_entity.build(feature_ph, is_training=True)
    train_step = model_entity.train(model_logit, label_ph)
    eval_step = model_entity.evaluate(model_logit, label_ph)

    return train_step, eval_step
