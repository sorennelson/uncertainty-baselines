import tensorflow as tf
from classification_models.tfkeras import Classifiers

class ResNetEnsemble(tf.keras.Model):

    def __init__(self, resnet_depth='18', ensemble_size=1, pretrained=False, l2=0.0, num_classes=100, reset_stage_list=None):
        super(ResNetEnsemble, self).__init__()
        self.ensemble_members = []
        self.l2 = l2
        self.num_classes = num_classes
        self.ensemble_size = ensemble_size
        if reset_stage_list is not None:
            if len(reset_stage_list) != self.ensemble_size:
                raise ValueError("Make sure reset stage list is correct.")
        else:
            reset_stage_list = [None] * self.ensemble_size
        for i in range(ensemble_size):
            ResNet, preprocess_input = Classifiers.get('resnet{}'.format(resnet_depth))
            weights = 'imagenet' if pretrained else None
            ensemble_member = {
                    'resnet_trunk': ResNet((224,224,3), l2=l2, weights=weights, include_top=False),
                    'global_pooling': tf.keras.layers.GlobalAveragePooling2D(),
                    'dense': tf.keras.layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.HeNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(l2),
                bias_regularizer=tf.keras.regularizers.l2(l2))
                    }
            if reset_stage_list[i] == 'None':
                reset_stage_idx = None
            else:
                reset_stage_idx = int(reset_stage_list[i])
            self.reinitialize_model(ensemble_member['resnet_trunk'], resnet_depth=resnet_depth, reset_stage_idx=reset_stage_idx)
            self.ensemble_members.append(ensemble_member)


    def call(self, inputs, training=False):
        outputs = []
        for  i, aug_batch in enumerate(tf.split(inputs, self.ensemble_size, axis=1)):
            reshaped_inputs = tf.image.resize(tf.squeeze(aug_batch), [224,224], method='bicubic')
            rescaled_inputs = reshaped_inputs * 255.
            resnet_trunk_out = self.ensemble_members[i]['resnet_trunk'](rescaled_inputs, training=training)
            pooled_resnet = self.ensemble_members[i]['global_pooling'](resnet_trunk_out)
            output = self.ensemble_members[i]['dense'](pooled_resnet)
            outputs.append(output)
        if training:
            outputs = tf.concat(outputs,axis=0)
        else:
            outputs = tf.stack(outputs)
            outputs = tf.nn.softmax(outputs, axis=-1)
            outputs =tf.reduce_mean(outputs, axis=0)
        return outputs

    def reinitialize_model(self, model,reset_stage_idx=None, resnet_depth='18'):
      if resnet_depth == '18':
        residual_blocks = ['stage1_unit1','stage1_unit2','stage2_unit1','stage2_unit2','stage3_unit1','stage3_unit2','stage4_unit1','stage4_unit2' ]
      elif resnet_depth =='34':
        config = [3,4,6,3]
        residual_blocks = []
        for i, num in enumerate(config):
            for j in range(num):
                residual_blocks.append('stage{}_unit{}'.format(i+1, j+1))
      else:
        raise ValueError('Incorrect resnet depth')
      if reset_stage_idx is not None:
          residual_blocks = residual_blocks[reset_stage_idx:reset_stage_idx + 1]
      else:
          return
      initializers = []
      weights = []
      for block in residual_blocks:
        for layer in model.layers:
          if block in layer.name:
            if isinstance(layer, (tf.keras.layers.Dense, tf.keras.layers.Conv2D)):
              weights += [layer.kernel]
              initializers += [layer.kernel_initializer]
              if layer.bias is not None:
                weights += [layer.bias]
                initializers += [layer.bias_initializer]
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
              weights += [layer.gamma, layer.beta, layer.moving_mean, layer.moving_variance]
              initializers += [layer.gamma_initializer,
                              layer.beta_initializer,
                              layer.moving_mean_initializer,
                              layer.moving_variance_initializer]
      for w, init in zip(weights, initializers):
        w.assign(init(w.shape, dtype=w.dtype))
