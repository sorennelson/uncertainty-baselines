import tensorflow as tf
from classification_models.tfkeras import Classifiers

class ResNetEnsemble(tf.keras.Model):

    def __init__(self, resnet_depth=18, ensemble_size=1, pretrained=False, l2=0.0, num_classes=100):
        super(ResNetEnsemble, self).__init__()
        self.ensemble_members = []
        self.l2 = l2
        self.num_classes = num_classes
        self.ensemble_size = ensemble_size
        for _ in range(ensemble_size):
            ResNet, preprocess_input = Classifiers.get('resnet{}'.format(resnet_depth))
            weights = 'imagenet' if pretrained else None
            ensemble_member = {
                    'resnet_trunk': ResNet((224,224,3), l2=l2, weights=weights, include_top=False),
                    'global_pooling': tf.keras.layers.GlobalAveragePooling2D(),
                    'dense': tf.keras.layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.HeNormal(),
                kernel_regularizer=tf.keras.regularizers.l2(l2),
                bias_regularizer=tf.keras.regularizers.l2(l2))
                    }
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

