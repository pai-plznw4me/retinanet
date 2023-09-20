import tensorflow as tf
from loss import RetinaNetLoss
from retinanet import get_backbone, RetinaNet
import os
import numpy as np


if __name__ == '__main__':
    # 모델 저장 장소
    model_dir = "retinanet/"

    # set hparam
    num_classes = 1
    batch_size = 2

    # set lr
    learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [125, 250, 500, 240000, 360000]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )

    # set model
    resnet50_backbone = get_backbone()
    loss_fn = RetinaNetLoss(num_classes)
    model = RetinaNet(num_classes, resnet50_backbone)

    # set optimizer
    optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=learning_rate_fn, momentum=0.9)
    model.compile(loss=loss_fn, optimizer=optimizer)

    # set os callback
    callbacks_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
            monitor="loss",
            save_best_only=False,
            save_weights_only=True,
            verbose=1,
        )
    ]

    # training
    save_encode_dir = os.path.join(os.getcwd(), 'sample_data', 'Riceseedlingdetection', 'Encode')
    xs = np.load(os.path.join(save_encode_dir, 'preproc_resnet_images.npy'))
    ys = np.load(os.path.join(save_encode_dir, 'labels.npy'))
    epochs = 1
    model.fit(
        xs,
        ys,
        epochs=epochs,
        callbacks=callbacks_list,
        verbose=1,
    )