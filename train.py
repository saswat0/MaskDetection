import os
import sys
import time

import tensorflow as tf
from absl import flags, logging, app
from absl.flags import FLAGS

from utils import config
from utils.lr_scheduler import MultiStepWarmUpLR
from utils.prior_box import priors_box
from utils.utils import set_memory_growth
from dataset.preprocess import load_dataset
from network.loss import MultiBoxLoss
from network.net import SSDModel

flags.DEFINE_string('gpu', '0', 'which gpu to use')


def main(_):
    global load_t1
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    logger = tf.get_logger()
    logger.disabled = True
    logger.setLevel(logging.FATAL)
    set_memory_growth()

    weights_dir = 'checkpoints/'
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)

    logging.info("Load configuration...")
    cfg = config.cfg
    label_classes = cfg['labels_list']
    logging.info(f"Total image sample:{cfg['dataset_len']},Total classes number:"
                 f"{len(label_classes)},classes list:{label_classes}")

    logging.info("Compute priors boxes...")
    priors, num_cell = priors_box(cfg)
    logging.info(f"Prior boxes number:{len(priors)},default anchor box number per feature map cell:{num_cell}")

    logging.info("Loading dataset...")
    train_dataset = load_dataset(cfg, priors, shuffle=True, train=True)
    val_dataset = load_dataset(cfg, priors, shuffle=False, train=False)

    logging.info("Create Model...")
    try:
        model = SSDModel(cfg=cfg, num_cell=num_cell, training=True)
        model.summary()
        tf.keras.utils.plot_model(model, to_file=os.path.join(os.getcwd(), 'model.png'),
                                  show_shapes=True, show_layer_names=True)
    except Exception as e:
        logging.error(e)
        logging.info("Create network failed.")
        sys.exit()

    init_epoch = -1

    steps_per_epoch = cfg['dataset_len'] // cfg['batch_size']
    val_steps_per_epoch = cfg['val_len'] // cfg['batch_size']

    logging.info(f"steps_per_epoch:{steps_per_epoch}")

    logging.info("Define optimizer and loss computation and so on...")

    learning_rate = MultiStepWarmUpLR(
        initial_learning_rate=cfg['init_lr'],
        lr_steps=[e * steps_per_epoch for e in cfg['lr_decay_epoch']],
        lr_rate=cfg['lr_rate'],
        warmup_steps=cfg['warmup_epoch'] * steps_per_epoch,
        min_lr=cfg['min_lr'])

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=cfg['momentum'], nesterov=True)

    multi_loss = MultiBoxLoss(num_class=len(label_classes), neg_pos_ratio=3)

    train_log_dir = 'logs/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    @tf.function
    def train_step(inputs, labels):
        with tf.GradientTape() as tape:
            predictions = model(inputs, training=True)
            losses = {}
            losses['reg'] = tf.reduce_sum(model.losses)
            losses['loc'], losses['class'] = multi_loss(labels, predictions)
            total_loss = tf.add_n([l for l in losses.values()])

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return total_loss, losses

    for epoch in range(init_epoch + 1, cfg['epoch']):
        try:
            start = time.time()
            avg_loss = 0.0
            for step, (inputs, labels) in enumerate(train_dataset.take(steps_per_epoch)):

                load_t0 = time.time()
                total_loss, losses = train_step(inputs, labels)
                avg_loss = (avg_loss * step + total_loss.numpy()) / (step + 1)
                load_t1 = time.time()
                batch_time = load_t1 - load_t0

                steps = steps_per_epoch * epoch + step
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss/total_loss', total_loss, step=steps)
                    for k, l in losses.items():
                        tf.summary.scalar('loss/{}'.format(k), l, step=steps)
                    tf.summary.scalar('learning_rate', optimizer.lr(steps), step=steps)

                print(
                    f"\rEpoch: {epoch + 1}/{cfg['epoch']} | Batch {step + 1}/{steps_per_epoch} | Batch time {batch_time:.3f} || Loss: {total_loss:.6f} | loc loss:{losses['loc']:.6f} | class loss:{losses['class']:.6f} ",
                    end='', flush=True)

            print(
                f"\nEpoch: {epoch + 1}/{cfg['epoch']}  | Epoch time {(load_t1 - start):.3f} || Average Loss: {avg_loss:.6f}")

            with train_summary_writer.as_default():
                tf.summary.scalar('loss/avg_loss', avg_loss, step=epoch)

            if (epoch + 1) % cfg['save_freq'] == 0:
                filepath = os.path.join(weights_dir, f'weights_epoch_{(epoch + 1):03d}.h5')
                model.save_weights(filepath)
                if os.path.exists(filepath):
                    print(f">>>>>>>>>>Save weights file at {filepath}<<<<<<<<<<")

        except KeyboardInterrupt:
            print('interrupted')
            exit(0)


if __name__ == '__main__':
    app.run(main)