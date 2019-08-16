import os
import torch
from absl import app
from absl import flags
from absl import logging


from octpred.data.dataloader import OCTDataSet
from octpred.models.resnet import resnet50
from octpred.routines import train_model, eval_model, save_model, maybe_restore

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", "~/Downloads/OCT2017", "path to data directory")
flags.DEFINE_string("save_dir", "/tmp", "path to save weights")
flags.DEFINE_string("checkpoint", None, "best checkpoint")
flags.DEFINE_integer("epochs", 1, "number of epochs")
flags.DEFINE_integer("batch_size", 8, "batch size")
flags.DEFINE_float("learning_rate", 0.001, "learning rate")
flags.DEFINE_integer("num_workers", None, "number of workers")
flags.DEFINE_boolean("debug", False, "debug mode")
flags.DEFINE_boolean("train", False, "training")
flags.DEFINE_boolean("val", False, "inference on validation")
flags.DEFINE_boolean("test", False, "inference on testing")
flags.DEFINE_boolean("restore", False, "try to restore")
flags.DEFINE_string("gpu", "0", "gpu id")

use_gpu = torch.cuda.is_available()

def main(_):
    if FLAGS.gpu:
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
        os.environ["CUDA_VISIBLE_DEVICES"]=FLAGS.gpu

    if torch.cuda.is_available():
        logging.info("Using device: {}".format(FLAGS.gpu))
    ds = OCTDataSet(FLAGS.data_dir, FLAGS.batch_size, FLAGS.num_workers)
    logging.info("Dataset loaded")

    model = resnet50(num_classes=4)

    logging.info("model initialized")
    
    if FLAGS.restore:
        maybe_restore(model, FLAGS.save_dir)
        logging.info("model loaded")
        pass
    
    criterion = torch.nn.CrossEntropyLoss()
    
    if use_gpu:
        model.cuda()

    if FLAGS.train:
        logging.info("Training over {} epochs".format(FLAGS.epochs))    
        optimizer_ft = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)
        exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
        model = train_model(
            model, ds=ds, criterion=criterion, save_dir=FLAGS.save_dir,
            optimizer=optimizer_ft, scheduler=exp_lr_scheduler, 
            num_epochs=FLAGS.epochs, debug=FLAGS.debug)
        # save_model(model, FLAGS.checkpoint)
    
    if FLAGS.val:
        logging.info("Running validation")
        eval_model(model, criterion, ds=ds, mode="val")

    if FLAGS.test:
        logging.info("Running inference on test set")
        eval_model(model, criterion, ds=ds, mode="test")

    logging.info("Exiting...")


if __name__ == "__main__":
    app.run(main)