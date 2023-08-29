from __future__ import absolute_import, division, print_function, unicode_literals
from comet_ml import Experiment
from utils.utils import process_config, create_dirs, get_args, create
import sys, gc, time, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import multiprocessing
import tensorflow as tf


def train(args, cf, fold) -> None:
    print(f'Processing Config File : {cf}')
    path_to_config = os.path.join(args.config,cf)
    config = process_config(path_to_config)
    config.data_loader.fold = fold
    create_dirs([config.callbacks.checkpoint_dir])

    timestamp = time.strftime("%H-%M",time.localtime())

    print('Create the data generator.')
    data_loader = create("data_loader."+config.data_loader.name)(config, timestamp)

    print('Create the model.')
    model = create("models."+config.model.name)(config, data_loader.get_shape())
    model.printModel()
    # model.model.load_weights("D:/Henrik Riedel/VADER/experiments/2023-07-04/m01-00-dl07-checkpoints/11-52/Raw all 9 - 0 - strat - debugging")
    
    print('Create the trainer')
    trainer = create("trainers."+config.trainer.name)(config.trainer.run + f" {config.data_loader.fold}", path_to_config, model.model, data_loader.get_train_data(), data_loader.get_validation_data(), data_loader.get_testing_data(), config, timestamp)
    
    print('Start training the model.')
    trainer.train()

    trainer.test()
    
    trainer.end()


def main():

    try:
        args = get_args("3")
        list_config_files = os.listdir(args.config)
        if len(list_config_files) == 0:
            raise Exception('Empty config directory !')

        # for fold in [500,100,200,300,400]:
        # for fold in [0]:
        for fold in [0,1,2,3,4,500,100,200,300,400]:
            #for each config train
            for cf in list_config_files:
                p = multiprocessing.Process(target=train, args=(args, cf, fold), daemon=True)
                p.start()
                p.join()
                p.close()
                gc.collect()

    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == '__main__':
    main()