from __future__ import absolute_import, division, print_function, unicode_literals
import os
import pathlib
import multiprocessing
import comet_ml as comet
from utils.utils import process_config, create_dirs, get_args, create
import sys, gc, time, os

current_directory = pathlib.Path(__file__).parent.absolute() # Get the current script's directory
os.environ['MLIR_CRASH_REPRODUCER_DIRECTORY'] = str(current_directory / 'mlir99') # Set the environment variable to the 'mlir' folder within the current directory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def main():

    try:
        args = get_args("")
        list_config_files = os.listdir(args.config)
        if len(list_config_files) == 0:
            raise Exception('Empty config directory !')

        for fold in [0,1,2,3,4,100,200,300,400,500]:
            for pooling_size in [2,3,4,5]:
                for pooling_steps in [3,4]:
                    for kernel in [3,5,7,9,11,15,19,27]:
                        for cf in list_config_files:
                            p = multiprocessing.Process(target=train, args=(args, cf, fold, kernel, pooling_size, pooling_steps))#, daemon=True)
                            p.start()
                            p.join()
                            p.close()
                            gc.collect()

    except Exception as e:
        print(e)
        sys.exit(1)

def train(args, cf, fold, kernel, pooling_size, pooling_steps) -> None:
    import tensorflow as tf
    tf.config.set_soft_device_placement(True)
    physical_devices = tf.config.list_physical_devices('GPU')
    try:
        for gpu in physical_devices:
            pass
            #tf.config.experimental.set_memory_growth(gpu, True)
            #tf.config.set_logical_device_configuration(
            #    gpu,
            #    [tf.config.LogicalDeviceConfiguration(memory_limit=7168)])
    except:
        # Invalid device or cannot modify virtual devices once initialized.
        pass

    print(f'Processing Config File : {cf}')
    path_to_config = os.path.join(args.config,cf)
    config = process_config(path_to_config)
    config.data_loader.fold = fold
    config.model.pooling_size = pooling_size
    config.model.pooling_steps = pooling_steps
    config.model.kernel_start = kernel
    config.model.kernel_mid = kernel
    config.model.kernel_end = kernel
    create_dirs([config.callbacks.checkpoint_dir])

    timestamp = time.strftime("%H-%M",time.localtime())

    print('Create the data generator.')
    data_loader = create("data_loader."+config.data_loader.name)(config, timestamp)

    print('Create the model.')
    model = create("models."+config.model.name)(config, data_loader.get_shape())
    #model.printModel()
    # model.model.load_weights("D:/Henrik Riedel/VADER/experiments/2023-07-04/m01-00-dl07-checkpoints/11-52/Raw all 9 - 0 - strat - debugging")
    
    print('Create the trainer')
    trainer = create("trainers."+config.trainer.name)(config.trainer.run + f" f{config.data_loader.fold}" + f" k{config.model.kernel_start}" + f" p{config.model.pooling_size}" + f" ps{config.model.pooling_steps}",
                                                       path_to_config, model.model, data_loader.get_train_data(), data_loader.get_validation_data(), data_loader.get_testing_data(), config, timestamp)
    
    print('Start training the model.')
    trainer.train()

    trainer.test()
    
    trainer.end()


if __name__ == '__main__':
    main()