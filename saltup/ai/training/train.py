import os
import shutil
import copy
import gc

from typing import Iterator, Tuple, Any, List, Union

import matplotlib.pyplot as plt


import torch
from torch.utils.data import DataLoader as pytorch_DataGenerator

import tensorflow as tf

from saltup.saltup_env import SaltupEnv
from saltup.ai.base_dataformat.base_datagen import BaseDatagenerator, kfoldGenerator
from saltup.ai.classification.evaluate import evaluate_model
from saltup.ai.utils.keras.to_onnx import *
from saltup.ai.utils.keras.to_tflite import tflite_conversion
from saltup.ai.training.callbacks import *
from saltup.ai.training.callbacks import _KerasCallbackAdapter
from tqdm import tqdm


def _train_model(
    model:Union[tf.keras.models.Sequential, torch.nn.Module],
    train_gen:BaseDatagenerator,
    val_gen:BaseDatagenerator,
    output_dir:str,
    epochs:int,
    loss_function:Union[tf.keras.losses.Loss, torch.nn.Module],
    optimizer:Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer],
    scheduler:Union[torch.optim.lr_scheduler._LRScheduler, None],
    model_output_name:str=None,
    class_weight:dict=None,
    app_callbacks=[]
) -> str:
    """
    Train the model.
    Args:
        model (Union[tf.keras.models.Sequential, torch.nn.Module]): model to be trained.
        train_gen (Union[keras_ClassificationDataGenerator, DataLoader]): training data generator
        val_gen (Union[keras_ClassificationDataGenerator, DataLoader]): validation data generator
        output_dir (str): folder to save the model
        epochs (int): number of epochs for training
        loss_function (Union[tf.keras.losses.Loss, torch.nn.Module]): loss function for training
        optimizer (Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer]): optimizer for training
        scheduler (Union[torch.optim.lr_scheduler._LRScheduler, None]): scheduler for the optimizer
        model_output_name (str, optional): name of the model. Defaults to None.
        app_callbacks (list, optional): list of callbacks for training. Defaults to [].
    """
    if model_output_name is None:
        model_output_name = 'model'
    saved_models_folder_path = os.path.join(output_dir, "saved_models")
    os.makedirs(saved_models_folder_path, exist_ok=True)
    if isinstance(model, tf.keras.Model):
        # === Keras model ===
        if optimizer is None or loss_function is None:
            raise ValueError("For Keras models, both `optimizer` and `loss_function` must be provided.")
        
        # TODO @S0nFra: Allow possibility to define arguments via SaltupEnv
        # SALTUP_TRAINING_KERAS_COMPILE_ARGS = {
        #     "optimizer": optimizer,
        #     "loss": loss_function,
        #     'jit_compile': False,
        # }.update(SaltupEnv.SALTUP_TRAINING_KERAS_COMPILE_ARGS)   
        # model.compile(
        #     **SALTUP_TRAINING_KERAS_COMPILE_ARGS
        # )
        model.compile(
            optimizer=optimizer,
            loss=loss_function,
            jit_compile=False
        )
        
        keras_callbacks = [_KerasCallbackAdapter(cb) for cb in app_callbacks]
        
        best_model_path = os.path.join(saved_models_folder_path, f'{model_output_name}_best.keras')
        last_epoch_model = os.path.join(saved_models_folder_path, f'{model_output_name}_last_epoch.keras')

        # TODO @S0nFra: Allow possibility to define arguments via SaltupEnv
        # SALTUP_TRAINING_KERAS_FIT_ARGS = {
        #     "validation_data": val_gen,
        #     "epochs": epochs,
        #     "callbacks": keras_callbacks + [save_best_clbk],
        #     "class_weight": class_weight,
        #     "shuffle": SaltupEnv.SALTUP_KERAS_TRAIN_SHUFFLE,
        #     "verbose": SaltupEnv.SALTUP_KERAS_TRAIN_VERBOSE
        # }
        # SALTUP_TRAINING_KERAS_FIT_ARGS.update(SaltupEnv.SALTUP_TRAINING_KERAS_FIT_ARGS)
        # history = model.fit(
        #     train_gen,
        #     **SALTUP_TRAINING_KERAS_FIT_ARGS
        # )

        save_best_clbk = tf.keras.callbacks.ModelCheckpoint(filepath=best_model_path, save_best_only=True)
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=keras_callbacks + [save_best_clbk],
            class_weight=class_weight,
            shuffle=SaltupEnv.SALTUP_KERAS_TRAIN_SHUFFLE,
            verbose=SaltupEnv.SALTUP_KERAS_TRAIN_VERBOSE
        )

        # Plotting
        def plot_history(data, filename, ylabel):
            fig = plt.figure(figsize=(10, 10))
            plt.plot(data, '.-')
            plt.ylabel(ylabel)
            plt.xlabel('epoch')
            plt.savefig(os.path.join(saved_models_folder_path, filename + '_plot.png'), bbox_inches='tight')
            plt.close(fig)

        # TODO: Generalize to handle Calssification and Object Detection
        # plot_history(history.history['loss'], 'history_loss', 'Loss')
        # plot_history(history.history['accuracy'], 'history_accuracy', 'Accuracy')

        model.save(last_epoch_model)
        print('Saved trained model at {} '.format(best_model_path))
        return best_model_path

    elif isinstance(model, torch.nn.Module):
        
        context = CallbackContext(model,
                                  epochs=epochs,
                                  batch_size=train_gen.dataset.batch_size,
                                  best_model=None,
                                  best_epoch=0,
                                  misc={})

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        best_model_path = os.path.join(saved_models_folder_path, f'{model_output_name}_best.pt')
        last_epoch_model = os.path.join(saved_models_folder_path, f'{model_output_name}_last_epoch.pt')
        for callback in app_callbacks:
            callback.on_train_begin(context)

            best_val_loss = float('inf')

            for epoch in range(epochs):
                model.train()
                running_loss = 0.0
                correct = 0
                total = 0

                if SaltupEnv.SALTUP_KERAS_TRAIN_VERBOSE:
                    train_iter = tqdm(train_gen, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
                else:
                    train_iter = train_gen

                for inputs, targets in train_iter:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # For classification, convert one-hot to class indices; for others, leave as is.
                    if targets.ndim == 2 and targets.shape[1] > 1 and targets.dtype in (torch.float32, torch.float64):
                        # Likely one-hot encoded classification
                        targets = torch.argmax(targets, dim=1)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = loss_function(outputs, targets)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == targets).sum().item()
                    total += targets.size(0)

                    if SaltupEnv.SALTUP_KERAS_TRAIN_VERBOSE:
                        train_iter.set_postfix(loss=loss.item())

                epoch_loss = running_loss / total
                epoch_accuracy = correct / total

                # Validation
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, targets in val_gen:
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        # Handle both one-hot encoded and non-one-hot encoded labels
                        # For classification, convert one-hot to class indices; for others, leave as is.
                        if targets.ndim == 2 and targets.shape[1] > 1 and targets.dtype in (torch.float32, torch.float64):
                            # Likely one-hot encoded classification
                            targets = torch.argmax(targets, dim=1)
                        
                        outputs = model(inputs)
                        loss = loss_function(outputs, targets)

                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == targets).sum().item()
                        val_total += targets.size(0)

                epoch_val_loss = val_loss / val_total
                epoch_val_accuracy = val_correct / val_total

                # Print summary at end of epoch if verbose
                if SaltupEnv.SALTUP_KERAS_TRAIN_VERBOSE:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {epoch_val_loss:.4f}")

                # Update context
                context.loss = epoch_loss
                context.accuracy = epoch_accuracy
                context.val_loss = epoch_val_loss
                context.val_accuracy = epoch_val_accuracy

                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    context.best_model = copy.deepcopy(model)
                    context.best_epoch = epoch
                    scripted = torch.jit.script(model.cpu())
                    scripted.save(best_model_path)

                callback.on_epoch_end(epoch, context)

            callback.on_train_end(context)
            scripted = torch.jit.script(model.cpu())
            scripted.save(last_epoch_model)

        return best_model_path         
            
def training(
    train_DataGenerator:BaseDatagenerator,
    model:Union[tf.keras.Model, torch.nn.Module],
    loss_function:callable,
    optimizer:callable,
    epochs:int,
    batch_size:int,           
    output_dir:str,
    validation:Union[list[float], BaseDatagenerator]=[0.2, 0.8],
    kfold_param:dict = {'enable':True, 'split':[0.2, 0.8]},
    scheduler:callable=None,
    model_output_name:str=None,
    training_callback:list=[],
    **kwargs
) -> dict:
    """
    Classification training function.

    Args:
        train_DataGenerator (BaseDatagenerator): Training data generator.
        model (Union[tf.keras.Model, torch.nn.Module]): Model instance (Keras or PyTorch).
        loss_function (callable): Function that returns a loss function instance.
        optimizer (callable): Function that returns an optimizer instance.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        output_dir (str): Directory to save outputs and models.
        validation (Union[list[float], BaseDatagenerator], optional): Validation split or generator. Defaults to [0.2, 0.8].
        kfold_param (dict, optional): K-fold parameters. Defaults to {'enable':True, 'split':[0.2, 0.8]}.
        scheduler (callable, optional): Scheduler for the optimizer. Defaults to None.
        model_output_name (str, optional): Name for the output model. Defaults to None.
        training_callback (list, optional): List of callbacks for training. Defaults to [].
        **kwargs: Additional keyword arguments.

    Returns:
        list: List with paths of files generate during trainig.
    """
    results_dict = {
        'kfolds': kfold_param['enable'],
        'models_paths': [],
    }
    
    parameters_path = os.path.join(output_dir, "options.txt")
    with open(parameters_path, mode='w') as f:
        if kfold_param['enable']:
            f.write("The number of folds: {}".format(len(kfold_param['split'])))
        f.write("\nThe number of epochs: {}".format(epochs))
        f.write("\nThe batch size: {}".format(batch_size))
    
    if kfold_param['enable']:
        kfolds = train_DataGenerator.split(kfold_param['split'])
        acc_per_fold = []
        best_accuracy = -1
        golden_model_folder = os.path.join(output_dir, 'golden_model_folder')
        # TODO @Marc: Add function for validate golden model and add golden model folder to results_dict
        for fold in range(len(kfolds)):
            print(f"\n--- Fold {fold + 1} ---")
            train_generator, val_generator = kfoldGenerator(kfolds, fold).get_fold_generators()
            
            print("Number of training samples:", train_generator.dataloader.get_num_samples_per_class())
            print("Number of validation samples:", val_generator.dataloader.get_num_samples_per_class())
            
            fold_model = model
            fold_model_optimizer = optimizer
            fold_model_loss = loss_function
            if isinstance(fold_model, torch.nn.Module):
                train_generator = pytorch_DataGenerator(train_generator, batch_size=train_generator.batch_size, shuffle=True)
                val_generator = pytorch_DataGenerator(val_generator, batch_size=val_generator.batch_size, shuffle=True)
            
            fold_path = os.path.join(output_dir,'k_'+str(fold))
            os.mkdir(fold_path)
            print("\n--- Model training ---")
            classification_class_weight = kwargs.get('classification_class_weight', None)
            trained_model_path = _train_model(
                model=fold_model,
                train_gen=train_generator,
                val_gen=val_generator,
                output_dir=fold_path,
                epochs=epochs,
                loss_function=fold_model_loss,
                optimizer=fold_model_optimizer,
                scheduler=scheduler,
                model_output_name=model_output_name,
                class_weight=classification_class_weight,
                app_callbacks=training_callback
            )
            print("\n--- Model fold evaluation ---")
            current_fold_folder = os.path.dirname(trained_model_path)
            current_accuracy = evaluate_model(trained_model_path, val_generator, current_fold_folder, loss_function)
            acc_per_fold.append(current_accuracy)
            txt_performance_file_name = "performances.txt"
            
            txt_performance_file_path = os.path.join(current_fold_folder, txt_performance_file_name)
            with open(txt_performance_file_path, mode='w') as f:
                f.write("The accuracy of the test of the fold " + str(fold) + ": "+str(current_accuracy))
            
            if  current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                fold_number = fold
                golden_model_path = trained_model_path

            golden_fold_folder = os.path.dirname(golden_model_path)
            shutil.copytree(golden_fold_folder, golden_model_folder, dirs_exist_ok=True)
            
            if isinstance(fold_model, tf.keras.Model):
                golden_model_name = os.path.basename(golden_model_path).replace('.keras', '')       
                onnx_model_path = os.path.join(golden_model_folder, f'{golden_model_name}.onnx')
                onnx_model_path, _ = convert_keras_to_onnx(golden_model_path, onnx_model_path)
                results_dict['models_paths'].append(onnx_model_path)
                
                tflite_golden_model_path = os.path.join(golden_model_folder, f'{golden_model_name}.tflite')
                tflite_model_path = tflite_conversion(
                    golden_model_path, 
                    tflite_golden_model_path
                )
                results_dict['models_paths'].append(tflite_model_path)
    else:
        if isinstance(validation, list):
            val_datagenerator, train_datagenerator  = train_DataGenerator.split(validation)
        else:
            val_datagenerator = validation
            train_datagenerator = train_DataGenerator
        
        training_model = model
        if isinstance(training_model, torch.nn.Module):
            print("PyTorch model detected.")
            if loss_function is None or optimizer is None:
                    raise ValueError("For PyTorch models, both `loss_function` and `optimizer` must be provided.")
            train_datagenerator = pytorch_DataGenerator(train_datagenerator, batch_size=batch_size, shuffle=True)
            val_datagenerator = pytorch_DataGenerator(val_datagenerator, batch_size=batch_size, shuffle=True)

        print("\n--- Model training ---")
        classification_class_weight = kwargs.get('classification_class_weight', None)
        model_path = _train_model(
            model=training_model, 
            train_gen=train_datagenerator, 
            val_gen=val_datagenerator, 
            output_dir=output_dir, 
            loss_function=loss_function,
            epochs=epochs,
            optimizer=optimizer,
            scheduler=scheduler, 
            model_output_name=model_output_name,
            class_weight=classification_class_weight,
            app_callbacks=training_callback
        )
        results_dict['models_paths'].append(model_path)
        if isinstance(training_model, tf.keras.Model):
            model_folder = os.path.dirname(model_path)
            model_name = os.path.basename(model_path).replace('.keras', '')
            onnx_model_path = os.path.join(model_folder, f'{model_name}.onnx')
            model_proto, keras_model = convert_keras_to_onnx(model_path, onnx_model_path)
            results_dict['models_paths'].append(onnx_model_path)
            
            tflite_model_path = os.path.join(model_folder, f'{model_name}.tflite')
            tflite_model_path = tflite_conversion(
                model_path, 
                tflite_model_path
            )
            results_dict['models_paths'].append(tflite_model_path)
    return results_dict
