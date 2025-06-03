from saltup.ai.classification.datagenerator import keras_ClassificationDataGenerator, ClassificationDataloader, pytorch_ClassificationDataGenerator
from saltup.utils.data.image.image_utils import Image, ColorMode
from saltup.ai.keras_utils.keras_to_tflite_quantization import *
from saltup.ai.keras_utils.keras_to_onnx import *
from saltup.ai.training.training_callbacks import *
from saltup.ai.classification.evaluate import evaluate_model
from saltup.ai.base_dataformat.base_datagen import BaseDatagenerator, kfoldGenerator
from typing import Iterator, Tuple, Any, List, Tuple, Union
import os
import shutil
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import numpy as np
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import albumentations as A
from saltup.ai.object_detection.utils.metrics import Metric
import copy
import gc
import torch
from torch.utils.data import DataLoader as pytorch_DataGenerator
import tensorflow as tf

class _KerasCallbackAdapter(tf.keras.callbacks.Callback):
    def __init__(self, custom_callback: BaseCallback):
        super().__init__()
        self.cb = custom_callback

    def on_epoch_end(self, epoch, metrics=None):
        self.cb.on_epoch_end(epoch, metrics=self.cb.filter_metrics(metrics))

def _train_model(model:Union[tf.keras.models.Sequential, torch.nn.Module],
                train_gen:BaseDatagenerator,
                val_gen:BaseDatagenerator,
                output_dir:str,
                epochs:int,
                loss_function:Union[tf.keras.losses.Loss, torch.nn.Module],
                optimizer:Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer],
                scheduler:Union[torch.optim.lr_scheduler._LRScheduler, None],
                model_output_name:str=None,
                class_weight:dict=None,
                app_callbacks=[]) -> str:

    """Train the model.
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
    pytorch_callbacks = [cb for cb in app_callbacks if isinstance(cb, BaseCallback)]
    if model_output_name is None:
        model_output_name = 'model'
    if isinstance(model, tf.keras.Model):
        # === Keras model ===
        if optimizer is None or loss_function is None:
            raise ValueError("For Keras models, both `optimizer` and `loss_function` must be provided.")
        
        model.compile(optimizer=optimizer,
                      loss=loss_function,
                      metrics=['accuracy'])
        
        keras_callbacks = [_KerasCallbackAdapter(cb) for cb in app_callbacks]
        
        saved_models_folder_path = os.path.join(output_dir, "saved_models")
        os.makedirs(saved_models_folder_path, exist_ok=True)
        
        b_v_model_path = os.path.join(saved_models_folder_path, f'{model_output_name}_best_v_.keras')
        b_t_model_path = os.path.join(saved_models_folder_path, f'{model_output_name}_best_t_.keras')
        b_model_path = os.path.join(saved_models_folder_path, f'{model_output_name}_last_epoch_.keras')

        b_v = tf.keras.callbacks.ModelCheckpoint(filepath=b_v_model_path, save_best_only=True)
        b_t = tf.keras.callbacks.ModelCheckpoint(filepath=b_t_model_path, monitor='loss', save_best_only=True)
        history = model.fit(train_gen,
                            validation_data=val_gen,
                            epochs=epochs,
                            callbacks=[keras_callbacks, b_v, b_t],
                            class_weight=class_weight,
                            shuffle=True)

        # Plotting
        def plot_history(data, filename, ylabel):
            fig = plt.figure(figsize=(10, 10))
            plt.plot(data, '.-')
            plt.ylabel(ylabel)
            plt.xlabel('epoch')
            plt.savefig(os.path.join(saved_models_folder_path, filename + '_plot.png'), bbox_inches='tight')
            plt.close(fig)

        plot_history(history.history['loss'], 'history_loss', 'Loss')
        plot_history(history.history['accuracy'], 'history_accuracy', 'Accuracy')

        model.save(b_model_path)
        print('Saved trained model at {} '.format(b_v_model_path))
        return b_v_model_path

    elif isinstance(model, torch.nn.Module):
        # === PyTorch model ===
        pytorch_callbacks = [cb for cb in app_callbacks if isinstance(cb, BaseCallback)]
        saved_models_folder_path = os.path.join(output_dir, "saved_models")
        os.makedirs(saved_models_folder_path, exist_ok=True)

        b_v_model_path = os.path.join(saved_models_folder_path, f'{model_output_name}_best_v_.pt')
        b_t_model_path = os.path.join(saved_models_folder_path, f'{model_output_name}_best_t_.pt')
        b_model_path = os.path.join(saved_models_folder_path, f'{model_output_name}_last_epoch_.pt')

        best_val_loss = float('inf')
        best_train_loss = float('inf')

        history = {
            'loss': [],
            'val_loss': [],
            'accuracy': [],
            'val_accuracy': []
        }

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        for epoch in range(epochs):
            model.train()
            running_loss, correct, total = 0, 0, 0
            for x_batch, y_batch in train_gen:
                x_batch, y_batch = x_batch.to(device).float(), y_batch.to(device)
                
                # Handle both one-hot encoded and non-one-hot encoded labels
                if y_batch.ndim == 1:  # If labels are not one-hot encoded
                    labels = y_batch
                elif y_batch.ndim == 2:  # If labels are one-hot encoded
                    labels = torch.argmax(y_batch, dim=1)
                else:
                    raise ValueError(f"Unexpected label shape: {y_batch.shape}")

                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / len(train_gen.dataset)
            train_acc = correct / len(train_gen.dataset)
            
            # Validation
            model.eval()
            val_loss, val_correct = 0, 0
            with torch.no_grad():
                for x_batch, y_batch in val_gen:
                    x_batch, y_batch = x_batch.to(device).float(), y_batch.to(device)
                    
                    # Handle both one-hot encoded and non-one-hot encoded labels
                    if y_batch.ndim == 1:  # If labels are not one-hot encoded
                        labels = y_batch
                    elif y_batch.ndim == 2:  # If labels are one-hot encoded
                        labels = torch.argmax(y_batch, dim=1)
                    else:
                        raise ValueError(f"Unexpected label shape: {y_batch.shape}")

                    outputs = model(x_batch)
                    loss = loss_function(outputs, labels)
                    val_loss += loss.item()
                    val_correct += (torch.argmax(outputs, 1) == labels).sum().item()

            val_loss /= len(val_gen.dataset)
            val_acc = val_correct / len(val_gen.dataset)
                
            history['loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['accuracy'].append(train_acc)
            history['val_accuracy'].append(val_acc)

            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            for cb in pytorch_callbacks:
                cb.on_epoch_end(epoch, metrics={
                'loss': round(train_loss, 4),
                'val_loss': round(val_loss, 4),
                'accuracy': round(train_acc, 4),
                'val_accuracy': round(val_acc, 4)
            })
            # Save best models (JIT)
            if val_loss < best_val_loss:
                scripted = torch.jit.script(model.cpu())
                scripted.save(b_v_model_path)
                model.to(device)
                best_val_loss = val_loss

            if train_loss < best_train_loss:
                scripted = torch.jit.script(model.cpu())
                scripted.save(b_t_model_path)
                model.to(device)
                best_train_loss = train_loss

            if scheduler:
                scheduler.step(val_loss)

        # Save last epoch model
        scripted = torch.jit.script(model.cpu())
        scripted.save(b_model_path)
        model.to(device)

        def save_plot(data_key, filename, ylabel):
            fig = plt.figure(figsize=(10, 10))
            plt.plot(history[data_key], '.-', label='train')
            plt.plot(history[f'val_{data_key}'], '.-', label='val')
            plt.ylabel(ylabel)
            plt.xlabel('epoch')
            plt.legend(loc='upper right')
            plt.savefig(os.path.join(saved_models_folder_path, filename + '_plot.png'), bbox_inches='tight')
            plt.close(fig)

        save_plot('loss', 'history_loss', 'Loss')
        save_plot('accuracy', 'history_accuracy', 'Accuracy')

        print(f"Saved scripted model at {b_model_path}")
        return b_model_path         
            
def training(train_DataGenerator:BaseDatagenerator,
             model_fn:callable,
             loss_function:callable,
             optimizer:callable,
             epochs:int,
             batch_size:int,           
             output_dir:str,
             validation_split:list[float]=[0.2, 0.8],
             kfold_param:dict = {'enable':True, 'split':[0.2, 0.8]},
             scheduler:callable=None,
             model_output_name:str=None,
             training_callback:list=[],
             test_Datagenerator:BaseDatagenerator=None,
             quantization_param:dict={'enable':False, 'quantize_input':True, 'quantize_output':True},
             **kwargs) -> str:
    """Classification training function.
    
    Args:
        Dataloader (ClassificationDataloader): Dataloader object.
        model_fn(calable): model name to be trained.
        folds (int): number of folds for cross-validation.
        epochs (int): number of epochs for training.
        training_callback (_type_, optional): callback function for training. Defaults to train_model.
        output_dir (str, optional): _description_. Defaults to None.
        loss_function (Union[tf.keras.losses.Loss, torch.nn.Module], optional): loss function for training
        optimizer (Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer], optional): optimizer for training
        batch_size (int, optional): batch size for training. Defaults to None.
        scheduler (callable, optional): scheduler for the optimizer. Defaults to None..
    """
    #images_paths = train_Dataloader.image_paths
    #labels = train_Dataloader.labels
    parameters_name = "options.txt"
    parameters_path = os.path.join(output_dir, parameters_name)
    with open(parameters_path, mode='w') as f:
        if kfold_param['enable']:
            f.write("The number of folds: {}".format(len(kfold_param['split'])))
        f.write("\nThe model name: {}".format(model_fn.__name__))
        f.write("\nThe number of epochs: {}".format(epochs))
        f.write("\nThe batch size: {}".format(batch_size))
    f.close()
    if kfold_param['enable']:
        #kfold = KFold(n_splits=folds, shuffle=True)
        kfolds = train_DataGenerator.split(kfold_param['split'])
        acc_per_fold = []
        best_accuracy = -1
        golden_model_folder = os.path.join(output_dir, 'golden_model_folder')
        for fold in range(len(kfolds)):
            print(f"\n--- Fold {fold + 1} ---")
            train_generator, val_generator = kfoldGenerator(kfolds, fold).get_fold_generators()
            
            print("Number of training samples:", train_generator.dataloader.get_num_samples_per_class())
            print("Number of validation samples:", val_generator.dataloader.get_num_samples_per_class())
            
            fold_model = model_fn()
            fold_model_optimizer = optimizer()
            fold_model_loss = loss_function()
            if isinstance(fold_model, torch.nn.Module):
                train_generator = pytorch_DataGenerator(train_generator, batch_size=train_generator.batch_size, shuffle=True)
                val_generator = pytorch_DataGenerator(val_generator, batch_size=val_generator.batch_size, shuffle=True)
                test_Datagenerator = pytorch_DataGenerator(test_Datagenerator, batch_size=test_Datagenerator.batch_size, shuffle=True)
            
            fold_path = os.path.join(output_dir,'k_'+str(fold))
            os.mkdir(fold_path)
            print("\n--- Model training ---")
            classification_class_weight = kwargs.get('classification_class_weight', None)
            trained_model_path = _train_model(model=fold_model, 
                                              train_gen=train_generator, 
                                              val_gen=val_generator, 
                                              output_dir=fold_path, 
                                              epochs=epochs,
                                              loss_function=fold_model_loss,
                                              optimizer=fold_model_optimizer,
                                              scheduler=scheduler, 
                                              model_output_name=model_output_name,
                                              class_weight=classification_class_weight,
                                              app_callbacks=training_callback)
            
            print("\n--- Model fold evaluation ---")
            current_fold_folder = os.path.dirname(trained_model_path)
            current_accuracy = evaluate_model(trained_model_path, val_generator, current_fold_folder, loss_function)
            acc_per_fold.append(current_accuracy)
            txt_performance_file_name = "performances.txt"
            
            txt_performance_file_path = os.path.join(current_fold_folder, txt_performance_file_name)
            with open(txt_performance_file_path, mode='w') as f:
                f.write("The accuracy of the test of the fold " + str(fold) + ": "+str(current_accuracy))
            f.close()
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
                
                example_samples = np.array([train_generator[i][0] for i in range(50)])
                example_samples = np.reshape(example_samples, (example_samples.shape[0] * example_samples.shape[1],   example_samples.shape[2], example_samples.shape[3], example_samples.shape[4]))
                tflite_golden_model_path = os.path.join(golden_model_folder, f'{golden_model_name}.tflite')
                tflite_model_path = tflite_quantization(golden_model_path, 
                                                            tflite_golden_model_path,
                                                            example_samples,
                                                            False,
                                                            False)
            
            if quantization_param['enable'] and fold == len(kfold_param['split'])-1:
                if not golden_model_path.endswith('.keras'):
                    raise ValueError("The model is not a keras model.")
                golden_fold_folder = os.path.dirname(golden_model_path)
                #shutil.copytree(golden_fold_folder, golden_model_folder)
                
                print("calibration data shape",example_samples.shape)
                model_name = os.path.basename(golden_model_path)
                model_name = model_name.split('.')[0]
                quantized_model_path = os.path.join(golden_model_folder, 'quantization', f'quantized_{model_name}.tflite')
                
                tflite_model_path = tflite_quantization(golden_model_path, 
                                                        quantized_model_path,
                                                        example_samples,
                                                        quantization_param['quantize_input'],
                                                        quantization_param['quantize_output'])
                
                if test_Datagenerator is not None:  
                    accuracy_of_the_keras_golden_model = evaluate_model(golden_model_path,test_Datagenerator,golden_model_folder)
                    tflite_quantized_folder = os.path.dirname(tflite_model_path)
                    accuracy_of_the_tflite_golden_model = evaluate_model(tflite_model_path,test_Datagenerator, tflite_quantized_folder)
                
                golden_performance_file_path = os.path.join(golden_model_folder, txt_performance_file_name)
                with open(golden_performance_file_path, mode='w') as f:
                    f.write("The golden model's path:{}".format(golden_model_path))
                    f.write("\nThe keras golden model is trained with the fold " + str(fold_number))
                    f.write("\nThe accuracy of the golden model: "+ str(best_accuracy))
                    if test_Datagenerator is not None:
                        f.write("\nThe accuracy of the quantized tflite golden model on test:{}".format(accuracy_of_the_tflite_golden_model))
                f.close()
            
            elif fold == len(kfold_param['split'])-1 and test_Datagenerator is not None:
                print('\n\nEvaluation on Test set:',best_accuracy,'\n\n')
                accuracy_of_the_golden_model = evaluate_model(golden_model_path,test_Datagenerator, loss_function)
                
            if fold ==len(kfold_param['split'])-1:
                print('------------------------------------------------------------------------')
                print('Score per fold')
                for i in range(0, len(acc_per_fold)):
                    print('------------------------------------------------------------------------')
                    print(f'> Fold {i} - Accuracy: {acc_per_fold[i]}%')
                print('------------------------------------------------------------------------')
                print('Average scores for all folds:')
                print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')

                print('------------------------------------------------------------------------')
                global_performance_file_name = "Global statistical info.txt.txt"
                global_performance_file_path = os.path.join(output_dir, global_performance_file_name)
                with open(global_performance_file_path, mode='w') as f:
                    f.write("The average accuracy for all folds:{}".format(np.mean(acc_per_fold)))
                    f.write("\nThe variance of the accuracies for all folds:{}".format(np.std(acc_per_fold)))
    else:
        
        val_datagenerator, train_datagenerator  = train_DataGenerator.split(validation_split)
        
        print("Number of training samples:", train_datagenerator.dataloader.get_num_samples_per_class())
        print("Number of validation samples:", val_datagenerator.dataloader.get_num_samples_per_class())
        
        training_model = model_fn()
        optimizer = optimizer()
        loss_function = loss_function()
        if isinstance(training_model, torch.nn.Module):
            print("PyTorch model detected.")
            if loss_function is None or optimizer is None:
                    raise ValueError("For PyTorch models, both `loss_function` and `optimizer` must be provided.")
            train_datagenerator = pytorch_DataGenerator(train_datagenerator, batch_size=batch_size, shuffle=True)
            val_datagenerator = pytorch_DataGenerator(val_datagenerator, batch_size=batch_size, shuffle=True)
            if test_Datagenerator is not None:
                test_Datagenerator = pytorch_DataGenerator(test_Datagenerator, batch_size=batch_size, shuffle=True)

        print("\n--- Model training ---")
        classification_class_weight = kwargs.get('classification_class_weight', None)
        model_path = _train_model(model=training_model, 
                                train_gen=train_datagenerator, 
                                val_gen=val_datagenerator, 
                                output_dir=output_dir, 
                                loss_function=loss_function,
                                epochs=epochs,
                                optimizer=optimizer,
                                scheduler=scheduler, 
                                model_output_name=model_output_name,
                                class_weight=classification_class_weight,
                                app_callbacks=training_callback)
        if isinstance(training_model, tf.keras.Model):
            model_folder = os.path.dirname(model_path)
            model_name = os.path.basename(model_path).replace('.keras', '')
            onnx_model_path = os.path.join(model_folder, f'{model_name}.onnx')
            model_proto, keras_model = convert_keras_to_onnx(model_path, onnx_model_path)
            
            example_samples = np.array([train_datagenerator[i][0] for i in range(50)])
            example_samples = np.reshape(example_samples, (example_samples.shape[0] * example_samples.shape[1],   example_samples.shape[2], example_samples.shape[3], example_samples.shape[4]))
                
            tflite_model_path = os.path.join(model_folder, f'{model_name}.tflite')
            tflite_model_path = tflite_quantization(model_path, 
                                                    tflite_model_path,
                                                    example_samples,
                                                    False,
                                                    False)
        txt_performance_file_name = "performances.txt"
        
        if quantization_param['enable']:
            if not model_path.endswith('.keras'):
                raise ValueError("The model is not a keras model. Only keras model is supported for quantization.")
            model_folder = os.path.dirname(model_path)
            
            example_samples = np.array([train_datagenerator[i][0] for i in range(50)])
            example_samples = np.reshape(example_samples, (example_samples.shape[0] * example_samples.shape[1],   example_samples.shape[2], example_samples.shape[3], example_samples.shape[4]))
            print("calibration data shape",example_samples.shape)
            model_name = os.path.basename(model_path)
            model_name = model_name.split('.')[0]
            tflite_model_path = os.path.join(model_folder, 'quantization', f'quantized_{model_name}.tflite')
            tflite_model_path = tflite_quantization(model_path, 
                                                    tflite_model_path,
                                                    example_samples,
                                                    quantization_param['quantize_input'],
                                                    quantization_param['quantize_output'])
            if test_Datagenerator is not None:
                accuracy_of_the_keras_model = evaluate_model(model_path,test_Datagenerator, model_folder)
            
                tflite_model_folder = os.path.dirname(tflite_model_path)
                accuracy_of_the_tflite_model = evaluate_model(tflite_model_path,test_Datagenerator, tflite_model_folder)
            
            performance_file_path = os.path.join(model_folder, txt_performance_file_name)
            with open(performance_file_path, mode='w') as f:
                f.write("The model's path:{}".format(model_path))
                f.write("\nThe accuracy of the model: "+ str(accuracy_of_the_keras_model))
                f.write("\nThe accuracy of the quantized tflite model on test:{}".format(accuracy_of_the_tflite_model))
            f.close()
        else:
            if test_Datagenerator is not None:
                model_folder = os.path.dirname(model_path)
                accuracy_of_the_model = evaluate_model(model_path,test_Datagenerator, model_folder, loss_function)
                print('\n\nEvaluation on Test set:',accuracy_of_the_model,'\n\n')
                performance_file_path = os.path.join(model_folder, txt_performance_file_name)
                with open(performance_file_path, mode='w') as f:
                    f.write("The model's path:{}".format(model_path))
                    f.write("\nThe accuracy of the model: "+ str(accuracy_of_the_model))
                f.close()