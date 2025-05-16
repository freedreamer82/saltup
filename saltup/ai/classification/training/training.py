from saltup.ai.classification.datagenerator.classification_datagen import keras_ClassificationDataGenerator, ClassificationDataloader, pytorch_ClassificationDataGenerator
from saltup.utils.data.image.image_utils import Image, ColorMode
from saltup.ai.keras_utils.keras_to_tflite_quantization import *
from saltup.ai.keras_utils.keras_to_onnx import *
from saltup.ai.classification.training.training_callbacks import *
from typing import Iterator, Tuple, Any, List, Tuple, Union
import os
import shutil
from sklearn.metrics import confusion_matrix
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
from torch.utils.data import DataLoader
import tensorflow as tf


def evaluate_model(model_path:str, 
                   test_gen:Union[keras_ClassificationDataGenerator, DataLoader],
                   output_dir:str=None,
                   loss_function:callable=None) -> float:
    """function to evaluate the model on the test set.

    Args:
        model_path (str): path to the model
        test_gen (Union[keras_ClassificationDataGenerator, pytorch_ClassificationDataGenerator]): test data generator
        test_data_dir (str): folder containing the test data
        loss_function (callable, optional): loss_function for evaluation. Only used when evaluating a PyTorch model. Defaults to None.

    Raises:
        ValueError: if the model is not a keras or pytorch model
        ValueError: if the model is not a keras model and loss_function is None

    Returns:
        float: Accuracy of the model on the test set
    """
    # Evaluate the model on the test set

    global_metric = Metric()
    if isinstance(test_gen, keras_ClassificationDataGenerator):
        class_names = test_gen.dataloader.get_classes().keys()
    elif isinstance(test_gen, DataLoader):
        class_names = test_gen.dataset.dataloader.get_classes().keys()
    
    print("The class names are:", class_names)
    metric_per_class = {i: Metric() for i in range(len(class_names))}

    # Load model once before loop
    model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_path.endswith('.keras'):
        print("\n--- Evaluate Keras model ---")
        model = tf.keras.models.load_model(model_path)

    elif model_path.endswith('.pt'):
        print("\n--- Evaluate PyTorch model ---")
        if loss_function is None:
            raise ValueError("The model is not a keras model, please provide a loss_function.")

        model = torch.jit.load(model_path) if model_path.endswith(".pt") else torch.load(model_path)
        model.eval()
        model.to(device)

    elif model_path.endswith('.tflite'):
        print("\n--- Evaluate TFLite model ---")
        tflite_interpreter = tf.lite.Interpreter(model_path=model_path)


    else:
        raise ValueError("Unsupported model type. Please use a Keras, PyTorch or TFLite model.")

    pbar = tqdm(test_gen, desc="Processing data", dynamic_ncols=True)

    all_true_labels = []
    all_pred_labels = []

    for images, labels in pbar:
        if model_path.endswith('.keras'):
            predictions = model.predict(images, verbose=0)

        elif model_path.endswith('.pt'):
            with torch.no_grad():
                X_batch = images.to(device)
                y_batch = labels.to(device)
                # Handle both one-hot encoded and non-one-hot encoded labels
                if y_batch.ndim == 1:  # If labels are not one-hot encoded
                    true_labels = y_batch
                elif y_batch.ndim == 2:  # If labels are one-hot encoded
                    true_labels = torch.argmax(y_batch, dim=1)
                else:
                    raise ValueError(f"Unexpected label shape: {y_batch.shape}")

                outputs = model(X_batch)
                loss = loss_function(outputs, true_labels)
                predictions = outputs.cpu().numpy()
                labels = true_labels.cpu().numpy()

        elif model_path.endswith('.tflite'):
            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()  
            #tflite_interpreter.resize_tensor_input(input_details[0]['index'], (images.shape[0], images.shape[1],images.shape[2],images.shape[3]))
            #tflite_interpreter.resize_tensor_input(output_details[0]['index'], (1, 4))
            tflite_interpreter.allocate_tensors()

            input_index = input_details[0]['index']
            output_index = output_details[0]['index']
            print(input_index)
            
            # Ensure the input tensor is of type FLOAT32
            if input_details[0]['dtype'] == np.float32:
                x_test_lite = images.astype(np.float32)
            else:
                scale, zero_point = input_details[0]["quantization"]
                x_test_lite = np.uint8(images / scale + zero_point)
            
            # Resize the input tensor only if necessary
            if x_test_lite.shape != tuple(input_details[0]['shape']):
                print('Resizing input tensor to:', x_test_lite.shape)
                tflite_interpreter.resize_tensor_input(input_index, x_test_lite.shape)
                tflite_interpreter.allocate_tensors()
            
            tflite_interpreter.set_tensor(input_index, x_test_lite)
            tflite_interpreter.invoke()

            predictions = tflite_interpreter.get_tensor(output_index)
            #scale, zero_point = output_details[0]['quantization']
            #predictions = (output.astype(np.float32) - zero_point) * scale
            # Ensure predictions and labels are aligned
            if len(predictions) != len(labels):
                raise ValueError(f"Mismatch between predictions ({len(predictions)}) and labels ({len(labels)})")

        # === Evaluation loop ===
        for i, pred in enumerate(predictions):
            pred_class = int(np.argmax(pred))
            true_class = int(np.argmax(labels[i]) if isinstance(labels[i], (np.ndarray, list)) else labels[i])

            all_true_labels.append(true_class)
            all_pred_labels.append(pred_class)

            if pred_class == true_class:
                # True Positive for the predicted class
                global_metric.addTP(1)
                metric_per_class[pred_class].addTP(1)
            else:
                # False Positive for the predicted class
                global_metric.addFP(1)
                metric_per_class[pred_class].addFP(1)

                # False Negative for the true class
                global_metric.addFN(1)
                metric_per_class[true_class].addFN(1)

        pbar.set_postfix(**global_metric.get_metrics())

    # ==== Confusion Matrix ====
    model_extension = os.path.splitext(model_path)[1].replace('.', '')
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        cm_path = os.path.join(output_dir, f"_{model_extension}_confusion_matrix.png")
        plt.savefig(cm_path, bbox_inches="tight")
        print(f"Confusion matrix saved at {cm_path}")
        plt.show()

    # ==== Final Report ====
    print("\n" + "="*80)
    print(f"{'METRICS SUMMARY':^80}")
    print("="*80 + "\n")
    print(f"{'Images processed:':<20} {len(test_gen.dataset) if hasattr(test_gen, 'dataset') else len(test_gen)}")

    print("\nPer class:")
    print("+" * 80)
    print(f"{'Label':<18} | {'Precision':<10} | {'Recall':<10} | {'Accuracy':<10}")
    print("-" * 80)
    for class_id, class_label in enumerate(class_names):
        metrics = metric_per_class[class_id]
        print(f"{class_label:<18} | {metrics.getPrecision():<10.4f} | {metrics.getRecall():<10.4f} | {metrics.getAccuracy():<10.4f}")

    print("\nOverall:")
    print(f"{'True Positives (TP):':<25} {global_metric.getTP()}")
    print(f"{'False Positives (FP):':<25} {global_metric.getFP()}")
    print(f"{'False Negatives (FN):':<25} {global_metric.getFN()}")
    print(f"{'Overall Precision:':<25} {global_metric.getPrecision():.4f}")
    print(f"{'Overall Recall:':<25} {global_metric.getRecall():.4f}")
    print(f"{'Overall Accuracy:':<25} {global_metric.getAccuracy():.4f}")
    print("=" * 80)

    # Free memory
    torch.cuda.empty_cache()
    gc.collect()

    return global_metric.getAccuracy()

class _KerasCallbackAdapter(tf.keras.callbacks.Callback):
    def __init__(self, custom_callback: BaseCallback):
        super().__init__()
        self.cb = custom_callback

    def on_epoch_end(self, epoch, metrics=None):
        self.cb.on_epoch_end(epoch, metrics=self.cb.filter_metrics(metrics))

def _train_model(model:Union[tf.keras.models.Sequential, torch.nn.Module],
                train_gen:Union[keras_ClassificationDataGenerator, DataLoader],
                val_gen:Union[keras_ClassificationDataGenerator, DataLoader],
                output_dir:str,
                epochs:int, 
                loss_function:Union[tf.keras.losses.Loss, torch.nn.Module],
                optimizer:Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer],
                scheduler:Union[torch.optim.lr_scheduler._LRScheduler, None],
                model_output_name:str=None,
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
    
def training(train_Dataloader:ClassificationDataloader,
             test_Dataloader:ClassificationDataloader,
             model_fn:callable,
             loss_function:Union[tf.keras.losses.Loss, torch.nn.Module],
             optimizer:Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer],
             epochs:int,
             target_size:tuple,
             batch_size:int,           
             output_dir:str,
             scheduler:callable=None,
             preprocess:callable=None,
             transform:A.Compose=None,
             model_output_name:str=None,
             folds:int=None,
             validation_split:float=0.2,
             training_callback:list=[],
             quantization:bool=False,
             quantize_input:bool=True,
             quantize_output:bool=True) -> str:
    """Classification training function.
    
    Args:
        Dataloader (ClassificationDataloader): Dataloader object.
        model_fn(calable): model name to be trained.
        folds (int): number of folds for cross-validation.
        epochs (int): number of epochs for training.
        training_callback (_type_, optional): callback function for training. Defaults to train_model.
        output_dir (str, optional): _description_. Defaults to None.
        validation_split (float, optional): _description_. Defaults to 0.2.
        loss_function (Union[tf.keras.losses.Loss, torch.nn.Module], optional): loss function for training
        optimizer (Union[tf.keras.optimizers.Optimizer, torch.optim.Optimizer], optional): optimizer for training
        target_size (tuple, optional): target size for the model. Defaults to None.
        batch_size (int, optional): batch size for training. Defaults to None.
        scheduler (callable, optional): scheduler for the optimizer. Defaults to None.
        preprocess (callable, optional): preprocessing function for the model. Defaults to None.
        transform (A.Compose, optional): augmentation function for the model. Defaults to None.
        test_Dataloader (ClassificationDataloader): Dataloader object for test data.
        quantization (bool, optional): quantization for the model. Defaults to False.
    """
    images_paths = train_Dataloader.image_paths
    parameters_name = "options.txt"
    parameters_path = os.path.join(output_dir, parameters_name)
    with open(parameters_path, mode='w') as f:
        if folds:
            f.write("The number of folds: {}".format(folds))
        f.write("\nThe training samples location: {}".format(train_Dataloader.root_dir))
        f.write("\nThe test samples location: {}".format(test_Dataloader.root_dir))
        f.write("\nThe model name: {}".format(model_fn.__name__))
        f.write("\nThe number of epochs: {}".format(epochs))
        f.write("\nThe batch size: {}".format(batch_size))
        f.write("\nThe target size: {}".format(target_size))
    f.close()
    if folds:
        kfold = KFold(n_splits=folds, shuffle=True)
        acc_per_fold = []
        best_accuracy = -1
        golden_model_folder = os.path.join(output_dir, 'golden_model_folder')
        for fold, (train_idx, val_idx) in enumerate(kfold.split(images_paths)):
            print(f"\n--- Fold {fold + 1} ---")
            
            train_dataloader = copy.deepcopy(train_Dataloader)
            val_dataloader = copy.deepcopy(train_Dataloader)
            
            train_dataloader.image_paths = [images_paths[i] for i in train_idx]
            val_dataloader.image_paths = [images_paths[i] for i in val_idx]
            print("Number of training samples:", len(train_dataloader))
            print("Number of validation samples:", len(val_dataloader))
            
            fold_model = model_fn()
            if isinstance(fold_model, tf.keras.Model):
                print("Keras model detected.")
                train_gen = keras_ClassificationDataGenerator(
                    dataloader=train_dataloader,
                    target_size=target_size,
                    num_classes=train_dataloader.get_num_classes(),
                    batch_size=batch_size,
                    preprocess=preprocess,
                    transform=transform
                )

                val_gen = keras_ClassificationDataGenerator(
                    dataloader=val_dataloader,
                    target_size=target_size,
                    num_classes=val_dataloader.get_num_classes(),
                    batch_size=batch_size,
                    preprocess=preprocess,
                    transform=None  # no augmentation
                )
                test_gen = keras_ClassificationDataGenerator(
                        dataloader=test_Dataloader,
                        target_size=target_size,
                        num_classes=test_Dataloader.get_num_classes(),
                        batch_size=batch_size,
                        preprocess=preprocess,
                        transform=None  # no augmentation
                )                
            elif isinstance(fold_model, torch.nn.Module):
                print("PyTorch model detected.")
                if loss_function is None or optimizer is None:
                    raise ValueError("For PyTorch models, both `loss_function` and `optimizer` must be provided.")

                train_gen = pytorch_ClassificationDataGenerator(
                    dataloader=train_dataloader,
                    target_size=target_size,
                    num_classes=train_dataloader.get_num_classes(),
                    batch_size=1,
                    preprocess=preprocess,
                    transform=transform
                )
                train_gen = DataLoader(train_gen, batch_size=batch_size, shuffle=True)
                val_gen = pytorch_ClassificationDataGenerator(
                    dataloader=val_dataloader,
                    target_size=target_size,
                    num_classes=val_dataloader.get_num_classes(),
                    batch_size=1,
                    preprocess=preprocess,
                    transform=None  # no augmentation
                )
                val_gen = DataLoader(val_gen, batch_size=batch_size, shuffle=True)
                test_gen = pytorch_ClassificationDataGenerator(
                    dataloader=test_Dataloader,
                    target_size=target_size,
                    num_classes=test_Dataloader.get_num_classes(),
                    batch_size=batch_size,
                    preprocess=preprocess,
                    transform=None  # no augmentation
                )
                test_gen = DataLoader(test_gen, batch_size=batch_size, shuffle=True)
            else:
                raise TypeError("Unsupported model type. Expected Keras or PyTorch model.")
            
                
            fold_path = os.path.join(output_dir,'k_'+str(fold))
            os.mkdir(fold_path)
            print("\n--- Model training ---")
            trained_model_path = _train_model(model=fold_model, 
                                              train_gen=train_gen, 
                                              val_gen=val_gen, 
                                              output_dir=fold_path, 
                                              epochs=epochs,
                                              
                                              loss_function=loss_function,
                                              optimizer=optimizer,
                                              scheduler=scheduler, 
                                              model_output_name=model_output_name,
                                              app_callbacks=training_callback)
            
            print("\n--- Model fold evaluation ---")
            current_fold_folder = os.path.dirname(trained_model_path)
            current_accuracy = evaluate_model(trained_model_path, val_gen, current_fold_folder, loss_function)
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
                
                example_samples = np.array([train_gen[i][0] for i in range(50)])
                example_samples = np.reshape(example_samples, (example_samples.shape[0] * example_samples.shape[1],   example_samples.shape[2], example_samples.shape[3], example_samples.shape[4]))
                tflite_golden_model_path = os.path.join(golden_model_folder, f'{golden_model_name}.tflite')
                tflite_model_path = tflite_quantization(golden_model_path, 
                                                            tflite_golden_model_path,
                                                            example_samples,
                                                            False,
                                                            False)
            
            if quantization and fold == folds-1:
                if not golden_model_path.endswith('.keras'):
                    raise ValueError("The model is not a keras model.")
                golden_fold_folder = os.path.dirname(golden_model_path)
                shutil.copytree(golden_fold_folder, golden_model_folder)
                
                #example_samples = np.array([test_gen[i][0] for i in range(50)])
                #example_samples = np.reshape(example_samples, (example_samples.shape[0] * example_samples.shape[1],   example_samples.shape[2], example_samples.shape[3], example_samples.shape[4]))
                print("calibration data shape",example_samples.shape)
                model_name = os.path.basename(golden_model_path)
                model_name = model_name.split('.')[0]
                quantized_model_path = os.path.join(golden_model_folder, 'quantization', f'quantized_{model_name}.tflite')
                #tflite_golden_model_path = os.path.join(golden_model_folder, f'{model_name}.tflite')
                tflite_model_path = tflite_quantization(golden_model_path, 
                                                        quantized_model_path,
                                                        example_samples,
                                                        quantize_input,
                                                        quantize_output)
                
            
                accuracy_of_the_keras_golden_model = evaluate_model(golden_model_path,test_gen,golden_model_folder)
                tflite_quantized_folder = os.path.dirname(tflite_model_path)
                accuracy_of_the_tflite_golden_model = evaluate_model(tflite_model_path,test_gen, tflite_quantized_folder)
                
                golden_performance_file_path = os.path.join(golden_model_folder, txt_performance_file_name)
                with open(golden_performance_file_path, mode='w') as f:
                    f.write("The golden model's path:{}".format(golden_model_path))
                    f.write("\nThe keras golden model is trained with the fold " + str(fold_number))
                    f.write("\nThe accuracy of the golden model: "+ str(best_accuracy))
                    f.write("\nThe accuracy of the quantized tflite golden model on test:{}".format(accuracy_of_the_tflite_golden_model))
                f.close()
            
            elif fold == folds-1:
                print('\n\nEvaluation on Test set:',best_accuracy,'\n\n')
                accuracy_of_the_golden_model = evaluate_model(golden_model_path,test_gen, loss_function)
                
            if fold ==folds-1:
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
        val_split = int(validation_split * len(train_Dataloader))
        train_dataloader = copy.deepcopy(train_Dataloader)
        val_dataloader = copy.deepcopy(train_Dataloader)
        train_dataloader.image_paths = images_paths[val_split:]
        val_dataloader.image_paths = images_paths[:val_split]
        print("Number of training samples:", len(train_dataloader))
        print("Number of validation samples:", len(val_dataloader))
        
        training_model = model_fn()
        if isinstance(training_model, tf.keras.Model):
            print("Keras model detected.")
            train_gen = keras_ClassificationDataGenerator(
                dataloader=train_dataloader,
                target_size=target_size,
                num_classes=train_dataloader.get_num_classes(),
                batch_size=batch_size,
                preprocess=preprocess,
                transform=transform
            )

            val_gen = keras_ClassificationDataGenerator(
                dataloader=val_dataloader,
                target_size=target_size,
                num_classes=val_dataloader.get_num_classes(),
                batch_size=batch_size,
                preprocess=preprocess,
                transform=None  # no augmentation
            )
            test_gen = keras_ClassificationDataGenerator(
                dataloader=test_Dataloader,
                target_size=target_size,
                num_classes=test_Dataloader.get_num_classes(),
                batch_size=batch_size,
                preprocess=preprocess,
                transform=None  # no augmentation
            )
        
        elif isinstance(training_model, torch.nn.Module):
            print("PyTorch model detected.")
            if loss_function is None or optimizer is None:
                    raise ValueError("For PyTorch models, both `loss_function` and `optimizer` must be provided.")
            train_gen = pytorch_ClassificationDataGenerator(
                dataloader=train_dataloader,
                target_size=target_size,
                num_classes=train_dataloader.get_num_classes(),
                batch_size=1,
                preprocess=preprocess,
                transform=transform
            )
            train_gen = DataLoader(train_gen, batch_size=batch_size, shuffle=True)
            val_gen = pytorch_ClassificationDataGenerator(
                dataloader=val_dataloader,
                target_size=target_size,
                num_classes=val_dataloader.get_num_classes(),
                batch_size=1,
                preprocess=preprocess,
                transform=None  # no augmentation
            )
            val_gen = DataLoader(val_gen, batch_size=batch_size, shuffle=True)
            test_gen = pytorch_ClassificationDataGenerator(
                dataloader=test_Dataloader,
                target_size=target_size,
                num_classes=test_Dataloader.get_num_classes(),
                batch_size=1,
                preprocess=preprocess,
                transform=None  # no augmentation
            )
            test_gen = DataLoader(test_gen, batch_size=batch_size, shuffle=True)
        else:
            raise TypeError("Unsupported model type. Expected Keras or PyTorch model.")

        print("\n--- Model training ---")
        model_path = _train_model(model=training_model, 
                                train_gen=train_gen, 
                                val_gen=val_gen, 
                                output_dir=output_dir, 
                                loss_function=loss_function,
                                epochs=epochs,
                                optimizer=optimizer,
                                scheduler=scheduler, 
                                model_output_name=model_output_name,
                                app_callbacks=training_callback)
        if isinstance(training_model, tf.keras.Model):
            model_folder = os.path.dirname(model_path)
            model_name = os.path.basename(model_path).replace('.keras', '')
            onnx_model_path = os.path.join(model_folder, f'{model_name}.onnx')
            model_proto, keras_model = convert_keras_to_onnx(model_path, onnx_model_path)
            
            example_samples = np.array([train_gen[i][0] for i in range(50)])
            example_samples = np.reshape(example_samples, (example_samples.shape[0] * example_samples.shape[1],   example_samples.shape[2], example_samples.shape[3], example_samples.shape[4]))
                
            tflite_model_path = os.path.join(model_folder, f'{model_name}.tflite')
            tflite_model_path = tflite_quantization(model_path, 
                                                    tflite_model_path,
                                                    example_samples,
                                                    False,
                                                    False)
        txt_performance_file_name = "performances.txt"
        
        if quantization:
            if not model_path.endswith('.keras'):
                raise ValueError("The model is not a keras model. Only keras model is supported for quantization.")
            model_folder = os.path.dirname(model_path)
            
            example_samples = np.array([test_gen[i][0] for i in range(50)])
            example_samples = np.reshape(example_samples, (example_samples.shape[0] * example_samples.shape[1],   example_samples.shape[2], example_samples.shape[3], example_samples.shape[4]))
            print("calibration data shape",example_samples.shape)
            model_name = os.path.basename(model_path)
            model_name = model_name.split('.')[0]
            tflite_model_path = os.path.join(model_folder, 'quantization', f'quantized_{model_name}.tflite')
            tflite_model_path = tflite_quantization(model_path, 
                                                    tflite_model_path,
                                                    example_samples,
                                                    quantize_input,
                                                    quantize_output)
            
            accuracy_of_the_keras_model = evaluate_model(model_path,test_gen, model_folder)
            
            tflite_model_folder = os.path.dirname(tflite_model_path)
            accuracy_of_the_tflite_model = evaluate_model(tflite_model_path,test_gen, tflite_model_folder)
            
            performance_file_path = os.path.join(model_folder, txt_performance_file_name)
            with open(performance_file_path, mode='w') as f:
                f.write("The model's path:{}".format(model_path))
                f.write("\nThe accuracy of the model: "+ str(accuracy_of_the_keras_model))
                f.write("\nThe accuracy of the quantized tflite model on test:{}".format(accuracy_of_the_tflite_model))
            f.close()
        else:
            model_folder = os.path.dirname(model_path)
            accuracy_of_the_model = evaluate_model(model_path,test_gen, model_folder, loss_function)
            print('\n\nEvaluation on Test set:',accuracy_of_the_model,'\n\n')
            performance_file_path = os.path.join(model_folder, txt_performance_file_name)
            with open(performance_file_path, mode='w') as f:
                f.write("The model's path:{}".format(model_path))
                f.write("\nThe accuracy of the model: "+ str(accuracy_of_the_model))
            f.close()