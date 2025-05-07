from saltup.ai.classification.datagenerator.classification_datagen import keras_ClassificationDataGenerator, ClassificationDataloader
from saltup.utils.data.image.image_utils import Image, ColorMode
from typing import Iterator, Tuple, Any, List, Tuple, Union
import os
import cv2
import shutil
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import albumentations as A
from saltup.ai.object_detection.utils.metrics import Metric
import copy
from datetime import datetime
import torch
import tensorflow as tf
from saltup.ai.keras_utils.keras_to_tflite_quantization import *
from saltup.ai.classification.datagenerator.classification_datagen import keras_ClassificationDataGenerator, pytorch_ClassificationDataGenerator
# Save the model
def evaluate_model(model_path:str, test_gen:Union[keras_ClassificationDataGenerator, pytorch_ClassificationDataGenerator], test_data_dir:str) -> float:
    # Evaluate the model on the test set

    global_metric = Metric()

    class_names = sorted(os.listdir(test_data_dir))
    print(class_names)
    metric_per_class = {i: Metric() for i in range(len(class_names))}
    list_images = [images for _, (images, _) in enumerate(test_gen)]

    
    if model_path.endswith('.keras'):
        print("\n--- Evaluate Keras model ---")
    elif model_path.endswith('.pt'):
        print("\n--- Evaluate PyTorch model ---")
    elif model_path.endswith('.tflite'):
        print("\n--- Evaluate Tflite model ---")
    pbar = tqdm(test_gen, desc="Processing data", position=0, leave=True, dynamic_ncols=True)
    for i, (images, labels) in enumerate(pbar):

        if model_path.endswith('.keras'):
            model = tf.keras.models.load_model(model_path)
            predictions = model.predict(images, verbose=0)
        elif model_path.endswith('.pt'):
            pass
            #images = torch.from_numpy(images).float().to(device)
            #predictions = model(images).cpu().detach().numpy()
        elif model_path.endswith('.tflite'):
        
            tflite_interpreter = tf.lite.Interpreter(model_path=model_path)

            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()
            tflite_interpreter.resize_tensor_input(input_details[0]['index'], (images.shape[0], images.shape[1],images.shape[2],images.shape[3]))
            tflite_interpreter.resize_tensor_input(output_details[0]['index'], (1, 4))
            tflite_interpreter.allocate_tensors()

            input_details = tflite_interpreter.get_input_details()
            output_details = tflite_interpreter.get_output_details()

            scale, zero_point = input_details[0]["quantization"]
            x_test_lite = np.uint8(images / scale + zero_point)


            #Making the predictions
            tflite_interpreter.set_tensor(input_details[0]['index'], x_test_lite)

            tflite_interpreter.invoke()
            output_details = tflite_interpreter.get_output_details()[0]
            tflite_model_predictions = tflite_interpreter.get_tensor(output_details['index'])
            #y_pred = np.argmax(tflite_model_predictions)
        
            scale, zero_point = output_details['quantization']

            predictions = (tflite_model_predictions.astype(np.float32) - zero_point) * scale    
        
        else:
            raise ValueError("Unsupported model type. Please use a Keras or PyTorch model.")
        for i, pred in enumerate(predictions):
            binary_pred = np.argmax(pred)
            if binary_pred == np.argmax(labels[i]):
                if binary_pred:
                    global_metric.addTP(1)
                    metric_per_class[binary_pred].addTP(1)
                    metric_per_class[1 - binary_pred].addTN(1)
                else:
                    global_metric.addTN(1)
                    metric_per_class[binary_pred].addTP(1)
                    metric_per_class[1 - binary_pred].addTN(1)
            else:
                if binary_pred:
                    global_metric.addFP(1)
                    metric_per_class[binary_pred].addFP(1)
                    metric_per_class[1 - binary_pred].addFN(1)
                else:
                    global_metric.addFN(1)
                    metric_per_class[binary_pred].addFP(1)
                    metric_per_class[1 - binary_pred].addFN(1)
                            # Update progress bar with current metrics
        pbar.set_postfix(**global_metric.get_metrics())
        
    print("\n")
    print("="*80)
    print(f"{'METRICS SUMMARY':^80}")
    print("="*80)
    print("\n")
    num_images = len(test_gen)
    print(f"{'Images processed:':<20} {num_images}")
    print(f"\nPer class:")
    print("+"*80)

    if class_names:
        for class_id, class_label in enumerate(class_names):
            if class_id == 0:
                print(f"     {'label':<12} | {'Precision':<12} {'':>12} {'Recall':<12} {'':>6}{'Accuracy':<12}")
                print("+"*80)
            print(f"  o {class_label:<12} | {metric_per_class[class_id].getPrecision():.4f} {'':<12}| {metric_per_class[class_id].getRecall():.4f} {'':<12}| {metric_per_class[class_id].getAccuracy():.4f} {'':<12}")
            print("-"*80)
    else:
        for class_id in range(len(class_names)):
            if class_id == 0:
                print(f"     {'id':<12} | {'Precision':<12} {'':>12} {'Recall':<12} {'':>6}{'Accuracy':<12}")
                print("+"*80)
            print(f"  o {class_id:<12} | {metric_per_class[class_id].getPrecision():.4f} {'':<12}| {metric_per_class[class_id].getRecall():.4f} {'':<12}| {metric_per_class[class_id].getAccuracy():.4f}")
            print("-"*80)

    print("\nOverall:")
    print(f"  - {'True Positives (TP):':<25} {global_metric.getTP()}")
    print(f"  - {'False Positives (FP):':<25} {global_metric.getFP()}")
    print(f"  - {'True Negatives (TN):':<25} {global_metric.getTN()}")
    print(f"  - {'False Negatives (FN):':<25} {global_metric.getFN()}")
    print(f"  - {'Overall Precision:':<25} {global_metric.getPrecision():.4f}")
    print(f"  - {'Overall Recall:':<25} {global_metric.getRecall():.4f}")
    print(f"  - {'Overall Accuracy:':<25} {global_metric.getAccuracy():.4f}")
    print("="*80)
    return global_metric.getAccuracy()

from sklearn.model_selection import KFold

def train_model(model:Union[tf.keras.models.Sequential, torch.nn.Module], train_gen, val_gen, output_dir, epochs=10) -> str:

    if 'keras' in str(type(model)).lower():
        # === Keras model ===
        
        saved_models_folder_name = "saved_models"
        saved_models_folder_path = os.path.join(output_dir, saved_models_folder_name)
        os.mkdir(saved_models_folder_path)
        b_v_model_path = os.path.join(saved_models_folder_path,'_keras_best_v_' +'.keras')
        b_t_model_path = os.path.join(saved_models_folder_path,'_keras_best_t_' +'.keras')
        #b_model_path = os.path.join(saved_models_folder_path,'_keras_last_epoch_' + str(k)+'.h5')
        b_model_path = os.path.join(saved_models_folder_path,'_keras_last_epoch_' + '.keras')
        b_v = tf.keras.callbacks.ModelCheckpoint(filepath=b_v_model_path, save_best_only=True)
        b_t = tf.keras.callbacks.ModelCheckpoint(filepath=b_t_model_path, monitor='loss', save_best_only=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0)

        history = model.fit(train_gen,
                    validation_data=val_gen,
                    epochs=epochs,
                    callbacks=[b_v, b_t], 
                    shuffle=True)
        
        # summarize history for loss
        fig = plt.figure(figsize=(10, 10))
        plt.plot(history.history['loss'], '.-')
        plt.plot(history.history['val_loss'], '.-')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        #plt.show()

        plt_filename = "history_loss"
        plt.savefig(os.path.join(saved_models_folder_path, plt_filename + '_plot.png'), bbox_inches='tight')
        plt.close(fig)
        
        fig1 = plt.figure(figsize=(10, 10))
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt_acc_filename = "history_accuracy"
        plt.savefig(os.path.join(saved_models_folder_path, plt_acc_filename + '_plot.png'), bbox_inches='tight')
        plt.close(fig1)
        
        model.save(b_v_model_path)
        print('Saved trained model at {} '.format(b_v_model_path))

        return b_v_model_path

    elif 'pytorch' in str(type(model)).lower():
        pass
    else:
        raise ValueError("Invalid model type. Supported types are 'Keras' and 'PyTorch'.")
    
def training(train_Dataloader:ClassificationDataloader,
             test_Dataloader:ClassificationDataloader,
             model:Union[tf.keras.models.Sequential, torch.nn.Module],
             folds:int,
             epochs:int,
             target_size:tuple,
             batch_size:int,           
             output_dir:str,
             preprocess:callable,
             transform:A.Compose,
             training_callback=train_model,
             validation_split:float=0.2,
             quantization:bool=False):
    """Classification training function.

    Args:
        Dataloader (ClassificationDataloader): Dataloader object.
        model (str): model name to be trained.
        folds (int): number of folds for cross-validation.
        epochs (int): number of epochs for training.
        training_callback (_type_, optional): callback function for training. Defaults to train_model.
        output_dir (str, optional): _description_. Defaults to None.
        validation_split (float, optional): _description_. Defaults to 0.2.
    """

    images_paths = train_Dataloader.image_paths
    test_gen = keras_ClassificationDataGenerator(
        dataloader=test_Dataloader,
        target_size=target_size,
        num_classes=test_Dataloader.get_num_classes(),
        batch_size=batch_size,
        preprocess=preprocess,
        transform=None  # no augmentation
    )
    if folds:
        kfold = KFold(n_splits=folds, shuffle=True)
        loss_per_fold = []
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
            fold_model = model
            fold_path = os.path.join(output_dir,'k_'+str(fold))
            os.mkdir(fold_path)
            print("\n--- Model training ---")
            trained_model_path = training_callback(fold_model, train_gen, val_gen, fold_path, epochs)
            
            
            print("\n--- Evaluate keras model ---")
            trained_model = tf.keras.models.load_model(trained_model_path)
            test_scores = trained_model.evaluate(val_gen, verbose=2)
            
            print("Test loss of the keras model:", test_scores[0])
            print("Test accuracy of the keras model:", test_scores[1])
            loss_per_fold.append(test_scores[0])
            acc_per_fold.append(test_scores[1]*100)
            txt_performance_file_name = "performances.txt"
            current_fold_folder = os.path.dirname(trained_model_path)
            txt_performance_file_path = os.path.join(current_fold_folder, txt_performance_file_name)
            with open(txt_performance_file_path, mode='w') as f:
                f.write("The accuracy of the test of the fold " + str(fold) + ": "+str(test_scores[1]))
                f.write("\nThe loss of the test of the fold " + str(fold) + ": "+str(test_scores[0]))
            f.close()  
            current_accuracy = test_scores[1]
            current_loss = test_scores[0]
            if  current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_loss = current_loss
                fold_number = fold
                golden_model_path = trained_model_path
        
            if quantization and fold == folds-1:
                if not golden_model_path.endswith('.keras'):
                    raise ValueError("The model is not a keras model.")
                golden_fold_folder = os.path.dirname(golden_model_path)
                shutil.copytree(golden_fold_folder, golden_model_folder)
    
                example_samples = np.array([test_gen[i][0] for i in range(50)])
                example_samples = np.reshape(example_samples, (example_samples.shape[0] * example_samples.shape[1],   example_samples.shape[2], example_samples.shape[3], example_samples.shape[4]))
                print("calibration data shape",example_samples.shape)
                model_name = os.path.basename(golden_model_path)
                tflite_model_path = os.path.join(golden_model_folder, model_name.replace('.keras', '.tflite'))
                tflite_model_path = tflite_quantization(golden_model_path, tflite_model_path,example_samples)
                
                accuracy_of_the_keras_golden_model = evaluate_model(golden_model_path,test_gen, test_Dataloader.root_dir)
                accuracy_of_the_tflite_golden_model = evaluate_model(tflite_model_path,test_gen, test_Dataloader.root_dir)
                
                golden_performance_file_path = os.path.join(golden_model_folder, txt_performance_file_name)
                with open(golden_performance_file_path, mode='w') as f:
                    f.write("The golden model's path:{}".format(golden_model_path))
                    f.write("\nThe keras golden model is trained with the fold " + str(fold_number))
                    f.write("\nThe accuracy of the golden model: "+ str(best_accuracy))
                    f.write("\nThe loss of the golden model: "+ str(best_loss))
                    f.write("\nThe accuracy of the tflite golden model on test:{}".format(accuracy_of_the_tflite_golden_model))
                f.close()
            elif fold == folds-1:
                print('\n\nEvaluation on Test set:',best_accuracy,'\n\n')
                accuracy_of_the_keras_golden_model = evaluate_model(golden_model_path,test_gen, test_Dataloader.root_dir)
                
            if fold ==folds-1:
                print('------------------------------------------------------------------------')
                print('Score per fold')
                for i in range(0, len(acc_per_fold)):
                    print('------------------------------------------------------------------------')
                    print(f'> Fold {i} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
                print('------------------------------------------------------------------------')
                print('Average scores for all folds:')
                print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
                print(f'> Loss: {np.mean(loss_per_fold)}')
                print('------------------------------------------------------------------------')
                global_performance_file_name = "Global statistical info.txt.txt"
                global_performance_file_path = os.path.join(output_dir, global_performance_file_name)
                with open(global_performance_file_path, mode='w') as f:
                    f.write("The average accuracy for all folds:{}".format(np.mean(acc_per_fold)))
                    f.write("\nThe variance of the accuracies for all folds:{}".format(np.std(acc_per_fold)))
                    f.write("\nThe average loss for all folds:{}".format(np.mean(loss_per_fold)))
    else:

        val_split = int(validation_split * len(train_Dataloader))
        
        train_dataloader = copy.deepcopy(train_Dataloader)
        val_dataloader = copy.deepcopy(train_Dataloader)
        train_dataloader.image_paths = images_paths[val_split:]
        val_dataloader.image_paths = images_paths[:val_split]
        print("Number of training samples:", len(train_dataloader))
        print("Number of validation samples:", len(val_dataloader))
        val_gen = keras_ClassificationDataGenerator(
            dataloader=val_dataloader,
            target_size=target_size,
            num_classes=val_dataloader.get_num_classes(),
            batch_size=batch_size,
            preprocess=preprocess,
            transform=None  # no augmentation
        )
        train_gen = keras_ClassificationDataGenerator(
            dataloader=train_dataloader,
            target_size=target_size,
            num_classes=train_dataloader.get_num_classes(),
            batch_size=batch_size,
            preprocess=preprocess,
            transform=transform
        )
        training_model = model
        print("\n--- Model training ---")
        model_path = training_callback(training_model, train_gen, val_gen, output_dir, epochs)
        txt_performance_file_name = "performances.txt"
        
        if quantization:
            if not model_path.endswith('.keras'):
                raise ValueError("The model is not a keras model. Only keras model is supported for quantization.")
            model_folder = os.path.dirname(model_path)
            
            example_samples = np.array([test_gen[i][0] for i in range(50)])
            example_samples = np.reshape(example_samples, (example_samples.shape[0] * example_samples.shape[1],   example_samples.shape[2], example_samples.shape[3], example_samples.shape[4]))
            print("calibration data shape",example_samples.shape)
            tflite_model_path = model_path.replace('.keras', '.tflite')
            tflite_model_path = tflite_quantization(model_path, tflite_model_path,example_samples)
            
            accuracy_of_the_keras_model = evaluate_model(model_path,test_gen, test_Dataloader.root_dir)
            accuracy_of_the_tflite_model = evaluate_model(tflite_model_path,test_gen, test_Dataloader.root_dir)
            
            performance_file_path = os.path.join(model_folder, txt_performance_file_name)
            with open(performance_file_path, mode='w') as f:
                f.write("The model's path:{}".format(model_path))
                f.write("\nThe accuracy of the model: "+ str(accuracy_of_the_keras_model))
                f.write("\nThe accuracy of the tflite model on test:{}".format(accuracy_of_the_tflite_model))
            f.close()
        else:
            model_folder = os.path.dirname(model_path)
            print('\n\nEvaluation on Test set:',best_accuracy,'\n\n')
            accuracy_of_the_model = evaluate_model(model_path,test_gen, test_Dataloader.root_dir)
            performance_file_path = os.path.join(model_folder, txt_performance_file_name)
            with open(performance_file_path, mode='w') as f:
                f.write("The model's path:{}".format(model_path))
                f.write("\nThe accuracy of the model: "+ str(accuracy_of_the_model))
            f.close()
