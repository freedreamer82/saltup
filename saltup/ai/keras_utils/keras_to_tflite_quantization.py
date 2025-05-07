import tensorflow as tf
import os
import keras

def tflite_quantization(golden_model_path, output_tflite_path, x_train, quantizing_input=True, quantizing_output=True):
    
    print("\n--- Tflite quantization ---")
    model = keras.models.load_model(golden_model_path)
    x_train = x_train.astype("float32")

    # Defining the calibration data
    def representative_data_gen():
        for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(20).take(700):
            yield [input_value]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT] # converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    
    converter.representative_dataset = representative_data_gen

    # Set the input and output tensors to uint8 (APIs added in r2.3)
    if quantizing_input:
        converter.inference_input_type = tf.uint8                   #quantizing the input
        
    if quantizing_output:
        converter.inference_output_type = tf.uint8                 #quantizing the output
    
    tflite_model_quantInt = converter.convert()

    # Quantize the model.

    interpreter = tf.lite.Interpreter(model_content=tflite_model_quantInt)
    input_type = interpreter.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output: ', output_type)
    
    # Save model and weights
    os.makedirs(os.path.dirname(output_tflite_path), exist_ok=True)
    
    with open(output_tflite_path, 'wb') as f:
        f.write(tflite_model_quantInt)
    print('Saved trained model at %s ' %output_tflite_path)
    return output_tflite_path
