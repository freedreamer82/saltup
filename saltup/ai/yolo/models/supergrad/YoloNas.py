import os
import torch
import json
import logging
import json

from saltup.ai.yolo.BaseConfigHandler import BaseConfigHandler
from saltup.ai.yolo.BaseYoloHandler import BaseYoloHandler

logging.info("Importing SuperGradients Trainer and dataset classes...")
from super_gradients.training import Trainer, models
import super_gradients.training.processing as sg_processing
from super_gradients.training.datasets.detection_datasets.yolo_format_detection import YoloDarknetFormatDetectionDataset
from super_gradients.training.datasets.detection_datasets.coco_format_detection import COCOFormatDetectionDataset
from super_gradients.training.dataloaders import dataloaders


class YoloNasConfig(BaseConfigHandler):
    """
    A class to handle loading, validating, and processing configuration files for the training process.
    """ 
    
    def validate_configs(self):
        """
        Validates the configuration dictionary to ensure all required parameters are present and valid.

        Raises:
        ValueError: If any required parameters are missing or invalid.
        """
        required_keys = ["annotations_format", "train_dataset_params", "val_dataset_params",
                         "train_dataloader_params", "val_dataloader_params", "model", "training_hyperparams"]
        for key in required_keys:
            if key not in self._configs:
                logging.error(f"Missing required key in configuration file: {key}")
                raise ValueError(f"Missing required key in configuration file: {key}")
        
        # Check if num_classes matches the length of class_names
        if ("num_classes" in self._configs) and ("class_names" in self._configs) and \
            (len(self._configs["class_names"]) != self._configs["num_classes"]):
            logging.error(f"Expected {len(self._configs['class_names'])} but got {self._configs['num_classes']}")
            raise ValueError(f"Expected {len(self._configs['class_names'])} but got {self._configs['num_classes']}")


class YoloNas(BaseYoloHandler):
    
    def __init__(self, args):
        """
        Initialize the YOLO NAS handler with configuration arguments.
        """
        self.__args = args
        
        self.config_filepath = self.__args.config_filepath
        logging.info("Loading configuration for YOLO training.")
        
        # Load configuration file
        self.configs = YoloNasConfig(self.config_filepath).get_configs()
        self.__model_card = self.configs['model']

    def train(self):
        """
        Train the YOLO model using the provided configuration and datasets.
        """
        # Check if resuming training with a valid checkpoint file
        if self.__args.resume:
            logging.info("Resume flag is set. Checking for checkpoint at path: %s", self.__args.weight)
            if not os.path.isfile(self.__args.weight) or not self.__args.weight.endswith('.pth'):
                logging.error("Invalid checkpoint path: '%s'", self.__args.weight)
                raise RuntimeError(f"'{self.__args.weight}' is not a correct path.")
            logging.info("Checkpoint file found and valid.")
            ckpt_dir, name = self.__args.weight.rsplit('/', 3)[:-2]
        else:
            name = self.__args.name
            ckpt_dir = self.__args.ckpt_dir
        
        # Select the dataset class based on the annotations format
        annotations_format = self.configs['annotations_format'].lower()
        if annotations_format == 'yolo':
            DatasetClass = YoloDarknetFormatDetectionDataset
            logging.info("Using YOLO Darknet format for dataset.")
        elif annotations_format == 'coco':
            DatasetClass = COCOFormatDetectionDataset
            logging.info("Using COCO format for dataset.")
        else:
            logging.error("Annotations format '%s' is not supported.", annotations_format)
            raise ValueError("Annotations format not supported. Available formats are 'yolo' or 'coco'.")

        # Initialize datasets
        logging.info("Initializing training and validation datasets...")
        trainset = DatasetClass(**self.configs['train_dataset_params'])
        valset = DatasetClass(**self.configs['val_dataset_params'])
        
        # Initialize DataLoader for training and validation
        logging.info("Creating data loaders for training and validation...")
        train_loader = dataloaders.get(
            dataset=trainset,
            dataloader_params=self.configs['train_dataloader_params']
        )
        valid_loader = dataloaders.get(
            dataset=valset,
            dataloader_params=self.configs['val_dataloader_params']
        )

        # Initialize Trainer
        logging.info("Initializing Trainer with experiment name: %s", name)
        trainer = Trainer(experiment_name=name, ckpt_root_dir=ckpt_dir)

        # Determine device (GPU or CPU)
        DEVICE = self.configs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Training on device: %s", DEVICE)

        # Set up the model configuration
        logging.info("Setting up model configuration...")
        if self.__args.resume:
            self.__model_card['checkpoint_path'] = self.__args.weight
            if 'pretrained_weights' in self.__model_card:
                del self.__model_card['pretrained_weights']
            logging.info("Resuming from checkpoint: %s", self.__args.weight)
        else:
            logging.info("Training from scratch with model configuration.")

        # Instantiate and move model to the correct device
        model = models.get(**self.__model_card).to(DEVICE)
        logging.info("Model initialized and moved to device.")

        # Set training parameters and start training
        logging.info("Starting training process...")
        training_hyperparams = YoloNasConfig.instantiate_classes(self.configs['training_hyperparams'])
        if self.__args.resume:
            training_hyperparams['resume'] = True
        trainer.train(model=model, training_params=training_hyperparams, train_loader=train_loader, valid_loader=valid_loader)

        # Log the path to the best model checkpoint
        best_checkpoint_path = os.path.join(trainer.checkpoints_dir_path, "ckpt_best.pth")
        logging.info("Best model checkpoint saved at: %s", best_checkpoint_path)
        
    def model_info(self):
        """
        Generate and export model metadata, including preprocessing steps and model details.
        """
        def __get_preprocessing_steps(preprocessing, processing: sg_processing):
            """
            Map preprocessing step objects to corresponding dictionary representations.
            """
            if isinstance(preprocessing, processing.StandardizeImage):
                return {"Standardize": {"max_value": preprocessing.max_value}}
            elif isinstance(preprocessing, processing.DetectionRescale):
                return {"DetRescale": None}
            elif isinstance(preprocessing, processing.DetectionLongestMaxSizeRescale):
                return {"DetLongMaxRescale": None}
            elif isinstance(preprocessing, processing.DetectionBottomRightPadding):
                return {
                    "BotRightPad": {
                        "pad_value": preprocessing.pad_value,
                    }
                }
            elif isinstance(preprocessing, processing.DetectionCenterPadding):
                return {
                    "CenterPad": {
                        "pad_value": preprocessing.pad_value,
                    }
                }
            elif isinstance(preprocessing, processing.NormalizeImage):
                return {
                    "Normalize": {"mean": preprocessing.mean.tolist(), "std": preprocessing.std.tolist()}
                }
            elif isinstance(preprocessing, processing.ImagePermute):
                return None
            elif isinstance(preprocessing, processing.ReverseImageChannels):
                return None
            else:
                logging.error("Model has processing steps that haven't been implemented.")
                raise NotImplementedError("Model has processing steps that haven't been implemented")

        import numpy as np
        
        logging.info(
            "🚀 \033[1m\033[94m"
            + "Generate Metadata: "
            + "\033[0m\n\t"
            + "\n\t".join([f"{x}: {y}" for x, y in vars(self.__args).items() if y is not None]) 
        )
        
        # Extract model details
        opt_type, num_classes, model_path = self.__model_card['model_name'], self.__model_card['num_classes'], self.__args.model_path
        net = models.get(opt_type, num_classes=num_classes, checkpoint_path=model_path)

        # Dummy image to simulate preprocessing
        dummy = np.random.randint(0, 255, (1000, 800, 3), dtype=np.uint8)

        # Extract model information
        labels = net._class_names
        iou = net._default_nms_iou
        conf = net._default_nms_conf
        preprocessing_steps = [
            __get_preprocessing_steps(st, sg_processing) for st in net._image_processor.processings
        ]
        imgsz = np.expand_dims(net._image_processor.preprocess_image(dummy)[0], 0).shape

        # Prepare result dictionary
        res = {
            "type": opt_type,
            "original_insz": imgsz,
            "iou_thres": iou,
            "score_thres": conf,
            "prep_steps": preprocessing_steps,
            "labels": labels,
        }

        # Save metadata to file
        filepath = os.path.join(self.__args.output_dir, f"custom-{opt_type}-metadata.json")
        logging.info(f"Export metadata to: {filepath}")
        with open(filepath, "w") as f:
            f.write(json.dumps(res, indent=3))
    
    def preprocess(self, input):
        """
        Placeholder method for model preprocess (not implemented yet).
        """
        logging.error("Preprocess method not implemented yet.")
        raise NotImplementedError("Preprocess method not implemented yet.")

    def postprocess(self, input):
        """
        Placeholder method for model postprocess (not implemented yet).
        """
        logging.error("Postprocess method not implemented yet.")
        raise NotImplementedError("Postprocess method not implemented yet.")
      
    def inference(self, input):
        """
        Placeholder method for model inference (not implemented yet).
        """
        logging.error("Inference method not implemented yet.")
        raise NotImplementedError("Inference method not implemented yet.")
    
    def export_onnx(self):
        """
        Placeholder method for model export_onnx (not implemented yet).
        """
        logging.error("Method export_onnx not implemented yet.")
        raise NotImplementedError("Method export_onnx not implemented yet.")
            
    def qat(self):
        """
        Placeholder method for model quantization (not implemented yet).
        """
        logging.error("Quantization method not implemented yet.")
        raise NotImplementedError("Quantization method not implemented yet.")
