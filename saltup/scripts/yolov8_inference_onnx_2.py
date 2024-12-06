# Ultralytics YOLO 🚀, AGPL-3.0 license

import argparse

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import time
import signal
import sys
import logging
import os
from tqdm import tqdm 

from ultralytics.utils import ASSETS
from ultralytics.utils.checks import check_requirements

def get_onnx_provider():
    return ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]

class YOLO:
    """YOLO object detection model class for handling inference and visualization."""

    def __init__(self, onnx_model_path, motherhouse, class_names:list=None, input_image=None, preprocess_img:bool=True, confidence_thres=0.5, iou_thres=0.5, provider=None, show_bbox:bool=False):
        """
        Initializes an instance of the YOLO class.

        Args:
            onnx_model: Path to the ONNX model.
            input_image: Path to the input image.
            confidence_thres: Confidence threshold for filtering detections.
            iou_thres: IoU (Intersection over Union) threshold for non-maximum suppression.
        """
        self.onnx_model_path = onnx_model_path
        self.motherhouse = motherhouse
        self.input_image = input_image
        self.preprocess_img = preprocess_img
        self.confidence_thres = confidence_thres
        self.iou_thres = iou_thres
        self.pp = 0         # pre-processing time
        self.run = 0        # inference time
        self.posttime = 0   # post-processing time
        self.show_bbox = show_bbox
        self.provider = provider if provider else get_onnx_provider()

        # Create an inference session using the ONNX model and specify execution providers
        self.session = ort.InferenceSession(self.onnx_model_path, providers=self.provider)
        logging.info(f"Using provider: {", ".join(self.provider)}")

        # Extract classes informations
        if class_names:
            self.class_num = len(class_names)
            self.classes = class_names
        else:
            # Load the ONNX model
            model = onnx.load(onnx_model_path)

            # Find the output tensor
            output_tensor = model.graph.output[0]

            # Extract the number of classes from the output tensor shape
            self.class_num = output_tensor.type.tensor_type.shape.dim[-1].dim_value
            self.classes = [c for c in range(self.class_num)]

        # Generate a color palette for the classes
        self.__color_palette = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def draw_detections(self, img, box, score, class_id):
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img: The input image to draw detections on.
            box: Detected bounding box.
            score: Corresponding detection score.
            class_id: Class ID for the detected object.

        Returns:
            None
        """
        # Extract the coordinates of the bounding box
        x1, y1, w, h = box

        # Retrieve the color for the class ID
        color = self.__color_palette[class_id]

        # Draw the bounding box on the image
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

       # print(class_id , score)
        # Create the label text with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"
        #print(label)
        # Calculate the dimensions of the label text
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Calculate the position of the label text
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw a filled rectangle as the background for the label text
        cv2.rectangle(
            img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED
        )

        # Draw the label text on the image
        cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        
    def resize_image_and_black_container_rgb(self, image: np.ndarray, final_width: int, final_height: int, img_position:str):
        """
	    Resizes an image to fit within the given width and height, keeping the aspect ratio,
	    and places it inside a black background of the specified size (RGB format).
	    
	    Args:
		image (np.ndarray): The input image as an OpenCV array (in BGR format).
		final_width (int): The width of the final black container.
		final_height (int): The height of the final black container.
		img_position (str): Define where to paste the image. The options are: 'top-left' and 'center' 
		path_to_save (str, optional): The file path to save the resulting image. Defaults to None.
	    
	    Returns:
		np.ndarray: The resulting image with the resized image centered on a black background.
	    """ 
        
	    #Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	    # Get original image dimensions
        h, w = image_rgb.shape[:2]

        # Calculate aspect ratio for resizing
        aspect_ratio = min(final_width / w, final_height / h)
        new_width = int(w * aspect_ratio)
        new_height = int(h * aspect_ratio)

        # Resize the image
        resized_image = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_AREA)

        # Create a black RGB background image
        #black_image = np.zeros((final_height, final_width, 3), dtype=np.uint8)
        grey_image = 114 * np.ones((final_height, final_width, 3), dtype=np.uint8)
        # Calculate position to center the resized image on the black background
        x_offset = (final_width - new_width) // 2
        y_offset = (final_height - new_height) // 2

        if img_position == 'top-left':
            # Place the resized image at the top-left corner of the black background
            grey_image[0:new_height, 0:new_width] = resized_image
        elif img_position == 'center': 
            # Place the resized image onto the black background
            grey_image[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_image
        else:
            raise ValueError("image_position not valid")

        return grey_image
    
    def preprocess(self, target_shape:tuple):
        """
        Preprocesses the input image before performing inference.

        Returns:
            image_data: Preprocessed image data ready for inference.
        """
        # Read the input image using OpenCV
        self.img = cv2.imread(self.input_image)

        # Get the height and width of the input image
        self.img_height, self.img_width = self.img.shape[:2]
        img = self.img
        # Normalize the image data by dividing it by 255.0
        if self.motherhouse == 'damo':
            # Convert the image color space from BGR to RGB
            img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            # Resize the image to match the input shape
            img = cv2.resize(img, (self.input_width, self.input_height))
            image_data = img
    
        elif self.motherhouse == 'supergrad':
	        # Paddings
            img = self.resize_image_and_black_container_rgb(img, final_height=target_shape[0], final_width=target_shape[1],
		                                       img_position='top-left')
            # Normalize the image data by dividing it by 255.0
            image_data = np.array(img) / 255.0
            # Transpose the image to have the channel dimension as the first dimension
            image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

            # Expand the dimensions of the image data to match the expected input shape
            image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
        
        elif self.motherhouse == 'supergrad-qat':
	        # Padding
            img = self.resize_image_and_black_container_rgb(img, final_height=target_shape[0], final_width=target_shape[1],
		                                       img_position='top-left')
            # Normalize the image data by dividing it by 255.0
            #image_data = np.array(img) / 255.0
            image_data = img
        else:
	        # Resize the image to match the input shape
            img = cv2.resize(img, (self.input_width, self.input_height))
	        # Display the output image in a window
	        #cv2.imshow("Input", img)
            image_data = np.array(img) / 255.0
            # Transpose the image to have the channel dimension as the first dimension
            image_data = np.transpose(image_data, (2, 0, 1))  # Channel first

            # Expand the dimensions of the image data to match the expected input shape
            image_data = np.expand_dims(image_data, axis=0).astype(np.float32)

        # Return the preprocessed image data
        return image_data

    def postprocess(self, input_image, output):
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            input_image (numpy.ndarray): The input image.
            output (numpy.ndarray): The output of the model.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        # Transpose and squeeze the output to match the expected shape
        outputs = np.transpose(np.squeeze(output[0]))

        # Get the number of rows in the outputs array
        rows = outputs.shape[0]

        # Lists to store the bounding boxes, scores, and class IDs of the detections
        boxes = []
        scores = []
        class_ids = []

        # Calculate the scaling factors for the bounding box coordinates
        x_factor = self.img_width / self.input_width
        y_factor = self.img_height / self.input_height

        # Iterate over each row in the outputs array
        for i in range(rows):
            # Extract the class scores from the current row
            classes_scores = outputs[i][4:]  #(1,6300,4)

            # Find the maximum score among the class scores
            max_score = np.amax(classes_scores)

            # If the maximum score is above the confidence threshold
            if max_score >= self.confidence_thres:
                # Get the class ID with the highest score
                class_id = np.argmax(classes_scores)

                # Extract the bounding box coordinates from the current row
                x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]  #(1,6300,4)
 
                # Calculate the scaled coordinates of the bounding box
                left = int((x - w / 2) * x_factor)
                top = int((y - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)

                # Add the class ID, score, and box coordinates to the respective lists
                class_ids.append(class_id)
                scores.append(max_score)
                boxes.append([left, top, width, height])

        # Apply non-maximum suppression to filter out overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_thres, self.iou_thres)

        # Iterate over the selected indices after non-maximum suppression
        for i in indices:
            # Get the box, score, and class ID corresponding to the index
            box = boxes[i]
            score = scores[i]
            class_id = class_ids[i]
            print(" box {} score {} , class {}".format(box,score,class_id))
            # Draw the detection on the input image
            self.draw_detections(input_image, box, score, class_id)

        # Return the modified input image
        return input_image
    
    def postprocess2(self, input_image, output):
        """
        Function used to convert RAW output from YOLOv8 to an array
        of detected objects. Each object contains the bounding box of
        this object, the type of object and the probability.
        
        Args:
            output (numpy.ndarray): Raw output of YOLOv8 network which is an array of shape (1,84,8400)
        
        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        if self.motherhouse == 'damo':
            class_score = output[0].squeeze(0)
            bboxes = output[1].squeeze(0)
            rows = class_score.shape[0]
            boxes = []
            for i in range(rows):
                # Extract the class scores from the current row
                classes_scores = class_score[i]  #(1,8400,4)
                #print(classes_scores)
                
                
                # Find the maximum score among the class scores
                prob = np.max(classes_scores)
                
                # If the maximum score is above the confidence threshold
                if prob >= self.confidence_thres:
                    #print(i)
                    # Get the class ID with the highest score
                    class_id = np.argmax(classes_scores)
                    label = self.classes[class_id]
                    # Extract the bounding box coordinates from the current row
                    x1, y1, x2, y2 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]  #(1,8400,4)
                    
                    #print(f'scrore: {max_score}, bboxes: {bboxes[i]}')#, x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}')
                    x1 = x1 / self.input_width * self.img_width
                    y1 = y1 / self.input_height * self.img_height
                    x2 = x2 / self.input_width * self.img_width
                    y2 = y2 / self.input_height * self.img_height

                    # Add the class ID, score, and box coordinates to the respective lists
                    boxes.append([x1, y1, x2, y2, label, prob])
            boxes.sort(key=lambda x: x[5], reverse=True)  # Sort by confidence
            result = []
            while len(boxes) > 0:
                result.append(boxes[0])
                boxes = [box for box in boxes if self.iou(box, boxes[0]) < self.iou_thres]
        
            out = []
            # Convert BB in x,y,w,h and draw detections
            for detection in result:
                x1, y1, x2, y2, label, prob = detection

                # Convert from (x1, y1, x2, y2) to (x1, y1, width, height)
                width = x2 - x1
                height = y2 - y1
                x1 = x1.item() if isinstance(x1, np.ndarray) else x1
                y1 = y1.item() if isinstance(y1, np.ndarray) else y1
                width = width.item() if isinstance(width, np.ndarray) else width
                height = height.item() if isinstance(height, np.ndarray) else height

                box = [int(x1), int(y1), int(width), int(height)]

                # Find the class_id from the label
                class_id = self.classes.index(label)

                # Call the draw_detections method to draw the bounding box on the image
                self.draw_detections(input_image, box, prob, class_id)
                out.append([int(x1), int(y1), int(width), int(height),prob,class_id,label])

            # Return the modified input image (like postprocess)
            return input_image,out, result
        
        elif self.motherhouse == 'supergrad':
            class_score = output[1].squeeze(0)
            bboxes = output[0].squeeze(0)
            rows = class_score.shape[0]
            boxes = []
            x_factor = self.img_width / self.input_width
            y_factor = self.img_height / self.input_height
            for i in range(rows):
                # Extract the class scores from the current row
                classes_scores = class_score[i]  #(1,8400,4)
                #print(classes_scores)
                
                # Find the maximum score among the class scores
                prob = np.max(classes_scores)
                
                # If the maximum score is above the confidence threshold
                if prob >= self.confidence_thres:
                    #print(i)
                    # Get the class ID with the highest score
                    class_id = np.argmax(classes_scores)
                    label = self.classes[class_id]
                    # Extract the bounding box coordinates from the current row
                    x1, y1, x2, y2 = bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]  #(1,8400,4)

                    # Add the class ID, score, and box coordinates to the respective lists
                    boxes.append([x1, y1, x2, y2, label, prob])
            boxes.sort(key=lambda x: x[5], reverse=True)  # Sort by confidence
            result = []
            while len(boxes) > 0:
                result.append(boxes[0])
                boxes = [box for box in boxes if self.iou(box, boxes[0]) < self.iou_thres]
        
            out = []
            # Convert BB in x,y,w,h and draw detections
            for detection in result:
                x1, y1, x2, y2, label, prob = detection

                # Convert from (x1, y1, x2, y2) to (x1, y1, width, height)
                width = x2 - x1
                height = y2 - y1
                x1 = x1.item() if isinstance(x1, np.ndarray) else x1
                y1 = y1.item() if isinstance(y1, np.ndarray) else y1
                width = width.item() if isinstance(width, np.ndarray) else width
                height = height.item() if isinstance(height, np.ndarray) else height

                box = [int(x1), int(y1), int(width), int(height)]

                # Find the class_id from the label
                class_id = self.classes.index(label)

                # Call the draw_detections method to draw the bounding box on the image
                self.draw_detections(input_image, box, prob, class_id)
                out.append([int(x1), int(y1), int(width), int(height),prob,class_id,label])

            # Return the modified input image (like postprocess)
            return input_image, out, result
        
        else:
            output = output[0].astype(float)
            output = output.transpose()

            boxes = []
            for row in output:
                prob = row[4:].max()
                if prob < self.confidence_thres:  # Use the confidence threshold set in the class
                    continue
                class_id = row[4:].argmax()
                label = self.classes[class_id]
                xc, yc, w, h = row[:4]
                x1 = (xc - w/2) / self.input_width * self.img_width
                y1 = (yc - h/2) / self.input_height * self.img_height
                x2 = (xc + w/2) / self.input_width * self.img_width
                y2 = (yc + h/2) / self.input_height * self.img_height

                boxes.append([x1, y1, x2, y2, label, prob])

            boxes.sort(key=lambda x: x[5], reverse=True)  # Sort by confidence
            result = []
            while len(boxes) > 0:
                result.append(boxes[0])
                boxes = [box for box in boxes if self.iou(box, boxes[0]) < self.iou_thres]
        
            out = []
            # Convert BB in x,y,w,h and draw detections
            for detection in result:
                x1, y1, x2, y2, label, prob = detection

                # Convert from (x1, y1, x2, y2) to (x1, y1, width, height)
                width = x2 - x1
                height = y2 - y1
                x1 = x1.item() if isinstance(x1, np.ndarray) else x1
                y1 = y1.item() if isinstance(y1, np.ndarray) else y1
                width = width.item() if isinstance(width, np.ndarray) else width
                height = height.item() if isinstance(height, np.ndarray) else height

                box = [int(x1), int(y1), int(width), int(height)]

                # Find the class_id from the label
                class_id = self.classes.index(label)

                # Call the draw_detections method to draw the bounding box on the image
                self.draw_detections(input_image, box, prob, class_id)
                out.append([ int(x1), int(y1), int(width), int(height),prob,class_id,label])

            # Return the modified input image (like postprocess)
            return input_image, out, result

    def intersection(self,box1,box2):
        """
        Function calculates intersection area of two boxes
        :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
        :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
        :return: Area of intersection of the boxes as a float number
        """
        box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
        box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
        x1 = max(box1_x1,box2_x1)
        y1 = max(box1_y1,box2_y1)
        x2 = min(box1_x2,box2_x2)
        y2 = min(box1_y2,box2_y2)
        return (x2-x1)*(y2-y1)
    
    def union(self,box1,box2):
        """
        Function calculates union area of two boxes
        :param box1: First box in format [x1,y1,x2,y2,object_class,probability]
        :param box2: Second box in format [x1,y1,x2,y2,object_class,probability]
        :return: Area of the boxes union as a float number
        """
        box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
        box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
        box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
        box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
        return box1_area + box2_area - self.intersection(box1,box2)

    def iou(self,box1,box2):
        """
        Function calculates "Intersection-over-union" coefficient for specified two boxes
        https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
        :param box1: First box in format: [x1,y1,x2,y2,object_class,probability]
        :param box2: Second box in format: [x1,y1,x2,y2,object_class,probability]
        :return: Intersection over union ratio as a float number
        """
        return self.intersection(box1,box2)/self.union(box1,box2)

    def main(self):
        """
        Performs inference using an ONNX model and returns the output image with drawn detections.

        Returns:
            output_img: The output image with drawn detections.
        """
            # Get the model inputs
        model_inputs = self.session.get_inputs()
 
        # Store the shape of the input for later use
        input_shape = model_inputs[0].shape
        logging.debug(input_shape)
        self.input_width = input_shape[3]
        self.input_height = input_shape[2]
     
        s = time.time()
        # Preprocess the image data
        if self.preprocess_img:
            img_data = self.preprocess((self.input_height, self.input_width))
        else:
            # Read the input image using OpenCV
            self.img = cv2.imread(self.input_image)
            # Get the height and width of the input image
            self.img_height, self.img_width = self.img.shape[:2]
            # Convert the image color space from BGR to RGB
            img_data = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            img_data = np.transpose(img_data, (2, 0, 1))
            img_data = np.expand_dims(img_data, axis=0).astype(np.uint8)
        
        now = time.time()    
        self.pp = now - s  # Salva il tempo di preprocessamento

        startrun = time.time()
        # Run inference using the preprocessed image data
        outputs = self.session.run(None, {model_inputs[0].name: img_data})
        self.run = time.time() - startrun
        poststart = time.time()
        post, bb, result = self.postprocess2(self.img, outputs)  # output image

        # Perform post-processing on the outputs to obtain output image.
        #post = self.postprocess(self.img, outputs)  # output image
        self.posttime = time.time() - poststart  # Salva il post-processing time

        if self.show_bbox:
            print(f"Found {len(bb)} Bounding Box:")
            for e in bb:
                print(e)

        #print("inference [sec] {}, pre-process {} , post-process {}".format(self.run, self.pp, self.posttime))
        bb_out = []
        for bbox in result:
            # x1, y1, x2, y2, label, prob = result
            if self.motherhouse == 'damo':
                cc = bbox[:4]
            elif self.motherhouse == 'supergrad':
                cc = bbox[:4]
            else:
                cc = [int(elt.item()) for elt in bbox if isinstance(elt, np.ndarray)]
            cc.append(bbox[-1])
            cc.append(self.classes.index(bbox[-2]))
            cc.append(bbox[-2])
            bb_out.append(cc)
         
        return bb_out, post


def signal_handler(sig, frame):
    """Funzione di gestione del segnale di interruzione."""
    print('Interruzione ricevuta (Ctrl+C). Uscita...')
    sys.exit(0)

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    box1Area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2Area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    iou = interArea / float(box1Area + box2Area - interArea) if float(box1Area + box2Area - interArea) > 0 else 0
    return iou

def calculate_metrics(pred_bb, gt_bb, iou_threshold=0.5):
    TP, FP, FN = 0, 0, 0
    matched_gt = set()

    # For each predicted box
    for pred_box in pred_bb:
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_bb):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Check if the IoU is above the threshold
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            TP += 1
            matched_gt.add(best_gt_idx)  # Mark this GT box as matched
        else:
            FP += 1

    # Count ground truth boxes that were not matched
    FN = len(gt_bb) - len(matched_gt)

    return TP, FP, FN

def calculate_metrics_per_class(pred_bb, gt_bb, class_id, iou_threshold=0.5):
    TP, FP, FN = 0, 0, 0
    matched_gt = set()

    # Filter predicted and ground truth boxes for the current class
    pred_class_bb = [box for box in pred_bb if box[5] == class_id]
    gt_class_bb = [box for box in gt_bb if box[5] == class_id]

    # For each predicted box of this class
    for pred_box in pred_class_bb:
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_class_bb):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        # Check if IoU is above threshold
        if best_iou >= iou_threshold and best_gt_idx not in matched_gt:
            TP += 1
            matched_gt.add(best_gt_idx)
        else:
            FP += 1

    # Count ground truth boxes that were not matched
    FN = len(gt_class_bb) - len(matched_gt)

    return TP, FP, FN

def read_yolo_labels(label_path, img_width, img_height):
    """Read YOLO format labels from a file"""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                class_id, xc, yc, w, h = map(float, line.strip().split())
                x1 = (xc - w/2) * img_width
                y1 = (yc - h/2) * img_height
                x2 = (xc + w/2) * img_width
                y2 = (yc + h/2) * img_height
                boxes.append([int(x1), int(y1), int(x2), int(y2), 1.0, int(class_id)])  # 1.0 is a dummy confidence
    return boxes

def yolo_metrics(TP, FP, FN):
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 

    return precision, recall, f1_score

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8n.onnx", help="Input your ONNX model.")
    parser.add_argument("--type", type=str, default="", help="Input your yolo model type.")
    parser.add_argument("--img", type=str, default=str(ASSETS / "bus.jpg"), help="Path to input image or folder.")
    parser.add_argument("--preprocess", type=str, default=True, help="To preprocess or not the image before entering the model")
    parser.add_argument("--label", type=str, help="Path to image label or folder.")
    parser.add_argument("--gui",  action='store_true', help="open gui to draw bbox")
    parser.add_argument("--conf-thres", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--show_bbox", type=str, help="to print bounding box")
    parser.add_argument("--max-images", type=int, default=0, help="Maximum number of images to process")
    parser.add_argument("--device", type=str, choices=["cpu", "gpu"], default="", help="Force device selection: 'cpu', 'gpu', or auto-select if not specified")

    return parser.parse_args()

def main(args):
    # Registra la funzione di gestione del segnale
    signal.signal(signal.SIGINT, signal_handler)
    
    # Determine the provider based on the device argument and GPU availability
    if args.device == "gpu":
        if torch.cuda.is_available():
            print("Forcing GPU as per user request.")
            provider = ["CUDAExecutionProvider"]
        else:
            print("GPU requested but not available. Falling back to CPU.")
            provider = ["CPUExecutionProvider"]
    elif args.device == "cpu":
        print("Forcing CPU as per user request.")
        provider = ["CPUExecutionProvider"]
    else:
        provider = get_onnx_provider()
    
    # Print chosen provider for confirmation
    print(f"Using provider: {provider}")
 
    # Check the requirements and select the appropriate backend (CPU or GPU)
    check_requirements("onnxruntime-gpu" if torch.cuda.is_available() else "onnxruntime")

    if args.label:
        if os.path.isdir(args.img):
            image_files = [f for f in os.listdir(args.img) if f.endswith('.jpg')]
            total_preprocess_time = 0
            total_postprocess_time = 0
            total_run_time = 0
            total_TP, total_FP, total_FN = 0, 0, 0
            classes = ["mouse", "drinking", "climbing", "feeding"]  # Assuming 4 classes with class IDs 0, 1, 2, 3
            class_metrics = {c: {'TP': 0, 'FP': 0, 'FN': 0} for c in classes}
            tot= len(image_files) if args.max_images == 0 else len(image_files[:args.max_images])
            with tqdm(total=tot, desc="Inference images", unit="image") as pbar:

                for i in range(tot):
                    img_path = os.path.join(args.img, image_files[i])
                    # Read the input image using OpenCV
                    img = cv2.imread(img_path)
                    # Get the height and width of the input image
                    img_height, img_width = img.shape[:2]
                    real_bb_path = os.path.join(args.label, image_files[i].replace('jpg','txt'))
                    detection = YOLO(args.model, args.type, img_path, args.preprocess, args.conf_thres,
                                       args.iou_thres, args.gui, provider, args.show_bbox)
                    bb, output_image = detection.main()
                    real_bb = read_yolo_labels(real_bb_path, img_width, img_height)

                    # Calculate metrics for each class
                    for iteration, class_name in enumerate(classes):
                        TP, FP, FN = calculate_metrics_per_class(bb, real_bb, iteration, iou_threshold=0.5)
                        # Accumulate totals for the current class
                        class_metrics[class_name]['TP'] += TP
                        class_metrics[class_name]['FP'] += FP
                        class_metrics[class_name]['FN'] += FN
                    
                    # Calculate overall Precision, Recall, F1-score across all classes with the accumulated iteration
                    total_TP = sum(class_metrics[c]['TP'] for c in classes)
                    total_FP = sum(class_metrics[c]['FP'] for c in classes)
                    total_FN = sum(class_metrics[c]['FN'] for c in classes)
                    
                    overall_precision, overall_recall, overall_f1_score = yolo_metrics(total_TP, total_FP, total_FN)
                    total_preprocess_time += detection.pp
                    total_postprocess_time += detection.posttime
                    total_run_time += detection.run
                    inf_time = total_run_time / (i+1)
                    
                    pbar.set_postfix(model_inf_time_in_sec = inf_time, precision=overall_precision, recall=overall_recall, f1 = overall_f1_score)

                    # Update the progress bar
                    pbar.update(1)
            
            # Calculate Precision, Recall, F1-score per class
            for class_name in classes:
                TP = class_metrics[class_name]['TP']
                FP = class_metrics[class_name]['FP']
                FN = class_metrics[class_name]['FN']
                precision, recall, f1_score = yolo_metrics(TP, FP, FN)
                print(f"Class {class_name} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")
            	
            num_images = tot
            print(f"Processed {num_images} images:")
            print(f"Average Pre-process Time: {total_preprocess_time / num_images:.4f} sec")
            print(f"Average Inference Time: {total_run_time / num_images:.4f} sec")
            print(f"Average Post-process Time: {total_postprocess_time / num_images:.4f} sec")
            
            # Calculate overall Precision, Recall, F1-score across all classes
            total_TP = sum(class_metrics[c]['TP'] for c in classes)
            total_FP = sum(class_metrics[c]['FP'] for c in classes)
            total_FN = sum(class_metrics[c]['FN'] for c in classes)
            
            overall_precision, overall_recall, overall_f1_score = yolo_metrics(total_TP, total_FP, total_FN)

            print(f"Overall - Precision: {overall_precision:.4f}, Recall: {overall_recall:.4f}, F1-Score: {overall_f1_score:.4f}")
            print(f"The model for inference: {args.model}")
    
    # Controlla se il percorso dell'immagine è una directory
    elif os.path.isdir(args.img):
        image_files = [f for f in os.listdir(args.img) if f.endswith('.jpg')]
        total_preprocess_time = 0
        total_postprocess_time = 0
        total_run_time = 0
        for img_file in image_files:
            img_path = os.path.join(args.img, img_file)
            detection = YOLO(args.model, args.type, img_path, args.preprocess, args.conf_thres, args.iou_thres,
                               args.gui, provider, args.show_bbox)

            # Perform object detection and obtain the output image
            bb, output_image = detection.main()

            # Accumula i tempi di preprocessamento e postprocessamento
            total_preprocess_time += detection.pp
            total_postprocess_time += detection.posttime
            total_run_time += detection.run

        # Calcola la media dei tempi
        num_images = len(image_files)
        print(f"Processed {num_images} images:")
        print(f"Average Pre-process Time: {total_preprocess_time / num_images:.4f} sec")
        print(f"Average Inference Time: {total_run_time / num_images:.4f} sec")
        print(f"Average Post-process Time: {total_postprocess_time / num_images:.4f} sec")

    else:
        # Create an instance of the YOLOv8 class with the specified arguments
        detection = YOLO(args.model, args.type, args.img, args.preprocess, args.conf_thres, args.iou_thres,
                           args.gui, provider, args.show_bbox)

        # Perform object detection and obtain the output image
        bb, output_image = detection.main()
        print(f"Found {len(bb)} Bounding Box:")
        for e in bb:
            print(e)
        if args.gui:
            print('Press Ctrl + C to stop')
            # Display the output image in a window
            cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
            cv2.imshow("Output", output_image)

            # Wait for a key press to exit
            while True:
                key = cv2.waitKey(1)  # Attende 1 millisecondo
                if key == 27:  # Esc key
                    break

            cv2.destroyAllWindows()  # Chiudi tutte le finestre di OpenCV
            
if __name__ == "__main__":
    args = get_params()
    main(args)
