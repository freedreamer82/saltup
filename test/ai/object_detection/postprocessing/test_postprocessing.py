# import pytest
# import numpy as np
# import os

# from saltup.ai.object_detection.postprocessing import (
#     PostprocessingFactory,
#     PostprocessingType,
#     Postprocessing
# )
# from saltup.ai.object_detection.postprocessing.impl import (
#     AnchorsBasedPostprocess,
#     DamoPostprocess,
#     SupergradPostprocess,
#     UltralyticsPostprocess
# )


# class TestBasePostprocessing:
#     """Test the abstract base Postprocessing class."""

#     class ConcretePostprocessing(Postprocessing):
#         """Concrete class for testing abstract base class."""

#         def __call__(self):
#             return super().__call__()

#     def test_validate_input_none(self):
#         """Test validation of None input."""
#         processor = self.ConcretePostprocessing()
#         with pytest.raises(ValueError, match="model output cannot be None"):
#             processor._validate_input(None)

#     def test_validate_input_wrong_type(self):
#         """Test validation of non-numpy or tf array input."""
#         processor = self.ConcretePostprocessing()
#         with pytest.raises(TypeError, match="Input must be numpy or tensorflow array"):
#             processor._validate_input([1, 2, 3])

#     def test_abstract_process_method(self):
#         """Test that abstract process method raises NotImplementedError."""
#         processor = self.ConcretePostprocessing()
#         with pytest.raises(NotImplementedError):
#             processor.__call__()


# class TestPostprocessingFactory:
#     """Test the Postprocessing factory class."""

#     def test_valid_processor_creation(self):
#         """Test creation of all valid processor types."""
#         valid_types = [
#             (PostprocessingType.ANCHORS_BASED, AnchorsBasedPostprocess),
#             (PostprocessingType.ULTRALYTICS, UltralyticsPostprocess),
#             (PostprocessingType.SUPERGRAD, SupergradPostprocess),
#             (PostprocessingType.DAMO, DamoPostprocess)
#         ]

#         for proc_type, expected_class in valid_types:
#             processor = PostprocessingFactory.create(proc_type)
#             assert isinstance(processor, expected_class)

#     def test_invalid_processor_type(self):
#         """Test factory behavior with invalid processor type."""
#         with pytest.raises(ValueError, match="Unknown processor type"):
#             PostprocessingFactory.create(999)


# class TestDamoPostprocess:
#     """Test the DAMO Postprocessing implementation."""

#     @pytest.fixture
#     def processor(self):
#         return DamoPostprocess()

#     @pytest.fixture
#     def sample_input(self):
#         """Create an example tensor."""
#         # Generate the first tensor with scores between 0 and 1
#         scores = np.random.rand(1, 100, 4)  # Random values between 0 and 1

#         # Generate the second tensor with bounding box values between 0 and max pixel coordonate value
#         bboxes = np.random.randint(0, 20, (1, 100, 4))  # Random integers between 0 and max pixel coordonate value

#         # Combine into a list
#         model_output_simulation = [scores, bboxes]
        
#         return model_output_simulation

#     def test_process_random_input(self, processor, sample_input):
#         """Test processing of valid model ouput."""
#         assert sample_input[0].shape[-1] == 4
#         assert sample_input[1].shape[-1] == 4
        
#         classes_name = ['red', 'blue', 'green', 'yellow']
#         model_input_height = 120 
#         model_input_width = 160 
#         image_height = 120 
#         image_width = 160
#         confidence_thr = 0.5 
#         iou_threshold = 0.5
        
#         try:
#             result = processor(sample_input, classes_name, model_input_height, 
#                             model_input_width, image_height, image_width, confidence_thr, iou_threshold)
#         except Exception as e:
#             print(f"Error during processing: {e}")
#             raise

#         assert isinstance(result, list)
#         if len(result) != 0:
#             for bbox in result:
#                 assert len(bbox) == 6

#     @pytest.fixture
#     def real_output(self, root_dir):
#         damo_model_output = np.load(os.path.join(str(root_dir), 'results', 'damo_output.npy'))
#         return damo_model_output
    
#     def test_process_real_input(self, processor, real_output):
#         """Test processing of valid model ouput."""
#         assert real_output[0].shape[-1] == 4
#         assert real_output[1].shape[-1] == 4
#         classes_name = ['red', 'blue', 'green', 'yellow']
#         model_input_height = 480 
#         model_input_width = 640 
#         image_height = 480 
#         image_width = 640
#         confidence_thr = 0.5 
#         iou_threshold = 0.5
        
#         result = processor(real_output, classes_name, model_input_height, 
#                             model_input_width, image_height, image_width, confidence_thr, iou_threshold)

#         assert isinstance(result, list)
#         if len(result) != 0:
#             for bbox in result:
#                 assert len(bbox) == 6

# class TestSupergradPostprocess:
#     """Test the Supergrad Postprocessing implementation."""

#     @pytest.fixture
#     def processor(self):
#         return SupergradPostprocess()

#     @pytest.fixture
#     def sample_input(self):
#         """Create an example tensor."""
#         # Generate the first tensor with scores between 0 and 1
#         scores = np.random.uniform(0.5, 1, (1, 10, 4)) # Random values between 0 and 1

#         # Generate the second tensor with bounding box values between 0 and max pixel coordonate value
#         bboxes = np.random.randint(0, 20, (1, 10, 4))  # Random integers between 0 and max pixel coordonate value

#         # Combine into a list
#         model_output = [bboxes, scores]
        
#         return model_output

#     def test_process_random_input(self, processor, sample_input):
#         """Test processing of valid model ouput."""
#         assert sample_input[0].shape[-1] == 4
#         assert sample_input[1].shape[-1] == 4
#         classes_name = ['red', 'blue', 'green', 'yellow']
#         model_input_height = 120 
#         model_input_width = 160 
#         image_height = 120 
#         image_width = 160
#         confidence_thr = 0.5 
#         iou_threshold = 0.5
        
#         result = processor(sample_input, classes_name, model_input_height, 
#                             model_input_width, image_height, image_width, confidence_thr, iou_threshold)

#         assert isinstance(result, list)
#         if len(result) != 0:
#             for bbox in result:
#                 assert len(bbox) == 6
    
#     @pytest.fixture
#     def real_output(self, root_dir):
#         supergrad_model_output = np.load(os.path.join(str(root_dir), 'results', 'supergrad_output.npy'))
#         return supergrad_model_output
    
#     def test_process_real_input(self, processor, real_output):
#         """Test processing of valid model ouput."""
#         assert real_output[0].shape[-1] == 4
#         assert real_output[1].shape[-1] == 4
#         classes_name = ['red', 'blue', 'green', 'yellow']
#         model_input_height = 480 
#         model_input_width = 640 
#         image_height = 480 
#         image_width = 640
#         confidence_thr = 0.5 
#         iou_threshold = 0.5
        
#         result = processor(real_output, classes_name, model_input_height, 
#                             model_input_width, image_height, image_width, confidence_thr, iou_threshold)

#         assert isinstance(result, list)
#         if len(result) != 0:
#             for bbox in result:
#                 assert len(bbox) == 6

# class TestUltralyticsPostprocess:
#     """Test the Ultralytics Postprocessing implementation."""

#     @pytest.fixture
#     def processor(self):
#         return UltralyticsPostprocess()

#     @pytest.fixture
#     def sample_input(self):
#         """Create an example tensor."""
#         # Generate random values for the format (8, 100)
#         # xcenter, ycenter, width, height (all between 0 and 640 pixels)
#         bbox_values = np.random.uniform(0, 160, (4, 100))  # 4 rows for bbox values

#         # Scores for 4 classes (values between 0 and 1)
#         class_scores = np.random.rand(4, 100)  # 4 rows for class scores

#         # Concatenate to form the final tensor of shape (8, 100)
#         model_output = [np.concatenate((bbox_values, class_scores), axis=0)]
        
#         return model_output

#     def test_process_random_input(self, processor, sample_input):
#         """Test processing of valid model ouput."""
#         assert sample_input[0].shape[0] == 8
#         classes_name = ['red', 'blue', 'green', 'yellow']
#         model_input_height = 120 
#         model_input_width = 160 
#         image_height = 120 
#         image_width = 160
#         confidence_thr = 0.5 
#         iou_threshold = 0.5
        
#         result = processor(sample_input, classes_name, model_input_height, 
#                            model_input_width,image_height, image_width,  confidence_thr, iou_threshold)

#         assert isinstance(result, list)
#         if len(result) != 0:
#             for bbox in result:
#                 assert len(bbox) == 6
    
#     @pytest.fixture
#     def real_output(self, root_dir):
#         ultralytics_model_output = np.load(os.path.join(str(root_dir), 'results', 'ultralytics_ouput.npy'))
#         ultralytics_model_output = ultralytics_model_output.squeeze(0)
#         return ultralytics_model_output
    
#     def test_process_real_input(self, processor, real_output):
#         """Test processing of valid model ouput."""
#         assert real_output[0].shape[0] == 8
        
#         classes_name = ['red', 'blue', 'green', 'yellow']
#         model_input_height = 480 
#         model_input_width = 640 
#         image_height = 480 
#         image_width = 640
#         confidence_thr = 0.5
#         iou_threshold = 0.5
        
#         result = processor(real_output, classes_name, model_input_height, 
#                             model_input_width, image_height, image_width, confidence_thr, iou_threshold)

#         assert isinstance(result, list)
#         if len(result) != 0:
#             for bbox in result:
#                 assert len(bbox) == 6

# class TestAnchorsBasedPostprocess:
#     """Test the AnchorsBased Postprocessing implementation."""

#     @pytest.fixture
#     def processor(self):
#         return AnchorsBasedPostprocess()

#     @pytest.fixture
#     def sample_input(self):
#         """Create an example tensor."""

#         model_output = np.random.uniform(-2, 12, (1, 15, 20, 5, 9))  

#         return model_output

#     def test_process_valid_input(self, processor, sample_input):
#         """Test processing of valid model ouput."""
        
        
#         classes_name = ['red', 'blue', 'green', 'yellow']
#         assert sample_input.shape[-1] == len(classes_name) + 5
#         model_input_height = 120 
#         model_input_width = 160 
#         image_height = 120 
#         image_width = 160
#         confidence_thr = 0.5 
#         iou_threshold = 0.5
#         max_output_boxes = 6
#         anchors = [0.14,0.19, 0.13,0.52, 0.16,0.31, 0.45,0.62, 0.28,0.38]
        
#         result = processor(sample_input, classes_name, anchors, model_input_height, model_input_width, image_height, 
#                                 image_width, max_output_boxes, confidence_thr, iou_threshold)

#         assert isinstance(result, list)
#         if len(result) != 0:
#             for bbox in result:
#                 assert len(bbox) == 6
                
#     @pytest.fixture
#     def real_output(self, root_dir):
#         anchorsBased_model_output = np.load(os.path.join(str(root_dir), 'results', 'anchorsBased_output.npy'))
#         return anchorsBased_model_output
    
#     def test_process_real_input(self, processor, real_output):
#         """Test processing of valid model ouput."""
        
#         classes_name = ['red', 'blue', 'green', 'yellow']
        
#         assert real_output.shape[-1] == len(classes_name) + 5
        
#         model_input_height = 160 
#         model_input_width = 160 
#         image_height = 160 
#         image_width = 160
#         confidence_thr = 0.5 
#         iou_threshold = 0.5
#         max_output_boxes = 6
#         anchors = [0.14,0.19, 0.13,0.52, 0.16,0.31, 0.45,0.62, 0.28,0.38]
        
#         result = processor(real_output, classes_name, anchors, model_input_height, model_input_width, image_height, 
#                                 image_width, max_output_boxes, confidence_thr, iou_threshold)

#         assert isinstance(result, list)
#         if len(result) != 0:
#             for bbox in result:
#                 assert len(bbox) == 6