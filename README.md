# CISC867Project

To run segmentation on DRIVE dataset, first set the path to your pre-trained model in line 204 of engine.py. Then run the command:
python main_segmentation.py --data_set DRIVE  \
--init ImageNet \
--train_data_dir path/to/train/images \
--train_mask_dir path/to/train/masks \
--valid_data_dir path/to/validation/images \
--valid_mask_dir path/to/validation/masks \
--test_data_dir path/to/test/images \
--test_mask_dir path/to/test/masks

