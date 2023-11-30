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


This runs the code from BenchmarkTransferLearning, however the following code was added to the original repository in engine.py:
```
if args.init.lower() == "imagenet":
            print("Training DiRA")
            # Initialize U-Net with the specified backbone and 'imagenet' pre-trained weights
            model = smp.Unet(args.backbone, encoder_weights='imagenet', activation=args.activate)

            # Load your custom weights
            weight = 'checkpoint.pth'
            state_dict = torch.load(weight, map_location="cpu")

            # Check if state dict is nested under 'state_dict' key
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

            # Adjust the keys by removing 'module.', 'backbone.', 'encoder_k.', and 'encoder_q.' prefixes
            state_dict = {k.replace("module.", "").replace("backbone.", "").replace("encoder_k.", "").replace("encoder_q.", ""): v for k, v in state_dict.items()}

            # Remove keys that do not belong to the U-Net architecture
            for k in list(state_dict.keys()):
                if k.startswith('fc') or k.startswith('segmentation_head'):
                    del state_dict[k]

            # Fetch the U-Net model's keys before loading the state_dict
            unet_state_dict = model.state_dict()
            unet_keys = set(unet_state_dict.keys())

            # Load the adjusted state dictionary into the model
            msg = model.load_state_dict(state_dict, strict=False)
            print("=> loaded pre-trained model '{}'".format(weight))
            print("missing keys:", msg.missing_keys)

            # Determine which keys from the pre-trained model were used
            modified_custom_keys = set(state_dict.keys())  # Get the modified keys from your custom state dictionary
            used_keys = modified_custom_keys.intersection(unet_keys)
            #print("Keys from the pre-trained model that were used:", used_keys)
```
The above code loads a standard imagenet model, and then updates its weights with the weights from the pre-trained model "checkpoint.pth"

## Classification

The following instructions were pulled in part from the [DiRA Github Page](https://github.com/fhaghighi/DiRA).

Clone the DiRA repository and install requirements by executing the following command:

```
$ git clone https://github.com/fhaghighi/DiRA.git
```

From here, please replace the necessary files with our files located in the DiRA folder.

Then, using our requirements.txt file, run the following:

```
$ pip3 install -r requirements.txt
```
From here, there are no additional changes that need to be made, and the rest can be run in a straightforward manner.

Run the following to warm up the DiRA encoder and decoder (change epochs as desired)

```
$ python main_DiRA_moco.py /path/to/images/folder --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
--world-size 1 --rank 0 --mlp --moco-t 0.2  --cos --mode dir --epochs 10
```

Then, run the following to train the DiRA model, varying the number of epochs as desired:

```
python main_DiRA_moco.py /path/to/images/folder --dist-url 'tcp://localhost:10001' --multiprocessing-distributed \
--world-size 1 --rank 0 --mlp --moco-t 0.2  --cos --mode dira --batch-size 16   --epochs 10 --generator_pre_trained_weights checkpoint/DiRA_moco/dir/checkpoint.pth 
```

From here, we refer to the [Benchmark Transfer Learning Github page](https://github.com/MR-HosseinzadehTaher/BenchmarkTransferLearning). Please ensure that the necessary files are downloaded from there, and that our provided files have been used to replace the files of the same name in their repo.

Run the following to download the necessary models:

```
python download_and_prepare_models.py
```

Then, copy and paste the "checkpoint.pth" file that was created from training DiRA (located at checkpoint/DiRA_moco/dira) into the BenchmarkTransferLearning/models folder with the other downloaded models, and rename it however you want. To complete the training and testing process run the following:

```
python main_classification.py --data_set ChestXray14  \
--init DiRA \
--proxy_dir path/to/pre-trained-model \
--data_dir path/to/dataset \
--train_list dataset/Xray14_train_official.txt \
--val_list dataset/Xray14_val_official.txt \
--test_list dataset/Xray14_test_official.txt 
```

This should train and test the model, outputting the AUC values over 14 trials.
