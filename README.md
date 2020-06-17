# DeepHealth 2nd Hackathon 04/06/2020

## Improving results on Pneumothorax challenge

The best performance is reached using a neural network ensemble of:

Trained with EDDL:
- SegNet baseline model with cross entropy loss - Dice Coefficient: `0.4788` - checkpoint [here](https://drive.google.com/uc?id=17aCogcCyBXBnFnFFIQ8-A8J960Remeop&export=download)
- Baseline further fine-tuned with the dice loss and reduced learning rate - Dice Coefficient: `0.4819` - checkpoint [here](https://drive.google.com/uc?id=1CN5-R_4jE5Yn6LFyfx8WnnTKNX7mAV85&export=download)

Trained with PyTorch:
- SegNet model with combo loss (BCE + dice + focal loss) - Dice Coefficient: `0.5179` - checkpoint [here](https://drive.google.com/uc?id=1HhKRiyRMse5y1gjvXnY11moqsdNLAR-B&export=download)
- UNet model trained with cross entropy loss - Dice Coefficient: `0.4910` - checkpoint [here](https://drive.google.com/uc?id=1eCa4sCIA3sDiAd5sE-EM-u8lm-HUFEfY&export=download)

Ensemble Dice Coefficient: `0.5665`

## Training

To re-train the networks as we did:
```bash
./PNEUMOTHORAX_SEGMENTATION_TRAINING --loss cross_entropy --model SegNetBN --batch_size 2 --learning_rate 0.0001 --dataset_path /path/to/siim/pneumothorax.yml --gpu 1 
./PNEUMOTHORAX_SEGMENTATION_TRAINING --loss dice --model SegNetBN --batch_size 2 --learning_rate 0.00001 --dataset_path /path/to/siim/pneumothorax.yml --gpu 1 --checkpoint ../checkpoints_pneumothorax/pneumothorax_model_SegNetBN_loss_cross_entropy_lr_0.0001_size_512_epoch_45.onnx

python ../pytorch/pneumothorax/train.py --loss_type Combo --model SegNet --batch_size 6 --scheduler plateau --lr 0.00001 --experiment_name SegNet_combo --dataset_filepath /path/to/siim/pneumothorax.yml --checkpoint_dir /path/where/you/store/checkpoints/
python ../pytorch/pneumothorax/train.py --loss_type BCE --model PadUNet --batch_size 6 --experiment_name UNet_BCE --dataset_filepath /path/to/siim/pneumothorax.yml --checkpoint_dir /path/where/you/store/checkpoints/
```

C++ arguments available for PNEUMOTHORAX_SEGMENTATION_TRAINING:

```
-e, --epochs        Number of training epochs (default: 50)
-b, --batch_size    Number of images for each batch (default: 2)
-n, --num_classes   Number of output classes (default: 1)
-s, --size          Size to which resize the input images (default: 512,512)
--loss              Loss function (default: cross_entropy)
-l, --learning_rate Learning rate (default: 0.0001)
--model             Model of the network (default: SegNetBN)
-g, --gpu           Which GPUs to use. If not given, the network will run on CPU. (default: 1, other examples: --gpu=0,1 or --gpu=1,1)
--lsb               How many batches are processed before synchronizing the model weights (default: 1)
-m, --mem           GPU memory usage configuration (default: low_mem, other possibilities: mid_mem, full_mem)
--save_images       Save validation images or not (default: false)
-r, --result_dir    Directory where the output images will be stored (default: ../output_images_pneumothorax)
--checkpoint_dir    Directory where the checkpoints will be stored (default: ../checkpoints_pneumothorax)
-d, --dataset_path  Dataset path (mandatory)
-c, --checkpoint    Path to the onnx checkpoint file (optional)
-h, --help          Print usage
```

Python arguments available for train.py:

```
--num_epochs            Number of training epochs (default: 100)
--batch_size            Number of images for each batch (default: 2)
--num_classes           Number of output classes (default: 1)
--resize_dims           Size to which resize the input images (default: 512)
--loss_type             Loss function (default: BCE, other possibilities: Focal, Combo, Dice)
--lr                    Learning rate (default: 0.0001)
--scheduler             Scheduler used (default: None, possibilities: plateau)
--model                 Model of the network (default: SegNet, other possibilities: PadUNet)
--experiment_name       Name of the experiment (default: exp1)
--checkpoint_dir        Directory where the checkpoints will be stored (default: None)
--dataset_filepath      Dataset path (default: None, mandatory)
--data_loader_workers   num_workers of Dataloader (default: 8)
--checkpoint_file       Path to the onnx checkpoint file (default: None, optional)
```

## Inference

C++ arguments available for PNEUMOTHORAX_SEGMENTATION_INFERENCE:

```
-b, --batch_size    Number of images for each batch (default: 2)
-n, --num_classes   Number of output classes (default: 1)
-g, --gpu           Which GPUs to use. If not given, the network will run on CPU. (default: 1, other examples: --gpu=0,1 or --gpu=1,1)
--lsb               How many batches are processed before synchronizing the model weights (default: 1)
-m, --mem           GPU memory usage configuration (default: low_mem, other possibilities: mid_mem, full_mem)
--save_images       Save validation images or not (default: false)
--save_gt           Save validation ground truth or not (default: false)
-r, --result_dir    Directory where the output images will be stored (default: ../output_images_pneumothorax)
-t, --gt_dir        Directory where the ground_truth images will be stored (default: ../ground_truth_images_pneumothorax)
-d, --dataset_path  Dataset path (mandatory)
-c, --checkpoint    Path to the onnx checkpoint file (optional)
-h, --help          Print usage
```

C++ arguments available for ENSEMBLE:

```
-s, --segnet_ce     Folder with output images from the baseline
-d, --segnet_dice   Folder with output images from SegNetBN with dice loss
-c, --segnet_combo  Folder with output images from SegNetBN with combo loss
-u, --unet          Folder with output images from the UNet with BCE
-g, --ground_truth  Folder with ground_truth images (mandatory)
-h, --help          Print usage
```

The outputs on the validation images must be saved for each of the four networks. 
```bash
./PNEUMOTHORAX_SEGMENTATION_INFERENCE --dataset_path /path/to/siim/pneumothorax.yml --checkpoint /path/to/pneumothorax_model_SegNetBN_loss_ce_lr_0.0001_size_512_epoch_45.onnx --save_images --result_dir ../output_images_SegNetBN_loss_ce_lr_0.0001 --save_gt --gpu 1
./PNEUMOTHORAX_SEGMENTATION_INFERENCE --dataset_path /path/to/siim/pneumothorax.yml --checkpoint /path/to/pneumothorax_model_SegNetBN_loss_dice_lr_0.00001_size_512_epoch_64.onnx --save_images --result_dir ../output_images_SegNetBN_loss_dice_lr_0.00001 --gpu 1
./PNEUMOTHORAX_SEGMENTATION_INFERENCE --dataset_path /path/to/siim/pneumothorax.yml --checkpoint /path/to/pneumothorax_model_SegNet_loss_Combo_lr_0.00001_size_512.onnx --save_images --result_dir ../output_images_SegNetBN_loss_combo_lr_0.00001_plateau --gpu 1
./PNEUMOTHORAX_SEGMENTATION_INFERENCE --dataset_path /path/to/siim/pneumothorax.yml --checkpoint /path/to/pneumothorax_model_UNet_loss_BCE_lr_0.0001_size_512.onnx --save_images --result_dir ../output_images_UNetBN_loss_BCE --gpu 1
```
Note that at least once (in the first execution in this example) you have to save also the ground_truth images with `--save_gt` in order to have them all in one folder.

Now, you can run the ensemble of all the networks:
```bash
./ENSEMBLE --ground_truth ../ground_truth_images_pneumothorax --segnet_ce ../output_images_SegNetBN_loss_ce_lr_0.0001 --segnet_dice ../output_images_SegNetBN_loss_dice_lr_0.00001 --segnet_combo ../output_images_SegNetBN_loss_combo_lr_0.00001_plateau --unet ../output_images_UNetBN_loss_BCE
```

Four metrics will be calculated:
- Mean dice coefficient 
- Mean dice coefficient with post processing with triplet (0.5, 300, 0.3) (see [this](https://github.com/sneddy/pneumothorax-segmentation#triplet-scheme-of-inference-and-validation) for more details)
- Mean dice coefficient with images that are binarized before building the ensembled image  
- Mean dice coefficient with images that are binarized before building the ensembled image and with post processing