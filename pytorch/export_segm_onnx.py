import os
import subprocess
import sys

import segmentation_models_pytorch as smp
import torch

if __name__ == '__main__':
    print('Require: `pip install torch segmentation-models-pytorch onnx-simplifier`')
    os.makedirs('segmmodels', exist_ok=True)

    encoders = ['resnet18', 'resnet34', 'resnet50', 'resnet101']
    models = [
        smp.DeepLabV3Plus,
        smp.Unet,
        smp.UnetPlusPlus,
        # smp.FPN,
        # smp.Linknet,
        # smp.PAN,
    ]
    dummy_input = torch.ones(12, 3, 224, 224, device='cpu')

    for encoder_name in encoders:
        for m in models:
            model = m(
                encoder_name=encoder_name,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=1,  # model output channels (number of classes in your dataset)
                aux_params=None,
            )
            model_name = model.__class__.__name__

            # export onnx
            fname = f'segmmodels/{model_name}_{encoder_name}.onnx'
            if os.path.exists(fname):
                continue
            print(f'Exporting {model_name}_{encoder_name}')
            model.train()
            try:
                torch.onnx.export(model, dummy_input, fname,
                                  verbose=False,
                                  export_params=True,
                                  training=torch.onnx.TrainingMode.TRAINING,
                                  opset_version=12,
                                  do_constant_folding=False,
                                  input_names=['input'],
                                  output_names=['output'],
                                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                                'output': {0: 'batch_size'}},
                                  # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
                                  )

                new_f = f'segmmodels/{model_name}_{encoder_name}_simpl.onnx'
                subprocess.run(
                    f'python -m onnxsim {fname} {new_f} --dynamic-input-shape --input-shape 12,3,224,224',
                    shell=True)
            except:
                print(f'Error exporting {model_name}_{encoder_name}')
                print("Unexpected error:", sys.exc_info()[0])
