import torch
import torchvision.models as models
from pathlib import Path
from torchvision.models import resnet18, ResNet18_Weights

def export_model(export_dir: Path):
    if not export_dir.exists():
        export_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path("model.onnx")
    model_path = export_dir / model_name
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    model.eval()

    example_input = torch.randn(1, 3, 224, 224)
    # batch_dim = torch.export.Dim("batch")
    # dynamic_shapes = ({0 : batch_dim},)   #!

    torch.onnx.export(
        model,
        (example_input,),
        str(model_path),
        input_names=["input"],
        output_names=["output"],
        dynamo=False,                     #!
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        # dynamo=True,
        # dynamic_shapes={"x" : {0 : batch_dim}},
        # dynamic_shapes=({0: batch_dim},),
        # dynamic_shapes=dynamic_shapes,   #!
        # opset_version=17,
        report=True     ##!!!
        ### or :dymamic_shapes = {"input" : {0 : batch_dim}}
        # verify=True
    )

    return model_path

if __name__ == "__main__":
    model_path = Path("/home/daydreamer/Desktop/study/python-onnx-demo/model")
    export_model(model_path)
