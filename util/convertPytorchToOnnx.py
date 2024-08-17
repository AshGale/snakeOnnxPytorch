import torch
import torch.nn as nn

class SnakeNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SnakeNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def convert_pth_to_onnx(pth_path, onnx_path, input_size, hidden_size, output_size):
    # Load the saved PyTorch model
    model = SnakeNN(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
    model.load_state_dict(torch.load(pth_path))
    model.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(1, input_size)

    # Export the model to ONNX format
    torch.onnx.export(model, dummy_input, onnx_path, verbose=True, input_names=['input'], output_names=['output'])
    print(f"Model converted and saved as {onnx_path}")

if __name__ == "__main__":
    pth_path = "snake_model.pth"  # Path to your saved PyTorch model
    onnx_path = "snake_model_converted.onnx"  # Path for the output ONNX model
    input_size = 5
    hidden_size = 64  # Changed from 64 to 128
    output_size = 4

    convert_pth_to_onnx(pth_path, onnx_path, input_size, hidden_size, output_size)