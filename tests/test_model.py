import torch


def test_model():
    test_input = torch.randn(3,200,200)
    model = ...
    out = model(test_input)
    assert out.shape == torch.Size([120]), "Output shape is incorrect"
    pass