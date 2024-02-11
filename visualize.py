import torch
from model import Transformer
from modelviz import draw_graphs


model = Transformer()
model.eval()

x_encoder = torch.rand(1, 3, 512)
x_decoder = torch.rand(1, 3, 512)
x_encoder.label = 'input-encoder'
x_decoder.label = 'input-decoder'


draw_graphs(
    model, 
    (x_encoder, x_decoder), 
    1, 10, 
    directory='./model_viz/', 
    hide_module_functions=False, 
    input_names=('input-encoder', 'input-decoder'), 
    output_names='output'
)
