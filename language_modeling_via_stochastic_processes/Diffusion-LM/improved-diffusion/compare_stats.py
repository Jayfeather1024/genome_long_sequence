import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import pdb; pdb.set_trace()
generated = torch.load('generation_outputs_highlevel/samples.pt')

generated = torch.load('/n/rush_lab/Lab/Users/yuntian/language_modeling_via_stochastic_processes/Diffusion-LM/improved-diffusion/generation_outputs_highlevel/samples.pt')
ground_truth = torch.load('/n/rush_lab/Lab/Users/yuntian/language_modeling_via_stochastic_processes/language_modeling_via_stochastic_processes/transformers/examples/pytorch/language-modeling/padded_sentence_embeddings.pt')
gt = torch.stack([item['padded_sentence_embeddings'] for item in ground_truth['test']], dim=0)

mean = generated.mean(0)
std = generated.std(0)
gt_mean = gt.mean(0)
gt_std = gt.std(0)


fig = plt.figure(figsize=(9.75, 3))
    
axes = ImageGrid(fig, 111,
                 nrows_ncols=(1,2),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
)
import pdb; pdb.set_trace()
im0 = axes[0].imshow(gt_mean.data.numpy(), cmap='jet', vmin=-12, vmax=7, interpolation='nearest')
im1 = axes[1].imshow(mean.data.numpy(), cmap='jet', vmin=-12, vmax=7, interpolation='nearest')
axes[0].set_axis_off()
axes[1].set_axis_off()
axes[1].cax.colorbar(im1)
axes[1].cax.toggle_label(True)
plt.savefig('mean_comp.png', bbox_inches='tight', pad_inches=0.1, dpi=200,)


fig = plt.figure(figsize=(9.75, 3))
    
axes = ImageGrid(fig, 111,
                 nrows_ncols=(1,2),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
)
import pdb; pdb.set_trace()
im0 = axes[0].imshow(gt_std.data.numpy(), cmap='jet', vmin=-12, vmax=7, interpolation='nearest')
im1 = axes[1].imshow(std.data.numpy(), cmap='jet', vmin=-12, vmax=7, interpolation='nearest')
axes[0].set_axis_off()
axes[1].set_axis_off()
axes[1].cax.colorbar(im1)
axes[1].cax.toggle_label(True)
plt.savefig('std_comp.png', bbox_inches='tight', pad_inches=0.1, dpi=200,)
