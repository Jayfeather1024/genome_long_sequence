import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
# import pdb; pdb.set_trace()
#generated = torch.load('generation_outputs_highlevel/samples.pt')
#
#generated = torch.load('/n/rush_lab/Lab/Users/yuntian/language_modeling_via_stochastic_processes/Diffusion-LM/improved-diffusion/generation_outputs_highlevel/samples.pt')
#ground_truth = torch.load('/n/rush_lab/Lab/Users/yuntian/language_modeling_via_stochastic_processes/language_modeling_via_stochastic_processes/transformers/examples/pytorch/language-modeling/padded_sentence_embeddings.pt')'



filename = '/n/holyscratch01/rush_lab/Users/yuntian/jiongxiao_roc_stories/encoder_28289/run_l0.0005_b32_cbow_diff/diffusion/seed_81482_lr0.00002/sample_21395_repaint/zs.pt.masked.seeds.1'
zs = torch.load(filename)[:50]

gt = torch.stack([item['padded_sentence_embeddings'] for item in zs], dim=0)

size = gt.shape[1]

zz = []
for item in zs:
    z = []
    for sss in range(size):
        z.append(item[f'masked_{sss}'])
    z = torch.stack(z, dim=0)
    zz.append(z)
generated = torch.stack(zz, dim=0)
#import pdb; pdb.set_trace()



mean = generated.mean(0).cpu()
std = generated.std(0).cpu()
gt_mean = gt.mean(0).cpu()
gt_std = gt.std(0).cpu()


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
vmin = gt_mean.min()
vmax = gt_mean.max()
# import pdb; pdb.set_trace()
vmin = -5
vmax = 5
im0 = axes[0].imshow(gt_mean.data.numpy(), cmap='jet', vmin=vmin, vmax=vmax, interpolation='nearest')
im1 = axes[1].imshow(mean.data.numpy(), cmap='jet', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[0].set_axis_off()
axes[1].set_axis_off()
axes[1].cax.colorbar(im1)
axes[1].cax.toggle_label(True)
plt.savefig('stats_repaint/infill_mean_comp.png', bbox_inches='tight', pad_inches=0.1, dpi=200,)


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
vmin = gt_std.min()
vmax = gt_std.max()
# import pdb; pdb.set_trace()
im0 = axes[0].imshow(gt_std.data.numpy(), cmap='jet', vmin=vmin, vmax=vmax, interpolation='nearest')
im1 = axes[1].imshow(std.data.numpy(), cmap='jet', vmin=vmin, vmax=vmax, interpolation='nearest')
axes[0].set_axis_off()
axes[1].set_axis_off()
axes[1].cax.colorbar(im1)
axes[1].cax.toggle_label(True)
plt.savefig('stats_repaint/infill_std_comp.png', bbox_inches='tight', pad_inches=0.1, dpi=200,)