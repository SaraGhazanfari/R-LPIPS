import argparse
import lpips
import torch

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-p0','--path0', type=str, default='./imgs/ex_ref.png')
parser.add_argument('-p1','--path1', type=str, default='./imgs/ex_p0.png')
parser.add_argument('-v','--version', type=str, default='0.1')
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()
my_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("mps")
## Initializing the model
loss_fn = lpips.LPIPS(net='alex',version=opt.version)

if(opt.use_gpu):
	loss_fn.to(my_device)

# Load images
img0 = lpips.im2tensor(lpips.load_image(opt.path0)) # RGB image from [-1,1]
img1 = lpips.im2tensor(lpips.load_image(opt.path1))

if(opt.use_gpu):
	img0 = img0.to(my_device)
	img1 = img1.to(my_device)

# Compute distance
dist01 = loss_fn.forward(img0, img1)
print('Distance: %.3f'%dist01)
