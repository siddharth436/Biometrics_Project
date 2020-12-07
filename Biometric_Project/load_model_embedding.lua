require 'torch'
require 'optim'

require 'paths'

require 'xlua'

require 'nn'
require 'dpnn'
require 'image'

imName = arg[1]
torch.setdefaulttensortype('torch.FloatTensor')

model = torch.load('nn4.v1.t7')
model:evaluate()

out = image.load(imName, 3, 'float')

out = image.scale(out, 96, 96)

embeddings = model:forward(out):float()

print (embeddings)
