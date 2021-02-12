##Practical tips

# 1. Didn't overfit a single Batch

# Get Neural network to be overfitted on single batch
# 
# --->> data, targets = next(iter(train_loader))
# To see neural networks have capability and no bugs : Sanity check of network is actually working


# 2. Forgot toggle train / eval

 
# model.eval()
# check_accuracy(test_loader,model)
# model.train()

# model.eval() --> We don't use dropout, Batchnorm ,,,

# 3. Forgot .zero_grad()
# 
# If not, we use all the accumulated gradient of previous batches
# We want gradient of current batch.

# 4. Softmax with CrossEntropyloss  
#
# CrossEntopyloss already contains softmax.

# 5. Using bias when using Batchnorm --> We don't need to use bias in conv layer when using Batch norm

# 6. Using view as permute

import torch
x = torch.tensor([[1,2,3],[4,5,6]])

print(x)
#tensor([[1, 2, 3],
#        [4, 5, 6]])
print(x.view(3,2)) 
#Just get elements sequentially and make shape
#tensor([[1, 2],
#        [3, 4],
#        [5, 6]])
print(x.permute(1,0)) ## Transpose
#tensor([[1, 4],
#        [2, 5],
#        [3, 6]])

# 7. Using bad augmentation
# --> Should not actually modify target

# 8. Not shuffling the data

# 9. Not Normalizing data

# 10. Not Clipping Gradients( RNNS, GRUS, LSTM)
# -> You might get gradient exploding problem
# torch.nn.utils.clip_grad_norm(model.parameters(), max_norm =1)