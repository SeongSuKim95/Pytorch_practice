# STEPS:

# 1. Specify how the preprocessing should be done -> Fields
# 2. Use dataset to load the data -> TabularDataset(JSON/CSV/TSV files)
# 3. Construct an iterator to do batching & padding -> BucketIterator

from torchtext.data import Field, TabularDataset, BucketIterator

tokenize = lambda x : x.split()

quote = Field(sequential=True, use_vocab = True, tokenize = tokenize, lower = True)
score = Field(sequential=False, use_vocab = False)

fields = {'quote':('q', quote), 'score':('s',score)}

train_data, test_data = TabularDataset.splits(
                            path = '/home/sungsu21/Project/Pytorch_practice',
                            train = 'example.json',
                            test = 'example.json',
                            #validation = 'validation.json,'
                            format = 'json',
                            fields = fields)

# train_data, test_data = TabularDataset.splits(
#                             path = 'mydata',
#                             train = 'example.csv',
#                             test = 'test.csv',
#                             format = 'csv',
#                             fields = fields)


print(train_data[0].__dict__.keys())