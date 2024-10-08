# Torchism 
General template for my PyTorch projects.
I will add more examples other than supervised mnist in the future.

## Dependency 
```
pip install -r requirements.txt
```

## Data folder 
Example `./data` structure
```
.
├── CIFAR10
│   └── cifar-10-batches-py
│       ├── batches.meta
│       ├── data_batch_1
│       ├── data_batch_2
│       ├── data_batch_3
│       ├── data_batch_4
│       ├── data_batch_5
│       ├── readme.html
│       └── test_batch
└── MNIST
    ├── mnist-csv-png
    │   ├── process_mnist.py
    │   ├── process.sh
    │   └── README.md
    ├── mnist_test.csv
    └── mnist_train.csv
```

## Sanity run 
Try using the script in `data_generator` or download mnist data (.csv) from `https://github.com/pjreddie/mnist-csv-png` and save it same as the paths in `configs/train/sample.yaml`. 
Remember to set env variable before run the train.py script, and change the trainer logger to Neptune logger (currently we are using neptune logger)
```bash
export NEPTUNE_API_TOKEN="<key>"
python train.py --config configs/train/sample.yaml --gpus 0
```

Or if you want a quick test, change the neptune logger in the trainer to tensorboard and run this:
```
python train.py --config configs/train/sample.yaml --gpus 0
tensorboard --logdir=runs 
```

## TODO: 
### Lazy mode training
I'm working on an idea of pushing the data splitting / processing to the model code only. 
At train loop, it is just 
```
train_iter = iter(dataloader)
batch = next(train_iter)

loss, _ = model(batch)
```
And the dataset class is like 
```
for metadata in filepath: {k:read(v)  for kv in metadat}
return dict / tensor
```
Remember to use collate_fn [x for x in batch] at the end of loader to merge them. 

Then the rest of the job about data is for model to play around.
### Versioning each exp
Even if it fails, commit a tag into it with comments. Remember this common practice.



