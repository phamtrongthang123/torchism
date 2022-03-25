# Torchism 
General template for my PyTorch projects.
I will add more examples other than supervised mnist in the future.

## Dependency 
```
pip install -r requirements.txt
```

## Sanity run 
Try using the script in `data_generator` or download mnist data (.csv) from `https://github.com/pjreddie/mnist-csv-png` and save it same as the paths in `configs/train/sample.yaml`. Then run:
```
python train.py --config configs/train/sample.yaml --gpus 0
tensorboard --logdir=runs 
```

### TODO 
- [ ] Save min val loss 
- [ ] consider remove supervised or make it the same as lit_mnist, then make train.py is just parsing arg. 
- [ ] then remove unnecessary files, like tsboard, loggers folder ... 
- [ ] to make it better than orginal torchan, make it cleaner and list a bunch of common module that I can use right away. Or try to write it in a way that I can see clearly its advantages.
    - I want to be able to write some module and it can run right away. 
    - I don't want to worry about looking for model to mask nnParallel when use multiple gpus
    - I want to be able to plug and play with multuple SoTA instrument to boost model or debug the model. (https://pytorch-lightning.readthedocs.io/en/stable/common/debugging.html)
