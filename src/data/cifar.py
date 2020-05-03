import torchvision
import numpy as np

def get_argdict(dataset):
    if dataset == 'cifar10':
        argdict = {'root' : './data_cifar10', 
                   'tvds_cls' : torchvision.datasets.CIFAR10,
                   }
    elif dataset == 'cifar100':
        argdict = {'root' : './data_cifar100', 
                   'tvds_cls' : torchvision.datasets.CIFAR100,
                   }
    return argdict

def get_dataset(dataset, is_train, download, Dataset_class, transform, 
                usage_rate_per_class=None, seed=None, 
                return_target=False):
    argdict = get_argdict(dataset)
    data = argdict['tvds_cls'](root=argdict['root'], train=is_train, download=download)

    inputs, target = [], []
    for dt in data:
        inp, tg = dt
        inputs.append(np.array(inp))
        target.append(tg)

    inputs = np.array(inputs)
    target = np.array(target).astype('int64')

    if usage_rate_per_class is not None:
        np.random.seed(seed)
        new_inps = []
        new_tgs = []

        all_idxs = np.arange(len(target))
        for i, ur in enumerate(usage_rate_per_class):
            tg_idxs = all_idxs[target==i]
            use_idxs = np.random.choice(tg_idxs, size=int(len(tg_idxs) * ur), replace=False)
            new_inps.append(inputs[use_idxs])
            new_tgs.append(target[use_idxs])
            print('label {0} : usage_rate {1}, num data {2}'.format(i, ur, len(use_idxs)))
        inputs = np.concatenate(new_inps, axis=0)
        target = np.concatenate(new_tgs, axis=0)

    ds_instance = Dataset_class(inputs, target, transform)

    if return_target:
        return ds_instance, target
    else:
        return ds_instance

def get_dataset_cifar100(is_train, download, Dataset_class, transform, usage_rate_per_class=None, seed=None, return_target=False):
    return get_dataset('cifar100', is_train, download, Dataset_class, transform, usage_rate_per_class, seed, return_target)

def get_dataset_cifar10(is_train, download, Dataset_class, transform, usage_rate_per_class=None, seed=None, return_target=False):
    return get_dataset('cifar10', is_train, download, Dataset_class, transform, usage_rate_per_class, seed, return_target)
