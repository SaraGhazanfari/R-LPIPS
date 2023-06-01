def CreateDataLoader(datafolder, data_root='./dataset', dataset_mode='2afc', load_size=64, batch_size=1,
                     serial_batches=True, nThreads=4):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(datafolder, dataroot=data_root + '/' + dataset_mode, dataset_mode=dataset_mode,
                           load_size=load_size, batch_size=batch_size, serial_batches=serial_batches, nThreads=nThreads)
    return data_loader
