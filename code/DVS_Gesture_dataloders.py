from torch import tensor

from transforms import *
import h5py
import os
from torch.utils.data import Dataset
from events_timeslices import *


mapping = {0: 'Hand Clapping',
           1: 'Right Hand Wave',
           2: 'Left Hand Wave',
           3: 'Right Arm CW',
           4: 'Right Arm CCW',
           5: 'Left Arm CW',
           6: 'Left Arm CCW',
           7: 'Arm Roll',
           8: 'Air Drums',
           9: 'Air Guitar',
           10: 'Other'}


class DVSGestureDataset(Dataset):
    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 chunk_size=500,
                 clip=10,
                 is_train_Enhanced=False,
                 dt=1000
                 ):
        super(DVSGestureDataset, self).__init__()

        self.n = 0
        self.root = root
        self.train = train
        self.chunk_size = chunk_size
        self.clip = clip
        self.is_train_Enhanced = is_train_Enhanced
        self.dt = dt
        self.transform = transform
        self.target_transform = target_transform

        if train:
            root_train = os.path.join(self.root, 'train')
            for _, _, self.files_train in os.walk(root_train):
                pass
            self.n = len(self.files_train)
        else:
            root_test = os.path.join(self.root, 'test')
            for _, _, self.files_test in os.walk(root_test):
                pass
            self.n = len(self.files_test)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):

        # Important to open and close in getitem to enable num_workers>0

        if self.train:
            assert idx < 1175
            root_test = os.path.join(self.root, 'train')

            with h5py.File(root_test + os.sep + self.files_train[idx], 'r', swmr=True, libver="latest") as f:
                data, target = sample_train(
                    f, T=self.chunk_size, is_train_Enhanced=self.is_train_Enhanced, dt=self.dt)

            if self.transform is not None:
                data = self.transform(data)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return data, target
        else:
            assert idx < 288
            root_test = os.path.join(self.root, 'test')

            with h5py.File(root_test + os.sep + self.files_test[idx], 'r', swmr=True, libver="latest") as f:
                data, target = sample_test(f,
                                           T=self.chunk_size,
                                           clip=self.clip,
                                           dt=self.dt)

            data_temp = []
            target_temp = []
            for i in range(self.clip):

                if self.transform is not None:
                    data_temp.append(self.transform(data[i]))

                if self.target_transform is not None:
                    target_temp.append(self.target_transform(target))

            for i in range(self.clip):
                data_temp[i] = data_temp[i].numpy()
                target_temp[i] = target_temp[i].numpy()

            data = np.array(data_temp)
            target = np.array(target_temp)
            data = torch.from_numpy(data)
            target = torch.from_numpy(target)

            return data, target


def sample_train(hdf5_file,
                 T=60,
                 dt=1000,
                 is_train_Enhanced=False):
    label = hdf5_file['labels'][()]

    tbegin = hdf5_file['times'][0]
    tend = np.maximum(0, hdf5_file['times'][-1] - T * dt)

    start_time = np.random.randint(tbegin, tend) if is_train_Enhanced else 0

    tmad = get_tmad_slice(hdf5_file['times'][()],
                          hdf5_file['addrs'][()],
                          tbegin,
                          T * dt)
    tmad[:, 0] -= tmad[0, 0]
    return tmad[:, [0, 3, 1, 2]], label


def sample_test(hdf5_file,
                T=60,
                clip=10,
                dt=1000
                ):

    label = hdf5_file['labels'][()]

    tbegin = hdf5_file['times'][0]
    tend = np.maximum(0, hdf5_file['times'][-1])

    tmad = get_tmad_slice(hdf5_file['times'][()],
                          hdf5_file['addrs'][()],
                          tbegin,
                          tend - tbegin)
    # 初试从零开始
    tmad[:, 0] -= tmad[0, 0]

    start_time = tmad[0, 0]
    end_time = tmad[-1, 0]

    start_point = []
    if clip * T * dt - (end_time - start_time) > 0:
        overlap = int(
            np.floor((clip * T * dt - (end_time - start_time)) / clip))
        for j in range(clip):
            start_point.append(j * (T * dt - overlap))
            if start_point[-1] + T * dt > end_time:
                diff = start_point[-1] + T * dt - end_time
                start_point[-1] = start_point[-1] - diff
    else:
        overlap = int(
            np.floor(((end_time - start_time) - clip * T * dt) / clip))
        for j in range(clip):
            start_point.append(j * (T * dt + overlap))
            if start_point[-1] + T * dt > end_time:
                diff = start_point[-1] + T * dt - end_time
                start_point[-1] = start_point[-1] - diff

    temp = []
    for start in start_point:
        idx_beg = find_first(tmad[:, 0], start)
        idx_end = find_first(tmad[:, 0][idx_beg:], start + T * dt) + idx_beg

        temp.append(np.column_stack([tmad[idx_beg:idx_end]])[:, [0, 3, 1, 2]])

    return temp, label


def create_datasets(root=None,
                    train=True,
                    chunk_size_train=60,
                    chunk_size_test=60,
                    ds=4,
                    dt=1000,
                    transform_train=None,
                    transform_test=None,
                    target_transform_train=None,
                    target_transform_test=None,
                    n_events_attention=None,
                    clip=10,
                    is_train_Enhanced=False,
                    ):

    if isinstance(ds, int):
        ds = [ds, ds]

    size = [2, 128 // ds[0], 128 // ds[1]]

    if n_events_attention is None:
        def default_transform(chunk_size): return Compose([
            Downsample(factor=[dt, 1, ds[0], ds[1]]),
            ToCountFrame(T=chunk_size, size=size),
            ToTensor()
        ])
    else:
        def default_transform(chunk_size): return Compose([
            Downsample(factor=[dt, 1, 1, 1]),
            Attention(n_events_attention, size=size),
            ToCountFrame(T=chunk_size, size=size),
            ToTensor()
        ])

    if transform_train is None:
        transform_train = default_transform(chunk_size_train)
    if transform_test is None:
        transform_test = default_transform(chunk_size_test)

    if target_transform_train is None:
        target_transform_train = Compose(
            [Repeat(chunk_size_train), toOneHot(11)])
    if target_transform_test is None:
        target_transform_test = Compose(
            [Repeat(chunk_size_test), toOneHot(11)])

    if train:

        train_d = DVSGestureDataset(root,
                                    train=train,
                                    transform=transform_train,
                                    target_transform=target_transform_train,
                                    chunk_size=chunk_size_train,
                                    is_train_Enhanced=is_train_Enhanced,
                                    dt=dt)

        # train_dl = torch.utils.data.DataLoader(
        #     train_d, batch_size=batch_size, shuffle=True, **dl_kwargs)
        return train_d
    else:
        test_d = DVSGestureDataset(root,
                                   transform=transform_test,
                                   target_transform=target_transform_test,
                                   train=train,
                                   chunk_size=chunk_size_test,
                                   clip=clip,
                                   dt=dt)

        # test_dl = torch.utils.data.DataLoader(
        #     test_d, batch_size=batch_size, **dl_kwargs)

        return test_d


if __name__ == '__main__':
    path = os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)))) + os.sep + 'dataset' + os.sep + 'DVS_Gesture'

    T = 60
    batch_size = 2
    train_dataset = create_datasets(path,
                                    train=True,
                                    is_train_Enhanced=False,
                                    ds=4,
                                    dt=1000,
                                    chunk_size_train=T,
                                    )
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False)

    test_dataset = create_datasets(path,
                                   train=False,
                                   ds=4,
                                   dt=1000,
                                   chunk_size_train=T,
                                   clip=2
                                   )
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False)


    ho = iter(train_loader)
    input, labels = next(ho)


    print(1)
