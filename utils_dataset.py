from utils_libs import *

class DatasetObject:
    def __init__(self, dataset, n_client, rule, rule_arg=''):
        self.dataset  = dataset
        self.n_client = n_client
        self.rule     = rule
        assert self.rule == 'ALID' or self.rule == 'ACID', 'Rule should be ALID or ACID.'
        self.rule_arg = rule_arg
        rule_arg_str = rule_arg if isinstance(rule_arg, str) else '%.3f' % rule_arg
        self.name = "%s_%d_%s_%s" %(self.dataset, self.n_client,self.rule, rule_arg_str)
        self.set_data()
        
    def set_data(self):
        # Prepare data if not ready
        if not os.path.exists('Data/%s' %(self.name)):
            # Get Raw data                            
            if self.dataset == 'CIFAR10':                
                transform = transforms.Compose([transforms.ToTensor()])

                trnset = torchvision.datasets.CIFAR10(root='Data/', train=True , download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR10(root='Data/', train=False, download=True, transform=transform)
                
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
    
            if self.dataset == 'CIFAR100':                
                transform = transforms.Compose([transforms.ToTensor()])

                trnset = torchvision.datasets.CIFAR100(root='Data/', train=True , download=True, transform=transform)
                tstset = torchvision.datasets.CIFAR100(root='Data/', train=False, download=True, transform=transform)
                
                trn_load = torch.utils.data.DataLoader(trnset, batch_size=50000, shuffle=False, num_workers=1)
                tst_load = torch.utils.data.DataLoader(tstset, batch_size=10000, shuffle=False, num_workers=1)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;

            if self.dataset == 'miniImageNet':
        
                # There are train.train, train.val, train.test, val and test files.
                # Train.train, val and test files has different classes with 600 datapoints each.
                # Combine train.train, val and test files into 60K images.
                # Split this as train and test as in CIFAR100 dataset. Train 50K, Test 10K
                total_data = []
                path_list = []
                
                assert os.path.exists('Data/miniImageNet/miniImageNet_category_split_train_phase_train.pickle'), 'Error download minImageNet dataset and unzip it...'
                assert os.path.exists('Data/miniImageNet/miniImageNet_category_split_val.pickle'), 'Error download minImageNet dataset and unzip it...'
                assert os.path.exists('Data/miniImageNet/miniImageNet_category_split_test.pickle'), 'Error download minImageNet dataset and unzip it...'
                
                path_list.append("Data/miniImageNet/miniImageNet_category_split_train_phase_train.pickle")
                path_list.append("Data/miniImageNet/miniImageNet_category_split_val.pickle")
                path_list.append("Data/miniImageNet/miniImageNet_category_split_test.pickle")     

                for file in path_list:
                    try:
                        with open(file, 'rb') as fo:
                            data = pickle.load(fo)
                    except:
                        with open(file, 'rb') as f:
                            u = pickle._Unpickler(f)
                            u.encoding = 'latin1'
                            data = u.load()
                    total_data.append(data)

                trn_trn = total_data[0]
                val_ = total_data[1]
                tst_ = total_data[2]
                data_x = np.concatenate((trn_trn['data'], val_['data'], tst_['data']), axis=0)
                data_y = np.concatenate(
                    (np.asarray(trn_trn['labels']), np.asarray(val_['labels']), np.asarray(tst_['labels'])),
                    axis=0).reshape(-1, 1)

                # Get idx of classes
                n_cls = 100
                cls_idx_list = list(range(n_cls))
                for i in range(n_cls):
                    cls_idx_list[i] = np.where(data_y[:, 0] == i)[0]

                trn_idx = [];
                tst_idx = []
                trn_per_cls = 500
                for i in range(n_cls):
                    curr_list = cls_idx_list[i]
                    np.random.shuffle(curr_list)
                    trn_idx.extend(curr_list[:trn_per_cls])
                    tst_idx.extend(curr_list[trn_per_cls:])

                # Set trn and tst, make images as Channel Height Width style
                trn_x = np.moveaxis(data_x[trn_idx], source=3, destination=1)
                trn_y = data_y[trn_idx]

                tst_x = np.moveaxis(data_x[tst_idx], source=3, destination=1)
                tst_y = data_y[tst_idx]
                self.channels = 3; self.width = 84; self.height = 84; self.n_cls = 100;

            if self.dataset != 'miniImageNet':
                trn_itr = trn_load.__iter__(); tst_itr = tst_load.__iter__() 
                # labels are of shape (n_data,)
                trn_x, trn_y = trn_itr.__next__()
                tst_x, tst_y = tst_itr.__next__()

                trn_x = trn_x.numpy(); trn_y = trn_y.numpy().reshape(-1,1)
                tst_x = tst_x.numpy(); tst_y = tst_y.numpy().reshape(-1,1)
            
                mean_ = np.mean(trn_x, axis=(0,2,3))
                std_  = np.std(trn_x, axis=(0,2,3))
            else:
                # Keep it in range 0-255 for the data augmentation part
                # PIL image takes it as 8 bytes (0-255 pixels) so normalize at the end.
                # print(mean_)
                # 120.56728471, 114.33188784, 102.66259233
                # print(std_)
                # 70.52597341, 68.42836057, 71.94073195
                # Divide these numbers with 255 since, we will normalize after transforming to Tensor.
                
                mean_ = np.mean(trn_x, axis=(0, 2, 3))  / 255
                std_ = np.std(trn_x, axis=(0, 2, 3))  / 255
                
            print('mean')
            print(mean_)
            print('std')
            print(std_)
            
            DatasetObject.mean = mean_ 
            DatasetObject.std  = std_ 
            
            # Shuffle Data
            rand_perm = np.random.permutation(len(trn_y))
            trn_x = trn_x[rand_perm]
            trn_y = trn_y[rand_perm]
            
            self.trn_x = trn_x
            self.trn_y = trn_y
            self.tst_x = tst_x
            self.tst_y = tst_y
                        
            assert len(trn_y) % self.n_client == 0
            n_data_per_clnt = int((len(trn_y)) / self.n_client)
            clnt_data_list = np.ones(self.n_client).astype(int) * n_data_per_clnt

            n_data_per_clnt_tst = int((len(tst_y)) / self.n_client)
            clnt_data_list_tst = np.ones(self.n_client).astype(int) * n_data_per_clnt_tst
                
            ###     
            cls_per_client = self.rule_arg
            n_cls = self.n_cls
            n_client = self.n_client

            # Distribute training datapoints
            idx_list = [np.where(trn_y==i)[0] for i in range(self.n_cls)]
            idx_count_list = [0 for i in range(self.n_cls)]
            cls_amount = np.asarray([len(idx_list[i]) for i in range(self.n_cls)])
            n_data = np.sum(cls_amount)
            total_clnt_data_list = np.asarray([0 for i in range(n_client)])
            clnt_cls_idx = [[[] for kk in range(n_cls)] for jj in range(n_client)] # Store the indeces of data points

            while np.sum(total_clnt_data_list) != n_data:
                # Still there are data to distibute
                # Get a random client that among the ones that has the least # of data with respect to totat data it is supposed to have
                min_amount = np.min(total_clnt_data_list - clnt_data_list)
                min_idx_list = np.where(total_clnt_data_list - clnt_data_list ==min_amount)[0]
                np.random.shuffle(min_idx_list)
                cur_clnt = min_idx_list[0]
                print('Current client %d, total remaining amount %d' %(cur_clnt, n_data - np.sum(total_clnt_data_list)))

                # Get its class list
                cur_cls_list = np.asarray([(cur_clnt+jj)%n_cls for jj in range(cls_per_client)])
                # Get the class that has minumum amount of data on the client
                cls_amounts = np.asarray([len(clnt_cls_idx[cur_clnt][jj]) for jj in range(n_cls)])
                min_to_max = cur_cls_list[np.argsort(cls_amounts[cur_cls_list])]
                cur_idx = 0
                while cur_idx!=len(min_to_max) and cls_amount[min_to_max[cur_idx]] == 0:
                    cur_idx += 1
                if cur_idx==len(min_to_max):
                    # This client is not full, it needs data but there is no class data left
                    # Pick a random client and assign its data to this client
                    while True:
                        rand_clnt = np.random.randint(n_client)    
                        print('Random client %d' %rand_clnt)                            
                        if rand_clnt == cur_clnt: # Pick a different client
                            continue
                        rand_clnt_cls = np.asarray([(rand_clnt+jj)%n_cls for jj in range(cls_per_client)])
                        # See if random client has an intersection class with the current client
                        cur_list = np.asarray([(cur_clnt+jj)%n_cls for jj in range(cls_per_client)])
                        np.random.shuffle(cur_list)
                        cls_idx = 0
                        is_found = False
                        while cls_idx != cls_per_client:
                            if cur_list[cls_idx] in rand_clnt_cls and len(clnt_cls_idx[rand_clnt][cur_list[cls_idx]]) > 1:
                                is_found = True
                                break
                            cls_idx += 1
                        if not is_found: # No class intersection, choose another client
                            continue
                        found_cls = cur_list[cls_idx]
                        # Assign this class instance to curr client
                        total_clnt_data_list[cur_clnt]  += 1
                        total_clnt_data_list[rand_clnt] -= 1
                        transfer_idx = clnt_cls_idx[rand_clnt][found_cls][-1]
                        del clnt_cls_idx[rand_clnt][found_cls][-1]
                        clnt_cls_idx[cur_clnt][found_cls].append(transfer_idx)
                        print('Class %d is transferred from %d to %d' %(found_cls, rand_clnt, cur_clnt))
                        break
                else:
                    cur_cls = min_to_max[cur_idx]
                    # Assign one data point from this class to the task
                    total_clnt_data_list[cur_clnt] += 1
                    cls_amount[cur_cls] -= 1
                    clnt_cls_idx[cur_clnt][cur_cls].append(idx_list[cur_cls][cls_amount[cur_cls]])
                    # print('Chosen client: %d, chosen class: %d' %(cur_clnt, cur_cls))

            for i in range(n_cls):
                assert 0 == cls_amount[i], 'Missing datapoints'
            assert n_data == np.sum(total_clnt_data_list), 'Missing datapoints'

            clnt_x = np.asarray([ np.zeros((clnt_data_list[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ])
            clnt_y = np.asarray([ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ])
            for jj in range(n_client):
                clnt_x[jj] = trn_x[np.concatenate(clnt_cls_idx[jj]).astype(np.int32)]
                clnt_y[jj] = trn_y[np.concatenate(clnt_cls_idx[jj]).astype(np.int32)]

            # Distribute test datapoints
            idx_list = [np.where(tst_y==i)[0] for i in range(self.n_cls)]
            idx_count_list = [0 for i in range(self.n_cls)]
            cls_amount = np.asarray([len(idx_list[i]) for i in range(self.n_cls)])
            n_data = np.sum(cls_amount)
            total_clnt_data_list = np.asarray([0 for i in range(n_client)])
            clnt_cls_idx = [[[] for kk in range(n_cls)] for jj in range(n_client)] # Store the indeces of data points

            while np.sum(total_clnt_data_list) != n_data:
                # Still there are data to distibute
                # Get a random client that among the ones that has the least # of data with respect to totat data it is supposed to have
                min_amount = np.min(total_clnt_data_list - clnt_data_list_tst)
                min_idx_list = np.where(total_clnt_data_list - clnt_data_list_tst==min_amount)[0]
                np.random.shuffle(min_idx_list)
                cur_clnt = min_idx_list[0]
                print('Current client %d, total remaining amount %d' %(cur_clnt, n_data - np.sum(total_clnt_data_list)))

                # Get its class list
                cur_cls_list = np.asarray([(cur_clnt+jj)%n_cls for jj in range(cls_per_client)])

                # Get the class that has minumum amount of data on the client
                cls_amounts = np.asarray([len(clnt_cls_idx[cur_clnt][jj]) for jj in range(n_cls)])
                min_to_max = cur_cls_list[np.argsort(cls_amounts[cur_cls_list])]
                cur_idx = 0
                while cur_idx!=len(min_to_max) and cls_amount[min_to_max[cur_idx]] == 0:
                    cur_idx += 1
                if cur_idx==len(min_to_max):
                    # This client is not full, it needs data but there is no class data left
                    # Pick a random client and assign its data to this client
                    while True:
                        rand_clnt = np.random.randint(n_client)
                        print('Random client %d' %rand_clnt)                            
                        if rand_clnt == cur_clnt: # Pick a different client
                            continue
                        rand_clnt_cls = np.asarray([(rand_clnt+jj)%n_cls for jj in range(cls_per_client)])
                        # See if random client has an intersection class with the current client
                        cur_list = np.asarray([(cur_clnt+jj)%n_cls for jj in range(cls_per_client)])
                        np.random.shuffle(cur_list)
                        cls_idx = 0
                        is_found = False
                        while cls_idx != cls_per_client:
                            if cur_list[cls_idx] in rand_clnt_cls and len(clnt_cls_idx[rand_clnt][cur_list[cls_idx]]) > 1:
                                is_found = True
                                break
                            cls_idx += 1
                        if not is_found: # No class intersection, choose another client
                            continue
                        found_cls = cur_list[cls_idx]
                        # Assign this class instance to curr client
                        total_clnt_data_list[cur_clnt]  += 1
                        total_clnt_data_list[rand_clnt] -= 1
                        transfer_idx = clnt_cls_idx[rand_clnt][found_cls][-1]
                        del clnt_cls_idx[rand_clnt][found_cls][-1]
                        clnt_cls_idx[cur_clnt][found_cls].append(transfer_idx)
                        print('Class %d is transferred from %d to %d' %(found_cls, rand_clnt, cur_clnt))
                        break
                else:
                    cur_cls = min_to_max[cur_idx]
                    # Assign one data point from this class to the task
                    total_clnt_data_list[cur_clnt] += 1
                    cls_amount[cur_cls] -= 1
                    clnt_cls_idx[cur_clnt][cur_cls].append(idx_list[cur_cls][cls_amount[cur_cls]])
                    # print('Chosen client: %d, chosen class: %d' %(cur_clnt, cur_cls))
            for i in range(n_cls):
                assert 0 == cls_amount[i], 'Missing datapoints'

            assert n_data == np.sum(total_clnt_data_list), 'Missing datapoints'

            clnt_tst_x = np.asarray([ np.zeros((clnt_data_list_tst[clnt__], self.channels, self.height, self.width)).astype(np.float32) for clnt__ in range(self.n_client) ])
            clnt_tst_y = np.asarray([ np.zeros((clnt_data_list_tst[clnt__], 1)).astype(np.int64) for clnt__ in range(self.n_client) ])
            for jj in range(n_client):
                clnt_tst_x[jj] = tst_x[np.concatenate(clnt_cls_idx[jj]).astype(np.int32)]
                clnt_tst_y[jj] = tst_y[np.concatenate(clnt_cls_idx[jj]).astype(np.int32)]

            tst_x = clnt_tst_x; tst_y = clnt_tst_y

            if self.rule == 'ALID':
                # Map class indices to actual label index. eg. cls_map_list[i][j]=k label j of device i corresponds to label k
                cls_map_list = np.zeros((self.n_client, cls_per_client))
                for clnt in range(self.n_client):
                    unique_cls_list = np.unique(clnt_y[clnt].reshape(-1))
                    # Flip labels
                    np.random.shuffle(unique_cls_list)
                    cls_map_list[clnt] = unique_cls_list

                    reverse_map = {}
                    for ii in range(len(unique_cls_list)):
                        reverse_map[unique_cls_list[ii]] = ii

                    for data_idx in range(len(clnt_y[clnt])):
                        clnt_y[clnt][data_idx][0] = reverse_map[clnt_y[clnt][data_idx][0]]

                    for data_idx in range(len(tst_y[clnt])):
                        tst_y[clnt][data_idx][0] = reverse_map[tst_y[clnt][data_idx][0]]

                self.cls_map_list = cls_map_list    
                
            self.clnt_x = clnt_x; self.clnt_y = clnt_y
            self.tst_x  = tst_x;  self.tst_y  = tst_y
            
            # Save data
            os.mkdir('Data/%s' %(self.name))
            
            np.save('Data/%s/mean.npy' %(self.name), DatasetObject.mean)
            np.save('Data/%s/std.npy' %(self.name), DatasetObject.std)
            
            np.save('Data/%s/clnt_x.npy' %(self.name), clnt_x)
            np.save('Data/%s/clnt_y.npy' %(self.name), clnt_y)

            np.save('Data/%s/tst_x.npy'  %(self.name),  tst_x)
            np.save('Data/%s/tst_y.npy'  %(self.name),  tst_y)
            
            if not os.path.exists('Model'):
                os.mkdir('Model')

        else:
            print("Data is already downloaded")
            DatasetObject.mean = np.load('Data/%s/mean.npy' %(self.name))
            DatasetObject.std  = np.load('Data/%s/std.npy' %(self.name))
            
            self.clnt_x = np.load('Data/%s/clnt_x.npy' %(self.name))
            self.clnt_y = np.load('Data/%s/clnt_y.npy' %(self.name))
            self.n_client = len(self.clnt_x)

            self.tst_x  = np.load('Data/%s/tst_x.npy'  %(self.name))
            self.tst_y  = np.load('Data/%s/tst_y.npy'  %(self.name))
                
            if self.dataset == 'CIFAR10':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;
            if self.dataset == 'CIFAR100':
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 100;    
            if self.dataset == 'miniImageNet':
                self.channels = 3; self.width = 84; self.height = 84; self.n_cls = 100;

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, data_x, data_y=True, train=False, dataset_name=''):
        self.name = dataset_name
        
        if self.name == 'CIFAR10' or self.name == 'CIFAR100':
            self.train = train
            self.transform = transforms.Compose([transforms.ToTensor()])
        
            self.X_data = torch.tensor(data_x).float()#data_x
            self.y_data = data_y
            
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()
                
            self.augment_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=DatasetObject.mean, std=DatasetObject.std)
            ])
            self.noaugmt_transform = transforms.Compose([
                transforms.Normalize(mean=DatasetObject.mean, std=DatasetObject.std)
            ])
        elif self.name == 'miniImageNet':
            self.X_data = data_x.astype(np.uint8)  # In range 0-255
            self.X_data = np.moveaxis(self.X_data, source=1, destination=3)  # Make it H,W,C
            self.y_data = data_y

            self.train = train
            if not isinstance(data_y, bool):
                self.y_data = torch.tensor(data_y).float()

            self.augment_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomCrop(84, padding=8),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=DatasetObject.mean, std=DatasetObject.std)
            ])
            self.noaugmt_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=DatasetObject.mean, std=DatasetObject.std)
            ])
           
    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        img = self.X_data[idx]
        img = self.augment_transform(img) if self.train else self.noaugmt_transform(img)

        if isinstance(self.y_data, bool):
            return img
        else:
            y = self.y_data[idx]
            return img, y