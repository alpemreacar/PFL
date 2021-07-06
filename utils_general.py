from utils_libs import *
from utils_dataset import *
from utils_models import *
# Global parameters
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
max_norm = 10
num_workers = 0

# --- Helper methods
weight = 0.8 
def smooth_filter(arr):  # Weight between 0 and 1
    last = arr[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in arr:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
    return smoothed

# Check if the model has NaN values
def is_model_NaN(model):
    for name, param in model.named_parameters():
        isNan = torch.sum(torch.isnan(param.data)).item()
        if isNan > 0:
            return True
    return False

def get_mdl_params(model_list, n_par=0):
    if n_par==0:
        for name, param in model_list[0].named_parameters():
            n_par += len(param.data.reshape(-1))
    param_mat = torch.zeros((len(model_list), n_par), dtype=torch.float32, device=device)
    for i, mdl in enumerate(model_list):
        idx = 0
        for name, param in mdl.named_parameters():
            temp = param.data.detach().reshape(-1)
            param_mat[i, idx:idx + len(temp)] = temp
            idx += len(temp)

    return param_mat

# --- Evaluate a NN model
def get_acc_loss(data_x, data_y, model, dataset_name, w_decay = 0):
    acc_overall = 0; loss_overall = 0;
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    batch_size = min(3000, data_x.shape[0])
    n_tst = data_x.shape[0]
    tst_load = torch.utils.data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), 
                                           batch_size=batch_size, shuffle=False, num_workers=num_workers)
    model.eval(); model = model.to(device)
    with torch.no_grad():
        for data in tst_load:
            batch_x, batch_y = data[0].to(device), data[1].to(device)
            y_pred = model(batch_x)
            # Loss calculation
            loss = loss_fn(y_pred, batch_y.reshape(-1).long())
            loss_overall += loss.item()
            # Accuracy calculation
            y_pred = y_pred.cpu().numpy()            
            y_pred = np.argmax(y_pred, axis=1).reshape(-1)
            batch_y = batch_y.cpu().numpy().reshape(-1).astype(np.int32)
            batch_correct = np.sum(y_pred == batch_y)
            acc_overall += batch_correct
    
    loss_overall /= n_tst
    if w_decay != 0:
        # Add L2 loss
        params = get_mdl_params([model]).cpu().numpy()
        loss_overall += w_decay/2 * np.sum(params * params)
        
    model.train()
    return loss_overall, acc_overall / n_tst

def get_acc_loss_over_clients(data_x, data_y, model, dataset_name, w_decay = 0):
    acc_ = 0; loss_ = 0; n_clnt = len(data_y); n_total = 0; 
    acc_list = np.zeros(n_clnt)
    for idx in range(n_clnt):
        loss_clnt, acc_clnt = get_acc_loss(data_x[idx], data_y[idx], model, dataset_name, w_decay)
        loss_ += loss_clnt*len(data_y); acc_ += acc_clnt*len(data_y); n_total+=len(data_y)
        acc_list[idx] = acc_clnt
    
    acc_ = acc_ / n_total; loss_= loss_/ n_total
    return loss_, acc_, np.max(acc_list), np.min(acc_list)

## MAML
def get_maml_acc_loss(data_x, data_y, model, model_func, learning_rate, num_grad_step, dataset_name, tst_x=False, tst_y=False, weight_decay_data=0,  weight_decay_tst=False):
    _model = model_func().to(device)
    _model.load_state_dict(copy.deepcopy(dict(model.named_parameters())))
    for params in _model.parameters():
        params.requires_grad = True
    optimizer_ = torch.optim.SGD(_model.parameters(), lr=learning_rate, weight_decay=weight_decay_data)
    # Do Fine Tuning on all dataset
    trn_load = torch.utils.data.DataLoader(Dataset(data_x, data_y, dataset_name=dataset_name), 
                                           batch_size=len(data_y), shuffle=False, num_workers=num_workers)
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    _model.train(); _model = _model.to(device)
    
    for _ in range(num_grad_step):
        for data in trn_load:
            batch_x, batch_y = data[0].to(device), data[1].to(device)
            y_pred = _model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long()) / list(batch_y.size())[0]
            optimizer_.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=_model.parameters(), max_norm=max_norm) # Clip gradients
            optimizer_.step()
 
    if isinstance(tst_x, bool):
        # Get train acc
        loss_, acc_ = get_acc_loss(data_x, data_y, _model, dataset_name, weight_decay_data)
    else:
        # Get test acc
        weight_decay_tst = weight_decay_data if isinstance(weight_decay_tst, bool) else weight_decay_tst
        loss_, acc_ = get_acc_loss(tst_x, tst_y, _model, dataset_name, weight_decay_tst)
    del _model
    return loss_, acc_

# Meta update and evaluate, Update based on train and evaluate based on test
def get_maml_acc_loss_over_clients(trn_x, trn_y, tst_x, tst_y, model_func, meta_learning_rate, meta_model, dataset_name, num_grad_step, w_decay = 0):
    acc_ = 0; loss_ = 0; n_clnt = len(trn_x); n_total = 0; 
    acc_list = np.zeros(n_clnt)
    for idx in range(n_clnt):
        loss_clnt, acc_clnt = get_maml_acc_loss(data_x=trn_x[idx], data_y=trn_y[idx], model=meta_model, model_func=model_func, 
                                                learning_rate=meta_learning_rate, num_grad_step=num_grad_step, 
                                                dataset_name=dataset_name, tst_x=tst_x[idx], tst_y=tst_y[idx], 
                                                weight_decay_data=w_decay)
        loss_ += loss_clnt*len(tst_y); acc_ += acc_clnt*len(tst_y); n_total+=len(tst_y)
        acc_list[idx] = acc_clnt
    
    acc_ = acc_ / n_total; loss_= loss_/ n_total
    return loss_, acc_, np.max(acc_list), np.min(acc_list)

## Proto
def get_proto_acc_loss(data_x, data_y, model, model_func, dataset_name, tst_x=False, tst_y=False, weight_decay=0):
    # Get one hot vectors for the labels
    unique, unique_indices = np.unique(data_y.reshape(-1), return_inverse=True)
    data_y_one_hot = np.zeros((len(data_x), np.max(unique_indices) + 1))
    data_y_one_hot[np.arange(len(data_x)), unique_indices] = 1
    trn_load = torch.utils.data.DataLoader(Dataset(data_x, data_y_one_hot, dataset_name=dataset_name),
                                           batch_size=len(data_x), shuffle=False, num_workers=num_workers)
    
    model = model.to(device)
    model.proto = True
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    if isinstance(tst_x, bool):
        tst_x = data_x; tst_y_one_hot = data_y_one_hot
    else:
        unique, unique_indices = np.unique(tst_y.reshape(-1), return_inverse=True)
        tst_y_one_hot = np.zeros((len(tst_x), np.max(unique_indices) + 1))
        tst_y_one_hot[np.arange(len(tst_x)), unique_indices] = 1

    tst_load = torch.utils.data.DataLoader(Dataset(tst_x, tst_y_one_hot, dataset_name=dataset_name),
                                           batch_size=len(tst_x), shuffle=False, num_workers=num_workers)
    with torch.no_grad():
        for data in trn_load:
            batch_trn_x, batch_trn_y = data[0].to(device), data[1].to(device)
        for data in tst_load:
            batch_tst_x, batch_tst_y = data[0].to(device), data[1].to(device)
                
        feature_space_sampl = model(batch_trn_x)
        feature_space_query = model(batch_tst_x)
        logits_cen = get_logits_proto_cen(feature_space_sampl, batch_trn_y, feature_space_query)
        loss_proto = (loss_fn(logits_cen, torch.argmax(batch_tst_y, 1).reshape(-1).long()) / list(batch_tst_y.size())[0]).item()
        acc  = np.mean(np.argmax(logits_cen.cpu().numpy(), axis=1) == np.argmax(batch_tst_y.cpu().numpy(), axis=1))
    model.proto = False
    return loss_proto, acc

# Meta update and evaluate, Update based on train and evaluate based on test
def get_proto_acc_loss_over_clients(trn_x, trn_y, tst_x, tst_y, model_func, meta_model, dataset_name, w_decay = 0):
    acc_ = 0; loss_ = 0; n_clnt = len(trn_x); n_total = 0
    acc_list = np.zeros(n_clnt)
    for idx in range(n_clnt):
        loss_clnt, acc_clnt = get_proto_acc_loss(data_x=trn_x[idx], data_y=trn_y[idx], model=meta_model, model_func=model_func, 
                                                 dataset_name=dataset_name,
                                                 tst_x=tst_x[idx], tst_y=tst_y[idx], weight_decay=w_decay)
        loss_ += loss_clnt*len(tst_y); acc_ += acc_clnt*len(tst_y); n_total+=len(tst_y)
        acc_list[idx] = acc_clnt
    acc_ = acc_ / n_total; loss_= loss_/ n_total
    return loss_, acc_, np.max(acc_list), np.min(acc_list)

#  --- Helper for the printing and recording performance
def get_all_results_maml(meta_learning_rate, num_grad_step, clnt_x, clnt_y, tst_x, tst_y, dataset_name, model_func, avg_model, all_model, fast_exec, i):
    print('Meta Lr: %f, Number of Gradient Steps: %d' %(meta_learning_rate, num_grad_step))
    loss_1, acc_1, acc_1_max, acc_1_min = get_maml_acc_loss_over_clients(clnt_x, clnt_y, tst_x, tst_y, model_func, meta_learning_rate, avg_model, dataset_name, num_grad_step)
    print("**** Communication sel %3d, Test Accuracy: %.4f, Max: %.3f, Min: %.2f, Loss: %.4f" %(i+1, acc_1, acc_1_max, acc_1_min, loss_1))
    ###
    loss_2, acc_2, acc_2_max, acc_2_min = get_maml_acc_loss_over_clients(clnt_x, clnt_y, tst_x, tst_y, model_func, meta_learning_rate, all_model, dataset_name, num_grad_step)          
    print("**** Communication all %3d, Test Accuracy: %.4f, Max: %.3f, Min: %.2f, Loss: %.4f" %(i+1, acc_2, acc_2_max, acc_2_min, loss_2))
          
    ###
    loss_3=0;acc_3=0;acc_3_max=0;acc_3_min=0
    loss_4=0;acc_4=0;acc_4_max=0;acc_4_min=0
    if not fast_exec:
        loss_3, acc_3, acc_3_max, acc_3_min = get_maml_acc_loss_over_clients(clnt_x, clnt_y, clnt_x, clnt_y, model_func, meta_learning_rate, avg_model, dataset_name, num_grad_step)
        print("**** Communication sel %3d, Cent Accuracy: %.4f, Max: %.3f, Min: %.2f, Loss: %.4f" %(i+1, acc_3, acc_3_max, acc_3_min, loss_3))
    ###
    if not fast_exec:
        loss_4, acc_4, acc_4_max, acc_4_min = get_maml_acc_loss_over_clients(clnt_x, clnt_y, clnt_x, clnt_y, model_func, meta_learning_rate, all_model, dataset_name, num_grad_step)
        print("**** Communication all %3d, Cent Accuracy: %.4f, Max: %.3f, Min: %.2f, Loss: %.4f" %(i+1, acc_4, acc_4_max, acc_4_min, loss_4))
    return [[loss_1, acc_1, acc_1_max, acc_1_min], [loss_2, acc_2, acc_2_max, acc_2_min], [loss_3, acc_3, acc_3_max, acc_3_min], [loss_4, acc_4, acc_4_max, acc_4_min]]

def get_all_results_plain(clnt_x, clnt_y, tst_x, tst_y, dataset_name, avg_model, all_model, fast_exec, i):
    print('No Meta learning only global model')
    
    loss_1, acc_1, acc_1_max, acc_1_min = get_acc_loss_over_clients(tst_x, tst_y, avg_model, dataset_name, w_decay = 0)
    print("**** Communication sel %3d, Test Accuracy: %.4f, Max: %.3f, Min: %.2f, Loss: %.4f" %(i+1, acc_1, acc_1_max, acc_1_min, loss_1))
    ###
    loss_2, acc_2, acc_2_max, acc_2_min = get_acc_loss_over_clients(tst_x, tst_y, all_model, dataset_name, w_decay = 0)
    print("**** Communication all %3d, Test Accuracy: %.4f, Max: %.3f, Min: %.2f, Loss: %.4f" %(i+1, acc_2, acc_2_max, acc_2_min, loss_2))
    ###
    loss_3=0;acc_3=0;acc_3_max=0;acc_3_min=0
    loss_4=0;acc_4=0;acc_4_max=0;acc_4_min=0
    if not fast_exec:
        loss_3, acc_3, acc_3_max, acc_3_min = get_acc_loss_over_clients(clnt_x, clnt_y, avg_model, dataset_name, w_decay = 0)
        print("**** Communication sel %3d, Cent Accuracy: %.4f, Max: %.3f, Min: %.2f, Loss: %.4f" %(i+1, acc_3, acc_3_max, acc_3_min, loss_3))
    ###
    if not fast_exec:
        loss_4, acc_4, acc_4_max, acc_4_min= get_acc_loss_over_clients(clnt_x, clnt_y, all_model, dataset_name, w_decay = 0)
        print("**** Communication all %3d, Cent Accuracy: %.4f, Max: %.3f, Min: %.2f, Loss: %.4f" %(i+1, acc_4, acc_4_max, acc_4_min, loss_4))
    
    return [[loss_1, acc_1, acc_1_max, acc_1_min], [loss_2, acc_2, acc_2_max, acc_2_min], [loss_3, acc_3, acc_3_max, acc_3_min], [loss_4, acc_4, acc_4_max, acc_4_min]]

def get_all_results_proto(clnt_x, clnt_y, tst_x, tst_y, dataset_name, model_func, avg_model, all_model, fast_exec, i):
    print('Proto')
    loss_1, acc_1, acc_1_max, acc_1_min = get_proto_acc_loss_over_clients(clnt_x, clnt_y, tst_x, tst_y, model_func, avg_model, dataset_name)
    print("**** Communication sel %3d, Test Accuracy: %.4f, Max: %.3f, Min: %.2f, Loss: %.4f" %(i+1, acc_1, acc_1_max, acc_1_min, loss_1))
    ###
    loss_2, acc_2, acc_2_max, acc_2_min = get_proto_acc_loss_over_clients(clnt_x, clnt_y, tst_x, tst_y, model_func, all_model, dataset_name)
    print("**** Communication all %3d, Test Accuracy: %.4f, Max: %.3f, Min: %.2f, Loss: %.4f" %(i+1, acc_2, acc_2_max, acc_2_min, loss_2))
    ###
    loss_3=0;acc_3=0;acc_3_max=0;acc_3_min=0
    loss_4=0;acc_4=0;acc_4_max=0;acc_4_min=0
    if not fast_exec:
        loss_3, acc_3, acc_3_max, acc_3_min = get_proto_acc_loss_over_clients(clnt_x, clnt_y, clnt_x, clnt_y, model_func,  avg_model, dataset_name)
        print("**** Communication sel %3d, Cent Accuracy: %.4f, Max: %.3f, Min: %.2f, Loss: %.4f" %(i+1, acc_3, acc_3_max, acc_3_min, loss_3))
    ###
    if not fast_exec:
        loss_4, acc_4, acc_4_max, acc_4_min = get_proto_acc_loss_over_clients(clnt_x, clnt_y, clnt_x, clnt_y, model_func,  all_model,  dataset_name)
        print("**** Communication all %3d, Cent Accuracy: %.4f, Max: %.3f, Min: %.2f, Loss: %.4f" %(i+1, acc_4, acc_4_max, acc_4_min, loss_4))
    return [[loss_1, acc_1, acc_1_max, acc_1_min], [loss_2, acc_2, acc_2_max, acc_2_min], [loss_3, acc_3, acc_3_max, acc_3_min], [loss_4, acc_4, acc_4_max, acc_4_min]]

# --- Train methods
def train_model(model, trn_x, trn_y, tst_x, tst_y, learning_rate, batch_size, K, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_load = torch.utils.data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                           batch_size=batch_size, shuffle=True, num_workers=num_workers)                    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device)

    if print_per != 0:
        print_test = not isinstance(tst_x, bool)
        loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
        if print_test:
            loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
            print("Step %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f" 
                  %(0, acc_trn, loss_trn, acc_tst, loss_tst))
        else:
            print("Step %3d, Training Accuracy: %.4f, Loss: %.4f" %(0, acc_trn, loss_trn))
    
        model.train()
    k=0
    while(k < K):
        for data in trn_load:
            batch_x, batch_y = data[0].to(device), data[1].to(device)
            
            y_pred = model(batch_x)
            loss = loss_fn(y_pred, batch_y.reshape(-1).long()) / list(batch_y.size())[0]
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients
            
            optimizer.step()

            k += 1
            if print_per != 0 and (k % print_per == 0):
                loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
                if print_test:
                    loss_tst, acc_tst = get_acc_loss(tst_x, tst_y, model, dataset_name, 0)
                    print("Step %3d, Training Accuracy: %.4f, Loss: %.4f, Test Accuracy: %.4f, Loss: %.4f" 
                          %(k, acc_trn, loss_trn, acc_tst, loss_tst))
                else:
                    print("Step %3d, Training Accuracy: %.4f, Loss: %.4f" 
                          %(k,acc_trn,loss_trn))
                model.train()
                
            if k == K:
                break
        
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def get_logits_proto_cen(feature_space_sampl, curr_sampl_y, feature_space_query):
    
    size_sampl = len(feature_space_sampl) # S
    size_query = len(feature_space_query) # Q
    n_dim_feat = feature_space_sampl.shape[1] # D
    
    assert len(feature_space_sampl) == len(curr_sampl_y), 'Error inconsistent input'
    assert feature_space_sampl.shape[1] == feature_space_query.shape[1], 'Error inconsistent input'
    
    feature_space_sampl_mean = torch.matmul(curr_sampl_y.T, feature_space_sampl) # CxS, SxD => CxD
    feature_space_sampl_mean = feature_space_sampl_mean.div(torch.sum(curr_sampl_y, 0).reshape(-1,1))
    
    # Distance Matrix Vectorization Trick
    AB = torch.mm(feature_space_query, feature_space_sampl_mean.transpose(0, 1))                       # QxD, DxC -> QxC
    AA = (feature_space_query * feature_space_query).sum(dim=1, keepdim=True)                          # QxD, QxD -> Qx1
    BB = (feature_space_sampl_mean * feature_space_sampl_mean).sum(dim=1, keepdim=True).reshape(1, -1) # CxD, CxD -> 1xC
    dist_ = AA.expand_as(AB) - 2 * AB + BB.expand_as(AB) # QxC
    logits = -1*dist_

    return logits

def train_proto_model(model, trn_x, trn_y, learning_rate, batch_size, K, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    # Get one hot vectors for the labels
    unique, unique_indices = np.unique(trn_y.reshape(-1), return_inverse=True)
    trn_y_one_hot = np.zeros((n_trn, np.max(unique_indices) + 1))
    trn_y_one_hot[np.arange(n_trn), unique_indices] = 1

    trn_load = torch.utils.data.DataLoader(Dataset(trn_x, trn_y_one_hot, train=True, dataset_name=dataset_name),
                                           batch_size=batch_size, shuffle=True, num_workers=num_workers)                       
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device); model.proto = True
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    k = 0
    while(k < K):        
        # Get two batches of data, one as sample and one as query.
        while True:
            data_list = []
            while len(data_list) != 2:
                for data in trn_load:
                    data_list.append([data[0], data[1]])
                    if len(data_list) == 2:
                        break
            # To get a meaningful loss, we need to have at least one sample in the sample set for all classes in query set
            sampl_cls = np.unique(data_list[0][1]); query_cls = np.unique(data_list[1][1]); is_violated = False
            for elem in query_cls:
                if not elem in sampl_cls:
                    is_violated = True
            if not is_violated:
                break
        
        # curr_sampl_y is BS x nCls. What if we do not have one class in the batch? That will make the loss NaN.
        # make curr_sampl_y as BS x Current Classes
        curr_sampl_y = data_list[0][1]; curr_query_y = data_list[1][1]
        non_zero_classes = torch.where(torch.sum(curr_sampl_y, 0).reshape(-1,1) != 0)[0]
        curr_sampl_y = curr_sampl_y[:,non_zero_classes].to(device)
        curr_query_y = curr_query_y[:,non_zero_classes].to(device)

        # Concatenate input
        curr_sampl_x = data_list[0][0].to(device)
        curr_query_x = data_list[1][0].to(device)
        curr_x = torch.cat((curr_sampl_x, curr_query_x), 0)
        feature_ = model(curr_x)
        feature_space_sampl = feature_[:len(curr_sampl_x)]
        feature_space_query = feature_[len(curr_sampl_x):]
        
        # Get logits of query batch based on applying kNN to sample batch            
        logits_cen = get_logits_proto_cen(feature_space_sampl, curr_sampl_y, feature_space_query) 
                    
        # Get negative log loss of the query class
        loss = loss_fn(logits_cen, torch.argmax(curr_query_y, 1).reshape(-1).long()) / list(curr_query_y.size())[0]
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients 
        ###
        optimizer.step()
        k += 1
        if print_per != 0 and (k % print_per == 0):
            acc = np.mean(np.argmax(logits_cen.detach().cpu().numpy(), axis=1) == np.argmax(curr_query_y.cpu().numpy(), axis=1))
            loss = loss.item()
            print("Step %3d, Batch Acc: %.4f, Loss: %.4f" %(k, acc, loss))
        
        model.proto = False
    return model

def set_client_from_params(mdl, params):
    dict_param = copy.deepcopy(dict(mdl.named_parameters()))
    idx = 0
    for name, param in mdl.named_parameters():
        weights = param.data
        length = len(weights.reshape(-1))
        dict_param[name].data.copy_(params.data[idx:idx+length].reshape(weights.shape))
        idx += length
    
    mdl.load_state_dict(dict_param)    
    return mdl

###
def train_meta_model_MAML(model_func, model, trn_x, trn_y, num_grad_step, meta_learning_rate, learning_rate, batch_size, K, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_load = torch.utils.data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                           batch_size=batch_size, shuffle=True, num_workers=num_workers)                       

    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    model.train(); model = model.to(device)
    n_par = len(get_mdl_params([model])[0])    
    optimizer_ = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    inner_opt = torch.optim.SGD(model.parameters(), lr=meta_learning_rate, weight_decay=weight_decay)

    if print_per != 0:   
        loss_trn, acc_trn = get_maml_acc_loss(trn_x, trn_y, model, model_func, meta_learning_rate, num_grad_step, dataset_name, weight_decay_data=weight_decay)
        print("Step %3d, Training Accuracy: %.4f, Loss: %.4f, Higher Library" %(0, acc_trn, loss_trn))
        model.train()
    
    for k in range(K):
        data_list = []
        while len(data_list) != 2:
            for data in trn_load:
                data_list.append([data[0], data[1]])
                if len(data_list) == 2:
                    break
        curr_trn_x, curr_trn_y = data_list[0][0].to(device), data_list[0][1].to(device)
        curr_val_x, curr_val_y = data_list[1][0].to(device), data_list[1][1].to(device)

        # Higher library
        optimizer_.zero_grad()
        with torch.backends.cudnn.flags(enabled=False):
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(num_grad_step):
                    trn_logits = fnet(curr_trn_x)
                    trn_loss = loss_fn(trn_logits, curr_trn_y.reshape(-1).long()) / list(curr_trn_y.size())[0]
                    diffopt.step(trn_loss)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                val_logits = fnet(curr_val_x)
                val_loss = loss_fn(val_logits, curr_val_y.reshape(-1).long()) / list(curr_val_y.size())[0]

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                val_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients 
        optimizer_.step()
        
        if print_per != 0 and ((k+1) % print_per == 0):
            loss_trn, acc_trn = get_maml_acc_loss(trn_x, trn_y, model, model_func, meta_learning_rate, num_grad_step, dataset_name, weight_decay_data=weight_decay) 
            print("Step %3d, Training Accuracy: %.4f, Loss: %.4f" 
                  %((k+1), acc_trn, loss_trn))
            model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

### FedDyn methods

####
def train_dyn_model(alpha, lambda_model, server_model, model, trn_x, trn_y, learning_rate, batch_size, K, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_load = torch.utils.data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                           batch_size=batch_size, shuffle=True, num_workers=num_workers)                       
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train(); model = model.to(device)
    
    if print_per != 0:
        loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
        print("Step %3d, Training Accuracy: %.4f, Loss: %.4f, alpha: %.3f" %(0,acc_trn,loss_trn,alpha))
        model.train()
    k=0
    while(k < K):
        for data in trn_load:
            batch_x, batch_y = data[0].to(device), data[1].to(device)
            
            y_pred = model(batch_x)
            optimizer.zero_grad()
            loss = loss_fn(y_pred, batch_y.reshape(-1).long()) / list(batch_y.size())[0]
            
            # FedDyn version!
            # Add Dynamic loss
            # Get model parameter
            mld_pars = []
            for name, param in model.named_parameters():
                mld_pars.append(param.reshape(-1))
            mld_pars = torch.cat(mld_pars)
            loss_lambda = -torch.sum(mld_pars * lambda_model)
            loss_server = -alpha*torch.sum(mld_pars * server_model) + alpha/2 * torch.sum(mld_pars*mld_pars)
            loss = loss + loss_lambda + loss_server
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients to prevent exploding
            optimizer.step()

            k += 1
            
            if print_per != 0 and (k % print_per == 0):
                loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
                print("Step %3d, Training Accuracy: %.4f, Loss: %.4f, alpha: %.3f" %(k,acc_trn,loss_trn,alpha))
                model.train()
                
            if k == K:
                break
        
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

# ###
def train_dyn_meta_model_MAML(alpha, lambda_model, server_model, model_func, model, trn_x, trn_y, num_grad_step, meta_learning_rate, learning_rate, batch_size, K, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_load = torch.utils.data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                           batch_size=batch_size, shuffle=True, num_workers=num_workers)                       

    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    model.train(); model = model.to(device)
    n_par = len(get_mdl_params([model])[0])    
    optimizer_ = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    inner_opt = torch.optim.SGD(model.parameters(), lr=meta_learning_rate, weight_decay=weight_decay)

    if print_per != 0:   
        loss_trn, acc_trn = get_maml_acc_loss(trn_x, trn_y, model, model_func, meta_learning_rate, num_grad_step, dataset_name, weight_decay_data=weight_decay)
        print("Step %3d, Training Accuracy: %.4f, Loss: %.4f, Higher Library, alpha %f" %(0, acc_trn, loss_trn, alpha))
        model.train()
    
    for k in range(K):
        data_list = []
        while len(data_list) != 2:
            for data in trn_load:
                data_list.append([data[0], data[1]])
                if len(data_list) == 2:
                    break
        curr_trn_x, curr_trn_y = data_list[0][0].to(device), data_list[0][1].to(device)
        curr_val_x, curr_val_y = data_list[1][0].to(device), data_list[1][1].to(device)

        # Higher library
        optimizer_.zero_grad()
        with torch.backends.cudnn.flags(enabled=False):
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(num_grad_step):
                    trn_logits = fnet(curr_trn_x)
                    trn_loss = loss_fn(trn_logits, curr_trn_y.reshape(-1).long()) / list(curr_trn_y.size())[0]
                    diffopt.step(trn_loss)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                val_logits = fnet(curr_val_x)
                val_loss = loss_fn(val_logits, curr_val_y.reshape(-1).long()) / list(curr_val_y.size())[0]

                # FedDyn version!
                # Add Dynamic loss
                # Get model parameter
                mld_pars = []
                for name, param in model.named_parameters():
                    mld_pars.append(param.reshape(-1))
                mld_pars = torch.cat(mld_pars)
                loss_lambda = -torch.sum(mld_pars * lambda_model)
                loss_server = -alpha*torch.sum(mld_pars * server_model) + alpha/2 * torch.sum(mld_pars*mld_pars)
                val_loss = val_loss + loss_lambda + loss_server

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                val_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients 
        optimizer_.step()
        
        if print_per != 0 and ((k+1) % print_per == 0):
            loss_trn, acc_trn = get_maml_acc_loss(trn_x, trn_y, model, model_func, meta_learning_rate, num_grad_step, dataset_name, weight_decay_data=weight_decay) 
            print("Step %3d, Training Accuracy: %.4f, Loss: %.4f, alpha %f" 
                  %((k+1), acc_trn, loss_trn, alpha))
            model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def train_dyn_proto_model(alpha, lambda_model, server_model, model, trn_x, trn_y, learning_rate, batch_size, K, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    # Get one hot vectors for the labels
    unique, unique_indices = np.unique(trn_y.reshape(-1), return_inverse=True)
    trn_y_one_hot = np.zeros((n_trn, np.max(unique_indices) + 1))
    trn_y_one_hot[np.arange(n_trn), unique_indices] = 1
    
    trn_load = torch.utils.data.DataLoader(Dataset(trn_x, trn_y_one_hot, train=True, dataset_name=dataset_name),
                                           batch_size=batch_size, shuffle=True, num_workers=num_workers)                       
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device); model.proto = True
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    k = 0
    while(k < K):        
        # Get two batches of data, one as sample and one as query.
        while True:
            data_list = []
            while len(data_list) != 2:
                for data in trn_load:
                    data_list.append([data[0], data[1]])
                    if len(data_list) == 2:
                        break
            # To get a meaningful loss, we need to have at least one sample in the sample set for all classes in query set
            sampl_cls = np.unique(data_list[0][1]); query_cls = np.unique(data_list[1][1]); is_violated = False
            for elem in query_cls:
                if not elem in sampl_cls:
                    is_violated = True
            if not is_violated:
                break

        # curr_sampl_y is BS x nCls. What if we do not have one class in the batch? That will make the loss NaN.
        # make curr_sampl_y as BS x Current Classes
        curr_sampl_y = data_list[0][1]; curr_query_y = data_list[1][1]
        non_zero_classes = torch.where(torch.sum(curr_sampl_y, 0).reshape(-1,1) != 0)[0]
        curr_sampl_y = curr_sampl_y[:,non_zero_classes].to(device)
        curr_query_y = curr_query_y[:,non_zero_classes].to(device)

        # Concatenate input
        curr_sampl_x = data_list[0][0].to(device)
        curr_query_x = data_list[1][0].to(device)
        curr_x = torch.cat((curr_sampl_x, curr_query_x), 0)
        feature_ = model(curr_x)
        feature_space_sampl = feature_[:len(curr_sampl_x)]
        feature_space_query = feature_[len(curr_sampl_x):]
        
        # Get logits of query batch based on applying kNN to sample batch            
        logits_cen = get_logits_proto_cen(feature_space_sampl, curr_sampl_y, feature_space_query)            
        # Get negative log loss of the query class
        loss = loss_fn(logits_cen, torch.argmax(curr_query_y, 1).reshape(-1).long()) / list(curr_query_y.size())[0]

        # FedDyn version!
        # Add Dynamic loss
        # Get model parameter
        mld_pars = []
        for name, param in model.named_parameters():
            mld_pars.append(param.reshape(-1))
        mld_pars = torch.cat(mld_pars)
        loss_lambda = -torch.sum(mld_pars * lambda_model)
        loss_server = -alpha*torch.sum(mld_pars * server_model) + alpha/2 * torch.sum(mld_pars*mld_pars)
        loss = loss + loss_lambda + loss_server
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients 
        optimizer.step()
        k += 1
        if print_per != 0 and (k % print_per == 0):
            acc = np.mean(np.argmax(logits_cen.detach().cpu().numpy(), axis=1) == np.argmax(curr_query_y.cpu().numpy(), axis=1))
            loss = loss.item()
            print("Step %3d, Batch Acc: %.4f, Loss: %.4f, alpha: %.4f" %(k, acc, loss, alpha))
        
    model.proto = False        
    return model

### SCAFFOLD methods

def train_SCAF_model(curr_state_params_diff, model, trn_x, trn_y, learning_rate, batch_size, K, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_load = torch.utils.data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                           batch_size=batch_size, shuffle=True, num_workers=num_workers)                       
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.train(); model = model.to(device)
    

    if print_per != 0:
        loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
        print("Step %3d, Training Accuracy: %.4f, Loss: %.4f" %(0,acc_trn,loss_trn))
        model.train()
    k=0
    while(k < K):
        for data in trn_load:
            batch_x, batch_y = data[0].to(device), data[1].to(device)
            
            y_pred = model(batch_x)
            optimizer.zero_grad()
            loss = loss_fn(y_pred, batch_y.reshape(-1).long()) / list(batch_y.size())[0]
            
            # Get model parameter
            mld_pars = []
            for name, param in model.named_parameters():
                mld_pars.append(param.reshape(-1))
            mld_pars = torch.cat(mld_pars)
            loss_inner = torch.sum(curr_state_params_diff * mld_pars)
            loss = loss + loss_inner
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients to prevent exploding
            optimizer.step()

            k += 1
            
            if print_per != 0 and (k % print_per == 0):
                loss_trn, acc_trn = get_acc_loss(trn_x, trn_y, model, dataset_name, weight_decay)
                print("Step %3d, Training Accuracy: %.4f, Loss: %.4f" %(k,acc_trn,loss_trn))
                model.train()
                
            if k == K:
                break
        
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def train_SCAF_meta_model_MAML(curr_state_params_diff, model_func, model, trn_x, trn_y, num_grad_step, meta_learning_rate, learning_rate, batch_size, K, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    trn_load = torch.utils.data.DataLoader(Dataset(trn_x, trn_y, train=True, dataset_name=dataset_name),
                                           batch_size=batch_size, shuffle=True, num_workers=num_workers)                       

    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    model.train(); model = model.to(device)
    n_par = len(get_mdl_params([model])[0])    
    optimizer_ = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    inner_opt = torch.optim.SGD(model.parameters(), lr=meta_learning_rate, weight_decay=weight_decay)

    if print_per != 0:   
        loss_trn, acc_trn = get_maml_acc_loss(trn_x, trn_y, model, model_func, meta_learning_rate, num_grad_step, dataset_name, weight_decay_data=weight_decay)
        print("Step %3d, Training Accuracy: %.4f, Loss: %.4f, Higher Library" %(0, acc_trn, loss_trn))
        model.train()
    
    for k in range(K):
        data_list = []
        while len(data_list) != 2:
            for data in trn_load:
                data_list.append([data[0], data[1]])
                if len(data_list) == 2:
                    break
        curr_trn_x, curr_trn_y = data_list[0][0].to(device), data_list[0][1].to(device)
        curr_val_x, curr_val_y = data_list[1][0].to(device), data_list[1][1].to(device)

        # Higher library
        optimizer_.zero_grad()
        with torch.backends.cudnn.flags(enabled=False):
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                # Optimize the likelihood of the support set by taking
                # gradient steps w.r.t. the model's parameters.
                # This adapts the model's meta-parameters to the task.
                # higher is able to automatically keep copies of
                # your network's parameters as they are being updated.
                for _ in range(num_grad_step):
                    trn_logits = fnet(curr_trn_x)
                    trn_loss = loss_fn(trn_logits, curr_trn_y.reshape(-1).long()) / list(curr_trn_y.size())[0]
                    diffopt.step(trn_loss)

                # The final set of adapted parameters will induce some
                # final loss and accuracy on the query dataset.
                # These will be used to update the model's meta-parameters.
                val_logits = fnet(curr_val_x)
                val_loss = loss_fn(val_logits, curr_val_y.reshape(-1).long()) / list(curr_val_y.size())[0]

                # Get model parameter
                mld_pars = []
                for name, param in model.named_parameters():
                    mld_pars.append(param.reshape(-1))
                mld_pars = torch.cat(mld_pars)
                loss_inner = torch.sum(mld_pars * curr_state_params_diff)
                val_loss = val_loss + loss_inner

                # Update the model's meta-parameters to optimize the query
                # losses across all of the tasks sampled in this batch.
                # This unrolls through the gradient steps.
                val_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients 
        optimizer_.step()
        
        if print_per != 0 and ((k+1) % print_per == 0):
            loss_trn, acc_trn = get_maml_acc_loss(trn_x, trn_y, model, model_func, meta_learning_rate, num_grad_step, dataset_name, weight_decay_data=weight_decay) 
            print("Step %3d, Training Accuracy: %.4f, Loss: %.4f" 
                  %((k+1), acc_trn, loss_trn))
            model.train()
    
    # Freeze model
    for params in model.parameters():
        params.requires_grad = False
    model.eval()
            
    return model

def train_SCAF_proto_model(curr_state_params_diff, model, trn_x, trn_y, learning_rate, batch_size, K, print_per, weight_decay, dataset_name):
    n_trn = trn_x.shape[0]
    # Get one hot vectors for the labels
    unique, unique_indices = np.unique(trn_y.reshape(-1), return_inverse=True)
    trn_y_one_hot = np.zeros((n_trn, np.max(unique_indices) + 1))
    trn_y_one_hot[np.arange(n_trn), unique_indices] = 1
    
    trn_load = torch.utils.data.DataLoader(Dataset(trn_x, trn_y_one_hot, train=True, dataset_name=dataset_name),
                                           batch_size=batch_size, shuffle=True, num_workers=num_workers)                       
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    model.train(); model = model.to(device); model.proto = True
    
    loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
    
    k = 0
    while(k < K):        
        # Get two batches of data, one as sample and one as query.
        while True:
            data_list = []
            while len(data_list) != 2:
                for data in trn_load:
                    data_list.append([data[0], data[1]])
                    if len(data_list) == 2:
                        break
            # To get a meaningful loss, we need to have at least one sample in the sample set for all classes in query set
            sampl_cls = np.unique(data_list[0][1]); query_cls = np.unique(data_list[1][1]); is_violated = False
            for elem in query_cls:
                if not elem in sampl_cls:
                    is_violated = True
            if not is_violated:
                break

        # curr_sampl_y is BS x nCls. What if we do not have one class in the batch? That will make the loss NaN.
        # make curr_sampl_y as BS x Current Classes
        curr_sampl_y = data_list[0][1]; curr_query_y = data_list[1][1]
        non_zero_classes = torch.where(torch.sum(curr_sampl_y, 0).reshape(-1,1) != 0)[0]
        curr_sampl_y = curr_sampl_y[:,non_zero_classes].to(device)
        curr_query_y = curr_query_y[:,non_zero_classes].to(device)

        # Concatenate input
        curr_sampl_x = data_list[0][0].to(device)
        curr_query_x = data_list[1][0].to(device)
        curr_x = torch.cat((curr_sampl_x, curr_query_x), 0)
        feature_ = model(curr_x)
        feature_space_sampl = feature_[:len(curr_sampl_x)]
        feature_space_query = feature_[len(curr_sampl_x):]
        
        # Get logits of query batch based on applying kNN to sample batch            
        logits_cen = get_logits_proto_cen(feature_space_sampl, curr_sampl_y, feature_space_query)
        # Get negative log loss of the query class
        loss = loss_fn(logits_cen, torch.argmax(curr_query_y, 1).reshape(-1).long()) / list(curr_query_y.size())[0]

        # Get model parameter
        mld_pars = []
        for name, param in model.named_parameters():
            mld_pars.append(param.reshape(-1))
        mld_pars = torch.cat(mld_pars)
        loss_inner = torch.sum(mld_pars * curr_state_params_diff)
        loss = loss + loss_inner
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_norm) # Clip gradients 
        optimizer.step()
        k += 1
        if print_per != 0 and (k % print_per == 0):
            acc = np.mean(np.argmax(logits_cen.detach().cpu().numpy(), axis=1) == np.argmax(curr_query_y.cpu().numpy(), axis=1))
            loss = loss.item()
            print("Step %3d, Batch Acc: %.4f, Loss %.4f"  %((k+1), acc, loss))
    model.proto = False
    return model