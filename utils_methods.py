from utils_libs import *
from utils_dataset import *
from utils_models import *
from utils_general import *

# fast_exec disables training statistics

### Methods
def train_FedAvg(data_obj, act_prob, learning_rate, batch_size, K, com_amount, print_per, weight_decay, lr_decay,
                 model_func, init_model, save_period, meta_learning_rate_list=False, 
                 num_grad_step_list=False, do_proto=False, do_plain=False,
                 rand_seed=0, save_models=False, fast_exec=False):
    suffix = 'FedAvg_S%d_F%f_Lr%f_B%d_K%d_W%f_lrdecay%f_seed%d' %(save_period, act_prob, learning_rate, batch_size, K, weight_decay, lr_decay, rand_seed)
    
    if meta_learning_rate_list != False:
        l1_str = [str(elem) for elem in meta_learning_rate_list]
        suffix += '_MetaLr_[' + ', '.join(l1_str) + ']'

        l2_str = [str(elem) for elem in num_grad_step_list]
        suffix += '_GS_[' + ', '.join(l2_str) + ']'
        
    if do_proto:
        suffix += '_Proto'
        
    if do_plain:
        suffix += '_Plain'

    n_clnt=data_obj.n_client

    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
    
    cent_x = np.concatenate(clnt_x, axis=0)
    cent_y = np.concatenate(clnt_y, axis=0)
        
    if (not os.path.exists('Model/%s/%s' %(data_obj.name, suffix))):
        os.mkdir('Model/%s/%s' %(data_obj.name, suffix))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))

    metaLr_numGrad = []
    if meta_learning_rate_list != False:
        for meta_learning_rate in meta_learning_rate_list:
                for num_grad_step in num_grad_step_list:
                    metaLr_numGrad.append([meta_learning_rate, num_grad_step])
                    
    n_cases = len(metaLr_numGrad)
    n_cases = n_cases + 1 if do_proto else n_cases
    n_cases = n_cases + 1 if do_plain else n_cases
    
    trn_perf_sel = np.zeros((n_cases, com_amount, 4));
    trn_perf_all = np.zeros((n_cases, com_amount, 4));
    tst_perf_sel = np.zeros((n_cases, com_amount, 4));
    tst_perf_all = np.zeros((n_cases, com_amount, 5));
    
    n_par = len(get_mdl_params([model_func()])[0])
    init_par_list=get_mdl_params([init_model], n_par)[0].cpu().numpy()
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par

    saved_itr = -1
    
    # Check if there are past saved iterates
    for i in range(com_amount):
        if os.path.exists('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)):
            saved_itr = i
            if save_models:
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False
                fed_mdls_sel[saved_itr//save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False
                fed_mdls_all[saved_itr//save_period] = fed_model


        if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1))):
            trn_perf_sel[:,:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            trn_perf_all[:,:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, (i+1)))

            tst_perf_sel[:,:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            tst_perf_all[:,:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, (i+1)))
            if save_models:
                clnt_params_list = np.load('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1))

    
    if (not os.path.exists('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, com_amount))):         
        avg_model = model_func().to(device)
        if saved_itr == -1:
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        else:
            # Load recent one
            avg_model.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt'%(data_obj.name, suffix, saved_itr+1)))
          
        for i in range(saved_itr+1, com_amount):
            ### Fix randomness
            np.random.seed(i + rand_seed)
            clnt_list = np.arange(n_clnt)
            np.random.shuffle(clnt_list)
            selected_clnts = clnt_list[:int(act_prob*n_clnt)]
            
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]; trn_y = clnt_y[clnt]; tst_x = False; tst_y = False

                cur_model = model_func().to(device)
                cur_model.load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in cur_model.parameters():
                    params.requires_grad = True
                cur_model = train_model(cur_model, trn_x, trn_y, tst_x, tst_y, learning_rate * (lr_decay ** i),
                                        batch_size, K, print_per, weight_decay, data_obj.dataset)

                is_diverged = is_model_NaN(cur_model)
                if is_diverged:
                    # If model has NaN do not update the list put the average model
                    clnt_params_list[clnt] = get_mdl_params([avg_model], n_par)[0].cpu().numpy()
                    tst_perf_all[0][i][-1] += 1 
                else:
                    clnt_params_list[clnt] = get_mdl_params([cur_model], n_par)[0].cpu().numpy()

            # Scale with weights
            
            avg_selected = np.mean(clnt_params_list[selected_clnts], axis = 0)
            avg_model = set_client_from_params(model_func().to(device), 
                                               torch.tensor(avg_selected, dtype=torch.float32).to(device))
            
            avg_all = np.mean(clnt_params_list, axis = 0)
            all_model = set_client_from_params(model_func().to(device), torch.tensor(avg_all, dtype=torch.float32).to(device))

            for idx_, [meta_learning_rate, num_grad_step] in enumerate(metaLr_numGrad):
                [list_1, list_2, list_3, list_4] = get_all_results_maml(meta_learning_rate,
                                     num_grad_step, data_obj.clnt_x, data_obj.clnt_y, data_obj.tst_x,
                                     data_obj.tst_y, data_obj.dataset, model_func, avg_model, all_model, fast_exec, i)
                tst_perf_sel[idx_,i,:] = list_1; tst_perf_all[idx_,i,:len(list_2)] = list_2
                trn_perf_sel[idx_,i,:] = list_3; trn_perf_all[idx_,i,:] = list_4
            
            offset_ = len(metaLr_numGrad)
            if do_proto:
                [list_1, list_2, list_3, list_4] = get_all_results_proto(
                                     data_obj.clnt_x, data_obj.clnt_y, data_obj.tst_x, data_obj.tst_y, data_obj.dataset,
                                     model_func, avg_model, all_model, fast_exec, i)
                tst_perf_sel[idx_+offset_,i,:] = list_1; tst_perf_all[idx_+offset_,i,:len(list_2)] = list_2
                trn_perf_sel[idx_+offset_,i,:] = list_3; trn_perf_all[idx_+offset_,i,:] = list_4
                offset_ = len(metaLr_numGrad) + 1
            
            
            if do_plain:
                [list_1, list_2, list_3, list_4] = get_all_results_plain(data_obj.clnt_x,
                                               data_obj.clnt_y, data_obj.tst_x, data_obj.tst_y, data_obj.dataset,
                                               avg_model, all_model, fast_exec, i)
                tst_perf_sel[offset_,i,:] = list_1; tst_perf_all[offset_,i,:len(list_2)] = list_2
                trn_perf_sel[offset_,i,:] = list_3; trn_perf_all[offset_,i,:] = list_4 
                        
            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if ((i+1) % save_period == 0):
                if save_models:
                    torch.save(avg_model.state_dict(), 'Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, (i+1)))                
                    torch.save(all_model.state_dict(), 'Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, (i+1)))
                    np.save('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, (i+1)), clnt_params_list)
                
                np.save('Model/%s/%s/%dcom_trn_perf_sel.npy'%(data_obj.name, suffix, (i+1)), trn_perf_sel[:,:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_sel.npy'%(data_obj.name, suffix, (i+1)), tst_perf_sel[:,:i+1])
                
                np.save('Model/%s/%s/%dcom_trn_perf_all.npy'%(data_obj.name, suffix, (i+1)), trn_perf_all[:,:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_all.npy'%(data_obj.name, suffix, (i+1)), tst_perf_all[:,:i+1])
                
                
            if (i+1) > save_period:
                if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period)):
                    # Delete the previous saved arrays
                    os.remove('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                    os.remove('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))

                    os.remove('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                    os.remove('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                    if save_models:
                        os.remove('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1-save_period))
                                    
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model
                
            # See if all clients in consecutive int(1/act_prob) rounds is diverging. If so stop execution.
            failure_arr = tst_perf_all[0,:,-1]
            total_fails = failure_arr[np.max([0,i-int(1/act_prob)]):i].sum()
            print('Total failures in this round: %d' %tst_perf_all[0,i, -1])
            if total_fails == int(act_prob*n_clnt)*int(1/act_prob):
                break
                
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all

###
def train_Meta_FedAvg_MAML(data_obj, act_prob ,learning_rate, batch_size, meta_learning_rate, K, com_amount, print_per, 
                           weight_decay, model_func, init_model, save_period, lr_decay, num_grad_step, 
                           rand_seed=0, save_models=False, fast_exec=False):
    suffix = 'PerAvg_MAML_S%d_F%f_Lr%f_B%d_K%d_W%f_MetaLr%f_GS%d_lrdecay%f_seed%d' %(save_period, act_prob, learning_rate, batch_size, K, weight_decay, meta_learning_rate, num_grad_step, lr_decay,rand_seed)
    
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
            
    if (not os.path.exists('Model/%s/%s' %(data_obj.name, suffix))):
        os.mkdir('Model/%s/%s' %(data_obj.name, suffix))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 4)); trn_perf_all = np.zeros((com_amount, 4))
    tst_perf_sel = np.zeros((com_amount, 4)); tst_perf_all = np.zeros((com_amount, 5))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0].cpu().numpy()
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    
    saved_itr = -1
    
    # Check if there are past saved iterates
    for i in range(com_amount):
        if os.path.exists('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)):
            saved_itr = i
            if save_models:
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr//save_period] = fed_model


        if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1))):
            trn_perf_sel[:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            trn_perf_all[:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, (i+1)))

            tst_perf_sel[:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            tst_perf_all[:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, (i+1)))
            if save_models:
                clnt_params_list = np.load('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1))

    
    if (not os.path.exists('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, com_amount))):         
        avg_model = model_func().to(device)
        if saved_itr == -1:
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        else:
            avg_model.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt'%(data_obj.name,suffix,saved_itr+1)))
          
        for i in range(saved_itr+1, com_amount):
            ### Fix randomness
            np.random.seed(i + rand_seed)
            clnt_list = np.arange(n_clnt)
            np.random.shuffle(clnt_list)
            selected_clnts = clnt_list[:int(act_prob*n_clnt)]
            
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                cur_model = model_func().to(device)
                cur_model.load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in cur_model.parameters():
                    params.requires_grad = True
                    
                cur_model = train_meta_model_MAML(model_func, cur_model, trn_x, trn_y, num_grad_step, meta_learning_rate,
                                                  learning_rate * (lr_decay ** i), batch_size, K, print_per,
                                                  weight_decay, data_obj.dataset)
                
                is_diverged = is_model_NaN(cur_model)
                if is_diverged:
                    # If model has NaN do not update the list put the average model
                    clnt_params_list[clnt] = get_mdl_params([avg_model], n_par)[0].cpu().numpy()
                    tst_perf_all[i][-1] += 1 
                else:
                    clnt_params_list[clnt] = get_mdl_params([cur_model], n_par)[0].cpu().numpy()

            # Scale with weights
            avg_selected = np.mean(clnt_params_list[selected_clnts], axis = 0)
            avg_model = set_client_from_params(model_func().to(device),
                                               torch.tensor(avg_selected, dtype=torch.float32).to(device))
            
            avg_all = np.mean(clnt_params_list, axis = 0)
            all_model = set_client_from_params(model_func().to(device), torch.tensor(avg_all, dtype=torch.float32).to(device))
                
            [list_1, list_2, list_3, list_4] = get_all_results_maml(meta_learning_rate,
                                     num_grad_step, data_obj.clnt_x, data_obj.clnt_y, data_obj.tst_x,
                                     data_obj.tst_y, data_obj.dataset, model_func, avg_model, all_model, fast_exec, i)
            tst_perf_sel[i] = list_1; tst_perf_all[i,:len(list_2)] = list_2
            trn_perf_sel[i] = list_3; trn_perf_all[i] = list_4
            
            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if ((i+1) % save_period == 0):
                if save_models:
                    torch.save(avg_model.state_dict(), 'Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, (i+1)))             
                    torch.save(all_model.state_dict(), 'Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, (i+1)))
                    np.save('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, (i+1)), clnt_params_list)
                
                np.save('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1)), trn_perf_sel[:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])
                
                np.save('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, (i+1)), trn_perf_all[:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, (i+1)), tst_perf_all[:i+1])
                
                
                if (i+1) > save_period:
                    if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                        os.remove('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                        
                        os.remove('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                        os.remove('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                        if save_models:
                            os.remove('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1-save_period))
                                    
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model
                
            # See if all clients in consecutive int(1/act_prob) rounds is diverging. If so stop execution.
            failure_arr = tst_perf_all[:, -1]
            total_fails = failure_arr[np.max([0,i-int(1/act_prob)]):i].sum()
            print('Total failures in this round: %d' %tst_perf_all[i, -1])
            if total_fails == int(act_prob*n_clnt)*int(1/act_prob):
                break
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all

###
def train_Meta_FedAvg_Proto(data_obj, act_prob ,learning_rate, batch_size, K, com_amount, print_per, weight_decay,
                            model_func, init_model, save_period, lr_decay,
                            rand_seed=0, save_models=False, fast_exec=False):
    suffix = 'PerAvg_Proto_S%d_F%f_Lr%f_B%d_K%d_W%f_lrdecay%f_seed%d' %(save_period, act_prob, learning_rate, batch_size, K, weight_decay, lr_decay, rand_seed)
    
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
        
    if (not os.path.exists('Model/%s/%s' %(data_obj.name, suffix))):
        os.mkdir('Model/%s/%s' %(data_obj.name, suffix))
    
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 4)); trn_perf_all = np.zeros((com_amount, 4))
    tst_perf_sel = np.zeros((com_amount, 4)); tst_perf_all = np.zeros((com_amount, 5))
    n_par = len(get_mdl_params([model_func()])[0])

    init_par_list=get_mdl_params([init_model], n_par)[0].cpu().numpy()
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    
    saved_itr = -1    
    # Check if there are past saved iterates
    for i in range(com_amount):
        if os.path.exists('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)):
            saved_itr = i
            if save_models:
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr//save_period] = fed_model


        if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1))):
            trn_perf_sel[:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            trn_perf_all[:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, (i+1)))

            tst_perf_sel[:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            tst_perf_all[:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, (i+1)))
            if save_models:
                clnt_params_list = np.load('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1))

    
    if (not os.path.exists('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, com_amount))):         
        avg_model = model_func().to(device)
        if saved_itr == -1:
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        else:
            # Load recent one
            avg_model.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt' %(data_obj.name,suffix,saved_itr+1)))
          
        for i in range(saved_itr+1, com_amount):
            ### Fix randomness
            np.random.seed(i + rand_seed)
            clnt_list = np.arange(n_clnt)
            np.random.shuffle(clnt_list)
            selected_clnts = clnt_list[:int(act_prob*n_clnt)]
            
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                cur_model = model_func().to(device)
                cur_model.load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in cur_model.parameters():
                    params.requires_grad = True
                    
                cur_model = train_proto_model(cur_model, trn_x, trn_y, learning_rate*(lr_decay**i),
                                              batch_size, K, print_per, weight_decay, data_obj.dataset)
                
                is_diverged = is_model_NaN(cur_model)
                if is_diverged:
                    # If model has NaN do not update the list put the average model
                    clnt_params_list[clnt] = get_mdl_params([avg_model], n_par)[0].cpu().numpy()
                    tst_perf_all[i][-1] += 1 
                else:
                    clnt_params_list[clnt] = get_mdl_params([cur_model], n_par)[0].cpu().numpy()

            # Scale with weights
            avg_selected = np.mean(clnt_params_list[selected_clnts], axis = 0)
            avg_model = set_client_from_params(model_func().to(device),
                                               torch.tensor(avg_selected, dtype=torch.float32).to(device))
            
            avg_all = np.mean(clnt_params_list, axis = 0)
            all_model = set_client_from_params(model_func().to(device), torch.tensor(avg_all, dtype=torch.float32).to(device))
            
            [list_1, list_2, list_3, list_4] = get_all_results_proto(
                                 data_obj.clnt_x, data_obj.clnt_y, data_obj.tst_x,
                                 data_obj.tst_y, data_obj.dataset, model_func, avg_model, all_model, fast_exec, i)
            tst_perf_sel[i] = list_1; tst_perf_all[i,:len(list_2)] = list_2
            trn_perf_sel[i] = list_3; trn_perf_all[i] = list_4
                        
            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if ((i+1) % save_period == 0):
                if save_models:
                    torch.save(avg_model.state_dict(), 'Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, (i+1)))          
                    torch.save(all_model.state_dict(), 'Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, (i+1)))
                    np.save('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, (i+1)), clnt_params_list)
                
                np.save('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1)), trn_perf_sel[:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])
                
                np.save('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, (i+1)), trn_perf_all[:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, (i+1)), tst_perf_all[:i+1])
                
                
                if (i+1) > save_period:
                    if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                        os.remove('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                        
                        os.remove('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                        os.remove('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                        if save_models:
                            os.remove('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1-save_period))
                                    
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model

            # See if all clients in consecutive int(1/act_prob) rounds is diverging. If so stop execution.
            failure_arr = tst_perf_all[:, -1]
            total_fails = failure_arr[np.max([0,i-int(1/act_prob)]):i].sum()
            print('Total failures in this round: %d' %tst_perf_all[i, -1])
            if total_fails == int(act_prob*n_clnt)*int(1/act_prob):
                break
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all

###
# FedDyn methods..
###
def train_FedDyn(data_obj, act_prob, alpha, learning_rate, batch_size, K, com_amount, print_per, weight_decay, 
                 model_func, init_model, save_period, lr_decay, meta_learning_rate_list=False,
                 num_grad_step_list=False, do_proto=False, do_plain=False, rand_seed=0, save_models=False, fast_exec=False):
    suffix = 'FedDy_S%d_F%f_Lr%f_B%d_alpha%f_K%d_W%f_lrdecay%f_seed%d' %(save_period, act_prob, learning_rate, batch_size, alpha, K, weight_decay, lr_decay,rand_seed)
    
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
            
    if (not os.path.exists('Model/%s/%s' %(data_obj.name, suffix))):
        os.mkdir('Model/%s/%s' %(data_obj.name, suffix))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))
    
    metaLr_numGrad = []
    if meta_learning_rate_list != False:
        for meta_learning_rate in meta_learning_rate_list:
                for num_grad_step in num_grad_step_list:
                    metaLr_numGrad.append([meta_learning_rate, num_grad_step])
    
    n_cases = len(metaLr_numGrad)
    n_cases = n_cases + 1 if do_proto else n_cases
    n_cases = n_cases + 1 if do_plain else n_cases
    
    trn_perf_sel = np.zeros((n_cases, com_amount, 4)); 
    trn_perf_all = np.zeros((n_cases, com_amount, 4));
    tst_perf_sel = np.zeros((n_cases, com_amount, 4));
    tst_perf_all = np.zeros((n_cases, com_amount, 5));
    
    n_par = len(get_mdl_params([model_func()])[0])

    lambda_model_list=np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list=get_mdl_params([init_model], n_par)[0].cpu().numpy()
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    
    saved_itr = -1     
    # Check if there are past saved iterates
    for i in range(com_amount):
        if os.path.exists('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)):
            saved_itr = i
            if save_models:
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr//save_period] = fed_model

        if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1))):
            trn_perf_sel[:,:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            trn_perf_all[:,:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, (i+1)))

            tst_perf_sel[:,:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            tst_perf_all[:,:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, (i+1)))
            if save_models:
                clnt_params_list = np.load('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1))
                lambda_model_list= np.load('Model/%s/%s/%d_lambda_params_list.npy' %(data_obj.name, suffix, i+1))

    
    if (not os.path.exists('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, com_amount))):         
        cld_model = model_func().to(device)
        avg_selected = model_func().to(device)
        if saved_itr == -1:            
            cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            avg_selected = get_mdl_params([init_model], n_par)[0].cpu().numpy()
        else:
            cld_model.load_state_dict(torch.load('Model/%s/%s/%dcom_cld.pt' %(data_obj.name, suffix, (saved_itr+1))))
            
            avg_selected.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, (saved_itr+1))))
            avg_selected = get_mdl_params([avg_selected], n_par)[0].cpu().numpy()
        cld_mdl_param = get_mdl_params([cld_model], n_par)[0].cpu().numpy()
          
        for i in range(saved_itr+1, com_amount):
            ### Fix randomness
            np.random.seed(i + rand_seed)
            clnt_list = np.arange(n_clnt)
            np.random.shuffle(clnt_list)
            selected_clnts = clnt_list[:int(act_prob*n_clnt)]
            
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            server_model = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)
            server_model_object = set_client_from_params(model_func().to(device),server_model)

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                cur_model = model_func().to(device)
                cur_model.load_state_dict(copy.deepcopy(dict(server_model_object.named_parameters())))

                for params in cur_model.parameters():
                    params.requires_grad = True
                
                lambda_model = torch.tensor(lambda_model_list[clnt], dtype=torch.float32, device=device)
                
                cur_model = train_dyn_model(alpha, lambda_model, server_model, cur_model, trn_x, trn_y, 
                                            learning_rate * (lr_decay ** i), batch_size, K, print_per,
                                            weight_decay, data_obj.dataset)
                
                is_diverged = is_model_NaN(cur_model)
                if is_diverged:
                    # If model has NaN do not update the list put the average model, do not update the lambda model.
                    clnt_params_list[clnt] = np.copy(avg_selected)
                    tst_perf_all[0][i][-1] += 1
                else:
                    clnt_params_list[clnt] = get_mdl_params([cur_model], n_par)[0].cpu().numpy()
                    lambda_model_list[clnt] = lambda_model_list[clnt] - alpha * (clnt_params_list[clnt] - cld_mdl_param)

            # Scale with weights
            avg_selected = np.mean(clnt_params_list[selected_clnts], axis = 0)
            avg_model = set_client_from_params(model_func().to(device),
                                               torch.tensor(avg_selected, dtype=torch.float32).to(device))
            
            avg_all = np.mean(clnt_params_list, axis = 0)
            all_model = set_client_from_params(model_func().to(device), torch.tensor(avg_all, dtype=torch.float32).to(device))
            cld_mdl_param = avg_selected - 1/alpha*np.mean(lambda_model_list, axis=0)
            
            for idx_, [meta_learning_rate, num_grad_step] in enumerate(metaLr_numGrad):
                [list_1, list_2, list_3, list_4] = get_all_results_maml(meta_learning_rate,
                                     num_grad_step, data_obj.clnt_x, data_obj.clnt_y, data_obj.tst_x,
                                     data_obj.tst_y, data_obj.dataset, model_func, avg_model, all_model, fast_exec, i)
                tst_perf_sel[idx_,i,:] = list_1; tst_perf_all[idx_,i,:len(list_2)] = list_2
                trn_perf_sel[idx_,i,:] = list_3; trn_perf_all[idx_,i,:] = list_4
            
            offset_ = len(metaLr_numGrad)
            
            if do_proto:
                [list_1, list_2, list_3, list_4] = get_all_results_proto(
                                     data_obj.clnt_x, data_obj.clnt_y, data_obj.tst_x, data_obj.tst_y, data_obj.dataset,
                                     model_func, avg_model, all_model, fast_exec, i)
                tst_perf_sel[idx_+offset_,i,:] = list_1; tst_perf_all[idx_+offset_,i,:len(list_2)] = list_2
                trn_perf_sel[idx_+offset_,i,:] = list_3; trn_perf_all[idx_+offset_,i,:] = list_4
                offset_ = len(metaLr_numGrad) + 1
                
            if do_plain:
                [list_1, list_2, list_3, list_4] = get_all_results_plain(data_obj.clnt_x,
                                               data_obj.clnt_y, data_obj.tst_x, data_obj.tst_y, data_obj.dataset,
                                               avg_model, all_model, fast_exec, i)
                tst_perf_sel[offset_,i,:] = list_1; tst_perf_all[offset_,i,:len(list_2)] = list_2
                trn_perf_sel[offset_,i,:] = list_3; trn_perf_all[offset_,i,:] = list_4
            
            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if ((i+1) % save_period == 0):
                if save_models:
                    torch.save(avg_model.state_dict(), 'Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, (i+1)))             
                    torch.save(all_model.state_dict(), 'Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, (i+1)))
                    torch.save(cld_model.state_dict(), 'Model/%s/%s/%dcom_cld.pt' %(data_obj.name, suffix, (i+1)))
                    np.save('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, (i+1)), clnt_params_list)
                    np.save('Model/%s/%s/%d_lambda_params_list.npy' %(data_obj.name, suffix, (i+1)), lambda_model_list)
                
                np.save('Model/%s/%s/%dcom_trn_perf_sel.npy'%(data_obj.name, suffix, (i+1)), trn_perf_sel[:,:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_sel.npy'%(data_obj.name, suffix, (i+1)), tst_perf_sel[:,:i+1])
                
                np.save('Model/%s/%s/%dcom_trn_perf_all.npy'%(data_obj.name, suffix, (i+1)), trn_perf_all[:,:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_all.npy'%(data_obj.name, suffix, (i+1)), tst_perf_all[:,:i+1])
                
                
                if (i+1) > save_period:
                    if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                        os.remove('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                        
                        os.remove('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                        os.remove('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                        if save_models:
                            os.remove('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1-save_period))
                            os.remove('Model/%s/%s/%d_lambda_params_list.npy' %(data_obj.name, suffix, i+1-save_period))
                                    
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model
                
            # See if all clients in consecutive int(1/act_prob) rounds is diverging. If so stop execution.
            failure_arr = tst_perf_all[0,:,-1]
            total_fails = failure_arr[np.max([0,i-int(1/act_prob)]):i].sum()
            print('Total failures in this round: %d' %tst_perf_all[0,i, -1])
            if total_fails == int(act_prob*n_clnt)*int(1/act_prob):
                break
                
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all

# ###
def train_Meta_FedDyn_MAML(data_obj, act_prob, alpha, learning_rate, batch_size, meta_learning_rate, K, com_amount, print_per,
                           weight_decay, model_func, init_model, save_period, num_grad_step, lr_decay,
                           rand_seed=0, save_models=False, fast_exec=False):
    suffix = 'PFLDyn_MAML_S%d_F%f_Lr%f_B%d_alpha%f_K%d_W%f_MetaLr%f_GS%d_lrdecay%f_seed%d' %(save_period, act_prob, learning_rate, batch_size, alpha, K, weight_decay, meta_learning_rate, num_grad_step, lr_decay,rand_seed)
    
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
            
    if (not os.path.exists('Model/%s/%s' %(data_obj.name, suffix))):
        os.mkdir('Model/%s/%s' %(data_obj.name, suffix))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 4)); trn_perf_all = np.zeros((com_amount, 4))
    tst_perf_sel = np.zeros((com_amount, 4)); tst_perf_all = np.zeros((com_amount, 5))
    n_par = len(get_mdl_params([model_func()])[0])

    lambda_model_list=np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list=get_mdl_params([init_model], n_par)[0].cpu().numpy()
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    
    saved_itr = -1
    # Check if there are past saved iterates
    for i in range(com_amount):
        if os.path.exists('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)):
            saved_itr = i
            if save_models:
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr//save_period] = fed_model

        if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1))):
            trn_perf_sel[:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            trn_perf_all[:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, (i+1)))

            tst_perf_sel[:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            tst_perf_all[:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, (i+1)))
            if save_models:
                clnt_params_list = np.load('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1))
                lambda_model_list= np.load('Model/%s/%s/%d_lambda_params_list.npy' %(data_obj.name, suffix, i+1))

    
    if (not os.path.exists('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, com_amount))):
        cld_model = model_func().to(device)
        avg_selected = model_func().to(device)
        if saved_itr == -1:            
            cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            avg_selected = get_mdl_params([init_model], n_par)[0].cpu().numpy()
        else:
            cld_model.load_state_dict(torch.load('Model/%s/%s/%dcom_cld.pt' %(data_obj.name, suffix, (saved_itr+1))))
            
            avg_selected.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, (saved_itr+1))))
            avg_selected = get_mdl_params([avg_selected], n_par)[0].cpu().numpy()
        cld_mdl_param = get_mdl_params([cld_model], n_par)[0].cpu().numpy()
          
        for i in range(saved_itr+1, com_amount):
            ### Fix randomness
            np.random.seed(i + rand_seed)
            clnt_list = np.arange(n_clnt)
            np.random.shuffle(clnt_list)
            selected_clnts = clnt_list[:int(act_prob*n_clnt)]
            
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            server_model = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)
            server_model_object = set_client_from_params(model_func().to(device),server_model)

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                cur_model = model_func().to(device)
                cur_model.load_state_dict(copy.deepcopy(dict(server_model_object.named_parameters())))

                for params in cur_model.parameters():
                    params.requires_grad = True
                
                lambda_model = torch.tensor(lambda_model_list[clnt], dtype=torch.float32, device=device)
                
                cur_model = train_dyn_meta_model_MAML(alpha, lambda_model, server_model, model_func, cur_model, trn_x, trn_y, 
                                                      num_grad_step, meta_learning_rate, 
                                                      learning_rate * (lr_decay ** i), batch_size, K, print_per,
                                                      weight_decay, data_obj.dataset)                
                is_diverged = is_model_NaN(cur_model)
                if is_diverged:
                    # If model has NaN do not update the list put the average model, do not update the lambda model.
                    clnt_params_list[clnt] = np.copy(avg_selected)
                    tst_perf_all[i][-1] += 1 
                else:
                    clnt_params_list[clnt] = get_mdl_params([cur_model], n_par)[0].cpu().numpy()
                    lambda_model_list[clnt] = lambda_model_list[clnt] - alpha * (clnt_params_list[clnt] - cld_mdl_param)

            # Scale with weights
            avg_selected = np.mean(clnt_params_list[selected_clnts], axis = 0)
            avg_model = set_client_from_params(model_func().to(device),
                                               torch.tensor(avg_selected, dtype=torch.float32).to(device))
            
            avg_all = np.mean(clnt_params_list, axis = 0)
            all_model = set_client_from_params(model_func().to(device), torch.tensor(avg_all, dtype=torch.float32).to(device))
            cld_mdl_param = avg_selected - 1/alpha*np.mean(lambda_model_list, axis=0)
            
            [list_1, list_2, list_3, list_4] = get_all_results_maml(meta_learning_rate,
                                     num_grad_step, data_obj.clnt_x, data_obj.clnt_y, data_obj.tst_x,
                                     data_obj.tst_y, data_obj.dataset, model_func, avg_model, all_model, fast_exec, i)
            tst_perf_sel[i] = list_1; tst_perf_all[i,:len(list_2)] = list_2
            trn_perf_sel[i] = list_3; trn_perf_all[i] = list_4
               
            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if ((i+1) % save_period == 0):
                if save_models:
                    torch.save(avg_model.state_dict(), 'Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, (i+1)))             
                    torch.save(all_model.state_dict(), 'Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, (i+1)))
                    torch.save(cld_model.state_dict(), 'Model/%s/%s/%dcom_cld.pt' %(data_obj.name, suffix, (i+1)))
                    np.save('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, (i+1)), clnt_params_list)
                    np.save('Model/%s/%s/%d_lambda_params_list.npy' %(data_obj.name, suffix, (i+1)), lambda_model_list)
                
                np.save('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1)), trn_perf_sel[:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])
                
                np.save('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, (i+1)), trn_perf_all[:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, (i+1)), tst_perf_all[:i+1])
                
                
                if (i+1) > save_period:
                    if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                        os.remove('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                        
                        os.remove('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                        os.remove('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                        if save_models:
                            os.remove('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1-save_period))
                            os.remove('Model/%s/%s/%d_lambda_params_list.npy' %(data_obj.name, suffix, i+1-save_period))
                                    
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model
            
            # See if all clients in consecutive int(1/act_prob) rounds is diverging. If so stop execution.
            failure_arr = tst_perf_all[:, -1]
            total_fails = failure_arr[np.max([0,i-int(1/act_prob)]):i].sum()
            print('Total failures in this round: %d' %tst_perf_all[i, -1])
            if total_fails == int(act_prob*n_clnt)*int(1/act_prob):
                break
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all

####
def train_Meta_FedDyn_Proto(data_obj, act_prob, alpha, learning_rate, batch_size, K, com_amount, print_per, weight_decay,
                            model_func, init_model, save_period, lr_decay,
                            rand_seed=0, save_models=False, fast_exec=False):
    suffix = 'PFLDyn_Proto_S%d_F%f_Lr%f_B%d_alpha%f_K%d_W%f_lrdecay%f_seed%d' %(save_period, act_prob, learning_rate, batch_size, alpha, K, weight_decay, lr_decay, rand_seed)
    
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
            
    if (not os.path.exists('Model/%s/%s' %(data_obj.name, suffix))):
        os.mkdir('Model/%s/%s' %(data_obj.name, suffix))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 4)); trn_perf_all = np.zeros((com_amount, 4))
    tst_perf_sel = np.zeros((com_amount, 4)); tst_perf_all = np.zeros((com_amount, 5))
    n_par = len(get_mdl_params([model_func()])[0])

    lambda_model_list=np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list=get_mdl_params([init_model], n_par)[0].cpu().numpy()
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    
    saved_itr = -1
    # Check if there are past saved iterates
    for i in range(com_amount):
        if os.path.exists('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)):
            saved_itr = i
            if save_models:
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr//save_period] = fed_model

        if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1))):
            trn_perf_sel[:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            trn_perf_all[:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, (i+1)))

            tst_perf_sel[:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            tst_perf_all[:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, (i+1)))
            if save_models:
                clnt_params_list = np.load('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1))
                lambda_model_list= np.load('Model/%s/%s/%d_lambda_params_list.npy' %(data_obj.name, suffix, i+1))

    
    if not os.path.exists('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, com_amount)):         
        cld_model = model_func().to(device)
        avg_selected = model_func().to(device)
        if saved_itr == -1:            
            cld_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
            avg_selected = get_mdl_params([init_model], n_par)[0].cpu().numpy()
        else:
            cld_model.load_state_dict(torch.load('Model/%s/%s/%dcom_cld.pt' %(data_obj.name, suffix, (saved_itr+1))))
            
            avg_selected.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, (saved_itr+1))))
            avg_selected = get_mdl_params([avg_selected], n_par)[0].cpu().numpy()
        cld_mdl_param = get_mdl_params([cld_model], n_par)[0].cpu().numpy()
          
        for i in range(saved_itr+1, com_amount):
            ### Fix randomness
            np.random.seed(i + rand_seed)
            clnt_list = np.arange(n_clnt)
            np.random.shuffle(clnt_list)
            selected_clnts = clnt_list[:int(act_prob*n_clnt)]
            
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            server_model = torch.tensor(cld_mdl_param, dtype=torch.float32, device=device)
            server_model_object = set_client_from_params(model_func().to(device),server_model)

            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                cur_model = model_func().to(device)
                cur_model.load_state_dict(copy.deepcopy(dict(server_model_object.named_parameters())))

                for params in cur_model.parameters():
                    params.requires_grad = True
                
                lambda_model = torch.tensor(lambda_model_list[clnt], dtype=torch.float32, device=device)
                
                cur_model = train_dyn_proto_model(alpha, lambda_model, server_model, cur_model, trn_x, trn_y,
                                                  learning_rate*(lr_decay**i), batch_size, K, print_per,
                                                  weight_decay, data_obj.dataset)
                is_diverged = is_model_NaN(cur_model)
                if is_diverged:
                    # If model has NaN do not update the list put the average model, do not update the lambda model.
                    clnt_params_list[clnt] = np.copy(avg_selected)
                    tst_perf_all[i][-1] += 1 
                else:
                    clnt_params_list[clnt] = get_mdl_params([cur_model], n_par)[0].cpu().numpy()
                    lambda_model_list[clnt] = lambda_model_list[clnt] - alpha * (clnt_params_list[clnt] - cld_mdl_param)

            # Scale with weights
            avg_selected = np.mean(clnt_params_list[selected_clnts], axis = 0)
            avg_model = set_client_from_params(model_func().to(device),
                                               torch.tensor(avg_selected, dtype=torch.float32).to(device))
            
            avg_all = np.mean(clnt_params_list, axis = 0)
            all_model = set_client_from_params(model_func().to(device), torch.tensor(avg_all, dtype=torch.float32).to(device))
            cld_mdl_param = avg_selected - 1/alpha*np.mean(lambda_model_list, axis=0)
            
            [list_1, list_2, list_3, list_4] = get_all_results_proto(
                                 data_obj.clnt_x, data_obj.clnt_y, data_obj.tst_x,
                                 data_obj.tst_y, data_obj.dataset, model_func, avg_model, all_model, fast_exec, i)
            tst_perf_sel[i] = list_1; tst_perf_all[i,:len(list_2)] = list_2
            trn_perf_sel[i] = list_3; trn_perf_all[i] = list_4
                        
            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if ((i+1) % save_period == 0):
                if save_models:
                    torch.save(avg_model.state_dict(), 'Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, (i+1)))             
                    torch.save(all_model.state_dict(), 'Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, (i+1)))
                    torch.save(cld_model.state_dict(), 'Model/%s/%s/%dcom_cld.pt' %(data_obj.name, suffix, (i+1)))
                    np.save('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, (i+1)), clnt_params_list)
                    np.save('Model/%s/%s/%d_lambda_params_list.npy' %(data_obj.name, suffix, (i+1)), lambda_model_list)
                
                np.save('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1)), trn_perf_sel[:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])
                
                np.save('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, (i+1)), trn_perf_all[:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, (i+1)), tst_perf_all[:i+1])
                
                
                if (i+1) > save_period:
                    if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                        os.remove('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                        
                        os.remove('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                        os.remove('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                        if save_models:
                            os.remove('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1-save_period))
                            os.remove('Model/%s/%s/%d_lambda_params_list.npy' %(data_obj.name, suffix, i+1-save_period))
                                    
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model
                
            # See if all clients in consecutive int(1/act_prob) rounds is diverging. If so stop execution.
            failure_arr = tst_perf_all[:, -1]
            total_fails = failure_arr[np.max([0,i-int(1/act_prob)]):i].sum()
            print('Total failures in this round: %d' %tst_perf_all[i, -1])
            if total_fails == int(act_prob*n_clnt)*int(1/act_prob):
                break
                
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all

def train_SCAFFOLD(data_obj, act_prob, learning_rate, batch_size, K, com_amount, print_per, weight_decay, model_func,
                   init_model, save_period, lr_decay,
                   meta_learning_rate_list=False, num_grad_step_list=False, do_proto=False, do_plain=False,
                   rand_seed=0, save_models=False, fast_exec=False):
    suffix = 'SCAFFOLD_S%d_F%f_Lr%f_B%d_K%d_W%f_lrdecay%f_seed%d' %(save_period, act_prob, learning_rate, batch_size, K, weight_decay, lr_decay,rand_seed)       
    
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
            
    if (not os.path.exists('Model/%s/%s' %(data_obj.name, suffix))):
        os.mkdir('Model/%s/%s' %(data_obj.name, suffix))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))
    
    metaLr_numGrad = []
    if meta_learning_rate_list != False:
        for meta_learning_rate in meta_learning_rate_list:
                for num_grad_step in num_grad_step_list:
                    metaLr_numGrad.append([meta_learning_rate, num_grad_step])
    
                    
    n_cases = len(metaLr_numGrad)
    n_cases = n_cases + 1 if do_proto else n_cases
    n_cases = n_cases + 1 if do_plain else n_cases
    
    trn_perf_sel = np.zeros((n_cases, com_amount, 4)); 
    trn_perf_all = np.zeros((n_cases, com_amount, 4));
    tst_perf_sel = np.zeros((n_cases, com_amount, 4));
    tst_perf_all = np.zeros((n_cases, com_amount, 5));
    
    n_par = len(get_mdl_params([model_func()])[0])

    c_state_list=np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list=get_mdl_params([init_model], n_par)[0].cpu().numpy()
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    
    saved_itr = -1
    # Check if there are past saved iterates
    for i in range(com_amount):
        if os.path.exists('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)):
            saved_itr = i
            if save_models:
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr//save_period] = fed_model

        if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1))):
            trn_perf_sel[:,:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            trn_perf_all[:,:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, (i+1)))

            tst_perf_sel[:,:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            tst_perf_all[:,:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, (i+1)))
            if save_models:
                clnt_params_list = np.load('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1))
                c_state_list= np.load('Model/%s/%s/%d_c_state_list.npy' %(data_obj.name, suffix, i+1))

    
    if (not os.path.exists('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, com_amount))):         
        avg_model = model_func().to(device)
        if saved_itr == -1:            
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        else:
            avg_model.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, (saved_itr+1))))

        server_params = get_mdl_params([avg_model], n_par)[0].cpu().numpy()
          
        for i in range(saved_itr+1, com_amount):
            ### Fix randomness
            np.random.seed(i + rand_seed)
            clnt_list = np.arange(n_clnt)
            np.random.shuffle(clnt_list)
            selected_clnts = clnt_list[:int(act_prob*n_clnt)]
            
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            server_c_state = np.mean(c_state_list, axis=0)
            
            for clnt in selected_clnts:
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                cur_model = model_func().to(device)
                cur_model.load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in cur_model.parameters():
                    params.requires_grad = True
                
                curr_state_params_diff = torch.tensor(-c_state_list[clnt] + server_c_state, dtype=torch.float32, device=device)
                cur_model = train_SCAF_model(curr_state_params_diff, cur_model, trn_x, trn_y, 
                                                 learning_rate * (lr_decay ** i), batch_size, K, print_per, 
                                                 weight_decay, data_obj.dataset)
                
                is_diverged = is_model_NaN(cur_model)
                if is_diverged:
                    # If model has NaN do not update the list put the average model, do not update the lambda model.
                    clnt_params_list[clnt] = np.copy(server_params)
                    tst_perf_all[0][i][-1] += 1
                else:
                    clnt_params_list[clnt] = get_mdl_params([cur_model], n_par)[0].cpu().numpy()
                    c_state_list[clnt] += (-server_c_state + 1/K/learning_rate * (server_params - clnt_params_list[clnt]))
                    

            server_params = np.mean(clnt_params_list[selected_clnts], axis = 0)
            avg_model = set_client_from_params(model_func().to(device),
                                               torch.tensor(server_params, dtype=torch.float32).to(device))
            
            avg_all = np.mean(clnt_params_list, axis = 0)
            all_model = set_client_from_params(model_func().to(device), torch.tensor(avg_all, dtype=torch.float32).to(device))
            
            for idx_, [meta_learning_rate, num_grad_step] in enumerate(metaLr_numGrad):
                [list_1, list_2, list_3, list_4] = get_all_results_maml(meta_learning_rate,
                                     num_grad_step, data_obj.clnt_x, data_obj.clnt_y, data_obj.tst_x,
                                     data_obj.tst_y, data_obj.dataset, model_func, avg_model, all_model, fast_exec, i)
                tst_perf_sel[idx_,i,:] = list_1; tst_perf_all[idx_,i,:len(list_2)] = list_2
                trn_perf_sel[idx_,i,:] = list_3; trn_perf_all[idx_,i,:] = list_4
            
            offset_ = len(metaLr_numGrad)
            
            if do_proto:
                [list_1, list_2, list_3, list_4] = get_all_results_proto(
                                     data_obj.clnt_x, data_obj.clnt_y, data_obj.tst_x, data_obj.tst_y, data_obj.dataset,
                                     model_func, avg_model, all_model, fast_exec, i)
                tst_perf_sel[idx_+offset_,i,:] = list_1; tst_perf_all[idx_+offset_,i,:len(list_2)] = list_2
                trn_perf_sel[idx_+offset_,i,:] = list_3; trn_perf_all[idx_+offset_,i,:] = list_4
                offset_ = len(metaLr_numGrad) + 1
                
            if do_plain:
                [list_1, list_2, list_3, list_4] = get_all_results_plain(data_obj.clnt_x,
                                               data_obj.clnt_y, data_obj.tst_x, data_obj.tst_y, data_obj.dataset,
                                               avg_model, all_model, fast_exec, i)
                tst_perf_sel[offset_,i,:] = list_1; tst_perf_all[offset_,i,:len(list_2)] = list_2
                trn_perf_sel[offset_,i,:] = list_3; trn_perf_all[offset_,i,:] = list_4
            
            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if ((i+1) % save_period == 0):
                if save_models:
                    torch.save(avg_model.state_dict(), 'Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, (i+1)))             
                    torch.save(all_model.state_dict(), 'Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, (i+1)))
                    np.save('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, (i+1)), clnt_params_list)
                    np.save('Model/%s/%s/%d_c_state_list.npy' %(data_obj.name, suffix, (i+1)), c_state_list)
                
                np.save('Model/%s/%s/%dcom_trn_perf_sel.npy'%(data_obj.name, suffix, (i+1)), trn_perf_sel[:,:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_sel.npy'%(data_obj.name, suffix, (i+1)), tst_perf_sel[:,:i+1])
                
                np.save('Model/%s/%s/%dcom_trn_perf_all.npy'%(data_obj.name, suffix, (i+1)), trn_perf_all[:,:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_all.npy'%(data_obj.name, suffix, (i+1)), tst_perf_all[:,:i+1])
                
                
                if (i+1) > save_period:
                    if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                        os.remove('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                        
                        os.remove('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                        os.remove('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                        if save_models:
                            os.remove('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1-save_period))
                            os.remove('Model/%s/%s/%d_c_state_list.npy' %(data_obj.name, suffix, i+1-save_period))
                                    
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model
                
            # See if all clients in consecutive int(1/act_prob) rounds is diverging. If so stop execution.
            failure_arr = tst_perf_all[0,:,-1]
            total_fails = failure_arr[np.max([0,i-int(1/act_prob)]):i].sum()
            print('Total failures in this round: %d' %tst_perf_all[0,i, -1])
            if total_fails == int(act_prob*n_clnt)*int(1/act_prob):
                break
                
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all

####

def train_Meta_SCAFFOLD_MAML(data_obj, act_prob, learning_rate, batch_size, meta_learning_rate, K, com_amount, print_per,
                             weight_decay, model_func, init_model, save_period, num_grad_step, lr_decay,
                             rand_seed=0, save_models=False, fast_exec=False):
    suffix = 'PFLSCAF_MAML_S%d_F%f_Lr%f_B%d_K%d_W%f_MetaLr%f_GS%d_lrdecay%f_seed%d' %(save_period, act_prob, learning_rate, batch_size, K, weight_decay, meta_learning_rate, num_grad_step, lr_decay,rand_seed)
       
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
        
    if (not os.path.exists('Model/%s/%s' %(data_obj.name, suffix))):
        os.mkdir('Model/%s/%s' %(data_obj.name, suffix))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 4)); trn_perf_all = np.zeros((com_amount, 4))
    tst_perf_sel = np.zeros((com_amount, 4)); tst_perf_all = np.zeros((com_amount, 5))
    n_par = len(get_mdl_params([model_func()])[0])

    c_state_list=np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list=get_mdl_params([init_model], n_par)[0].cpu().numpy()
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    
    saved_itr = -1   
    # Check if there are past saved iterates
    for i in range(com_amount):
        if os.path.exists('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)):
            saved_itr = i
            if save_models:
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr//save_period] = fed_model

        if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1))):
            trn_perf_sel[:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            trn_perf_all[:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, (i+1)))

            tst_perf_sel[:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            tst_perf_all[:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, (i+1)))
            if save_models:
                clnt_params_list = np.load('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1))
                c_state_list= np.load('Model/%s/%s/%d_c_state_list.npy' %(data_obj.name, suffix, i+1))

    
    if (not os.path.exists('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, com_amount))):
        avg_model = model_func().to(device)
        if saved_itr == -1:            
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        else:            
            avg_model.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, (saved_itr+1))))
        server_model_param = get_mdl_params([avg_model], n_par)[0].cpu().numpy()
          
        for i in range(saved_itr+1, com_amount):
            ### Fix randomness
            np.random.seed(i + rand_seed)
            clnt_list = np.arange(n_clnt)
            np.random.shuffle(clnt_list)
            selected_clnts = clnt_list[:int(act_prob*n_clnt)]
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            server_c_state = np.mean(c_state_list, axis=0)
            for clnt in selected_clnts:                
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                cur_model = model_func().to(device)
                cur_model.load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in cur_model.parameters():
                    params.requires_grad = True
                
                curr_state_params_diff = torch.tensor(-c_state_list[clnt] + server_c_state, dtype=torch.float32, device=device)
                
                cur_model = train_SCAF_meta_model_MAML(curr_state_params_diff,model_func, cur_model, trn_x, trn_y,
                                                           num_grad_step, meta_learning_rate,
                                                           learning_rate * (lr_decay ** i), batch_size, K, print_per,
                                                           weight_decay, data_obj.dataset)
                
                is_diverged = is_model_NaN(cur_model)
                if is_diverged:
                    # If model has NaN do not update the list put the average model, do not update the lambda model.
                    clnt_params_list[clnt] = np.copy(server_model_param)
                    tst_perf_all[i][-1] += 1 
                else:
                    clnt_params_list[clnt] = get_mdl_params([cur_model], n_par)[0].cpu().numpy()
                    c_state_list[clnt] += (-server_c_state + 1/K/learning_rate * (server_model_param - clnt_params_list[clnt]))

            server_model_param = np.mean(clnt_params_list[selected_clnts], axis = 0)
            avg_model = set_client_from_params(model_func().to(device),
                                               torch.tensor(server_model_param, dtype=torch.float32).to(device))
            
            avg_all = np.mean(clnt_params_list, axis = 0)
            all_model = set_client_from_params(model_func().to(device), torch.tensor(avg_all, dtype=torch.float32).to(device))
            
            [list_1, list_2, list_3, list_4] = get_all_results_maml(meta_learning_rate,
                                     num_grad_step, data_obj.clnt_x, data_obj.clnt_y, data_obj.tst_x,
                                     data_obj.tst_y, data_obj.dataset, model_func, avg_model, all_model, fast_exec, i)
            tst_perf_sel[i] = list_1; tst_perf_all[i,:len(list_2)] = list_2
            trn_perf_sel[i] = list_3; trn_perf_all[i] = list_4

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if ((i+1) % save_period == 0):
                if save_models:
                    torch.save(avg_model.state_dict(), 'Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, (i+1)))             
                    torch.save(all_model.state_dict(), 'Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, (i+1)))
                    np.save('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, (i+1)), clnt_params_list)
                    np.save('Model/%s/%s/%d_c_state_list.npy' %(data_obj.name, suffix, (i+1)), c_state_list)
                
                np.save('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1)), trn_perf_sel[:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])
                
                np.save('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, (i+1)), trn_perf_all[:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, (i+1)), tst_perf_all[:i+1])
                
                
                if (i+1) > save_period:
                    if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                        os.remove('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                        
                        os.remove('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                        os.remove('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                        if save_models:
                            os.remove('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1-save_period))
                            os.remove('Model/%s/%s/%d_c_state_list.npy' %(data_obj.name, suffix, i+1-save_period))
                                    
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model
            
            # See if all clients in consecutive int(1/act_prob) rounds is diverging. If so stop execution.
            failure_arr = tst_perf_all[:, -1]
            total_fails = failure_arr[np.max([0,i-int(1/act_prob)]):i].sum()
            print('Total failures in this round: %d' %tst_perf_all[i, -1])
            if total_fails == int(act_prob*n_clnt)*int(1/act_prob):
                break
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all

####

def train_Meta_SCAFFOLD_Proto(data_obj, act_prob, learning_rate, batch_size, K, com_amount, print_per, weight_decay,
                              model_func, init_model, save_period, lr_decay, 
                              rand_seed=0, save_models=False, fast_exec=False):
    suffix = 'PFLSCAFD_Proto_S%d_F%f_Lr%f_B%d_K%d_W%f_lrdecay%f_seed%d' %(save_period, act_prob, learning_rate, batch_size, K, weight_decay, lr_decay,rand_seed)
    
    n_clnt=data_obj.n_client
    clnt_x = data_obj.clnt_x; clnt_y=data_obj.clnt_y
            
    if (not os.path.exists('Model/%s/%s' %(data_obj.name, suffix))):
        os.mkdir('Model/%s/%s' %(data_obj.name, suffix))
        
    n_save_instances = int(com_amount / save_period)
    fed_mdls_sel = list(range(n_save_instances)); fed_mdls_all = list(range(n_save_instances))
    
    trn_perf_sel = np.zeros((com_amount, 4)); trn_perf_all = np.zeros((com_amount, 4))
    tst_perf_sel = np.zeros((com_amount, 4)); tst_perf_all = np.zeros((com_amount, 5))
    n_par = len(get_mdl_params([model_func()])[0])

    c_state_list=np.zeros((n_clnt, n_par)).astype('float32')
    init_par_list=get_mdl_params([init_model], n_par)[0].cpu().numpy()
    clnt_params_list=np.ones(n_clnt).astype('float32').reshape(-1, 1) * init_par_list.reshape(1, -1) # n_clnt X n_par
    
    saved_itr = -1
    # Check if there are past saved iterates
    for i in range(com_amount):
        if os.path.exists('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)):
            saved_itr = i
            if save_models:
                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_sel[saved_itr//save_period] = fed_model

                ###
                fed_model = model_func()
                fed_model.load_state_dict(torch.load('Model/%s/%s/%dcom_all.pt' %( data_obj.name, suffix, i+1)))
                fed_model.eval()
                fed_model = fed_model.to(device)
                # Freeze model
                for params in fed_model.parameters():
                    params.requires_grad = False

                fed_mdls_all[saved_itr//save_period] = fed_model

        if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1))):
            trn_perf_sel[:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            trn_perf_all[:i+1] = np.load('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, (i+1)))

            tst_perf_sel[:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, (i+1)))
            tst_perf_all[:i+1] = np.load('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, (i+1)))
            if save_models:
                clnt_params_list = np.load('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1))
                c_state_list= np.load('Model/%s/%s/%d_c_state_list.npy' %(data_obj.name, suffix, i+1))

    
    if not os.path.exists('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, com_amount)):         
        avg_model = model_func().to(device)
        if saved_itr == -1:            
            avg_model.load_state_dict(copy.deepcopy(dict(init_model.named_parameters())))
        else:            
            avg_model.load_state_dict(torch.load('Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, (saved_itr+1))))
        server_model_param = get_mdl_params([avg_model], n_par)[0].cpu().numpy()
          
        for i in range(saved_itr+1, com_amount):
            ### Fix randomness
            np.random.seed(i + rand_seed)
            clnt_list = np.arange(n_clnt)
            np.random.shuffle(clnt_list)
            selected_clnts = clnt_list[:int(act_prob*n_clnt)]
            
            print('Selected Clients: %s' %(', '.join(['%2d' %item for item in selected_clnts])))
            server_c_state = np.mean(c_state_list, axis=0)

            for clnt in selected_clnts:                    
                print('---- Training client %d' %clnt)
                trn_x = clnt_x[clnt]
                trn_y = clnt_y[clnt]

                cur_model = model_func().to(device)
                cur_model.load_state_dict(copy.deepcopy(dict(avg_model.named_parameters())))

                for params in cur_model.parameters():
                    params.requires_grad = True
                
                curr_state_params_diff = torch.tensor(-c_state_list[clnt] + server_c_state, dtype=torch.float32, device=device)
                
                cur_model = train_SCAF_proto_model(curr_state_params_diff, cur_model, trn_x, trn_y,
                                                       learning_rate*(lr_decay**i), batch_size, K, print_per,
                                                       weight_decay, data_obj.dataset)

                is_diverged = is_model_NaN(cur_model)
                if is_diverged:
                    # If model has NaN do not update the list put the average model, do not update the lambda model.
                    clnt_params_list[clnt] = np.copy(server_model_param)
                    tst_perf_all[i][-1] += 1 
                else:
                    clnt_params_list[clnt] = get_mdl_params([cur_model], n_par)[0].cpu().numpy()
                    c_state_list[clnt] += (-server_c_state + 1/K/learning_rate * (server_model_param - clnt_params_list[clnt]))
                    

            server_model_param = np.mean(clnt_params_list[selected_clnts], axis = 0)
            avg_model = set_client_from_params(model_func().to(device),
                                               torch.tensor(server_model_param, dtype=torch.float32).to(device))
            
            avg_all = np.mean(clnt_params_list, axis = 0)
            all_model = set_client_from_params(model_func().to(device), torch.tensor(avg_all, dtype=torch.float32).to(device))
            
            [list_1, list_2, list_3, list_4] = get_all_results_proto(
                                 data_obj.clnt_x, data_obj.clnt_y, data_obj.tst_x,
                                 data_obj.tst_y, data_obj.dataset, model_func, avg_model, all_model, fast_exec, i)
            tst_perf_sel[i] = list_1; tst_perf_all[i,:len(list_2)] = list_2
            trn_perf_sel[i] = list_3; trn_perf_all[i] = list_4

            # Freeze model
            for params in avg_model.parameters():
                params.requires_grad = False
            if ((i+1) % save_period == 0):
                if save_models:
                    torch.save(avg_model.state_dict(), 'Model/%s/%s/%dcom_sel.pt' %(data_obj.name, suffix, (i+1)))
                    torch.save(all_model.state_dict(), 'Model/%s/%s/%dcom_all.pt' %(data_obj.name, suffix, (i+1)))
                    np.save('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, (i+1)), clnt_params_list)
                    np.save('Model/%s/%s/%d_c_state_list.npy' %(data_obj.name, suffix, (i+1)), c_state_list)
                
                np.save('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, (i+1)), trn_perf_sel[:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, (i+1)), tst_perf_sel[:i+1])
                
                np.save('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, (i+1)), trn_perf_all[:i+1])
                np.save('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, (i+1)), tst_perf_all[:i+1])
                
                
                if (i+1) > save_period:
                    if os.path.exists('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period)):
                        # Delete the previous saved arrays
                        os.remove('Model/%s/%s/%dcom_trn_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                        os.remove('Model/%s/%s/%dcom_tst_perf_sel.npy' %(data_obj.name, suffix, i+1-save_period))
                        
                        os.remove('Model/%s/%s/%dcom_trn_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                        os.remove('Model/%s/%s/%dcom_tst_perf_all.npy' %(data_obj.name, suffix, i+1-save_period))
                        if save_models:
                            os.remove('Model/%s/%s/%d_clnt_params_list.npy' %(data_obj.name, suffix, i+1-save_period))
                            os.remove('Model/%s/%s/%d_c_state_list.npy' %(data_obj.name, suffix, i+1-save_period))
                                    
            if ((i+1) % save_period == 0):
                fed_mdls_sel[i//save_period] = avg_model
                fed_mdls_all[i//save_period] = all_model
                
            # See if all clients in consecutive int(1/act_prob) rounds is diverging. If so stop execution.
            failure_arr = tst_perf_all[:, -1]
            total_fails = failure_arr[np.max([0,i-int(1/act_prob)]):i].sum()
            print('Total failures in this round: %d' %tst_perf_all[i, -1])
            if total_fails == int(act_prob*n_clnt)*int(1/act_prob):
                break
                
    return fed_mdls_sel, trn_perf_sel, tst_perf_sel, fed_mdls_all, trn_perf_all, tst_perf_all