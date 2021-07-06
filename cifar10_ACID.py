from utils_methods import *

n_client       = 100
cls_per_client = 5
rule           = 'ACID'
data_obj   = DatasetObject(dataset='CIFAR10', n_client=n_client, rule=rule, rule_arg=cls_per_client)
model_name = 'cifar10' # Model type

###
com_amount    = 1000
act_prob      = .1
save_period   = 100
batch_size    = 50
save_models   = True  # Save models if True
fast_exec     = False # Record only test performance if True
weight_decay  = 1e-3
lr_decay      = .997
learning_rate = 1e-1
K             = 50
print_per     = K//4

# Model function, change output layer to be cls_per_client
model_func = lambda : client_model(model_name, args = cls_per_client if rule=='ALID' else '')
init_model = model_func()

# Initalise the model for all methods with a random seed or load it from a saved initial model
init_model = model_func()
if not os.path.exists('Model/%s/%s_init_mdl.pt' %(data_obj.name, model_name)):
    if not os.path.exists('Model/%s/' %(data_obj.name)):
        print("Create a new directory")
        os.mkdir('Model/%s/' %(data_obj.name))
    torch.save(init_model.state_dict(), 'Model/%s/%s_init_mdl.pt' %(data_obj.name, model_name))
else:
    # Load model
    init_model.load_state_dict(torch.load('Model/%s/%s_init_mdl.pt' %(data_obj.name, model_name)))    

print('FedAvg')
meta_lr_rate_list   = [1e-1]
num_grad_step_list  = [1]
do_plain            = True
do_proto            = True              
[_, _,
 _, _,
 _, plainFedAvg] = train_FedAvg(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size, 
                      K=K, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                      model_func=model_func, init_model=init_model, save_period=save_period,
                      meta_learning_rate_list=meta_lr_rate_list, num_grad_step_list=num_grad_step_list,
                      do_proto=do_proto, do_plain=do_plain, lr_decay=lr_decay,
                      save_models=save_models, fast_exec=fast_exec)

####
print('PerAvg_MAML')
meta_lr_rate   = 1e-2
num_grad_step  = 5

[_, _,
 _, _,
 _, perAvg_MAML] = train_Meta_FedAvg_MAML(data_obj=data_obj, act_prob=act_prob , learning_rate=learning_rate, batch_size=batch_size, 
                                meta_learning_rate=meta_lr_rate, K=K, com_amount=com_amount, print_per=print_per,
                                weight_decay=weight_decay, model_func=model_func, init_model=init_model, 
                                save_period=save_period, num_grad_step=num_grad_step,
                                lr_decay=lr_decay, save_models=save_models, fast_exec=fast_exec)
###
print('PerAvg_Proto')
[_, _,
 _, _,
 _, perAvg_Proto] = train_Meta_FedAvg_Proto(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                                 batch_size=batch_size, K=K, com_amount=com_amount, print_per=print_per,
                                 weight_decay=weight_decay, model_func=model_func, init_model=init_model,
                                 save_period=save_period, lr_decay=lr_decay,
                                 save_models=save_models, fast_exec=fast_exec)
###

print('FedDyn')
alpha = 1e-2
meta_lr_rate_list   = [1e-1]
num_grad_step_list  = [1]
do_plain            = True
do_proto            = True

[_, _,
 _, _,
 _, plainFedDyn] = train_FedDyn(data_obj=data_obj, act_prob=act_prob, alpha=alpha, learning_rate=learning_rate, batch_size=batch_size, 
                      K=K, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay, model_func=model_func,
                      init_model=init_model, save_period=save_period, meta_learning_rate_list=meta_lr_rate_list,
                      num_grad_step_list=num_grad_step_list, do_proto=do_proto, do_plain=do_plain, 
                      lr_decay=lr_decay, save_models=save_models, fast_exec=fast_exec)
   
###
print('PFLDyn_MAML')
alpha          = 1e-2
meta_lr_rate   = 1e-2
num_grad_step  = 5

[_, _,
 _, _,
 _, PFLDyn_MAML] = train_Meta_FedDyn_MAML(data_obj=data_obj, act_prob=act_prob, alpha=alpha, learning_rate=learning_rate, 
                                batch_size=batch_size, meta_learning_rate=meta_lr_rate, K=K, com_amount=com_amount,
                                print_per=print_per, weight_decay=weight_decay, model_func=model_func, init_model=init_model,
                                save_period=save_period, num_grad_step=num_grad_step, lr_decay=lr_decay,
                                save_models=save_models, fast_exec=fast_exec)
###
print('PFLDyn_Proto')
alpha          = 1e-1
[_, _,
 _, _,
 _, PFLDyn_Proto] = train_Meta_FedDyn_Proto(data_obj=data_obj, act_prob=act_prob, alpha=alpha, learning_rate=learning_rate,
                                 batch_size=batch_size, K=K, com_amount=com_amount, print_per=print_per,
                                 weight_decay=weight_decay, model_func=model_func, init_model=init_model, 
                                 save_period=save_period, lr_decay=lr_decay, save_models=save_models,
                                 fast_exec=fast_exec)

print('SCAFFOLD')
meta_lr_rate_list   = [1e-1]
num_grad_step_list  = [1]
do_plain            = True
do_proto            = True              
[_, _,
 _, _,
 _, plainScaffold] = train_SCAFFOLD(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate, batch_size=batch_size, 
                      K=K, com_amount=com_amount, print_per=print_per, weight_decay=weight_decay,
                      model_func=model_func, init_model=init_model, save_period=save_period,
                      meta_learning_rate_list=meta_lr_rate_list, num_grad_step_list=num_grad_step_list,
                      do_proto=do_proto, do_plain=do_plain, lr_decay=lr_decay,
                      save_models=save_models, fast_exec=fast_exec)

####
print('PFLSCAF_MAML')
meta_lr_rate   = 1e-1
num_grad_step  = 1
[_, _,
 _, _,
 _, PFLScaf_MAML] = train_Meta_SCAFFOLD_MAML(data_obj=data_obj, act_prob=act_prob , learning_rate=learning_rate, batch_size=batch_size, 
                                meta_learning_rate=meta_lr_rate, K=K, com_amount=com_amount, print_per=print_per,
                                weight_decay=weight_decay, model_func=model_func, init_model=init_model, 
                                save_period=save_period, num_grad_step=num_grad_step,
                                lr_decay=lr_decay, save_models=save_models, fast_exec=fast_exec)
###
print('PFLSCAF_Proto')
[_, _,
 _, _,
 _, PFLScaf_Proto] = train_Meta_SCAFFOLD_Proto(data_obj=data_obj, act_prob=act_prob, learning_rate=learning_rate,
                                 batch_size=batch_size, K=K, com_amount=com_amount, print_per=print_per,
                                 weight_decay=weight_decay, model_func=model_func, init_model=init_model,
                                 save_period=save_period, lr_decay=lr_decay,
                                 save_models=save_models, fast_exec=fast_exec)

####
idx_ = 1
plt.figure(figsize=(8, 7))
plt.plot(np.arange(com_amount)+1, smooth_filter(PFLDyn_Proto[:com_amount,idx_]), label='PFLDyn (Proto)')
plt.plot(np.arange(com_amount)+1, smooth_filter(PFLDyn_MAML[:com_amount,idx_]), label='PFLDyn (MAML)')

plt.plot(np.arange(com_amount)+1, smooth_filter(perAvg_Proto[:com_amount,idx_]), label='P-Avg (Proto)')
plt.plot(np.arange(com_amount)+1, smooth_filter(perAvg_MAML[:com_amount,idx_]),  label='Fallah et al., 2020')

plt.plot(np.arange(com_amount//2)*2+2, smooth_filter(PFLScaf_Proto[:com_amount//2,idx_]), label='PFLScaf (Proto)')
plt.plot(np.arange(com_amount//2)*2+2, smooth_filter(PFLScaf_MAML[:com_amount//2,idx_]), label='PFLScaf (MAML)')

plt.plot(np.arange(com_amount)+1, smooth_filter(plainFedDyn[-2][:com_amount,idx_]), 'C1-.', label='No Per., PFLDyn - Proto A.')
plt.plot(np.arange(com_amount)+1, smooth_filter(plainFedAvg[-2][:com_amount,idx_]), 'C3-.', label='No Per., FedAvg - Proto A.')
plt.plot(np.arange(com_amount//2)*2+2, smooth_filter(plainScaffold[-2][:com_amount//2,idx_]), 'C4-.', label='No Per., PFLScaf - Proto A.')

plt.ylabel('Average Test Accuracy', fontsize=16)
plt.xlabel('# Models Transmitted', fontsize=16)
plt.legend(fontsize=13, loc='lower right')
plt.grid()
plt.title('CIFAR-10 ACID 5 classes per device', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('cifar10_5cls_ACID.pdf', dpi=1000, bbox_inches='tight')