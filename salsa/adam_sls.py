import copy
import time
import torch
import numpy as np

from .sls_base import StochLineSearchBase, get_grad_list, compute_grad_norm, random_seed_torch, try_sgd_update

#gets a nested list of parameters as input
class AdamSLS(StochLineSearchBase):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=0.1,
                 c=0.2,
                 gamma=2.0,
                 beta=0.999,
                 momentum=0.9,
                 gv_option='per_param',
                 base_opt='adam',
                 pp_norm_method='pp_armijo',
                 strategy = "cycle",
                 mom_type='standard',
                 clip_grad=False,
                 beta_b=0.9,
                 beta_s = 0.999,
                 reset_option=1,
                 timescale = 0.05,
                 line_search_fn="armijo",
                 combine_threshold = 0,
                 smooth = True,
                 smooth_after = 0,
                 only_decrease = False,
                 speed_up = False):
        params = list(params)
        super().__init__(params,
                         n_batches_per_epoch=n_batches_per_epoch,
                         init_step_size=init_step_size,
                         c=c,
                         beta_b=beta_b,
                         gamma=gamma,
                         reset_option=reset_option,
                         line_search_fn=line_search_fn)
        self.mom_type = mom_type
        self.pp_norm_method = pp_norm_method
        self.speed_up = speed_up

        self.init_step_sizes = [init_step_size for i in range(len(params))]
        # sps stuff
        # self.adapt_flag = adapt_flag

        self.smooth = smooth
        # sls stuff
        self.beta_b = beta_b
        self.beta_s = beta_s
        self.reset_option = reset_option
        self.combine_threshold = combine_threshold
        self.smooth_after = smooth_after
        self.only_decrease = only_decrease
        if not self.smooth_after == 0:
            self.smooth = False

        # others
        self.strategy = strategy
        self.nextcycle = 0
        self.params = params
        paramslist = []
        for param in self.params:
            paramslist = paramslist + param
        if self.mom_type == 'heavy_ball':
            self.params_prev = copy.deepcopy(params) 


        self.momentum = momentum
        self.beta = beta
        self.at_step = 0
        self.first_step = True
        self.timescale = timescale
        self.tslls = 100
        self.avg_step_size_slow = 1e-8
        self.avg_step_size_fast = 1e-8

        self.avg_decrease = torch.zeros(len(params))
        self.avg_gradient_norm = torch.zeros(len(params))

        self.clip_grad = clip_grad
        self.gv_option = gv_option
        self.base_opt = base_opt

        # gv options
        self.gv_option = gv_option
        if self.gv_option in ['scalar']:
            self.state['gv'] = [[0.]]

        elif self.gv_option == 'per_param':
            self.state['gv'] = [[torch.zeros(p.shape).to(p.device) for p in params] for params in self.params]
            self.state['mv'] = [[torch.zeros(p.shape).to(p.device) for p in params] for params in self.params]
            
        

    def step(self, closure, closure_with_backward = None):
        # deterministic closure
        seed = time.time()
        start = time.time()
        def closure_deterministic():
            with random_seed_torch(int(seed)):
                return closure()

        if closure_with_backward is not None:
            loss = closure_with_backward()
        else:
            loss = closure_deterministic()
            loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.params, 0.25)
        # increment # forward-backward calls
        self.state['n_forwards'] += 1

        # save the current parameters
        params_current = [copy.deepcopy(param) for param in self.params]
        grad_current = [get_grad_list(param) for param in self.params]

        rate_of_change = self.avg_step_size_fast /self.avg_step_size_slow
        if rate_of_change < 1:
            rate_of_change = 1/rate_of_change
        neededchecks = min(10,1/((rate_of_change-1)+ 1e-8))
        self.state["LS_freq"] = neededchecks
        self.tslls += 1
        #print("neededchecks:",neededchecks)
        if self.gv_option == 'per_param':
            # update gv
            for a, grad in enumerate(grad_current):
                for i, g in enumerate(grad):
                    if isinstance(g, torch.Tensor) and isinstance(self.state['gv'][a][i], torch.Tensor):
                        if g.device != self.state['gv'][a][i].device:
                            self.state['gv'][a][i] = self.state['gv'][a][i].to(g.device)
                    if isinstance(g, torch.Tensor) and isinstance(self.state['mv'][a][i], torch.Tensor):
                        if g.device != self.state['mv'][a][i].device:
                            self.state['mv'][a][i] = self.state['mv'][a][i].to(g.device)
                    if self.base_opt == 'adam':
                        self.state['gv'][a][i] = (1-self.beta)*(g**2) + (self.beta) * self.state['gv'][a][i]
                        self.state['mv'][a][i] = (1-self.momentum)*g + (self.momentum) * self.state['mv'][a][i]
                    if self.base_opt == 'lion':
                    # self.state['gv'][a][i] = (1-self.beta)*(g**2) + (self.beta) * self.state['gv'][a][i]
                        self.state['mv'][a][i] = (1-self.beta)*g + (self.beta) * self.state['mv'][a][i]
        step_sizes = self.state.get('step_sizes') or self.init_step_sizes
        step_sizes = [self.reset_step(step_size=step_size,
                                    n_batches_per_epoch=self.n_batches_per_epoch,
                                    gamma=self.gamma,
                                    reset_option=self.reset_option,
                                    init_step_size=self.init_step_size) for step_size in step_sizes]
        if self.state['step'] < 10 or self.tslls >= neededchecks or not self.speed_up:
            self.tslls = 0
            grad_norm = [compute_grad_norm(grad) for grad in grad_current]
            if self.base_opt == "scalar":
                pp_norm = grad_norm
            else:
                pp_norm =[self.get_pp_norm(g_cur,i) for i,g_cur in enumerate(grad_current)]


            # compute step size and execute step
            # =================
        #   print(self.at_step)
            for i in range(len(self.avg_gradient_norm)):
                if i == self.nextcycle:
                    self.avg_gradient_norm[i] = self.avg_gradient_norm[i] * self.beta_s + (pp_norm[i]) *(1-self.beta_s)

            self.state['gradient_norm'] =  [a.item() for a in pp_norm]
            self.pp_norm = pp_norm
            if self.smooth == False and not self.smooth_after == 0:
                if self.state['step'] > self.smooth_after:
                    self.smooth = True

            if self.first_step:
                step_size, loss_next = self.line_search(-1,step_sizes[0], params_current, grad_current, loss, closure_deterministic,  precond=True)
                step_sizes = [step_size for i in range(len(step_sizes))]
                self.try_sgd_precond_update(-1,self.params, step_size, params_current, grad_current, self.momentum)
                self.at_step = self.at_step +1
                if self.at_step > 5:
                    self.first_step = False
                    for i in range(len(self.avg_gradient_norm)):
                        self.avg_gradient_norm[i] = self.avg_gradient_norm[0]
                        self.avg_decrease[i] = self.avg_decrease[-1]   
            else:
                for i,step_size in enumerate(step_sizes):
                    if i == self.nextcycle:
                        step_size, loss_next = self.line_search(i,step_size, params_current[i], grad_current[i], loss, closure_deterministic, precond=True)
                        self.try_sgd_precond_update(i,self.params[i], step_size, params_current[i], grad_current[i], self.momentum)
                        step_sizes[i] = step_size
                    else:
                        self.try_sgd_precond_update(i,self.params[i], step_size, params_current[i], grad_current[i], self.momentum)
                self.nextcycle += 1
                if self.nextcycle >= len(self.params):
                    self.nextcycle = 0
            self.avg_step_size_slow = self.avg_step_size_slow * 0.99 + (step_sizes[0]) *(1-0.99)
            self.avg_step_size_fast = self.avg_step_size_fast * 0.9 + (step_sizes[0]) *(1-0.9)
        else:
            for i,step_size in enumerate(step_sizes):
                self.try_sgd_precond_update(i,self.params[i], step_size, params_current[i], grad_current[i], self.momentum)   
            loss_next = 0.0
            grad_norm = 0.0        

                
        self.save_state(step_sizes, loss, loss_next, grad_norm)

        if torch.isnan(self.params[0][0]).sum() > 0:
            raise ValueError('nans detected')

        return loss

    def get_pp_norm(self, grad_current, i):
        if self.pp_norm_method in ['pp_armijo', "just_pp"]:
            pp_norm = 0
            for g_i, gv_i in zip(grad_current, self.state['gv'][i]):
                if self.base_opt == 'adam':
                    gv_i_scaled = scale_vector(gv_i, self.beta, self.state['step']+1)
                    pv_i = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)
                    if self.pp_norm_method == 'pp_armijo':
                        layer_norm = ((g_i**2) * pv_i).sum()
                    elif self.pp_norm_method == "just_pp":
                        layer_norm = pv_i.sum()
                else:
                    raise ValueError('%s not found' % self.base_opt)
                pp_norm += layer_norm

        else:
            raise ValueError('%s does not exist' % self.pp_norm_method)

        return pp_norm

    @torch.no_grad()
    def try_sgd_precond_update(self, i,params, step_size, params_current, grad_current, momentum):
        if self.gv_option in ['scalar']:
            if i == -1:
                zipped = zip([item for sublist in params for item in sublist], [item for sublist in params_current for item in sublist], [item for sublist in grad_current for item in sublist] )
            else:
                zipped = zip(params, params_current, grad_current)

            for p_next, p_current, g_current in zipped:
                p_next.data[:] = p_current.data
                p_next.data.add_(g_current, alpha=- step_size)
        
        elif self.gv_option == 'per_param':
            if self.base_opt == 'adam':
                if i == -1:
                    zipped = zip([item for sublist in params for item in sublist], [item for sublist in params_current for item in sublist], [item for sublist in grad_current for item in sublist], 
                        [item for sublist in self.state['gv'] for item in sublist],[item for sublist in self.state['mv'] for item in sublist] )
                else:
                    zipped = zip(params, params_current, grad_current, self.state['gv'][i], self.state['mv'][i])
                for p_next, p_current, g_current, gv_i, mv_i in zipped:
                    gv_i_scaled = scale_vector(gv_i, self.beta, self.state['step']+1)
                    pv_list = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                    if momentum == 0. or  self.mom_type == 'heavy_ball':
                        mv_i_scaled = g_current
                    elif self.mom_type == 'standard':
                        mv_i_scaled = scale_vector(mv_i, momentum, self.state['step']+1)
                    
                    p_next.data[:] = p_current.data
                    p_next.data.add_((pv_list *  mv_i_scaled), alpha=- step_size)
            
            

            else:
                raise ValueError('%s does not exist' % self.base_opt)

        else:
            raise ValueError('%s does not exist' % self.gv_option)

def scale_vector(vector, alpha, step, eps=1e-8):
    scale = (1-alpha**(max(1, step)))
    return vector / scale

