import torch
import numpy as np
import copy
import time
import contextlib
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SaLSA(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 init_step_size=1,
                 c=0.5,
                 momentum=(0.9,0.999,0.9),
                 speed_up = False,
                 use_mv = True,):
        
        params = list(params)
        super().__init__(params, {})

        self.use_mv = use_mv
        self.params = params
        self.c = c
        self.momentum = momentum[0]
        self.beta = momentum[1]
        self.beta_3 = momentum[2]
        self.decrease_factor = 0.8
        self.growth_factor = 1.1
        self.init_step_size = init_step_size
        self.step_size = init_step_size
        self.state['step'] = 0
        self.state['n_forwards'] = 0
        self.state['n_backwards'] = 0
        self.state['n_backtr'] = 0
        self.speed_up = speed_up
        self.base_opt = 'adam'
       # self.params_prev = copy.deepcopy(params) 


        self.tslls = 100
        self.avg_step_size_slow = 1e-8
        self.avg_step_size_fast = 1e-8
        # self.state['step_size'] = init_step_size

        self.avg_decrease = torch.zeros(1, device = device)#(0.0 for i in range(len(params))]
        self.avg_gradient_norm = torch.zeros(1, device = device)#[0.0 for i in range(len(params))]

        # gv options
        
        if self.base_opt == 'adam':
            self.state['gv'] = [torch.zeros(p.shape).to(p.device)  for p in self.params]
            self.state['mv'] = [torch.zeros(p.shape).to(p.device)  for p in self.params]


    def step(self, closure):
        # deterministic closure
        seed = time.time()
        def closure_deterministic(backwards = False):
            with random_seed_torch(int(seed)):
                return closure(backwards = backwards)

        loss = closure_deterministic(backwards = True)
     #   loss.backward()

        # if self.clip_grad:
        #     torch.nn.utils.clip_grad_norm_(self.params, 0.25)

        # increment # forward-backward calls
        self.state['n_forwards'] += 1
        self.state['n_backwards'] += 1    
        self.state['step'] += 1    
        
        grad_current = [p.grad for p in self.params]

       # if self.speed_up:
        rate_of_change = self.avg_step_size_fast /self.avg_step_size_slow
        if rate_of_change < 1:
            rate_of_change = 1/rate_of_change
        neededchecks = min(10,1/((rate_of_change-1)+ 1e-8))
        self.state["LS_freq"] = neededchecks
        self.tslls += 1
            
        if self.base_opt == 'adam':
            # update gv
            for a, g in enumerate(grad_current):
                if isinstance(g, torch.Tensor) and isinstance(self.state['gv'][a], torch.Tensor):
                    if g.device != self.state['gv'][a].device:
                        self.state['gv'][a] = self.state['gv'][a].to(g.device)
                if isinstance(g, torch.Tensor) and isinstance(self.state['mv'][a], torch.Tensor):
                    if g.device != self.state['mv'][a].device:
                        self.state['mv'][a] = self.state['mv'][a].to(g.device)
                self.state['gv'][a] = (1-self.beta)*(g**2) + (self.beta) * self.state['gv'][a]
                self.state['mv'][a] = (1-self.momentum)*g + (self.momentum) * self.state['mv'][a]

        step_size = self.step_size * self.growth_factor


        if not self.speed_up or self.state['step'] < 10 or self.tslls >= neededchecks :
            params_current = copy.deepcopy(self.params) 
            self.tslls = 0
            grad_norm = compute_grad_norm(grad_current) 
            if self.base_opt == "scalar":
                pp_norm = grad_norm
            if self.base_opt == "adam":
                pp_norm = self.get_pp_norm(grad_current)

            self.avg_gradient_norm = self.avg_gradient_norm * self.beta_3 + pp_norm *(1-self.beta_3)
            self.state['gradient_norm'] =  pp_norm


            step_size, loss_next = self.line_search(step_size, params_current, grad_current, loss, closure_deterministic)
            self.step_size = step_size

            self.avg_step_size_slow = self.avg_step_size_slow * 0.99 + step_size *(1-0.99)
            self.avg_step_size_fast = self.avg_step_size_fast * 0.9 + step_size *(1-0.9)
        else:
            self.try_sgd_precond_update(self.params, step_size, params_current, grad_current, self.momentum)      

        if torch.isnan(self.params[0][0]).sum() > 0:
            raise ValueError('nans detected')
        self.state["lr"] = self.step_size
        return loss
    
    def get_pp_norm(self, grad_current):
        pp_norm = 0
        for g_i, gv_i, mv_i in zip(grad_current, self.state['gv'],  self.state['mv']):
            gv_i_scaled = scale_vector(gv_i, self.beta, self.state['step']+1)
            pv_i = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)
            if self.use_mv:
                d_i = scale_vector(mv_i, self.momentum, self.state['step']+1)
                g_i = g_i.flatten()
                d_i = (d_i*pv_i).flatten()
                layer_norm = (torch.inner(g_i,d_i)).sum()
            else:
                layer_norm = ((g_i**2) * pv_i).sum()
            pp_norm += layer_norm
        return pp_norm

    @torch.no_grad()
    def line_search(self, step_size, params_current, grad_current, loss, closure_deterministic):

        suff_dec = self.avg_gradient_norm

        if loss.item() != 0 and suff_dec >= 1e-8:
            # check if condition is satisfied
            found = 0

            
            for e in range(100):
                # try a prospective step
                if self.use_mv:
                    momentum = self.momentum
                else:
                    momentum = 0.
                self.try_sgd_precond_update(self.params, step_size, params_current, grad_current,  momentum=momentum)

                # compute the loss at the next step; no need to compute gradients.
                loss_next = closure_deterministic()
                self.state['n_forwards'] += 1
                self.state['loss_decrease'] = loss-loss_next
                decrease= (self.avg_decrease * self.beta_3 + (loss-loss_next) *(1-self.beta_3) )#/((1-self.beta)**((self.state['step']+1)/len(self.avg_decrease)))
                

                if loss - loss_next ==  0:
                    found = 1
                    print("had cancelation error loss was equal")
                    break

                found, step_size = self.check_armijo_conditions(step_size=step_size,
                                                                decrease=decrease,
                                                                suff_dec=suff_dec,
                                                                c=self.c,
                                                                beta_b=self.decrease_factor)
                if found == 1:
                    self.avg_decrease  = self.avg_decrease * self.beta_3 + (loss-loss_next) *(1-self.beta_3) 
                    break



            self.state['n_backtr'] += e

        else:
            print("loss is", loss.item(), "suff_dec", suff_dec)
            if loss.item() == 0:
                self.state['numerical_error'] += 1
            loss_next = closure_deterministic()

        return step_size, loss_next
    
    @torch.no_grad()
    def try_sgd_precond_update(self,params, step_size, params_current, grad_current, momentum):
   #     
        if self.base_opt  == "sgd":
            zipped = zip(params, params_current, grad_current)
            for p_next, p_current, g_current in zipped:
                p_next.data[:] = p_current.data
                p_next.data.add_(g_current, alpha=- step_size)
        
        elif self.base_opt == 'adam':
                zipped = zip(params, params_current, grad_current, self.state['gv'], self.state['mv'])
                for p_next, p_current, g_current, gv_i, mv_i in zipped:
                    gv_i_scaled = scale_vector(gv_i, self.beta, self.state['step']+1)
                    pv_list = 1. / (torch.sqrt(gv_i_scaled) + 1e-8)

                    if momentum == 0.:
                        mv_i_scaled = g_current
                    else:
                        mv_i_scaled = scale_vector(mv_i, momentum, self.state['step']+1)
                    
                    p_next.data[:] = p_current.data
                    p_next.data.add_((pv_list *  mv_i_scaled), alpha=- step_size)
         
            
        

    def check_armijo_conditions(self, step_size, decrease, suff_dec, c, beta_b):
        found = 0
        sufficient_decrease = step_size * c * suff_dec
        if (decrease >= sufficient_decrease):
            found = 1
        else:
            step_size = step_size * beta_b
        return found, step_size


def scale_vector(vector, alpha, step):
    scale = (1-alpha**(max(1, step)))
    return vector / scale


def compute_grad_norm(grad_list):
    grad_norm = torch.zeros(1, device=device)# 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm


@contextlib.contextmanager
def random_seed_torch( seed, device=0):
    cpu_rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        gpu_rng_state = torch.cuda.get_rng_state(0)

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(gpu_rng_state, device)        



