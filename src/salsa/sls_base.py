import torch
import contextlib
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
@contextlib.contextmanager
def random_seed_torch(seed, device=0):
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

#seems pretty slow like 0.1secs per iteration
def compute_grad_norm(grad_list):
    grad_norm = torch.zeros(1, device=device)# 0.
    for g in grad_list:
        if g is None:
            continue
        grad_norm += torch.sum(torch.mul(g, g))
   # print(grad_norm)
    grad_norm = torch.sqrt(grad_norm)
    return grad_norm

def get_grad_list(params):
    return [p.grad for p in params]


def try_sgd_update(params, step_size, params_current, grad_current):
    zipped = zip(params, params_current, grad_current)

    for p_next, p_current, g_current in zipped:
       # p_next.data[:] = p_current.data
        p_next.data = p_current - step_size * g_current

class StochLineSearchBase(torch.optim.Optimizer):
    def __init__(self,
                 params,
                 n_batches_per_epoch=500,
                 init_step_size=1,
                 c=0.1,
                 beta_b=0.9,
                 gamma=2.0,
                 reset_option=0,
                 line_search_fn="armijo"):
        params = list(params)
        paramslist = []
        for param in params:
            paramslist = paramslist + param
        super().__init__(paramslist, {})

        self.params = params
        self.c = c
        self.beta_b = beta_b
        self.gamma = gamma
        self.init_step_size = init_step_size
        self.n_batches_per_epoch = n_batches_per_epoch
        self.line_search_fn = line_search_fn
        self.state['step'] = 0
        self.state['n_forwards'] = 0
        self.state['n_backtr'] = []
        self.budget = 0
        self.reset_option = reset_option
        self.new_epoch()


    def step(self, closure):
        # deterministic closure
        raise RuntimeError("This function should not be called")

    def line_search(self,i, step_size, params_current, grad_current, loss, closure_deterministic, precond=False):
        with torch.no_grad():

            if self.first_step:
                suff_dec = torch.sum(self.avg_gradient_norm)
            else:
                suff_dec = self.avg_gradient_norm[i]
            if loss.item() != 0 and suff_dec >= 1e-8:
                # check if condition is satisfied
                found = 0

                
                for e in range(200):
                    # try a prospective step
                    if self.first_step:
                        if precond:
                            self.try_sgd_precond_update(i,self.params, step_size, params_current, grad_current,  momentum=0.)
                        else:
                            try_sgd_update(self.params, step_size, params_current, grad_current)
                    else:
                        if precond:
                            self.try_sgd_precond_update(i,self.params[i], step_size, params_current, grad_current, momentum=0.)
                        else:
                            try_sgd_update(self.params[i], step_size, params_current, grad_current)

                    # compute the loss at the next step; no need to compute gradients.
                    loss_next = closure_deterministic()
                    self.state['loss_decrease'] = loss-loss_next
                    decrease= (self.avg_decrease[i] * self.beta_s + (loss-loss_next) *(1-self.beta_s) )
                    self.state['n_forwards'] += 1

                    if loss - loss_next > 0.0 or not self.only_decrease:
                        if loss - loss_next == 0.0:
                            found = 1
                            print("had cancelation error loss was equal no decrease necessary")
                            break

                        if not self.smooth:
                            decrease = loss-loss_next
                            self.avg_decrease[i] = decrease
                            suff_dec = self.pp_norm[i]
                            self.avg_gradient_norm[i] = suff_dec

                        found, step_size = self.check_armijo_conditions(step_size=step_size,
                                                                        decrease=decrease,
                                                                        suff_dec=suff_dec,
                                                                        c=self.c,
                                                                        beta_b=self.beta_b)
                        

                        if found == 1:
                            self.avg_decrease[i]  = self.avg_decrease[i] * self.beta_s + (loss-loss_next) *(1-self.beta_s) 
                            break
                    else: 
                        step_size = step_size * self.beta_b

                self.state['backtracks'] += e
                self.state['f_eval'].append(e)
                self.state['n_backtr'].append(e)

            else:
                print("loss is", loss.item(), "suff_dec", suff_dec)
                if loss.item() == 0:
                    self.state['numerical_error'] += 1
                loss_next = closure_deterministic()

        return step_size, loss_next

    def check_armijo_conditions(self, step_size, decrease, suff_dec, c, beta_b):
        found = 0

        sufficient_decrease = step_size * c * suff_dec
        if (decrease >= sufficient_decrease):
            found = 1
        else:
            step_size = step_size * beta_b

        return found, step_size

    def reset_step(self, step_size, n_batches_per_epoch=None, gamma=None, reset_option=1, init_step_size=None):

        if reset_option == 0:
            pass
        elif reset_option == 1:
            step_size = step_size * gamma ** (1. / n_batches_per_epoch)
        elif reset_option == 11:
            step_size = min(step_size * gamma ** (1. / n_batches_per_epoch), 10)
        elif reset_option == 2:
            step_size = init_step_size
        elif reset_option == 3:
            if step_size < self.step_threshold:
                step_size = step_size * gamma 
            else:
                step_size = step_size * gamma ** (1. / n_batches_per_epoch)
        else:
            raise ValueError("reset_option {} does not existing".format(reset_option))

        return step_size

    def save_state(self, step_sizes, loss, loss_next, grad_norm):
       # if isinstance(step_sizes[0], torch.Tensor):
        step_sizes = [step_size.item() if isinstance(step_size, torch.Tensor) else step_size for step_size in step_sizes]
            #step_size = step_size.item()
     #   self.state['step_size'] = step_size
        self.state['step_sizes'] = step_sizes
        self.state['step'] += 1
        self.state['all_step_size'].append(step_sizes)
        self.state['all_losses'].append(loss.item())
        if isinstance(loss_next, torch.Tensor):
            loss_next = loss_next.item()
        self.state['all_new_losses'].append(loss_next)
        self.state['grad_norm_avg']  = [a.item() for a in self.avg_gradient_norm]
        self.state['loss_dec_avg'] = [a.item()  if isinstance(a, torch.Tensor) else a for a in self.avg_decrease]
        self.state['n_batches'] += 1
     #   self.state['avg_step'] += step_sizes
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()
        self.state['grad_norm'].append(grad_norm)

    def new_epoch(self):
        self.state['avg_step'] = 0
        self.state['semi_last_step_size'] = 0
        self.state['all_step_size'] = []
        self.state['all_losses'] = []
        self.state['grad_norm'] = []
        self.state['grad_norm_avg'] = []
        self.state['loss_dec_avg'] = []
        self.state['all_new_losses'] = []
        self.state['f_eval'] = []
        self.state['backtracks'] = 0
        self.state['n_batches'] = 0
        self.state['zero_steps'] = 0
        self.state['numerical_error'] = 0

    def gather_flat_grad(self, params):
        views = []
        for p in params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def flatten_vect(self, vect):
        views = []
        for p in vect:
            if p is None:
                view = p.new(p.numel()).zero_()
            elif p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)
