"""
Custom optimizers.
- do not update a weight if the gradient is zero.
- use parameter-wise learning rate specified by param.lr.
"""
import chainer


class AdamRule(chainer.optimizers.adam.AdamRule):
    def update_core_cpu(self, param):
        raise NotImplementedError

    def update_core_gpu(self, param):
        grad = param.grad
        if grad is None:
            return
        lr = self.lr * param.lr if hasattr(param, 'lr') else self.lr
        if lr != 0:
            chainer.cuda.elementwise(
                'T grad, T lr, T one_minus_beta1, T one_minus_beta2, T eps',
                'T param, T m, T v',
                '''
                    if (grad != 0.0) {
                        m += one_minus_beta1 * (grad - m);
                        v += one_minus_beta2 * (grad * grad - v);
                        if (v < 0) v = 0;
                        param -= lr * m / (sqrt(v) + eps);
                    }
                ''',
                'adam',
            )(
                grad, lr, 1 - self.hyperparam.beta1, 1 - self.hyperparam.beta2, self.hyperparam.eps, param.data,
                self.state['m'], self.state['v'],
            )


class Adam(chainer.optimizers.adam.Adam):
    def create_update_rule(self):
        return AdamRule(self.hyperparam)
