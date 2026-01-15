import MatrixOperations as mo
import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def choose(x):
    if x <0.5:
        return 0
    else:
        return 1
class LogisticRegressor:
    def __init__(self, alpha=0.01, epoch = 1000, features = 1):
        self.alpha = alpha
        self.epoch = epoch
        self.features = features
    def pred_data(self, data, WandB):
        pred = mo.Matrix_d(data.Rows,1)
        pred = mo.LinearRegressor().predict(data, pred, WandB)
        for i in range(pred.Rows()):
            pred.Set_element(i, 0, sigmoid(pred.Element(i, 0)))
        return pred
    def output(self, pred):
        f_pred = mo.Matrix_d(pred.Rows(), 1)
        for i in range (pred.Rows()):
            k = pred.element(i,0)
            if k < 0.5:
                f_pred.Set_element(i, 0, 0)
            else:
                f_pred.Set_element(i,0,1)
        return f_pred
    def individual_cost(self,f_pred, pred):
        cost = mo.Matrix_d(pred.Rows(),0)
        for i in range(pred.Rows()):
            k = f_pred.element(i,0)
            if k == 1:
                c = -1*math.log(pred.element(i,0))
                cost.set_element(i, 0 , c)
            elif k == 0:
                c = -1*math.log(1-pred.element(i,0))
                cost.set_element(i,0,c)
            else:
                "value error"
                break
        return cost
    def cost_function(self,f_pred, pred):
        k = 0
        r = pred.Rows()
        for i in range (pred.Rows()):
            k += (self.output(self, pred).Element(i,0))*pred.Element(i,0)
        t_cost = k*(1/r)
        return t_cost
    def gradient_vector(self, pred, data, expct_val):
        temp = expct_val
        temp.Scalar_multiplication(-1)
        error = mo.add(temp,pred)
        k = mo.transpose(data)
        gradient_vector = mo.mul(k, error)
        gradient_vector.Scalar_multiplication(1/self.features)
        return gradient_vector
    def gradient_descent(self, data, expct_val):
        WandB = mo.Matrix_d(self.features, 1)
        for i in range(self.epoch):
            pred = pred_data(self, data, WandB)
            gradient_vector = gradient_vector(self, pred, data, expct_val)
            WandB = mo.add(WandB, gradient_vector.Scalar_multiplication((-1)*self.alpha))
        return WandB
