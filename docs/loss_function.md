##### Loss functions
Loss functions are functions to be minimized during learning

```java
public interface CostFunction extends Serializable {
    MetricsResult cost(Model m, double[][] trainX, double[][] trainY);
    double[] backprop(double[] activations, double[] y);
}
```
###### Mean Square Error
Cost: <a href="https://www.codecogs.com/eqnedit.php?latex=C(w,b)=\frac{1}n{}&space;\sum(y(x)&space;-&space;a)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C(w,b)=\frac{1}n{}&space;\sum(y(x)&space;-&space;a)^2" title="C(w,b)=\frac{1}n{} \sum(y(x) - a)^2" /></a>

Derivative for one x: <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;C}{\partial&space;a}&space;=&space;y(x)-a" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;C}{\partial&space;a}&space;=&space;y(x)-a" title="\frac{\partial C}{\partial a} = y(x)-a" /></a>
```java
public class MSE implements CostFunction {
    @Override
    public MetricsResult cost(Model m, double[][] X, double[][] Y) {
        //
        for (int i = 0; i < X.length; i++) {
            cost += Math.pow(m.resultF.process(m.forward(X[i])) - m.resultF.process(Y[i]), 2) / X.length;
        }
       //
    }

    @Override
    public double[] backprop(double[] activations, double[] y) {
        return M.minusR(activations, y);
    }
}
 ``` 
###### Cross Entropy
Cost: <a href="https://www.codecogs.com/eqnedit.php?latex=C=-\frac{1}{n}\sum_x[y&space;ln(a)&plus;(1-y)ln(1-a)]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C=-\frac{1}{n}\sum_x[y&space;ln(a)&plus;(1-y)ln(1-a)]" title="C=-\frac{1}{n}\sum_x[y ln(a)+(1-y)ln(1-a)]" /></a>

Derivative for one x: <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\mathrm{dC}&space;}{\mathrm{d}&space;a}&space;=&space;-(\frac{y}{a}&plus;\frac{1-y}{1-a})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\mathrm{dC}&space;}{\mathrm{d}&space;a}&space;=&space;-(\frac{y}{a}&plus;\frac{1-y}{1-a})" title="\frac{\mathrm{dC} }{\mathrm{d} a} = -(\frac{y}{a}+\frac{1-y}{1-a})" /></a>
```java
public class MSE implements CostFunction {
    @Override
    public MetricsResult cost(Model m, double[][] X, double[][] Y) {
        //
        for (int i = 0; i < X.length; i++) {
            double a = m.resultF.process(m.forward(X[i]));
            double y = m.resultF.process(Y[i]);

            cost += (y * ln(a) + (1 - y) * ln(1 - a)) / X.length;
        }
       //
    }

    @Override
    public double[] backprop(double[] activations, double[] y) {
        return M.minusR(activations, y);
    }
}
 ```
