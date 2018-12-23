##### Loss functions
```java
public interface CostFunction extends Serializable {
    MetricsResult execute(Model m, double[][] trainX, double[][] trainY);
    double[] backprop(double[] activations, double[] y);
}
```
###### Mean Square Error
Forward: <a href="https://www.codecogs.com/eqnedit.php?latex=C(w,b)=\frac{1}n{}&space;\sum(y(x)&space;-&space;a)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?C(w,b)=\frac{1}n{}&space;\sum(y(x)&space;-&space;a)^2" title="C(w,b)=\frac{1}n{} \sum(y(x) - a)^2" /></a>

Derivative: <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;C}{\partial&space;a}&space;=&space;y(x)-a" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;C}{\partial&space;a}&space;=&space;y(x)-a" title="\frac{\partial C}{\partial a} = y(x)-a" /></a>
```java
public class MSE implements CostFunction {
    @Override
    public MetricsResult execute(Model m, double[][] X, double[][] Y) {
        double cost = 0;
        for (int i = 0; i < X.length; i++) {
            cost += Math.pow(m.evaluate(X[i]) - m.resultFunction.apply(Y[i]), 2) / X.length;
        }
        return new SimpleMetricsResult(cost, "MSE: %.3f");
    }

    @Override
    public double[] backprop(double[] activations, double[] y) {
        return M.minusR(activations, y);
    }
}
 ``` 
