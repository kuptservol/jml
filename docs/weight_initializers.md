##### Weight Initializers
```java
public interface WeightInitializer extends Serializable {
    double[][] init(int x, int y);
}
```
###### Gaussian
```java
    private final double limit;
    private final Random random = new Random();
    
    @Override
    public double[][] init(int x, int y) {
        double[][] vals = new double[x][y];
        return M.FR(v -> limit * random.nextGaussian(), vals);
    }
```
###### Sharp Gaussian
The idea is to have more sharp probability density function of activations on hidden layers - 
so standart deviation is not >> 1 or << 0 - so sigmoid activation do not saturate at start in hidden layers


