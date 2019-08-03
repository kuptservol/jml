##### Weight Initializers
We want weights to be 
neither big - cause multiple operations will lead out to Nan
nor small - cause multiple operations will lead out to 0
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


