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


