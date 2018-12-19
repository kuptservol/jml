# jml
### Yet Another Java ML Lib

Java ML library made for educational purposes mostly(no optimization, threads, gpu, etc) to study and practise ml theory things for people with strong java background - for those who find it difficult to study ml concepts(f.e. lin algebra) and python concepts(f.e. numpy) simultaneously - like me.

Made with pure java math without any libs except lombok, slf4j and junit, code organization is more common for java project with builders, interfaces, hierarchy and so on - so can be easily extended
 
Contains some standart datasets loader - like MNIST.

#### MNIST
MNIST - is a database of hand-written digits. Here i want to show that with 10 simple classes in java you can learn to determine digit on photo with 90% acccuracy within 15 minutes learning on one CPU.    
##### With default settings:
From `MNISTest.learnWithDefaultSettings`
```java
DataSet mnist = DataSets.MNIST(Paths.get("/tmp/mnist"));

Model model = Models.linear(784, 30, 10)
    .resultFunction(ResultFunctions.MAX_INDEX)
    .metrics(Metrics.ACCURACY)
    .build();

model.train(mnist);

Path path = Paths.get("/tmp/mnist_model");

model.save(path);
Model modelLoaded = Models.load(path);
        
logger.debug("X0: " + M.asPixels(M.to(mnist.train.x[1], 28, 28)));
logger.debug("Expected answer: " + modelLoaded.resultFunction.apply(mnist.train.y[1]));
logger.debug("Model answer: " + modelLoaded.evaluate(mnist.train.x[1]));
```
Output
```bash
2018-12-19 14:12:05:926 +0300 [main] INFO Epoch 0 train accuracy 26,893 % test accuracy 26,810 % MSE: 12,620
...
2018-12-19 14:41:55:975 +0300 [main] INFO Epoch 88 train accuracy 88,543 % test accuracy 88,940 % MSE: 1,954

2018-12-19 14:47:27:645 +0300 [main] DEBUG X1:                             
                                       
               *****        
              ******        
             *********      
           ***********      
           ***********      
          ************      
         *********  ***     
        ******      ***     
       *******      ***     
       ****         ***     
       ***          ***     
      ****          ***     
      ****        *****     
      ***        *****      
      ***       ****        
      ***      ****         
      *************         
      ***********           
      *********             
       *******              
                            
2018-12-19 14:47:27:646 +0300 [main] DEBUG Expected answer: 0.0
2018-12-19 14:47:27:662 +0300 [main] DEBUG Model answer: 0.0

```
