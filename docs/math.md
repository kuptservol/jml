#### Math
Let's have layer function - we transform input from prev layer to output like this:

<a href="https://www.codecogs.com/eqnedit.php?latex=\textup{a}{_{j}}^{l}&space;=&space;\sigma&space;(\sum&space;_k\textup{w}{_{jk}}^{l}&space;\textup{a}{_{k}}^{l-1}&space;&plus;&space;\textup{b}{_{j}}^{l})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textup{a}{_{j}}^{l}&space;=&space;\sigma&space;(\sum&space;_k\textup{w}{_{jk}}^{l}&space;\textup{a}{_{k}}^{l-1}&space;&plus;&space;\textup{b}{_{j}}^{l})" title="\textup{a}{_{j}}^{l} = \sigma (\sum _k\textup{w}{_{jk}}^{l} \textup{a}{_{k}}^{l-1} + \textup{b}{_{j}}^{l})" /></a>

we try to **minimize cost function**.

<a href="https://www.codecogs.com/eqnedit.php?latex=CostF&space;=&space;\sum_x&space;(a&space;-&space;y(x))&space;=&space;CostF&space;-&space;\bigtriangleup&space;CostF&space;\rightarrow&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?CostF&space;=&space;\sum_x&space;(a&space;-&space;y(x))&space;=&space;CostF&space;-&space;\bigtriangleup&space;CostF&space;\rightarrow&space;0" title="CostF = \sum_x (a - y(x)) = CostF - \bigtriangleup CostF \rightarrow 0" /></a>

we **want**
<a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangleup&space;CostF&space;<&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangleup&space;CostF&space;<&space;0" title="\bigtriangleup CostF < 0" /></a> **where**

<a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangleup&space;CostF&space;=&space;\bigtriangleup&space;x&space;\frac{\partial&space;CostF{}}{\partial&space;x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangleup&space;CostF&space;=&space;\bigtriangleup&space;x&space;\frac{\partial&space;CostF{}}{\partial&space;x}" title="\bigtriangleup CostF = \bigtriangleup x \frac{\partial CostF{}}{\partial x}" /></a>

if we **make** <a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangleup&space;x&space;=&space;-\eta&space;\frac{\partial&space;CostF{}}{\partial&space;x}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangleup&space;x&space;=&space;-\eta&space;\frac{\partial&space;CostF{}}{\partial&space;x}" title="\bigtriangleup x = -\eta \frac{\partial CostF{}}{\partial x}" /></a> **then** 

<a href="https://www.codecogs.com/eqnedit.php?latex=\bigtriangleup&space;CostF&space;=&space;-&space;\bigtriangleup&space;x&space;(\frac{\partial&space;CostF{}}{\partial&space;x})^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\bigtriangleup&space;CostF&space;=&space;-&space;\bigtriangleup&space;x&space;(\frac{\partial&space;CostF{}}{\partial&space;x})^2" title="\bigtriangleup CostF = - \bigtriangleup x (\frac{\partial CostF{}}{\partial x})^2" /></a>

**less than 0** by design.

So if we know **partial derivatives** of **w** and **b** on **each layer** - we can update them and expect **loss decreases** 

<a href="https://www.codecogs.com/eqnedit.php?latex=w_k&space;\rightarrow&space;w_k{}'&space;=&space;w_k&space;-&space;\eta&space;\bigtriangledown\frac{\partial&space;CostF}{\partial&space;w_k}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w_k&space;\rightarrow&space;w_k{}'&space;=&space;w_k&space;-&space;\eta&space;\bigtriangledown\frac{\partial&space;CostF}{\partial&space;w_k}" title="w_k \rightarrow w_k{}' = w_k - \eta \bigtriangledown\frac{\partial CostF}{\partial w_k}" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=b_l&space;\rightarrow&space;b_l{}'&space;=&space;b_l&space;-&space;\eta&space;\bigtriangledown\frac{\partial&space;CostF}{\partial&space;b_l}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b_l&space;\rightarrow&space;b_l{}'&space;=&space;b_l&space;-&space;\eta&space;\bigtriangledown\frac{\partial&space;CostF}{\partial&space;b_l}" title="b_l \rightarrow b_l{}' = b_l - \eta \bigtriangledown\frac{\partial CostF}{\partial b_l}" /></a>

To know **derivatives** we apply chain rule - cause result of cost function is

<a href="https://www.codecogs.com/eqnedit.php?latex=Cost&space;=&space;CostF(a_j(&space;{a_{j-1}}...)))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Cost&space;=&space;CostF(a_j(&space;{a_{j-1}}...)))" title="Cost = CostF(a_j( {a_{j-1}}...)))" /></a>

So **derivative** is  

<a href="https://www.codecogs.com/eqnedit.php?latex={\frac{\partial&space;Cost}{\partial&space;x}}&space;=&space;{\frac{\partial&space;Cost}{\partial&space;a_j}}&space;{\frac{\partial&space;a_j}{\partial&space;a_{j-1}}}&space;..." target="_blank"><img src="https://latex.codecogs.com/gif.latex?{\frac{\partial&space;Cost}{\partial&space;x}}&space;=&space;{\frac{\partial&space;Cost}{\partial&space;a_j}}&space;{\frac{\partial&space;a_j}{\partial&space;a_{j-1}}}&space;..." title="{\frac{\partial Cost}{\partial x}} = {\frac{\partial Cost}{\partial a_j}} {\frac{\partial a_j}{\partial a_{j-1}}} ..." /></a>

Remember that  <a href="https://www.codecogs.com/eqnedit.php?latex=\textup{a}{_{}}^{l}&space;=&space;\sigma&space;(\sum&space;\textup{w}{_{}}^{l}&space;\textup{a}{_{}}^{l-1}&space;&plus;&space;\textup{b}{_{}}^{l})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\textup{a}{_{}}^{l}&space;=&space;\sigma&space;(\sum&space;\textup{w}{_{}}^{l}&space;\textup{a}{_{}}^{l-1}&space;&plus;&space;\textup{b}{_{}}^{l})" title="\textup{a}{_{}}^{l} = \sigma (\sum \textup{w}{_{}}^{l} \textup{a}{_{}}^{l-1} + \textup{b}{_{}}^{l})" /></a>

And <a href="https://www.codecogs.com/eqnedit.php?latex=CostF&space;=&space;\frac{1}{n}\sum&space;(&space;\sigma&space;(\sum&space;_k\textup{w}{_{jk}}^{l}&space;\textup{a}{_{k}}^{l-1}&space;&plus;&space;\textup{b}{_{j}}^{l})&space;-&space;y)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?CostF&space;=&space;\frac{1}{n}\sum&space;(&space;\sigma&space;(\sum&space;_k\textup{w}{_{jk}}^{l}&space;\textup{a}{_{k}}^{l-1}&space;&plus;&space;\textup{b}{_{j}}^{l})&space;-&space;y)^2" title="CostF = \frac{1}{n}\sum ( \sigma (\sum _k\textup{w}{_{jk}}^{l} \textup{a}{_{k}}^{l-1} + \textup{b}{_{j}}^{l}) - y)^2" /></a>

If const-function = **MSE**, then for last linear layer:

<a href="https://www.codecogs.com/eqnedit.php?latex=CostF'&space;=&space;{((a_l-y)^2)}'*{(w_la_{l-1}&plus;b_l)}'*(\sigma(w_la_{l-1}&plus;b_l))'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?CostF'&space;=&space;{((a_l-y)^2)}'*{(w_la_{l-1}&plus;b_l)}'*(\sigma(w_la_{l-1}&plus;b_l))'" title="CostF' = {((a_l-y)^2)}'*{(w_la_{l-1}+b_l)}'*(\sigma(w_la_{l-1}+b_l))'" /></a>

**By w:**
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;w_l}&space;=&space;(a_l-y)*a_{l-1}*(\sigma(w_la_{l-1}&plus;b_l))'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;w_l}&space;=&space;(a_l-y)*a_{l-1}*(\sigma(w_la_{l-1}&plus;b_l))'" title="\frac{\partial CostF}{\partial w_l} = (a_l-y)*a_{l-1}*(\sigma(w_la_{l-1}+b_l))'" /></a>

**By b:**
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;b_l}&space;=&space;(a_l-y)*(\sigma(w_la_{l-1}&plus;b_l))'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;b_l}&space;=&space;(a_l-y)*(\sigma(w_la_{l-1}&plus;b_l))'" title="\frac{\partial CostF}{\partial b_l} = (a_l-y)*(\sigma(w_la_{l-1}+b_l))'" /></a>

Then going **back** from last layer to first

<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;a_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;\frac{\partial&space;a_{l&plus;1}}{\partial&space;a_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*\frac{\partial&space;(w_{l&plus;1}\sigma(w_l*a_{l-1}&space;&plus;&space;b_l))&plus;b_{l&plus;1}))}{\partial&space;a_l}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;\frac{\partial&space;(w_{l&plus;1}z_l&plus;b_{l&plus;1})}{\partial&space;a_{l}}*&space;(\sigma(w_l*a_{l-1}&space;&plus;&space;b_l))'&space;*&space;(w_l*a_{l-1}&space;&plus;&space;b_l)'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;a_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;\frac{\partial&space;a_{l&plus;1}}{\partial&space;a_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*\frac{\partial&space;(w_{l&plus;1}\sigma(w_l*a_{l-1}&space;&plus;&space;b_l))&plus;b_{l&plus;1}))}{\partial&space;a_l}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;\frac{\partial&space;(w_{l&plus;1}z_l&plus;b_{l&plus;1})}{\partial&space;a_{l}}*&space;(\sigma(w_l*a_{l-1}&space;&plus;&space;b_l))'&space;*&space;(w_l*a_{l-1}&space;&plus;&space;b_l)'" title="\frac{\partial CostF}{\partial a_{l}} = \frac{\partial CostF}{\partial a_{l+1}} * \frac{\partial a_{l+1}}{\partial a_{l}} = \frac{\partial CostF}{\partial a_{l+1}} *\frac{\partial (w_{l+1}\sigma(w_l*a_{l-1} + b_l))+b_{l+1}))}{\partial a_l} = \frac{\partial CostF}{\partial a_{l+1}} * \frac{\partial (w_{l+1}z_l+b_{l+1})}{\partial a_{l}}* (\sigma(w_l*a_{l-1} + b_l))' * (w_l*a_{l-1} + b_l)'" /></a>

By w:
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;w_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;(\sigma(z_l))'&space;*&space;a_{l-1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;w_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;(\sigma(z_l))'&space;*&space;a_{l-1}" title="\frac{\partial CostF}{\partial w_{l}} = \frac{\partial CostF}{\partial a_{l+1}} * w_{l+1}* (\sigma(z_l))' * a_{l-1}" /></a>

By b:
<a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;b_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;(\sigma(z_l))'" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;b_{l}}&space;=&space;\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}&space;*&space;w_{l&plus;1}*&space;(\sigma(z_l))'" title="\frac{\partial CostF}{\partial b_{l}} = \frac{\partial CostF}{\partial a_{l+1}} * w_{l+1}* (\sigma(z_l))'" /></a>

So for **each layer** we must know <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\partial&space;CostF}{\partial&space;a_{l&plus;1}}" title="\frac{\partial CostF}{\partial a_{l+1}}" /></a> from next layer,
**current activations**, **weights** from next layer and **activations** from prev layer - for first layer this is inputs in model
