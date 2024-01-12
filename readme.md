
# One repsitory, three subprojects
+ Readability: How easy is a text to read?
+ Deutscher Lyrik Korpus: At which year was a poem published?
+ Python autocomplete: Creates Python Code

### Readability
The kaggle readability dataset contains short passages of text suitable for school kids, labeled with a readability score which indicates or easy or difficult the text is to read. 
The readability score is here scaled between 0 (hardest) and 1 (easiest). Examples for scores:
+ Score 0.76 (easy to read): When Grandmother Lane was a little girl her father came in one day and said, „Wife, it is all settled at last. I have sold the farm. Next week we will start West...“
+ Score 0.49 (medium to read): Most people know that physical activity is important for their physical health. People who are more physically...
+ Score 0.33 (difficult to read): In a column of fluid, pressure increases with depth as a result of the weight of the overlying fluid. Thus the pressure at the bottom of a column of fluid is greater than at the top.


### DLK
The German Lyric Corpus (Deutscher Lyrik Koprus) is a dataset containing more than 60.000 poems, published between the years ~ 1600 and 1900. Each poem is labeled with a year in which it was published.<br>
Hence, by using supervised learning we can train an neural network to predict the year a poem was published.<br>
<img src="images/dlk correlation.png" width=500px/>


### Python copilot
A network was trained to autocomplete a line or even several lines of code:

```python
### Example 1 ###
import n
### becomes ###
import numpy as np # linear algebra
import pandas as pd # data processing

### Example 2 ###
# plot
### becomes ###
# plot
plt.show()

### Example 3 ###
# pytorch
import 
### becomes ###
# pytorch
import torch.nn as nn
import torch.nn
import torch.nn.functional as ...

### Example 4 ###
# neural network
class Net
### becomes ###
# neural network
class Network(nn.Module):
    def __init__(self, input_shape
```