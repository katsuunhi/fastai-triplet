基于fastai v2的triplet实现，在mnist上做了测试，降维可视化效果很好


nn中的triplet margin loss不能用，原因是输出pred为元组，被当做一个变量处理了，具体参考learner.py中的_do_one_batch()函数