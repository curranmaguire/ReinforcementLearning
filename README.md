# ReinforcementLearning

Experimentation

streight algorithm, > 1 day,
adding another linear layer > took longer to reach convergence. not really needed.

### Reward Scaling

We changed the -100 reward to -5.
(ii) We increased other rewards by a factor of 5.
(iii) We implemented a replay buffer where failed episodes, in which the bipedalwalker fell down at the
end, and successful episodes are added to the replay-buffer with 5:1 ratio.
(iii) because we found
failed episodes are more useful for learning than successful ones. The reason we believe is that when the
bipidedalwalker already knows how to handle a terrain, there is no need to further train using the same type
of terrain. When the training is near the end, most of the episodes are successful so adding these successful
episodes overwhelm the more useful episodes (failed ones), which slows down the learning.
suggested in this blog
https://mp.weixin.qq.com/s?__biz=MzA5MDMwMTIyNQ==&mid=2649294554&idx=1&sn=9f893801b8917575779430cae89829fb&scene=21#wechat_redirect
https://arxiv.org/pdf/2010.01652.pdf

### TD3

TD3 has be benefit of being deep and works over a continuous space however the algorithm doesn't work for bipedal walker hardcore bc it basically learns to stay away from ravines as they are high penalty zones and that just stand still. Potential use of a recurrent net is recommended however it sounds like this will take a lot of time to train. Let's start one and see.

### TD3_FORK

shows to be the best algorithm in terms of efficiency,

- need to understand how it works

transfer learning of the model to help it train more efficiently. We will see if it work.

- look at adapting the training params for the seperate learning envs potentially

-look at a recurrent net https://github.com/zhihanyang2022/off-policy-continuous-control/blob/pub/offpcc/basics/summarizer.py to help

-use this for tips and tricks https://agents.inf.ed.ac.uk/blog/reinforcement-learning-implementation-tricks/
