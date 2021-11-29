# w251-hw11

## Questions
1. What parameters did you change, and what values did you use?
I changed the density of the first layer to 512 (self.density_first_layer = 512). I also changed the density of the second layer to 256 (self.density_second_layer = 256).

### 1. Train Output:
455 	: Episode || Reward:  217.52771379075864 	|| Average Reward:  195.56893754421853 	 epsilon:  0.10170090558064004
456 	: Episode || Reward:  236.85010883334547 	|| Average Reward:  198.3017368389839 	 epsilon:  0.10119240105273684
457 	: Episode || Reward:  253.0415415888944 	|| Average Reward:  198.5849061277835 	 epsilon:  0.10068643904747315
DQN Training Complete...

real	55m36.094s
user	0m0.464s
sys	0m0.196s

### 1. Test Output:
95 	: Episode || Reward:  253.98970512520134
96 	: Episode || Reward:  202.5373148686308
97 	: Episode || Reward:  215.68187267778598
98 	: Episode || Reward:  196.2028627155266
99 	: Episode || Reward:  233.94552547681195
Average Reward:  212.92297662289934
Total tests above 200:  81

real	10m8.137s
user	0m0.152s
sys	0m0.072s

2. Did you try any other changes (like adding layers or changing the epsilon value) that made things better or worse?
I changed the batch size to 128 (self.batch_size = 128), which made things worse. Average reward bounced around in the -70's for a while then very slowly improved while bouncing around.

3. Did your changes improve or degrade the model? How close did you get to a test run with 100% of the scores above 200?
Changing the density of the layers improved the model. In the test run 81% of the scores were above 200.

4. Based on what you observed, what conclusions can you draw about the different parameters and their values?
Increasing the density of the layers improves the model's performance. In creasing the batch size from 64 to 128 is too big of a jump and degrades the model's performance.

5. What is the purpose of the epsilon value?
Epsilon randomly chooses the next action as either a random action, or the highest scoring predicted action.

6. Describe "Q-Learning".
Q-Learning is a form of Reinforcement Learning which uses Q-values (also called action values) to iteratively improve the behavior of the learning agent [https://www.geeksforgeeks.org/q-learning-in-python/]. The Q-learning algorithm gives the agent a memory in form of a Q-table. The Q-table stores values to estimate the reward we get by taking that action and are called Q-values. Thus Q-values represent the “quality” of an action taken from that state. [https://www.novatec-gmbh.de/en/blog/introduction-to-q-learning/]


## Videos:
### Train:
- https://w251-hw3-bucket.s3.us-west-1.amazonaws.com/episode0.mp4
- https://w251-hw3-bucket.s3.us-west-1.amazonaws.com/episode300.mp4
- https://w251-hw3-bucket.s3.us-west-1.amazonaws.com/episode550.mp4
### Test:
- https://w251-hw3-bucket.s3.us-west-1.amazonaws.com/testing_run0.mp4
- https://w251-hw3-bucket.s3.us-west-1.amazonaws.com/testing_run50.mp4

