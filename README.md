# w251-hw11: Fun with OpenAI Gym!

## Questions
1. What parameters did you change, and what values did you use?
I changed the density of the first layer to 512 (self.density_first_layer = 512). I also changed the density of the second layer to 256 (self.density_second_layer = 256).

> 1. Train Output:
> ```
> 455 	: Episode || Reward:  217.52771379075864 	|| Average Reward:  195.56893754421853 	 epsilon:  0.10170090558064004
> 456 	: Episode || Reward:  236.85010883334547 	|| Average Reward:  198.3017368389839 	 epsilon:  0.10119240105273684
> 457 	: Episode || Reward:  253.0415415888944 	|| Average Reward:  198.5849061277835 	 epsilon:  0.10068643904747315
> DQN Training Complete...
>
> real	55m36.094s
> user	0m0.464s
> sys	0m0.196s
> ```
>
> 1. Test Output:
> ```
> 95 	: Episode || Reward:  253.98970512520134
> 96 	: Episode || Reward:  202.5373148686308
> 97 	: Episode || Reward:  215.68187267778598
> 98 	: Episode || Reward:  196.2028627155266
> 99 	: Episode || Reward:  233.94552547681195
> Average Reward:  212.92297662289934
> Total tests above 200:  81
>
> real	10m8.137s
> user	0m0.152s
> sys	0m0.072s
> ```

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

## Assignment
In this homework, you will be training a Lunar Lander module to land properly **using your Xavier NX**. There is a video component to this file, so use a display or VNC.

First, some background reading: https://www.novatec-gmbh.de/en/blog/introduction-to-q-learning/

We are using a container base image with all the OpenAI Gym prerequisites installed. 

The python code to train the model is in agent_lunar_lander.py. The code to test the model is test_lander_model.py.

In the python code, the `env.step()` method directs the lander module to take another step (equivalent to one frame of video) and returns several fields: `state`, `reward`, `done`, and `info`. 

 - The `state` is a vector with eight values (x and y position, x and y velocity, lander angle and angular velocity, boolean for left leg contact with ground, boolean for right leg contact with ground). The state information is used to build the model using Keras.
 - The `reward` is a value indicating whether or not the step was "good" or "bad". A reward greater than 200 indicates a successful landing.
 - The `done` field is a boolean indicating whether or not the module has landed. 
 - `info` is not used.

The goal of this homework is to train the lunar module to land better. The model, as it is currently configured, will not converge and the lunar module will never learn to land well. By modifying the parameters (lines 30-43 of the python code), you should be able to train the module in fewer than 500 iterations.

```
        self.density_first_layer = 16
        self.density_second_layer = 8
        self.num_epochs = 1
        self.batch_size = 64
        self.epsilon_min = 0.01
```

We are using a Sequential model for the lander. A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor (current state) and one output tensor (best move to make). The "moves" that can be made are firing the thrusters (right, left, up) to adjust the speed and trajectory.

The current model has three layers. Consider the dimension of the input tensor for each layer. Is it optimal? Are the activations appropriate for the use case?

```
        model = Sequential()
        model.add(Dense(self.density_first_layer, input_dim=self.num_observation_space, activation=relu))
        model.add(Dense(self.density_second_layer, activation=relu))
        model.add(Dense(self.num_action_space, activation=linear))
```

To run the environment, use these commands (ensure you have all the files from the hw11 github folder in your current directory on the Jetson):

```
# If you haven't added your User to the docker group, do it now
sudo usermod -aG docker $USER

# reboot to make the previous step take effect

docker build -t hw11 -f Dockerfile.4.4 .

# enable video sharing from the container
xhost +

# Start the training
time docker run -it --rm --net=host --runtime nvidia  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix:rw --privileged -v /data/videos:/tmp/videos hw11

```

When the process starts, you will see the animation of the lunar lander on your screen and the training will start.

Training output looks like this (ignore any WARNING messages):

```
0 	: Episode || Reward:  -355.4552185273774 	|| Average Reward:  -355.4552185273774 	 epsilon:  0.995
1 	: Episode || Reward:  -302.69548515410156 	|| Average Reward:  -329.0753518407395 	 epsilon:  0.990025
2 	: Episode || Reward:  -197.1461440026914 	|| Average Reward:  -285.09894922805677 	 epsilon:  0.985074875
3 	: Episode || Reward:  -251.29447991556844 	|| Average Reward:  -276.64783189993466 	 epsilon:  0.9801495006250001
4 	: Episode || Reward:  -312.69842116384507 	|| Average Reward:  -283.85794975271676 	 epsilon:  0.9752487531218751
5 	: Episode || Reward:  -193.10620553981315 	|| Average Reward:  -268.73265905056616 	 epsilon:  0.9703725093562657
6 	: Episode || Reward:  -125.35339813322857 	|| Average Reward:  -248.2499074909465 	 epsilon:  0.9655206468094844
7 	: Episode || Reward:  -95.87496167296544 	|| Average Reward:  -229.20303926369886 	 epsilon:  0.960693043575437
8 	: Episode || Reward:  -10.731355125180073 	|| Average Reward:  -204.92840769275233 	 epsilon:  0.9558895783575597
```

The training will end when either the Average Reward is greater than 200, or 2000 iterations have passed. I would recommend killing the model if it ever hits 800, though.

After the training, you can run a test process. 

To build the test container, copy your model (can be found with `ls -l /data/videos/*.h5`) to your local Docker build directory and rename it to `mymodel.h5`. 

The container can be built with:

```
docker build -t testlander -f Dockerfile.test .
```

The container can be run with:

```
time docker run -it --rm --net=host --runtime nvidia  -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix:rw --privileged -v /data/videos:/tmp/videos testlander
```

The output will look like this:

```
DQN Training Complete...
Starting Testing of the trained model...
0       : Episode || Reward:  219.64614710147364
1       : Episode || Reward:  204.5401595978414
2       : Episode || Reward:  191.82778586724473
3       : Episode || Reward:  300.26513457499857
4       : Episode || Reward:  265.38375246986914
5       : Episode || Reward:  231.17971859331598
6       : Episode || Reward:  158.1286447553571
.
.
.
Average Reward:  243.09916996497867
```

**The assignment**: Modify the parameters in the python file with your best (educated) guess to improve the model training. A well tuned model will start landing the module after about 300 iterations and consistently land it after about 400 iterations. If you are feeling creative, you can change other aspects of the model training (like batch size and epsilon value). You can re-build the Docker image after changing the python file to test your changes. 
