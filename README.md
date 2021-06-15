# Masters Project

## Reinforcement Learning in Fault Tolerant Control 
## Author: Fotinos Kyriakides - The University of Cambridge


### Installing the necessary packages 

The project requires that custom libraries are used, to act as ai-gym environments, 
as well as some complementary libraries

#### Installing gym_Boeing

```console
cd Environment/gym-Boeing
pip install -e . 
```

#### Installing the Lunar Lander

This lunar lander package is a modified version of the one found in AI Gym. 

```console
cd "Lunar Lander/Lunar-gym"
pip install -e . 
```

#### Installing Partial Observability Libraries

These libraries are auxiliary to the lunar lander. One intercepts the env outputs, making them partially observable and the other augments the state input to the agent. 

```console
cd "Lunar Lander/pomdp_libs/gym_pomdp"
pip install -e . 
```

```console
cd "Lunar Lander/pomdp_libs/gym-pomdp-wrappers"
pip install -e . 
```

### Scenarios - Plane

1. "boeing-danger-v1"
  -- Modes: [A,B,C,D]
2. "failure-train-v1"
  -- Modes: [A,B,C,D], [A,0.5B,C,D] 
3. "failure-train-v2" **FAILS**
  -- Modes: [A,B,C,D], [A,-B,C,D]  
4. "faultyA-train-v0"
  -- Modes: [A,B,C,D], [0.8A,B,C,D]
5. "failure-train-v3"
  -- Modes: [A,B,C,D], [A,0.5B,C,D], [0.8A,B,C,D]
6. "actuation-train-v0"
  -- Modes: Normal, Elevator Lag =  2sec
7. "four-modes-train-v0"
  -- Modes: [A,B,C,D], [A,0.5B,C,D], [0.8A,B,C,D], Throttle Lag = 5sec & Elevator Lag = 1sec

### Scenarios - Lunar Lander

1. "partially-obs-v0"
  -- Modes: Default 
2. "mass-train-v0"
  -- Modes: Default, Mass = 3 
3. "inertia-train-v0"
  -- Modes: Default, Moment of Inerta = 8
4. "inertia-mass-train-v0"
  -- Modes: Default, Moment of Inerta = 8, Mass = 3
5. "sticky-train-v0"
  --Modes: Probability of stuck controler = 0.1
6. "sticky-im-train-v0"
  -- Modes: Default, Moment of Inerta = 8, Mass = 3, Probability of stuck controler = 0.1

These are the availble training environments. 

Replacing "train", with "test" in all the above environment names will give their respective training environments, which include additional test modes, novel to the trained agent. 

Before testing any one agent on the environments, ensure that the exists a file with the name of the corresponding training environemnent in the agent folder (ie ensure that the agent has been trained on the environment).
