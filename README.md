# Masters Project

## Reinforcement Learning in Fault Tolerant Control 
## Auther: Fotinos Kyriakides - The University of Cambridge


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

These libraries are auxiliary to the lunar lander 

```console
cd "Lunar Lander/pomdp_libs/gym_pomdp"
pip install -e . 
```

```console
cd "Lunar Lander/pomdp_libs/gym-pomdp-wrappers"
pip install -e . 
```
