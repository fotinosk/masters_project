"""
Taken from:
https://towardsdatascience.com/deep-deterministic-policy-gradient-ddpg-theory-and-implementation-747a3010e82f
"""

"""
In the original DDPG paper for the critic network the action enters the network in middle layers instead of 
entering the network from the beginning. This is only done to increase performance/stability, and we would 
not resort to this trick. For us the action and state input will enter the critic network from the beginning.
"""

import tensorflow as tf
from replay import BasicBuffer, BasicBuffer2
import numpy as np
from datetime import datetime


def ANN2(input_shape, layer_sizes, hidden_activation='relu', output_activation=None):
    """Function that generates both the actor and critic"""

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    for h in layer_sizes[:-1]:
        x = model.add(tf.keras.layers.Dense(units=h, activation='relu'))
    model.add(tf.keras.layers.Dense(units=layer_sizes[-1], activation=output_activation))
    return model


"""
The output layer for the actor will be a ‘tanh’, ( to map continuous action -1 to 1) and the output layer 
for critic will be ‘None’ as its the Q-value. The output for the actor-network can be scaled by a factor
 to make the action correspond to the environment action range.
"""


##############################################
#            Model Initialization            #
##############################################

def ddpg(env_fn, ac_kwargs=dict(), seed=0, save_folder=None, num_train_episodes=100,
         test_agent_every=25, replay_size=int(1e6), gamma=0.99, decay=0.99, mu_lr=1e-3,
         q_lr=1e-3, batch_size=32, start_steps=10, action_noise=0.0, max_episode_length=1500):

    tf.random.set_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()

    # get size of state space and action space
    num_states = env.observation_space.shape[0]
    num_actions = env.action_space.shape[0]
    action_max = env.action_space.high[0]

    # Network parameters
    X_shape = (num_states)
    QA_shape = (num_states + num_actions)
    hidden_sizes_1 = (1000, 500, 200)
    hidden_sizes_2 = (400, 200)

    # Main network outputs
    mu = ANN2(X_shape, list(hidden_sizes_1) + [num_actions], hidden_activation='relu', output_activation='tanh')
    q_mu = ANN2(QA_shape, list(hidden_sizes_2) + [1], hidden_activation='relu')

    # Target networks
    mu_target = ANN2(X_shape, list(hidden_sizes_1) + [num_actions], hidden_activation='relu', output_activation='tanh')
    q_mu_target = ANN2(QA_shape, list(hidden_sizes_2) + [1], hidden_activation='relu')

    # Copying weights in,
    mu_target.set_weights(mu.get_weights())
    q_mu_target.set_weights(q_mu.get_weights())

    # Todo: maybe try basic buffer b
    replay_buffer = BasicBuffer2(size=replay_size, obs_dim=num_states, act_dim=num_actions)

    # Train each network separately
    mu_optimizer = tf.keras.optimizers.Adam(learning_rate=mu_lr)
    q_optimizer = tf.keras.optimizers.Adam(learning_rate=q_lr)

    def get_actions(s, noise_scale):
        a = action_max * mu.predict(np.array(s).reshape(1, -1))[0]
        a += noise_scale * np.random.randn(num_actions)
        return np.clip(a, -action_max, action_max)

    test_returns = []

    def test_agent(num_episodes=5):
        t0 = datetime.now()
        n_steps = 0

        for j in range(num_episodes):
            s, episode_return, episode_length, d = test_env.reset(), 0, 0, False
            while not (d or (episode_length == max_episode_length)):
                test_env.render()  # maybe comment out this line
                print(s, get_actions(s,0))
                s, r, d, _ = test_env.step(get_actions(s, 0))
                episode_return += r
                episode_length += 1
                n_steps += 1
            print('test return:', episode_return, 'episode_length:', episode_length)
            test_returns.append(episode_return)

    # Main loop: train
    returns = []
    q_losses = []
    mu_losses = []
    num_steps = 0

    for i_episode in range(num_train_episodes):

        # reset
        s, episode_return, episode_length, d = env.reset(), 0, 0, False

        while not (d or (episode_length == max_episode_length)):
            # use randomly sampled actions for 1st steps

            if num_steps > start_steps:
                a = get_actions(s, action_noise)
            else:
                a = env.action_space.sample()

            num_steps += 1
            if num_steps == start_steps:
                print('USING AGENT ACTIONS NOW')

            s2, r, d, _ = env.step(a)
            episode_return += r
            episode_length += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d_store = False if episode_length == max_episode_length else d

            # store to replay buffer
            replay_buffer.push(s, a, r, s2, d_store)

            s = s2

        for _ in range(episode_length):

            X, A, R, X2, D = replay_buffer.sample(batch_size)
            X = np.asarray(X, dtype=np.float32)
            A = np.asarray(A, dtype=np.float32)
            R = np.asarray(R, dtype=np.float32)
            X2 = np.asarray(X2, dtype=np.float32)
            Xten = tf.convert_to_tensor(X)

            # Actor optim
            with tf.GradientTape() as tape2:
                Aprime = action_max * mu(X)
                temp = tf.keras.layers.concatenate([Xten, Aprime], axis=1)
                Q = q_mu(temp)
                mu_loss = -tf.reduce_mean(Q)
                grads_mu = tape2.gradient(mu_loss, mu.trainable_variables)
            mu_losses.append(mu_loss)
            mu_optimizer.apply_gradients(zip(grads_mu, mu.trainable_variables))

            # Critic optim
            with tf.GradientTape() as tape:
                next_a = action_max * mu_target(X2)
                temp = np.concatenate((X2, next_a), axis=1)
                q_target = R + gamma * (np.array([1]*len(D)) - np.array(D)) * q_mu_target(temp)
                temp2 = np.concatenate((X, A), axis=1)
                qvals = q_mu(temp2)
                q_loss = tf.reduce_mean((qvals - q_target)**2)
                grads_q = tape.gradient(q_loss, q_mu.trainable_variables)

            q_optimizer.apply_gradients(zip(grads_q, q_mu.trainable_variables))
            q_losses.append(q_loss)

            # updating critic target (soft)
            temp1 = np.array(q_mu_target.get_weights())
            temp2 = np.array(q_mu.get_weights())
            temp3 = decay * temp1 + (1-decay)* temp2
            q_mu_target.set_weights(temp3)

            # updating actor target (soft)
            temp1 = np.array(mu_target.get_weights())
            temp2 = np.array(mu.get_weights())
            temp3 = decay * temp1 + (1 - decay) * temp2
            mu_target.set_weights(temp3)

        print("Episode:", i_episode + 1, "Return:", episode_return, 'episode_length:', episode_length)
        returns.append(episode_return)

        # Test the agent
        # if i_episode > 0 and i_episode % test_agent_every == 0:
        #   test_agent()

    return (returns, q_losses, mu_losses)
    