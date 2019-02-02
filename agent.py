import numpy as np
from environment import Environment
import pickle
import cma
"""
Contains the definition of the agent that will run in an
environment.
"""

class RandomAgent:
    def __init__(self):
        """
        Init a new agent.
        """
        self.train() # Do not remove this line!!

    def train(self):
        """
        Learn your (final) policy.

        Use evolution strategy algortihm CMA-ES: https://pypi.org/project/cma/

        Possible action: [0, 1, 2]
        Range observation (tuple):
            - position: [-1.2, 0.6]
            - velocity: [-0.07, 0.07]
        """
        # 1- Define state features
        # 2- Define search space (to define a policy)
        # 3- Define objective function (for policy evaluation)
        # 4- Use CMA-ES to optimize the objective function
        # 5- Save optimal policy

        # This is an example of using Envrironment class (No learning is done yet!)
        for i in range(10):
            env = Environment()
            done = False
            while not done:
                reward, done = env.act(env.sample_action())

    def act(self, observation):
        """
        Acts given an observation of the environment (using learned policy).

        Takes as argument an observation of the current state, and
        returns the chosen action.
        See environment documentation: https://github.com/openai/gym/wiki/MountainCar-v0
        Possible action: [0, 1, 2]
        Range observation (tuple):
            - position: [-1.2, 0.6]
            - velocity: [-0.07, 0.07]
        """
        return np.random.choice([0, 1, 2])


np.random.seed(0)

class Patrick:

    def __init__(self):
        """
        Init a new agent.
        """

        self.positions = np.linspace(start=-1.2, stop=0.6, num=50)
        self.velocities = np.linspace(start=-0.07, stop=0.07, num=4)
        #self.policy = np.array([list([0, 1, 1, 2, 1, 2, 2, 1, 1, 1]), list([1, 2, 2, 1, 2, 2, 2, 1, 2, 1]), list([1, 2, 1, 1, 2, 2, 1, 2, 1, 2]), list([1, 1, 2, 0, 2, 2, 2, 2, 1, 2]), list([2, 0, 2, 2, 0, 1, 2, 1, 2, 1]), list([0, 2, 0, 1, 2, 2, 1, 2, 1, 2]), list([2, 2, 2, 2, 1, 2, 2, 1, 2, 2]), list([2, 1, 2, 1, 1, 2, 1, 2, 1, 1]), list([0, 2, 2, 2, 2, 1, 2, 1, 2, 2]), list([2, 1, 2, 2, 2, 1, 2, 0, 2, 2])])
        self.policy = np.array([2 for _ in range(self.positions.shape[0]*self.velocities.shape[0])]).reshape(self.positions.shape[0], self.velocities.shape[0])
        # self.policy = np.random.randint(low=0, high=3,
        #                                 size=(self.positions.shape[0],
        #                                 self.velocities.shape[0]))
        self.init = self.policy.reshape(self.policy.shape[0]*self.policy.shape[1])
        self.train() # Do not remove this line!!


    def train(self):

        """
        Learn your (final) policy.

        Use evolution strategy algortihm CMA-ES: https://pypi.org/project/cma/

        Possible action: [0, 1, 2]
        Range observation (tuple):
            - position: [-1.2, 0.6]
            - velocity: [-0.07, 0.07]
        """
        def policy_action(position, velocity, policy):
                i_position, i_velocity = get_discretized_env(position, velocity)
                # print(i_position, i_velocity)
                action = policy[i_position][i_velocity]
                return action

        def get_discretized_env(position, velocity, velocities=self.velocities, positions=self.positions):
                i = 0
                while velocity > velocities[i]:
                    i += 1
                velocity_index = i - 1
                j = 0
                while position > positions[j]:
                    j += 1
                position_index = j - 1
                return position_index, velocity_index
        # 1- Define state features
        # 2- Define search space (to define a policy)
        # 3- Define objective function (for policy evaluation)

        # 3- Define objective function (for policy evaluation)
        env = Environment()
        def obj_function(policy):
            env.reset()
            iter_counter = 0
            # Instanciate a testing environement

            # Create a x solution for CMA (reshape 1D) and discretize the policy actions
            x = policy.reshape(self.positions.shape[0]*self.velocities.shape[0], -1)
            x = np.floor(x)
            d_policy = x.reshape(self.positions.shape[0], self.velocities.shape[0])
            testing_steps = 10001
            distances = []

            for i in range(testing_steps):

                # We compute the distance to the optimal as the difference
                # between the optimal position and the actual position
                distances.append(0.5 - env.state[0])
                #print(np.absolute(0.5 - env.state[0]))
                # We take an action according to the given policy
                position, velocity = env.state
                action = policy_action(position, velocity, d_policy)
                #done = False
                #self.state, reward, done, info = self.env.step(action)
                _, _ = env.act(int(np.floor(action)))
                iter_counter = i
                if distances[-1:][0] < 1e-2:
                    print('WON')
                    value =  min(distances)*iter_counter
                    #print(value)
                    return value
                #reward, _ = self.env.act(d_action)
                #value += reward
                # if (0.5 - position) < self.closest:
                #     value = 0.5 - position
            #print('Current best value = '+str(value))

            value = min(distances) * iter_counter
            #print(min(distances), iter_counter)
            #print("Evaluated on %d steps value = %f"%(testing_steps, value))
            #print(value)
            return value

        best_policy, _ = cma.fmin(obj_function, self.init, 0.1,{
        'BoundaryHandler': 'BoundTransform',
        'bounds':[0,3],
        #'bounds': [[0 for _ in range(self.positions.shape[0]*self.velocities.shape[0])], [3 for _ in range(self.positions.shape[0]*self.velocities.shape[0])]],
        'popsize':1000,
        'CMA_mu':10,
        'verbose':1
        })
        self.policy = np.floor(best_policy)
        print("Best Policy updated"+str(self.policy))
        pickle.dump(self.policy, open('bestPol.p', 'wb'))
        # def obj_function(x):
        #     env = Environment()
        #     # This is an example of using Envrironment class (No learning is done yet!)
        #     dist = []
        #     done = False
        #     for i in range(200):
        #         dist.append(self.state[0])
        #         reward, done = env.act(env.sample_action())
        #         if done:
        #             print("Goal Reached")

        # 4- Use CMA-ES to optimize the objective function
        # 5- Save optimal policy

        # This is an example of using Envrironment class (No learning is done yet!)
        # for i in range(10):
        #     env.reset()
        #     done = False
        #     while not done:
        #         reward, done = env.act(env.sample_action())

    def act(self, observation):
        """
        Acts given an observation of the environment (using learned policy).

        Takes as argument an observation of the current state, and
        returns the chosen action.
        See environment documentation: https://github.com/openai/gym/wiki/MountainCar-v0
        Possible action: [0, 1, 2]
        Range observation (tuple):
            - position: [-1.2, 0.6]
            - velocity: [-0.07, 0.07]
        """
        print("Test")
        return np.random.choice([0, 1, 2])

Agent = Patrick
