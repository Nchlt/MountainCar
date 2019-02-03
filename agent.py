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
        self.cto_policy = None
        self.b2_policy = np.array([list([0., 1., 0., 0., 0., 1.]), list([0., 0., 1., 2., 0., 0.]), list([0., 0., 0., 0., 2., 0.]), list([1., 1., 2., 0., 2., 2.]), list([2., 0., 0., 0., 2., 1.]), list([1., 0., 0., 2., 0., 0.])])
        # self.b_policy = np.array([list([1., 1., 0., 1., 1., 1.]), list([1., 0., 1., 2., 0., 0.]), list([0., 1., 0., 0., 2., 0.]), list([2., 1., 2., 0., 2., 2.]), list([2., 0., 0., 0., 2., 2.]), list([1., 0., 1., 2., 0., 0.])])
        self.b_policy = np.array([list([0., 1., 0., 0., 0., 1.]), list([0., 0., 1., 2., 0., 0.]), list([0., 0., 0., 0., 2., 0.]), list([1., 1., 2., 0., 2., 2.]), list([2., 0., 0., 0., 2., 1.]), list([1., 0., 0., 2., 0., 0.])])
        self.positions = np.linspace(start=-1.2, stop=0.6, num=6)
        self.velocities = np.linspace(start=-0.07, stop=0.07, num=6)
        # Here we initialize the policy with a previous solution
        #self.policy = pickle.load(open('bestPol.p', 'rb')).reshape(50, 4)
        #self.policy = np.array([list([0, 1, 1, 2, 1, 2, 2, 1, 1, 1]), list([1, 2, 2, 1, 2, 2, 2, 1, 2, 1]), list([1, 2, 1, 1, 2, 2, 1, 2, 1, 2]), list([1, 1, 2, 0, 2, 2, 2, 2, 1, 2]), list([2, 0, 2, 2, 0, 1, 2, 1, 2, 1]), list([0, 2, 0, 1, 2, 2, 1, 2, 1, 2]), list([2, 2, 2, 2, 1, 2, 2, 1, 2, 2]), list([2, 1, 2, 1, 1, 2, 1, 2, 1, 1]), list([0, 2, 2, 2, 2, 1, 2, 1, 2, 2]), list([2, 1, 2, 2, 2, 1, 2, 0, 2, 2])])
        #self.policy = np.array([2 for _ in range(self.positions.shape[0]*self.velocities.shape[0])]).reshape(self.positions.shape[0], self.velocities.shape[0])
        # self.policy = np.random.randint(low=0, high=3,
        #                                 size=(self.positions.shape[0],
        #                                 self.velocities.shape[0]))

        #self.policy = np.array([list([1., 0., 0., 0., 2., 0.]), list([2., 1., 1., 2., 0., 0.]), list([0., 2., 0., 0., 2., 0.]), list([2., 2., 2., 0., 2., 1.]), list([2., 0., 0., 1., 2., 2.]), list([1., 0., 2., 2., 0., 0.])])
        self.policy = np.array([list([1., 1., 0., 1., 1., 1.]), list([1., 0., 1., 2., 0., 0.]), list([0., 1., 0., 0., 2., 0.]), list([2., 1., 2., 0., 2., 2.]), list([2., 0., 0., 0., 2., 2.]), list([1., 0., 1., 2., 0., 0.])])
        # self.policy = pickle.load(open('b2p.p', 'rb'))
        self.init = self.policy.reshape(self.positions.shape[0]*self.velocities.shape[0])
        #self.init = self.policy.reshape(self.policy.shape[0]*self.policy.shape[1])
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
                velocity_index = i
                j = 0
                while position > positions[j]:
                    j += 1
                position_index = j
                # print("Debug pos vel input :")
                # print(position, velocity)
                # print("Debug output : ")
                # print(positions[position_index], velocities[velocity_index])
                #exit()
                return position_index, velocity_index

        # 1- Define state features
        # 2- Define search space (to define a policy)
        # 3- Define objective function (for policy evaluation)

        # 3- Define objective function (for policy evaluation)
        env = Environment()
        self.min_min_distance = 10
        self.best_value = 999999999
        self.min_iter_win = 99999999999
        self.calls_to_obj = 0
        self.previous_value = 9999999999999
        def obj_function(policy):

            self.calls_to_obj += 1
            print(self.calls_to_obj)
            print(self.previous_value)
            env.reset()
            #print(env.state[0])
            iter_counter = 0
            # Instanciate a testing environement

            # Create a x solution for CMA (reshape 1D) and discretize the policy actions
            x = policy.reshape(self.positions.shape[0]*self.velocities.shape[0], -1)
            x = np.floor(x)
            d_policy = x.reshape(self.positions.shape[0], self.velocities.shape[0])

            # if self.calls_to_obj == 7:
            #     print("Recording policy 8")
            #     self.cto_policy = d_policy
            #     pickle.dump(d_policy, open('calls_to_obj_policy.p', 'wb'))

            testing_steps = 200
            distances = []
            distances_mid = []
            energy = []
            middle = 0
            nb_iter = testing_steps

            for i in range(testing_steps):
                env.render()
                # We compute the distance to the optimal as the difference
                # between the optimal position and the actual position
                #distance_to_goal = env.state[0] - 0.5
                #distances.append(distance_to_goal)
                speed = env.state[1]
                pos = env.state[0]
                # print(pos, speed)
                # exit()
                if ((pos >= -0.5) and (pos <= -0.4)):
                    middle += 10
                #distances.append(np.absolute(0.5 - pos))
                distances.append(np.absolute(0.6 - pos))
                distances_mid.append(np.absolute(-0.56 - pos))
                energy.append(0.5*(speed**2))

                #print(distances[-1:])
                #print(np.absolute(0.5 - env.state[0]))
                # We take an action according to the given policy
                position, velocity = env.state
                action = policy_action(position, velocity, d_policy)
                # done = False
                #self.state, reward, done, info = self.env.step(action)

                _, done = env.act(int(np.floor(action)))
                # print(done)
                #_, _ = env.act(int(np.ceil(action)))
                iter_counter = i
                if min(distances) ==0:

                    # print("WON "+str(iter_counter))
                    if iter_counter < self.min_iter_win:
                        self.min_iter_win = iter_counter
                        print('New record WON at %d iteration.'%iter_counter)
                        self.b2_policy = policy
                        pickle.dump(self.b2_policy, open('b2p.p', 'wb'))
                        # exit()
                        # print("Updated b2_policy")
                    value = iter_counter * sum(distances)
                    # print(value)
                    self.previous_value = value
                    return value
                    # nb_iter = iter_counter
                    # value =(-0.8*sum(distances_mid)+0.2*sum(distances))
                    # if value < self.best_value:
                    #     self.best_value = value
                    #     print(self.best_value, self.min_min_distance)
                    # return value
                #     value = (-sum(energy)*100+min(distances)*100)/-nb_iter
                #
                #     if value < self.best_value:
                #         self.best_value = value
                #         print(self.best_value, self.min_min_distance)
                #     return value

                    #value =  min(distances)*iter_counter
                    #print(value)
                    #return value
                #reward, _ = self.env.act(d_action)
                #value += reward
                # if (0.5 - position) < self.closest:
                #     value = 0.5 - position
            #print('Current best value = '+str(value))
            #min_iter = np.argmin(np.array(distances))
            #value = (min(distances) * min_iter) + min_iter
            #value = -sum(speed)
            #value = middle + -sum(energy)
            #value = -sum(energy)*1000 + middle + nb_iter*100 -nb_iter*min(distances)*20
            #value = -sum(energy)*100/ + nb_iter/5 +min(distances)*100
            #value = (-sum(energy)*100+min(distances)*100)/-nb_iter
            # value = -sum(energy)*0.3 + sum(distances) -sum(distances_mid)
            # value =(-0.8*sum(distances_mid)+0.2*sum(distances))
            # value = iter_counter
            value = iter_counter * sum(distances)
            # value = -sum(energy)
            #print(value, middle, -sum(energy))
            if min(distances) < self.min_min_distance:
                self.min_min_distance = min(distances)
                #print(value, -sum(energy)*100, nb_iter/2, min(distances)*100, self.min_min_distance)
                #print(value, -sum(energy)*1000,middle, nb_iter,-nb_iter*min(distances)*20,self.min_min_distance)

            if value < self.best_value:
                self.best_value = value
                # print(self.best_value, self.min_min_distance)
            # print(value)
            self.previous_value = value
            return value

            #print(min(distances), iter_counter)
            #print("Evaluated on %d steps value = %f"%(testing_steps, value))
            #print(value)
            #print(value, min_iter)
            # print(value)
            # return value
        # On good seed 239448 237591
        best_policy, _ = cma.fmin2(obj_function, self.init, 0.08,{
        #'BoundaryHandler': 'BoundPenalty',
        'BoundaryHandler': 'BoundTransform',
        'bounds':[0,3],
        #'bounds': [[0 for _ in range(self.positions.shape[0]*self.velocities.shape[0])], [3 for _ in range(self.positions.shape[0]*self.velocities.shape[0])]],
        'popsize':200,
        'CMA_mu':5,
        'verbose':1,
        'ftarget':8.307408575667165e3,
        'seed': 237591
        })
        print("Optimization FINISHED")
        #self.policy = best_policy
        self.b_policy = best_policy
        self.policy = np.floor(best_policy).reshape(self.positions.shape[0], self.velocities.shape[0])
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

        def policy_action(position, velocity, policy):
                i_position, i_velocity = get_discretized_env(position, velocity)
                # print(i_position, i_velocity)
                action = policy[i_position][i_velocity]
                return action

        def get_discretized_env(position, velocity, velocities=self.velocities, positions=self.positions):
                i = 0
                while velocity > velocities[i]:
                    i += 1
                velocity_index = i
                j = 0
                while position > positions[j]:
                    j += 1
                position_index = j
                # print("Debug pos vel input :")
                # print(position, velocity)
                # print("Debug output : ")
                # print(positions[position_index], velocities[velocity_index])
                #exit()
                return position_index, velocity_index
        #print("Training over, now using optimal policy:")

        x = self.cto_policy.reshape(self.positions.shape[0]*self.velocities.shape[0], -1)
        x = np.floor(x)
        d_policy = x.reshape(self.positions.shape[0], self.velocities.shape[0])
        action = policy_action(observation[0], observation[1], d_policy)
        action = (int(np.floor(action)))


        # position, velocity = observation
        # action = int(policy_action(position, velocity, self.policy))

        return action

Agent = Patrick
