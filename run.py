import random
import numpy as np
from numpy import random as rnd
from functools import reduce
import operator

#lambda_ = 0.1
#while lambda_ < 20:

print(f"\n                 SIMULATION vs ANALYTIC                  ")
print(f"*** Note: Theta=2 , Mu=1 are preferred for Simulation *** \n")
theta = float(input("Enter value of theta: "))
theta = np.float128(theta)
mu = float(input("Enter value of mu: "))
mu = np.float128(mu)
lambda_ = 5     # if you want print out Pb,Pd Values from Lambda 0.1 to 20 change initial values to 0.1
num_customers = 10 ** 5 #There was I/O issue on powers more than 6 on my system
queue_size = 12
expdrop = True
######################################### Analytical Module ###########
class Analyzer:
    def __init__(self, lambda_, mu, theta, queue_size, expdrop = True):
        self.lambda_ = lambda_
        self.mu = mu
        self.theta = theta
        self.queue_size = queue_size
        self.is_exp = expdrop
        self.coefficients = []
        self.pb = 0
        self.pd = 0
        self.p0 = 0
                         # https://github.com/NeuralEnsemble/PyNN/blob/master/pyNN/random.py
    def phi(self, n):
        if self.is_exp:
            return np.math.factorial(n) / reduce(operator.mul, [(self.mu + (i / self.theta)) for i in range(0, n + 1)], 1)
        else:
            return (np.math.factorial(n)/(self.mu ** (n + 1))) \
                   * (1 - (np.exp(-1 * self.mu * self.theta)) * sum([(((self.mu * self.theta) ** i)/np.math.factorial(i)) for i in range(0, n)]))

    def run(self):
        for i in range(0,self.queue_size+1):
            if i == 0:
                the_coef = 1
            elif i == 1:
                the_coef = self.lambda_ / self.mu
            else:
                the_coef = ((self.lambda_ ** i) * self.phi(i - 1)) / np.math.factorial(i - 1)
            self.coefficients.append(the_coef)

        self.p0 = 1/sum(self.coefficients)
        self.pb = self.p0 * self.coefficients[-1]
        self.pd = (1 - (self.mu / self.lambda_) * (1 - self.p0)) - self.pb
        return self.pb, self.pd

########################################### Simulation Module #########

class task:
    def __init__(self, task_id, arrive_time, mu, theta, expdrop):
        self.task_id = task_id
        self.expdrop = expdrop
        self.arrive_time = arrive_time
        self.service_time = random.expovariate(1 / mu)
        self.drop_time = rnd.exponential(theta) if self.expdrop else theta
        self.service_end_time = self.arrive_time + self.service_time
        self.limit_time = self.arrive_time + self.drop_time


class event:
    def __init__(self, task_id, event_time, status=0):
        self.task_id = task_id
        self.event_time = event_time
        self.status = status


class system:
    def __init__(self, queue_size):
        self.queue_size = queue_size
        self.num_exited = 0
        self.num_dropped = 0
        self.num_blocked = 0
        self.priority_queue = []
        self.events_queue = []
        self.all_tasks = []
        self.queue_status = 0

    def add_event(self, the_new_event):
        if the_new_event:
            i = 0
            j = len(self.events_queue)
            while i < j:
                mid = (i + j) // 2
                if self.events_queue[mid].event_time < the_new_event.event_time:
                    i = mid + 1
                else:
                    j = mid
            self.events_queue.insert(i, the_new_event)

    def add_task(self, the_new_task):
        self.all_tasks.append(the_new_task)
        the_new_event = self.handle_events(the_new_task)
        self.add_event(the_new_event)

    def handle_events(self, the_task, event_time=0, status=0):
        if status == 0:
            self.queue_status = len(self.priority_queue)
            if self.queue_status == self.queue_size:
                self.num_blocked += 1
                return False
            else:
                self.priority_queue.append(the_task)
                return event(the_task.task_id, the_task.service_end_time if self.queue_status == 0 else the_task.limit_time, 2 if self.queue_status == 0 else 1)

        elif status == 1:
            self.num_dropped += 1
            self.priority_queue = [q for q in self.priority_queue if q.task_id != the_task.task_id]
            self.events_queue = [q for q in self.events_queue if q.task_id != the_task.task_id]
            return False

        else:
            self.num_exited += 1
            self.priority_queue = [q for q in self.priority_queue if q.task_id != the_task.task_id]
            self.events_queue = [q for q in self.events_queue if q.task_id != the_task.task_id]
            if len(self.priority_queue) == 0:
                return False
            else:
                self.events_queue = [q for q in self.events_queue if q.task_id != self.priority_queue[0].task_id]
                return event(self.priority_queue[0].task_id, event_time + self.priority_queue[0].service_time, 2)

    def handle_tasks(self):
        the_event, self.events_queue = self.events_queue[0], self.events_queue[1:]
        the_new_event = self.handle_events(self.all_tasks[the_event.task_id], the_event.event_time, the_event.status)
        self.add_event(the_new_event)


class Simulator:
    def __init__(self, lambda_, mu, theta, queue_size, expdrop = True ):
        self.lambda_ = lambda_
        self.mu = mu
        self.theta = theta
        self.queue_size = queue_size
        self.system = system(self.queue_size)
        self.expdrop = expdrop

    def run(self, num_customers):
        current_time = 0
        task_id = 0
        while task_id < num_customers:
            if len(self.system.events_queue) == 0 or  current_time < self.system.events_queue[0].event_time:
                new_task = task(task_id, current_time, self.mu, self.theta, self.expdrop)
                self.system.add_task(new_task)
                current_time += rnd.exponential(1 / self.lambda_)
                task_id += 1
            else:
                self.system.handle_tasks()
        return self.system.num_blocked / num_customers, self.system.num_dropped / num_customers

#theta = int(input("Enter initial value of theta: "))
#mu,theta = input("Enter comma-separated values of mu and Theta: ").split(",") # for fixed values comment last line
while lambda_ < 20:

   if __name__ == "__main__":
        simulator = Simulator(lambda_, mu, theta, queue_size, expdrop)
        analyzer = Analyzer(lambda_, mu, theta, queue_size, expdrop)
        simulation_pb, simulation_pd = simulator.run(num_customers)
        analytic_pb, analytic_pd = analyzer.run()

        print(f"[Lambda]:{lambda_:.2f}    [Theta]:{theta:.10f}  [Simulation Pb]:{simulation_pb:.10f}  [Analytic Pb]:{analytic_pb:.10f}  [Simulation Pd]:{simulation_pd:.10f}  [Analytic Pd]:{analytic_pd:.10f}")
        lambda_ += 0.5 # for inrease by 0.1 factor or 5
        theta += rnd.exponential(2) # Increase theta values
