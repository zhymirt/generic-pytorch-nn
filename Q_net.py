from neural import *
import random

class Q_net(neural):

    def __init__(self, layers=None, optimizer=None, loss_fn=nn.MSELoss(), device=torch.device('cpu'), q_values={}):
        super(Q_net, self).__init__()
        self.q_values = q_values

    def getQValue(self, state, action):
        if (state, action) in self.q_values:
            return self.q_values[(state, action)]
        else:
            return 0.0

    def best_value(self, state):
        actions = self.getLegalActions(state)
        if actions is None or not actions:
            return 0.0
        max_val = -1e99
        for action in actions:
            q_value = self.getQValue(state, action)
            max_val = max(max_val, q_value)
        return max_val

    def best_action(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        best_action = []
        actions = self.getLegalActions(state)
        if actions is None or len(actions) == 0:
          return None
        max_val = -1e99
        for action in actions:
          qval = self.getQValue(state,action)
          if qval > max_val:
            best_action.clear()
            best_action.append(action)
            max_val = qval
          elif qval == max_val:
            best_action.append(action)
          # max_val = max(max_val,qval)
        return random.choice(best_action)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        pickRandom = random.random() < self.epsilon
        return random.choice(legalActions) if pickRandom else self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        discount = self.discount
        alpha = self.alpha
        actions = self.getLegalActions(nextState)
        stateVal = -1e99 if actions is not None and len(actions) > 0 else 0.0
        for a in actions:
          stateVal = max(stateVal,0) if self.qvalues[(nextState,a)] is None else max(stateVal,self.qvalues[(nextState,a)])
        sample = reward + discount * stateVal
        self.qvalues[(state,action)] = ((1-alpha)*self.qvalues[(state,action)])+(sample * alpha)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)
