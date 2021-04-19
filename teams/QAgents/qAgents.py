# baselineAgents.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from ast import Index
from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import distanceCalculator
import random, time, util
from game import Directions, Configuration
import keyboardAgents
import game
from util import nearestPoint

#############
# FACTORIES #
#############

NUM_KEYBOARD_AGENTS = 0
class BaselineAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, isRed, first='offense', second='defense', rest='offense'):
    AgentFactory.__init__(self, isRed)
    self.agents = [first, second]
    self.rest = rest

  def getAgent(self, index):
    if len(self.agents) > 0:
      return self.choose(self.agents.pop(0), index)
    else:
      return self.choose(self.rest, index)

  def choose(self, agentStr, index):
    if agentStr == 'keys':
      global NUM_KEYBOARD_AGENTS
      NUM_KEYBOARD_AGENTS += 1
      if NUM_KEYBOARD_AGENTS == 1:
        return keyboardAgents.KeyboardAgent(index)
      elif NUM_KEYBOARD_AGENTS == 2:
        return keyboardAgents.KeyboardAgent2(index)
      else:
        raise Exception('Max of two keyboard agents supported')
    elif agentStr == 'offense':
      return OffensiveQAgent(index)
    elif agentStr == 'defense':
      return DefensiveReflexAgent(index)
    else:
      raise Exception("No staff agent identified by " + agentStr)

class AllOffenseAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, **args):
    AgentFactory.__init__(self, **args)

  def getAgent(self, index):
    return OffensiveQAgent(index)

class OffenseDefenseAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, **args):
    AgentFactory.__init__(self, **args)
    self.offense = False

  def getAgent(self, index):
    self.offense = not self.offense
    if self.offense:
      return OffensiveQAgent(index)
    else:
      return DefensiveReflexAgent(index)

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent, object):

  def __init__(self, index, timeForComputing=0.1):
      super(ReflexCaptureAgent, self).__init__(index, timeForComputing=timeForComputing)

  def registerInitialState(self, gameState):
      super(ReflexCaptureAgent, self).registerInitialState(gameState)
      # generate legal coordinates
      self.legalPos = []
      for x in range(gameState.data.layout.width):
        for y in range(gameState.data.layout.height):
          if not gameState.hasWall(x,y):
            self.legalPos.append((x,y))
      # intialize particle filter
      self.particles = {}
      self.numParticles = 200
      self.initializeParticles(gameState)

  """
  A base class for reflex agents that chooses score-maximizing actions
  """
  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    #start = time.time()

    # particle filter updates
    self.observeState(gameState)
    self.getBeliefDistribution(gameState)
    self.getMLP(gameState)
    self.displayDistributionsOverPositions(self.beliefs)
    self.elapseTime(gameState)

    values = [self.evaluate(gameState, a) for a in actions]
    #print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    return random.choice(bestActions)

  def getSuccessor(self, gameState, index, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(index, action)
    pos = successor.getAgentState(index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, self.index, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

  # initialize particles at starting positions
  def initializeParticles(self, gameState):
    for o in self.getOpponents(gameState):
      particles = []
      for i in range(self.numParticles):
        particles.append(gameState.getInitialAgentPosition(o))
      self.particles[o] = particles

  # intializes particles uniformally across all possible states for an opponent
  def initializeParticlesUniformally(self, opponent):
    particles = []
    random.shuffle(self.legalPos)
    for i in range(self.numParticles):
      particles.append(self.legalPos[i % len(self.legalPos)])
    self.particles[opponent] = particles

  # update particles on observation based on true distance and noisy distance
  def observeState(self, gameState):
    myState = gameState.getAgentState(self.index)
    opponents = self.getOpponents(gameState)
    myPos = myState.getPosition()
    agentPositions = {i: self.getCurrentObservation().getAgentPosition(i) for i in opponents}
    noisyDistances = {i:self.getCurrentObservation().getAgentDistances()[i] for i in opponents}

    for o in opponents:
      if not agentPositions[o] is None:
        for i in range(self.numParticles):
          self.particles[o][i] = agentPositions[o]
      elif noisyDistances[o] is None:
        for i in range(self.numParticles):
          self.particles[o][i] = gameState.getInitialAgentPosition(o)
      else:
        weights = util.Counter()
        for particle in self.particles[o]:
          trueDistance = util.manhattanDistance(myPos, particle)
          weights[particle] += gameState.getDistanceProb(trueDistance, noisyDistances[o])
        if weights.totalCount() <= 0:
          self.initializeParticlesUniformally(o)
        else:
          weights.normalize()
          #self.particles[o] = util.nSample(weights, weights.keys,self.numParticles)
          for i in range(self.numParticles):
            self.particles[o][i] = util.sample(weights)

  # elapse time on particles
  def elapseTime(self, gameState):
    # copy state to be altered
    alteredState = gameState.deepCopy()
    opponents = self.getOpponents(gameState)
    # loop through sets of particles
    for o in opponents:
      newParticles = []
      # loop through particles
      for oldParticle in self.particles[o]:
        # alter game state to have opponent in position of particle
        pos = oldParticle
        newPosDist = util.Counter()
        alteredState.data.agentStates[o].configuration = Configuration(pos, None)
        actions = alteredState.getLegalActions(o)
        states = [self.getSuccessor(alteredState, o, a) for a in actions]
        # choose new pos at random
        newParticles.append(random.choice(states).data.agentStates[o].configuration.pos)
      self.particles[o] = newParticles
    
  # gets the belief distribution of particles
  def getBeliefDistribution(self, gameState):
    beliefs = []
    for i in range(gameState.getNumAgents()):
      belief = util.Counter()
      if i in self.getOpponents(gameState):
        for state in self.particles[i]:
          belief[state] += 1
        belief.normalize()
        beliefs.append(belief)
      else:
        belief[self.getCurrentObservation().getAgentPosition(i)] = 1
        beliefs.append(belief)
    self.beliefs = beliefs

  # get most likely pos with particles
  def getMLP(self, gameState):
    self.beliefMLP = []
    for i in range(gameState.getNumAgents()):
      self.beliefMLP.append(self.beliefs[i].argMax())


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, self.index, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # non-invaders/invaders
    invaders = []
    nonInvaders = []
    for i in self.getOpponents(successor): 
      if successor.getAgentState(i).isPacman:
        invaders.append(i)
      else:
        nonInvaders.append(i)

    # Computes distance to invaders we can see
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, self.beliefMLP[i]) for i in invaders]
      features['invaderDistance'] = min(dists)
    
    # min dist to non-invader
    if len(nonInvaders) > 0:
      dists = [self.getMazeDistance(myPos, self.beliefMLP[i]) for i in nonInvaders]
      features['nonInvaderDistance'] = min(dists)

    # avg dist to own food
    foodList = self.getFoodYouAreDefending(successor).asList()
    if len(foodList) > 0:
      avg = float(sum([self.getMazeDistance(myPos, food) for food in foodList]))/float(len(foodList))
      features['foodDist'] = avg

    # min dist to own food pellet
    pellets = self.getCapsulesYouAreDefending(successor)
    if len(pellets) > 0:
      dist = min([self.getMazeDistance(myPos, pellet) for pellet in pellets])
      features['pelletDist'] = dist

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    features['bias'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'nonInvaderDistance': -0.5, 'bias': 1, 'foodDist': -1, 'pelletDist':-1}

class OffensiveQAgent(CaptureAgent):
  
