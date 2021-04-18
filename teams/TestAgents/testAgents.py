# baselineAgents.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from capture import GameState, noisyDistance
from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import distanceCalculator
import random, time, util
from game import Directions
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
      return OffensiveReflexAgent(index)
    elif agentStr == 'defense':
      return DefensiveReflexAgent(index)
    else:
      raise Exception("No staff agent identified by " + agentStr)

class AllOffenseAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, **args):
    AgentFactory.__init__(self, **args)

  def getAgent(self, index):
    return OffensiveReflexAgent(index)

class OffenseDefenseAgents(AgentFactory):
  "Returns one keyboard agent and offensive reflex agents"

  def __init__(self, **args):
    AgentFactory.__init__(self, **args)
    self.offense = False

  def getAgent(self, index):
    self.offense = not self.offense
    if self.offense:
      return OffensiveReflexAgent(index)
    else:
      return DefensiveReflexAgent(index)

##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent, object):
  """
  A base class for reflex agents that chooses score-maximizing actions
  """
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
      for o in self.getOpponents(gameState):
        self.initializeParticles(o)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    # distance to nearest Ghost
    self.observeState(gameState)
    beliefs = self.getBeliefDistribution(gameState)
    self.displayDistributionsOverPositions(beliefs)
    self.elapseTime(gameState)
    self.beliefMLP = {o: beliefs[o].argMax() for o in self.getOpponents(gameState)}
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
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
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}
  
  def initializeParticles(self, opponent):
    particles = []
    random.shuffle(self.legalPos)
    for i in range(self.numParticles):
      particles.append(self.legalPos[i % len(self.legalPos)])
    self.particles[opponent] = particles

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
      else:
        weights = util.Counter()
        for particle in self.particles[o]:
          trueDistance = util.manhattanDistance(myPos, particle)
          weights[particle] += gameState.getDistanceProb(trueDistance, noisyDistances[o])
        if weights.totalCount() <= 0:
          self.initializeParticles(o)
        else:
          weights.normalize()
          for i in range(self.numParticles):
            self.particles[o][i] = util.sample(weights)
        
    
  def elapseTime(self, gameState):
    opponents = self.getOpponents(gameState)
    for o in opponents:
      newParticles = []
      for oldParticle in self.particles[o]:
          pos = oldParticle
          newPosDist = util.Counter()
          #stay
          newPosDist[pos] += 1
          # up
          if (pos[0], pos[1]+1) in self.legalPos:
            newPosDist[(pos[0], pos[1]+1)] += 1
          else:
            newPosDist[pos] +=1
          # down
          if (pos[0], pos[1]-1) in self.legalPos:
            newPosDist[(pos[0], pos[1]-1)] += 1
          else:
            newPosDist[pos] +=1
          # left
          if (pos[0]-1, pos[1]) in self.legalPos:
            newPosDist[(pos[0]-1, pos[1])] += 1
          else:
            newPosDist[pos] +=1
          # right
          if (pos[0]+1, pos[1]) in self.legalPos:
            newPosDist[(pos[0]+1, pos[1])] += 1
          else:
            newPosDist[pos] +=1
          newPosDist.normalize()
          newParticles.append(util.sampleFromCounter(newPosDist))
      self.particles[o] = newParticles

  def getBeliefDistribution(self, gameState):
    beliefs = []
    for i in range(gameState.getNumAgents()):
      if i in self.getOpponents(gameState):
        belief = util.Counter()
        for state in self.particles[i]:
          belief[state] += 1
        belief.normalize()
        beliefs.append(belief)
      else:
        beliefs.append(None)
    return beliefs



class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)

    # Compute distance to the nearest food
    foodList = self.getFood(successor).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    enemies = [(i,successor.getAgentState(i)) for i in self.getOpponents(successor)]
    ghosts = [(i, a) for i, a in enemies if not a.isPacman]
    if len(ghosts) > 0:
      dists = [self.getMazeDistance(myPos, self.beliefMLP[i]) for i, a in ghosts]
      distanceToOpponent = min(dists)
    features['distanceToOpponent'] = float('-inf') if distanceToOpponent == 0 else 1.0/distanceToOpponent
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -4, 'distanceToOpponent':-5}


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [(i,successor.getAgentState(i)) for i in self.getOpponents(successor)]
    invaders = [(i, a) for i, a in enemies if a.isPacman]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, self.beliefMLP[i]) for i, a in invaders]
      features['invaderDistance'] = min(dists)
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -200, 'stop': -100, 'reverse': -2}

