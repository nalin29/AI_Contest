# baselineAgents.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from os import stat
from capture import GameState, noisyDistance
from captureAgents import CaptureAgent
from captureAgents import AgentFactory
import distanceCalculator
import random, time, util
from game import Configuration, Directions
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
      self.depth = 1
      self.initializeParticles(gameState)

  def chooseAction(self, gameState):

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()

    # particle filter updates
    self.elapseTime(gameState)
    self.observeState(gameState)
    self.getBeliefDistribution(gameState)
    self.displayDistributionsOverPositions(self.beliefs)

    # generate the most likely position of every opponent
    self.getMLP(gameState)

    # alter gamestate to include opponents in the most likely positions
    alteredState = gameState.deepCopy()
    for o in self.getOpponents(gameState):
      alteredState.data.agentStates[o].configuration = Configuration(self.beliefMLP[o], None)

    # run expecitmax on the altered state
    return self.expectimax(alteredState,0,0)

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

  def evaluate(self, gameState):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState)
    weights = self.getWeights(gameState)
    return features * weights

  def getFeatures(self, gameState):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()
    features['score'] = self.getScore(gameState)
    return features

  def getWeights(self, gameState):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'score': 1.0}

  def initializeParticles(self, gameState):
    for o in self.getOpponents(gameState):
      particles = []
      for i in range(self.numParticles):
        particles.append(gameState.getInitialAgentPosition(o))
      self.particles[o] = particles

  def initializeParticlesUniformally(self, opponent):
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
          for i in range(self.numParticles):
            self.particles[o][i] = util.sample(weights)
        
    
  def elapseTime(self, gameState):
    alteredState = gameState.deepCopy()
    opponents = self.getOpponents(gameState)
    for o in opponents:
      newParticles = []
      for oldParticle in self.particles[o]:
          pos = oldParticle
          newPosDist = util.Counter()
          alteredState.data.agentStates[o].configuration = Configuration(pos, None)
          actions = alteredState.getLegalActions(o)
          states = [self.getSuccessor(alteredState, o, a) for a in actions]
          for state in states:
            newPos = state.data.agentStates[o].configuration.pos
            newPosDist[newPos] += 1
          newPosDist.normalize()
          newParticles.append(util.sampleFromCounter(newPosDist))
      self.particles[o] = newParticles

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

  def getMLP(self, gameState):
    self.beliefMLP = []
    for i in range(gameState.getNumAgents()):
     self.beliefMLP.append(self.beliefs[i].argMax())

  def expectimax(self, gameState, depth, index):
    indexTable =[self.index]+self.getOpponents(gameState)
    index %= len(indexTable)
    agent = indexTable[index]
    # max depth return
    if depth == self.depth:
      return self.evaluate(gameState)
    # possible actions
    actions = gameState.getLegalActions(agent)
    # if at terminating state return
    if len(actions) == 0:
      return self.evaluate(gameState)
    # calculate next depth
    nextDepth = depth if not index == len(indexTable) - 1 else depth + 1
    # expectimax of all actions
    expectimaxActions = [self.expectimax(self.getSuccessor(gameState, agent, a), nextDepth, index+1) for a in actions]
    # pacman (max state)
    if agent == self.index:
      # get index of action with max expectimax
      maxArg = max(range(len(actions)), key= lambda i: expectimaxActions[i])
      # if at starting of recursion then return action
      if(depth == 0):
        return actions[maxArg]
      # for recursion return max
      return expectimaxActions[maxArg]
    # chance node (ghost) return sum of weighted utilty of each action (In this case they are equally likely so just return avg)
    return float(sum(expectimaxActions))/float(len(expectimaxActions))


class OffensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState):
    features = util.Counter()

    features['score'] = self.getScore(gameState)

    myPos = gameState.getAgentState(self.index).getPosition()

    # get scared and not scared ghosts
    nonScaredGhosts = []
    scaredGhosts = []
    for o in self.getOpponents(gameState):
      if (not gameState.getAgentState(o).isPacman) and gameState.getAgentState(o).scaredTimer <= 0:
        nonScaredGhosts.append(o)
      elif (not gameState.getAgentState(o).isPacman) and gameState.getAgentState(o).scaredTimer > 0:
        scaredGhosts.append(o)

    # Compute distance to the nearest food
    foodList = self.getFood(gameState).asList()
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance
    
    # food left
    features['foodLeft'] = len(foodList)

    # min distance to nearest not-scared ghost
    if len(nonScaredGhosts) > 0:
      dists = [self.getMazeDistance(myPos, gameState.getAgentState(o).getPosition()) for o in nonScaredGhosts]
      distanceToOpponent = min(dists)
      features['distanceToOpponent'] = float('-inf') if distanceToOpponent == 0 else 1.0/distanceToOpponent

    # number of power pellets
    powerPellets = None
    if self.red:
      powerPellets = gameState.getBlueCapsules()
    else:
      powerPellets = gameState.getRedCapsules()
    features['powerPelletsLeft'] = len(powerPellets)

    # min distance to scared ghost
    if len(scaredGhosts) > 0:
      dists = [self.getMazeDistance(myPos, gameState.getAgentState(o).getPosition()) for o in scaredGhosts]
      distanceToGhost = min(dists)
      features['distanceToScared'] = distanceToGhost
    
    return features

  def getWeights(self, gameState):
    return {'score': 100, 'distanceToFood': -3, 'distanceToOpponent':-5, 'foodLeft': -10, 'powerPelletsLeft': -100, 'distanceToScared': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState):
    features = util.Counter()
    successor = gameState

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders
    invaders = [o for o in self.getOpponents(gameState) if gameState.getAgentState(o).isPacman]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, gameState.getAgentState(o).getPosition()) for o in invaders]
      features['invaderDistance'] = min(dists)

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

    """ if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1 """

    return features

  def getWeights(self, gameState):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -50, 'stop': -20, 'reverse': -2, 'foodDist': -1, 'pelletDist': -1}


