# AI_Contest

This project was the optional extra credit for Artificial Intelligence survey course. The game is  set up such that there are two sides of the map.
The goal is to eat all pellets on the opponenets side of the map. Agents becomes pacman when crossing to the opponents side and Ghosts on their side.
To acheive this I implemented from scratch a Q-learning agent. This agent used a variety of features that could be tweaked to estimate the value of next states.
The agent was given rewards based on the number of pellets remaining, winning, eating ghosts and the distance from the closest pellet. 
Additionally, a negative reward was applied when losing, being eaten or being close to a ghost. 
Ghosts or oppponents can be directly detected when within 5 blocks. To improve this we implemented a particle filter then we could estimate the locations of the opponents using peaks.

Q-Learning was run for several hundred rounds against the provided AI and across several maps. Then hand made tweaks were made to some of the features to improve performance. The result was a fifth place finish and a decent Q-learning reinforcement agent.
