# gym_nethack

OpenAI Gym-like environment for NetHack.

Current environments:

* basic combat, where you can specify a monster and various player attributes (inventory, experience level, strength, etc.), and the player and monster will face off in a small rectangular room.
* exploration of a level absent any items, monsters, locked doors, boulders.
* simplified combined combat/exploration ("level") environment with monsters and items present.

Note: This is not a typical OpenAI Gym environment. It is more like an OpenAI gym plus a wrapper that deals with setting up the learning model and policy. Environment, model and policy parameters are all located in the configs.py file.

## Installation

1. Download this repository.
2. Install Python (v3+) and the following Python libraries: pyzmq, dill, gym, keras, numpy.
3. Download this [modified version of keras-rl](https://github.com/campbelljc/keras-rl). Place the rl folder into the libs directory of this repo. Do a find-and-replace in the rl folder of "from rl" and change to "from libs.rl".
4. Download and build the [zmq library](http://zeromq.org/intro:get-the-software).
5. Download this [modified version of NetHack](https://github.com/campbelljc/NetHack). Follow the instructions for building NetHack from source, but first make the following modifications:
  * In nethack/include/config.h, edit line 320 to specify the path to the nethack/game directory. E.g., "/Users/jcampbell/Downloads/nethack/game".
  * (Mac/Linux): After the Makefile in the main NetHack directory is created during the regular NetHack build instructions, open it and make the following changes before continuing to build NetHack:
    * Change PREFIX from "/usr" to the path to your NetHack directory, e.g., "/Users/jcampbell/Downloads/nethack".
    * Change the GAMEUID to your current username, and GAMEGRP to staff.
    * Change HACKDIR to the full path to the "nethack/game" directory, and INSTDIR to the full path to the "nethack/install" directory.
	* Search for the line "sed -e 's;/usr/games/lib/nethackdir;$(HACKDIR);'" and replace /usr/games/lib/nethackdir with the full path to the "nethack/game" directory.	
  * (Windows): After the Makefile in the main NetHack directory is created during the regular NetHack build instructions, open the file "src/Makefile.gcc" and make the following changes before continuing to build NetHack:
    * Look for the "$(GAMEDIR)/NetHack.exe" rule. At the end of the $(link) command, add the full path to the zmq\bin\libzmq.dll file built in the step 4.
	* Look for the "$(O)allmain.o" rule and add the full path to the zmq\include\zmq.h file.
  * After making these changes, you should then be able to continue with the regular NetHack installation instructions.
6. Edit the file gym\_nethack/conn.py. At the top, the relative path to the NetHack directory and executable are listed. Modify them (only the ones for your OS) to where you have stored the nethack directory.

## Running

First, you have to decide on the particular gym environment (combat/exploration/level). Then, look in gym\_nethack/configs.py and choose, modify or add a config for that particular environment. (The set\_config method of each environment and policy describes the arguments that can be passed in.) Note the ID of the config (its index into the config array), which is in a comment above each config. You may have to alter the last line of the config file to point to the config array for the environment you chose. Then, inside the root repo directory, issue the following commands in **two separate console windows**:

* python3 -m gym_nethack.nhdaemon CONFIGNUM
* python3 ngym.py CONFIGNUM

That will start the daemon and the training script. The daemon runs as a separate process since memory issues arise if the train script launches a bunch of NH processes (even if they are perpetually closed).

You may want to adjust the VERBOSE variable at the top of gym\_nethack/misc.py to output much less stuff on the console.

The ngym.py file also has the capability to run multiple agents on multiple NetHack processes in parallel (e.g., for parameter grid search). I will have to document this in future, although there are some comments in the file already.

## Environments

This information is mostly in the documentation but I have reproduced it here with some additional clarifications for ease of use.

### Combat environment

**Description**: One player vs one monster in a closed-off square room.

**Episode end**: Ends on player or monster death. (NetHack will send a signal when monster dies.)

**Reward**: Positive for monster death, negative for player death.

**Actions**: approach monster, attach monster, line up with monster (for ranged attacks), wait, move in random direction, move towards ammo, pick up ammo, equip bare hands, equip [x] weapon, wear [x] armor, wear [x] ring on empty finger, quaff [x] potion, throw [x] potion at monster, throw [x] projectile at mnster, read [x] scroll, zap [x] wand.

**State**: list of monsters present (vector); number of monsters (categorical); player role (one-hot); player alignment (one-hot); if player lost health this game (boolean); if player has lycanthropy (boolean); if player is invisible (boolean); player stats (HP, PW, cLvl, str, dex, const, int, wis, cha, AC) & dungeon level (normalized vector); player status effects (vector); player inventory (vector); player current equipment: weapons, armor and rings (vector); player's normalized distance to monster, whether monster and/or player approached each other and/or changed positions in the last turn (vector); if monster is in player's line of fire, if ammo is on ground and if player is standing on ammo (vector of booleans).

**Parameters that be specified**: List of monsters from which each episode will select one randomly; difference between monster and player level; guaranteed player initial equipment; whether to additionally sample items from another list for player's inventory, and how many to sample; whether to give the player a fixed AC; what dungeon level the fight should take place on, affecting monster difficulty; whether only weapons can be used or if armor can also be equipped; whether to use a tabular (discrete) state representation.

#### Combat policies

**ApproachAttackPolicy**: Heuristic policy that randomly equips a weapon (and armor, if specified), then approaches the monster and attacks it at close range. (If ranged weapon equipped, it will attack from a distance instead of approaching.)

**ApproachAttackItemPolicy**: Heuristic policy that randomly equips a weapon (and armor, if specified), then uses a random item with probability 0.25, and approaches the monster and attacks it at close range with probability 0.75. (If ranged weapon equipped, it will attack from a distance instead of approaching.)

**FireAntPolicy**: Heuristic policy for fire ant, as described in my thesis.

**LinearAnnealedPolicy**: with inner policy as EpsGreedyPossibleQPolicy. Standard deep Q-learning policy.

### Exploration environment

**Description**: A standard, full NetHack map, absent any monsters or hunger limits.

**Episode end**: When the exploration policy decides it has finished exploring (done\_exploring()).

**Reward**: Not implemented (only heuristic policies used so far).

**Actions**: 8 movement directions (including diagonally).

**State**: Not implemented as such, but the NetHackInfo object contains all information about the current map.

**Parameters that be specified**: whether to disable generation of secret doors/corridors; whether to disable map randomization so that the same sequence of maps are always encountered.

#### Exploration policies

**GreedyExplorationPolicy**: Map exploration policy that always visits closest frontier to player until no frontiers remain.

**SecretGreedyExplorationPolicy**: Extension of greedy exploration algorithm to support searching for secret doors and corridors. Searches every room wall and dead-end corridor for a specified number of turns.

**OccupancyMapPolicy**: Occupancy map exploration algorithm for NetHack. Described in the paper "Exploration with Secret Discovery", J. Campbell & C. Verbrugge, IEEE Transactions on Games, 2018.

### Level environment

**Description**: An extension of the exploration environment, where monsters spawn randomly, and going down the stairs will enter the next level. The environment can detect when a monster is present, and switch to using combat reward/state until the monster is no longer visible, at which point will revert to exploration.

**Episode end**: Only on player death.

**Reward**: Uses combat rewards.

**Actions**: Uses both exploration and combat actions.

**State**: Same as above environments.

**Parameters that be specified**: whether to disable generation of secret doors/corridors; whether to disable map randomization so that the same sequence of maps are always encountered.

#### Level policies

**LevelPolicy**: Policy that can explore a level and enter combat with monsters, by using a different sub-policy for combat and exploration. It will default to the exploration policy, but engage the combat policy when a monster is visible and close to the player. When the exploration policy is finished exploring the level, this policy will move towards the down stair (looking under monsters if necessary) and go down to the next level.

## Code documentation

All environment and policy methods are commented with docstrings. You can view the documentation [here](http://www.campbelljc.com/research/nethackrl/docs), with the page on environment and policy methods located [here](http://www.campbelljc.com/research/nethackrl/docs/gym_nethack.html).

The framework is also mentioned in small part in my thesis, which you can access [here](http://campbelljc.com/research/thesis/).

## Known issues

* NetHack will sometimes give an unexpected message during combat and the episode will have to be terminated (it will be recorded as Terminals.CONN\_ERROR). E.g., if you try to move while paralyzed. These are corner cases and happen every once in a while depending on the monster you are facing and the items you have. Exploration is stable however.
* The full level environment has a few bugs. In particular, NetHack occasionally will crash and I haven't found the time to pinpoint where just yet. I sadly believe it to be a memory issue due to some code I introduced. Also, there are a couple of bugs with the default exploration algorithm getting stuck in shops (better shop parsing needed).
* Records may not be saved properly in the level environment.

## TODO

[ ] Check if records are being saved properly in level environment.

[ ] Add the occupancy map algorithm for detecting secret areas as described in paper (have to refactor it).

[ ] Update keras-rl branch to current version.

[ ] Add policy name to save directory name.

[ ] Document how to run multiple NH processes in parallel.

## Changes to NetHack

Several changes were made to NetHack to support integration with the gym environment, including removing animations that waste time for automated play, and quick hacks to stop NetHack asking for input outside of the main command loop.

You can view all the changes made by using the [GitHub compare function on my NetHack repository](https://github.com/campbelljc/NetHack/compare/585e9f1b35fda7b47f8d27d12f7e93e12a69a7bc...campbelljc:0631fec4c5ca6f2177841f361a8235667692a744).

Flags to enable/disable behaviors:

* Integration with the zmq library to send screen output (incl. inventory screen) and receive commands from a port instead of the console. Port is specified as a command-line option. This modification allows for much faster communication with the gym environment than traditional approaches which use console emulation.
* Added support for a one-on-one monster-vs-player arena combat. This involved changing dat/bigroom.des (used out of convenience) to a smaller room and adding several option flags to allow for specification of monster, etc. Can be enabled/disabled using a special options flag.
* Flag to enable or disable creation of secret doors/corridors.
* Flag to enable or disable generation of items and/or monsters.

Disabled mechanics (for now?):

* All objects generate as fully identified.
* Hunger set to not hungry every turn.
* Pushed special levels (gnomish mines branch, oracle level, big room level) to further in the game (quick hack so we don't deal with their layout as of yet).
* Disabled several input routines outside of the main loop, e.g., asking to name an item, which ring-finger to use (defaults to left), ask for item removal, ask for multi-item pickup, teleport control, paranoid attack query.
* Disable locked doors and unlit rooms; stop boulders, iron bars, traps or vaults from being generated; can always squeeze through passages.
* Never be encumbered.

Smaller changes:

* Top line always includes names of currently-visible monsters. (Can be done by a human player by pressing a key each turn.)
* Number of items in inventory, number of rooms, number of secret doors/corridors, and number of squares explored shown on bottom line. (Used only for after-game statistics purposes.)
* In addition to screen output, the glyph that the player is currently standing on is also sent, as well as the player's current x,y coordinates. (Due to difficulty in establishing door openings and also problems with player invisibility.)
* Allow for map seed to be specified as command-line option. (To enable reproducibility of experiments.)
* Disabled many "You feel" messages. (To remove top-line clutter.)
* Changed open doors to "." character, grave to "\_", sink to "\", spellbook to "&" to allow for easier recognition of topographical characters. (For more stable map parsing.)
* Added parsing of a "*" character in wizkit file to specify what items to equip at start. (For experiments.)
* Stopped several animations from playing to speed up playing. (Faster running time.)

## Contributing

Please feel free to submit any pull requests or issues if you have any bug fixes, feature enhancements or suggestions.

## References / citing

If you want to use this framework in a paper, I would be grateful if you could cite one or both of the following papers. I created this framework while doing the research for these papers. You could also cite this repository alternatively.

* Jonathan Campbell, Clark Verbrugge. "Exploration in NetHack With Secret Discovery." in IEEE Transactions on Games. 2018. To appear in a future issue. DOI: 10.1109/TG.2018.2861759.
  * [official site](https://doi.org/10.1109/TG.2018.2861759)
  * [paper pdf](http://campbelljc.com/research/expl_paper_tog.pdf)
  * [bibtex](http://campbelljc.com/research/expl_paper_tog.bib)

* Jonathan Campbell, Clark Verbrugge. "Learning Combat in NetHack." AIIDE'17: Proceedings of the 13th AAAI Conference on Artificial Intelligence and Interactive Digital Entertainment. Snowbird City, Utah. October 2017. pp.16–22.
  * [official site](https://aaai.org/ocs/index.php/AIIDE/AIIDE17/paper/view/15815)
  * [paper pdf](http://gram.cs.mcgill.ca/papers/campbell-17-learning.pdf)
  * [bibtex](http://gram.cs.mcgill.ca/bib/campbell-17-learning.bib)