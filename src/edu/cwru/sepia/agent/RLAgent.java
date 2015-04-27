package edu.cwru.sepia.agent;

import edu.cwru.sepia.action.Action;
import edu.cwru.sepia.action.ActionResult;
import edu.cwru.sepia.action.TargetedAction;
import edu.cwru.sepia.environment.model.history.DamageLog;
import edu.cwru.sepia.environment.model.history.DeathLog;
import edu.cwru.sepia.environment.model.history.History;
import edu.cwru.sepia.environment.model.history.History.HistoryView;
import edu.cwru.sepia.environment.model.state.State;
import edu.cwru.sepia.environment.model.state.State.StateView;
import edu.cwru.sepia.environment.model.state.Unit;
import edu.cwru.sepia.environment.model.state.Unit.UnitView;
import edu.cwru.sepia.util.DistanceMetrics;

import java.io.*;
import java.util.*;

public class RLAgent extends Agent {

	/**
	 * Set in the constructor. Defines how many learning episodes your agent should run for.
	 * When starting an episode. If the count is greater than this value print a message
	 * and call sys.exit(0)
	 */
	public final int numEpisodes;
	public int numEpisodesPlayed;
	/*
	 * when true, weights will not be updated and Q function will always be used to pick actions
	 * (no random action picking) (essentially determines whether we're in learning or evaluation mode)
	 */
	public boolean freeze;
	//each of our footmen has its id mapped to its cumulative discounted reward
	//TODO make sure these rewards are discounted
	public Map<Integer, Double> rewardMap;
	//ensures the +100 reward can only be claimed once for killing an enemy, so no other unit can also claim
	//the reward for kill the same enemy. (can occur when two or more footmen attack the same enemy at once)
	public LinkedList<Integer> enemyBlackList;
	//stores the set of actions last sent out to the footmen
	public Map<Integer, Action> currentActionMap;
	//store the set the actions sent out before that
	public Map<Integer, Action> previousActionMap;
	//the rewards from the 5 evaluation rounds
	public Double[] evaluationRewards;
	//counter to keep track of which evaluation round we're on
	public int evalRoundCounter;
	//the list of average rewards (one for each of evaluation session)
	public List<Double> avgRewards;

	/**
	 * List of your footmen and your enemies footmen
	 */
	private List<Integer> myFootmen;
	private List<Integer> enemyFootmen;

	/**
	 * Convenience variable specifying enemy agent number. Use this whenever referring
	 * to the enemy agent. We will make sure it is set to the proper number when testing your code.
	 */
	public static final int ENEMY_PLAYERNUM = 1;

	/**
	 * TODO Set this to whatever size your feature vector is.
	 */
	public static final int NUM_FEATURES = 5;

	/** Use this random number generator for your epsilon exploration. When you submit we will
	 * change this seed so make sure that your agent works for more than the default seed.
	 */
	//12345
	public final Random random = new Random(12345);

	/**
	 * Your Q-function weights.
	 */
	public Double[] weights;

	/**
	 * These variables are set for you according to the assignment definition. You can change them,
	 * but it is not recommended. If you do change them please let us know and explain your reasoning for
	 * changing them.
	 */
	public final double gamma = 0.9;
	public final double learningRate = .0001;
	public final double epsilon = .02;

	public RLAgent(int playernum, String[] args) {
		super(playernum);

		if (args.length >= 1) {
			numEpisodes = 65;//TODO revert: Integer.parseInt(args[0]);
			System.out.println("Running " + numEpisodes + " episodes.");
		} else {
			numEpisodes = 10;
			System.out.println("Warning! Number of episodes not specified. Defaulting to 10 episodes.");
		}

		//start with Q function frozen so that the first 5 rounds will be evaluation rounds
		//This will given us a good baseline for average cumulative reward (because the Q function
		//is garbage at the start)
		freeze = true;
		numEpisodesPlayed = 0;
		avgRewards = new LinkedList<>();
		evaluationRewards = new Double[5];

		boolean loadWeights = false;
		if (args.length >= 2) {
			loadWeights = Boolean.parseBoolean(args[1]);
		} else {
			System.out.println("Warning! Load weights argument not specified. Defaulting to not loading.");
		}

		if (loadWeights) {
			weights = loadWeights();
		} else {
			// initialize weights to random values between -1 and 1
			weights = new Double[NUM_FEATURES];
			for (int i = 0; i < weights.length; i++) {
				weights[i] = random.nextDouble() * 2 - 1;
			}
		}
	}

	/**
	 * We've implemented some setup code for your convenience. Change what you need to.
	 */
	@Override
	public Map<Integer, Action> initialStep(State.StateView stateView, History.HistoryView historyView) {

		// Find all of your units
		myFootmen = new LinkedList<>();
		for (Integer unitId : stateView.getUnitIds(playernum)) {
			Unit.UnitView unit = stateView.getUnit(unitId);

			String unitName = unit.getTemplateView().getName().toLowerCase();
			if (unitName.equals("footman")) {
				myFootmen.add(unitId);
			} else {
				System.err.println("Unknown unit type: " + unitName);
			}
		}

		//clear this stuff every round
		rewardMap = new HashMap<>();
		enemyBlackList = new LinkedList<>();
		currentActionMap = null;
		previousActionMap = null;

		//initialize each footman's reward to 0
		for (Integer id : myFootmen) {
			rewardMap.put(id, 0.0);
		}

		// Find all of the enemy units
		enemyFootmen = new LinkedList<>();
		for (Integer unitId : stateView.getUnitIds(ENEMY_PLAYERNUM)) {
			Unit.UnitView unit = stateView.getUnit(unitId);

			String unitName = unit.getTemplateView().getName().toLowerCase();
			if (unitName.equals("footman")) {
				enemyFootmen.add(unitId);
			} else {
				System.err.println("Unknown unit type: " + unitName);
			}
		}

		return middleStep(stateView, historyView);
	}

	/**
	 * You will need to calculate the reward at each step and update your totals. You will also need to
	 * check if an event has occurred. If it has then you will need to update your weights and select a new 
	 * action.
	 *
	 * If you are using the footmen vectors you will also need to remove killed units. To do so use the 
	 * historyView
	 * to get a DeathLog. Each DeathLog tells you which player's unit died and the unit ID of the dead unit. 
	 * To get
	 * the deaths from the last turn do something similar to the following snippet. Please be aware that on 
	 * the first
	 * turn you should not call this as you will get nothing back.
	 *
	 * for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() -1)) {
	 *     System.out.println("Player: " + deathLog.getController() + " unit: " + deathLog.getDeadUnitID());
	 * }
	 *
	 * You should also check for completed actions using the history view. Obviously you never want a footman
	 *  just
	 * sitting around doing nothing (the enemy certainly isn't going to stop attacking). So at the minimum 
	 * you will
	 * have an even whenever one your footmen's targets is killed or an action fails. Actions may fail if the
	 *  target
	 * is surrounded or the unit cannot find a path to the unit. To get the action results from the previous 
	 * turn
	 * you can do something similar to the following. Please be aware that on the first turn you should not
	 *  call this
	 *
	 * Map<Integer, ActionResult> actionResults = historyView.getCommandFeedback(playernum, 
	 * stateView.getTurnNumber() - 1);
	 * for(ActionResult result : actionResults.values()) {
	 *     System.out.println(result.toString());
	 * }
	 *
	 * @return New actions to execute or nothing if an event has not occurred.
	 */
	@Override
	public Map<Integer, Action> middleStep(State.StateView stateView, History.HistoryView historyView) {

		//for each footman, calculate it's reward at this step and add it to it's total reward
		updateFootmenRewards(stateView, historyView);		
		cleanupDeadUnits(stateView, historyView);

		Map<Integer, Action> actionMap = new HashMap<>();

		if (eventHasOccurred(stateView, historyView)) {

			for (Integer id : myFootmen) {
				//reassign attack actions
				int enemyID = selectAction(stateView, historyView, id);

				//only update weights freeze == false
				if (!freeze) {
					weights = updateWeights(weights,
							calculateFeatureVector(stateView, historyView, id, enemyID), 
							rewardMap.get(id),
							stateView,
							historyView,
							id);
				}

				actionMap.put(id, Action.createCompoundAttack(id, enemyID));
			}
			previousActionMap = currentActionMap;
			currentActionMap = actionMap;
		}
		return actionMap;
	}

	/**
	 * determines whether the state has changed enough for an update.
	 * 
	 * TODO implement this more intelligently
	 */
	private boolean eventHasOccurred(StateView stateView, HistoryView historyView) {

		if (stateView.getTurnNumber() == 0) {
			//true on first turn
			//System.out.println("event has occurred: first turn");			
			return true;
		}
		//a death indicates a significant change
		if (historyView.getDeathLogs(stateView.getTurnNumber() - 1).size() > 0) {
			//System.out.println("event has occurred: somebody died");			
			return true;
		}

		//return true if any units are not in the middle of executing an action
		Map<Integer, ActionResult> actionResults =
				historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
		for (ActionResult result : actionResults.values()) {

			if(!result.getFeedback().toString().equals("INCOMPLETE")) {
				//				System.out.println("event has occurred: Somebody's action was: " +
				//						result.getFeedback().toString());			
				return true;
			}
		}
		return false;
	}

	/**
	 * Here you will calculate the cumulative average rewards for your testing episodes. If you have just
	 * finished a set of test episodes you will call out testEpisode.
	 *
	 * It is also a good idea to save your weights with the saveWeights function.
	 */
	@Override
	public void terminalStep(State.StateView stateView, History.HistoryView historyView) {

		//add the rewards for the last move
		updateFootmenRewards(stateView, historyView);		
		//remove the dead people so we can see who won and by how much
		cleanupDeadUnits(stateView, historyView);

		//say who wins
		if (myFootmen.size() == 0) {
			System.out.println("You Lose. Enemy has " + enemyFootmen.size() + " footmen remaining");
		}
		else if (enemyFootmen.size() == 0) {
			System.out.println("You Win. You have " + myFootmen.size() + " footmen remaining");
		}
		else {
			System.err.println("ERROR: Winner unknown");
		}

		numEpisodesPlayed++;
		System.out.println(numEpisodesPlayed + " episodes have been played\n");

		//count the total reward if we're in evaluation mode (freeze == true)
		if (freeze) {
			Double sum = 0.0;
			for (Double reward : rewardMap.values()) {
				sum += reward;
			}
			evaluationRewards[evalRoundCounter] = sum;
			evalRoundCounter++;
		}

		//Q function starts frozen for the first 5 rounds, so freeze needs to be set to true every 15 rounds
		//so freezing will occur at round 0, 15, 30, etc
		if (numEpisodesPlayed % 15 == 0) {
			System.out.println("Entering evaluation mode, freezing Q function");
			freeze = true;
			evaluationRewards = new Double[5];
			evalRoundCounter = 0;
		}
		//similarly unfreezing will occur at round 5, 20, 35, etc
		else if ((numEpisodesPlayed - 5) % 15 == 0) {
			freeze = false;

			//TODO cumulative rewards should be UNDISCOUNTED
			Double sum = 0.0;
			for (Double reward : evaluationRewards) {
				sum += reward;
			}

			Double avg = sum/evaluationRewards.length;
			avgRewards.add(avg);	
			printTestData(avgRewards);
			System.out.println("Entering learning mode, unfreezing Q function");
		}

		//quit if we've just played the last episode
		if (numEpisodesPlayed >= numEpisodes) {
			System.out.println("Session complete");			
			System.exit(0);
		}

		saveWeights(weights);
	}

	/**
	 * removes the units that were killed on the last turn from myFootmen and enemyFootmen
	 * 
	 * @param stateView
	 * @param historyView
	 */
	private void cleanupDeadUnits(State.StateView stateView, History.HistoryView historyView) {
		if (stateView.getTurnNumber() > 0) {
			//"bring out your dead, bring out your dead"
			for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() - 1)) {
				Integer deadUnitID = deathLog.getDeadUnitID();
				//System.out.println("Player: " + deathLog.getController() + " unit: " + deadUnitID);

				//remove the dead unit from whichever list its in
				if (myFootmen.contains(deadUnitID)) {
					myFootmen.remove(deadUnitID);
				}
				else if (enemyFootmen.contains(deadUnitID)) {
					enemyFootmen.remove(deadUnitID);
				}
				else {
					System.err.println("ERROR: dead unit not identified");
				}
			}
		}
	}

	/**
	 * Add the rewards from the last turn to each unit's totals
	 * 
	 * @param stateView
	 * @param historyView
	 */
	private void updateFootmenRewards(State.StateView stateView, History.HistoryView historyView) {
		for (Integer id : myFootmen) {
			double currentReward = calculateReward(stateView, historyView, id);
			double cumulativeReward = rewardMap.get(id);
			rewardMap.put(id, cumulativeReward + currentReward);
		}
	}

	/**
	 * Calculate the updated weights for this agent. 
	 * @param oldWeights Weights prior to update
	 * @param oldFeatures Features from (s,a)
	 * @param totalReward Cumulative discounted reward for this footman.
	 * @param stateView Current state of the game.
	 * @param historyView History of the game up until this point
	 * @param footmanId The footman we are updating the weights for
	 * @return The updated weight vector.
	 */
	public Double[] updateWeights(Double[] oldWeights, double[] oldFeatures, double totalReward,
			State.StateView stateView, History.HistoryView historyView, int footmanId) {

		//TODO not sure if this is doing exactly what we're supposed to
		//see lec 18 slide 58, and book 846

		Double[] newWeights = new Double[oldWeights.length];
		for (int i = 0; i < oldWeights.length; i++) {

			//compute the dot product to get the final qVal
			double dotProduct = 0;
			for (int j = 0; j < oldFeatures.length; j++) {
				dotProduct += oldFeatures[j] * oldWeights[j];
			}
			double currentQVal = dotProduct;

			double maxQVal = Double.NEGATIVE_INFINITY;
			for (Integer enemy : enemyFootmen) {
				double qVal = calcQValue(stateView, historyView, footmanId, enemy);
				if (qVal > maxQVal) {
					maxQVal = qVal;
				}
			}

			double targetQVal = totalReward + gamma * maxQVal;
			double dldw = -1 * (targetQVal - currentQVal) * oldFeatures[i];
			newWeights[i] = oldWeights[i] - learningRate * (dldw);	

			//TODO ask george if his diffs are fucking enormous
			//System.out.println("diff " + i + " is: " + (targetQVal - currentQVal));
		}

		//		System.out.println("\nOld weights: ");
		//		for (int i = 0; i < oldWeights.length; i++) {
		//			System.out.print(oldWeights[i] + ", ");
		//		}
		//		System.out.println();
		//		System.out.println("New weights: ");
		//		for (int i = 0; i < newWeights.length; i++) {
		//			System.out.print(newWeights[i] + ", ");
		//		}

		return newWeights;
	}

	/**
	 * Given a footman and the current state and history of the game select the enemy that this unit should
	 * attack. This is where you would do the epsilon-greedy action selection.
	 *
	 * @param stateView Current state of the game
	 * @param historyView The entire history of this episode
	 * @param attackerId The footman that will be attacking
	 * @return The enemy footman ID this unit should attack
	 */
	public int selectAction(State.StateView stateView, History.HistoryView historyView, int attackerId) {

		if (enemyFootmen.size() > 0) {
			//if not frozen and the rand number less than epsilon choose random action
			if (!freeze && random.nextDouble() < epsilon) {

				//choose a random index of enemyFootmen
				int index = (int) random.nextDouble() * enemyFootmen.size();
				//return the id
				return enemyFootmen.get(index);
			}
			//otherwise choose action that maxmizes Q value
			else {
				int bestEnemyToAttack = enemyFootmen.get(0);
				double bestQVal = calcQValue(stateView, historyView, attackerId, bestEnemyToAttack);
				for (int i = 1; i < enemyFootmen.size(); i++) {
					int enemyID = enemyFootmen.get(i);
					double qVal = calcQValue(stateView, historyView, attackerId, enemyID);
					if (qVal > bestQVal) {
						bestQVal = qVal;
						bestEnemyToAttack = enemyID;
					}
				}
				return bestEnemyToAttack;
			}
		}
		//No enemies left to attack
		return -1;
	}

	/**
	 * Given the current state and the footman in question calculate the reward received on the last turn.
	 * This is where you will check for things like Did this footman take or give damage? Did this footman 
	 * die
	 * or kill its enemy. Did this footman start an action on the last turn? See the assignment description
	 * for the full list of rewards.
	 *
	 * Remember that you will need to discount this reward based on the timestep it is received on. See
	 * the assignment description for more details.
	 *
	 * As part of the reward you will need to calculate if any of the units have taken damage. You can use
	 * the history view to get a list of damages dealt in the previous turn. Use something like the following.
	 *
	 * for(DamageLog damageLogs : historyView.getDamageLogs(lastTurnNumber)) {
	 *     System.out.println("Defending player: " + damageLog.getDefenderController() + 
	 *     " defending unit: " + \
	 *     damageLog.getDefenderID() + " attacking player: " + damageLog.getAttackerController() + \
	 *     "attacking unit: " + damageLog.getAttackerID());
	 * }
	 *
	 * You will do something similar for the deaths. See the middle step documentation for a snippet
	 * showing how to use the deathLogs.
	 *
	 * To see if a command was issued you can check the commands issued log.
	 *
	 * Map<Integer, Action> commandsIssued = historyView.getCommandsIssued(playernum, lastTurnNumber);
	 * for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
	 *     System.out.println("Unit " + commandEntry.getKey() + " was command to " + 
	 *     commandEntry.getValue().toString);
	 * }
	 *
	 * @param stateView The current state of the game.
	 * @param historyView History of the episode up until this turn.
	 * @param footmanId The footman ID you are looking for the reward from.
	 * @return The current reward
	 */
	//discounting is not necessary because we update the reward at every step, 
	public double calculateReward(State.StateView stateView, History.HistoryView historyView, int footmanId) {


		//TODO figure out how to implement discounting of rewards based on timestep, see lec 17 slide 16


		//no reward on the first turn because nothing has happened yet
		if (stateView.getTurnNumber() == 0) {
			return 0;
		}

		double reward = 0;

		//Here we only add -.1 to the reward if a new action is given to this footman
		//So if a new command is issued, but the target is the same, don't add -.1 because its not
		//really a new move
		Map<Integer, Action> commandsIssued =
				historyView.getCommandsIssued(playernum, stateView.getTurnNumber() - 1);
		for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {

			if (commandEntry.getKey() == footmanId) {

				if (previousActionMap != null) {
					//TODO cast works because no other actions can exist, right?
					TargetedAction oldAction = (TargetedAction) previousActionMap.get(footmanId);
					TargetedAction newAction = (TargetedAction) commandEntry.getValue();

					//new action started if the targets are different
					if (oldAction.getTargetId() != newAction.getTargetId()) {
						reward -= 0.1;
					}
				}
				//no previous action -> new action started
				else {
					reward -= 0.1;
				}

			}
		}

		for(DamageLog damageLog : historyView.getDamageLogs(stateView.getTurnNumber() - 1)) {

			int damageAmount = damageLog.getDamage();	
			Integer defenderID = damageLog.getDefenderID();
			Integer attackerID = damageLog.getAttackerID();

			if (defenderID == footmanId) {
				reward -= damageAmount;
			}

			if (attackerID == footmanId) {
				reward += damageAmount;
			}
		}

		for(DeathLog deathLog : historyView.getDeathLogs(stateView.getTurnNumber() - 1)) {

			int playerID = deathLog.getController();
			Integer deadUnitID = deathLog.getDeadUnitID();

			//check if it was an enemy that died
			if (ENEMY_PLAYERNUM == playerID) {

				Map<Integer, ActionResult> actionResults =
						historyView.getCommandFeedback(playernum, stateView.getTurnNumber() - 1);
				for (ActionResult result : actionResults.values()) {

					//TODO cast works because no other actions can exist, right?
					TargetedAction compoundAttack = (TargetedAction) result.getAction();
					//claim the reward if no one else has claimed it, and this footman was the killer
					if (!enemyBlackList.contains(deadUnitID) &&
							compoundAttack.getUnitId() == footmanId &&
							compoundAttack.getTargetId() == deadUnitID) {
						enemyBlackList.add(deadUnitID);
						reward += 100;
						//System.out.println("kill reward claimed");
						break;
					}
				}
			}
			else if (deadUnitID == footmanId) {
				reward -= 100;
			}
		}

		//TODO delete
//		if (stateView.getUnit(footmanId) != null) {
//			System.out.println("footman at " + stateView.getUnit(footmanId).getXPosition() + ", " + 
//					stateView.getUnit(footmanId).getYPosition() + " just got reward: " + reward);
//		}
//		else {
//			System.out.println("footman " + footmanId + " just got reward: " + reward);
//		}

		
		return reward;
	}

	/**
	 * Calculate the Q-Value for a given state action pair. The state in this scenario is the current
	 * state view and the history of this episode. The action is the attacker and the enemy pair for the
	 * SEPIA attack action.
	 *
	 * This returns the Q-value according to your feature approximation. This is where you will calculate
	 * your features and multiply them by your current weights to get the approximate Q-value.
	 *
	 * @param stateView Current SEPIA state
	 * @param historyView Episode history up to this point in the game
	 * @param attackerId Your footman. The one doing the attacking.
	 * @param defenderId An enemy footman that your footman would be attacking
	 * @return The approximate Q-value
	 */
	public double calcQValue(State.StateView stateView,
			History.HistoryView historyView,
			int attackerId,
			int defenderId) {

		double[] features = calculateFeatureVector(stateView, historyView, attackerId, defenderId);

		if (weights.length != features.length) {
			System.err.println("ERROR: weights and features not same length");
			System.exit(0);
		}

		//compute the dot product to get the final qVal
		double dotProduct = 0;
		for (int i = 0; i < features.length; i++) {
			dotProduct += features[i] * weights[i];
		}

		return dotProduct;
	}

	/**
	 * Given a state and action calculate your features here. Please include a comment explaining what 
	 * features
	 * you chose and why you chose them.
	 *
	 * All of your feature functions should evaluate to a double. Collect all of these into an array. You will
	 * take a dot product of this array with the weights array to get a Q-value for a given state action.
	 *
	 * It is a good idea to make the first value in your array a constant. This just helps remove any offset
	 * from 0 in the Q-function. The other features are up to you. Many are suggested in the assignment
	 * description.
	 *
	 * @param stateView Current state of the SEPIA game
	 * @param historyView History of the game up until this turn
	 * @param attackerId Your footman. The one doing the attacking.
	 * @param defenderId An enemy footman. The one you are considering attacking.
	 * @return The array of feature function outputs.
	 */
	public double[] calculateFeatureVector(State.StateView stateView,
			History.HistoryView historyView,
			int attackerId,
			int defenderId) {

		//our friendly unit
		UnitView attacker = stateView.getUnit(attackerId);
		//enemy unit that we're attacking
		UnitView defender = stateView.getUnit(defenderId);

		double[] featuresArray = new double[NUM_FEATURES];

		//f0 is a constant
		featuresArray[0] = 1;

		int targetDistance = DistanceMetrics.chebyshevDistance(attacker.getXPosition(),
				attacker.getYPosition(), defender.getXPosition(), defender.getYPosition());

		//indicates how close this footman is 
		int closenessRank = 0;
		for (int enemyID : enemyFootmen) {
			UnitView enemy = stateView.getUnit(enemyID);
			int distance = DistanceMetrics.chebyshevDistance(attacker.getXPosition(),
					attacker.getYPosition(), enemy.getXPosition(), enemy.getYPosition());
			if (distance < targetDistance) {
				closenessRank++;
			}
		}
		//f1 is the number of enemies left minus the rank of how close this enemy is to the footman
		//compared to the others in terms of chebyshev distance.
		//Chose to use the feature because it causes the footmen to favor attacking closer enemies
		featuresArray[1] = enemyFootmen.size() - closenessRank;


		//f2 is the health ratio of the friendly footman to its target
		//This features causes the footmen to favor attacking enemies weaker that themselves
		featuresArray[2] = attacker.getHP()/defender.getHP();
		//		featuresArray[2] = attacker.getHP() - defender.getHP();
		//		featuresArray[2] = defender.getHP() - attacker.getHP();


		//the number of friendly units also attacking your target
		int numFriendliesAlsoAttacking = 0;

		if (stateView.getTurnNumber() != 0) {

			Map<Integer, Action> commandsIssued =
					historyView.getCommandsIssued(playernum, stateView.getTurnNumber() - 1);
			for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {
				if (commandEntry.getKey() != attackerId) {
					TargetedAction action = (TargetedAction) commandEntry.getValue();
					if (action != null && action.getTargetId() == defenderId) {
						numFriendliesAlsoAttacking++;
					}
				}
			}
		}

		//f3 is the number of of friendly units also attacking the target divided by the total number of
		//friendlies 
		featuresArray[3] = numFriendliesAlsoAttacking/myFootmen.size();


		//determines if target was attacking you on the last turn
		double enemyIsAttackingFriendly = -1;
		if (stateView.getTurnNumber() != 0) {

			Map<Integer, Action> commandsIssued =
					historyView.getCommandsIssued(ENEMY_PLAYERNUM, stateView.getTurnNumber() - 1);
			for (Map.Entry<Integer, Action> commandEntry : commandsIssued.entrySet()) {

				if (commandEntry.getKey() == defenderId) {

					TargetedAction action = (TargetedAction) commandEntry.getValue();
					if (action != null && action.getTargetId() == attackerId) {
						//System.out.println("atttttaAACKCKCAKAKKK");
						enemyIsAttackingFriendly = 1;
					}
				}
			}
		}

		//Chose to use the feature because it encourages footmen to defend themselves
		featuresArray[4] = enemyIsAttackingFriendly;

		//consider avoiding those with higher health than you
		//consider attacking closest
		//consider attacking the those with lowest relative health
		//consider attacking one that is attacking you
		//consider attacking one that others are already attacking
		//consider attacking those attacking your homies
		//consider continuing to attack the one you attacked last time





		return featuresArray;
	}

	/**
	 * DO NOT CHANGE THIS!
	 *
	 * Prints the learning rate data described in the assignment. Do not modify this method.
	 *
	 * @param averageRewards List of cumulative average rewards from test episodes.
	 */
	public void printTestData (List<Double> averageRewards) {
		System.out.println("");
		System.out.println("Games Played      Average Cumulative Reward");
		System.out.println("-------------     -------------------------");
		for (int i = 0; i < averageRewards.size(); i++) {
			String gamesPlayed = Integer.toString(10*i);
			String averageReward = String.format("%.2f", averageRewards.get(i));

			int numSpaces = "-------------     ".length() - gamesPlayed.length();
			StringBuffer spaceBuffer = new StringBuffer(numSpaces);
			for (int j = 0; j < numSpaces; j++) {
				spaceBuffer.append(" ");
			}
			System.out.println(gamesPlayed + spaceBuffer.toString() + averageReward);
		}
		System.out.println("");
	}

	/**
	 * DO NOT CHANGE THIS!
	 *
	 * This function will take your set of weights and save them to a file. Overwriting whatever file is
	 * currently there. You will use this when training your agents. You will include the output of this
	 * function
	 * from your trained agent with your submission.
	 *
	 * Look in the agent_weights folder for the output.
	 *
	 * @param weights Array of weights
	 */
	public void saveWeights(Double[] weights) {
		File path = new File("agent_weights/weights.txt");
		// create the directories if they do not already exist
		path.getAbsoluteFile().getParentFile().mkdirs();

		try {
			// open a new file writer. Set append to false
			BufferedWriter writer = new BufferedWriter(new FileWriter(path, false));

			for (double weight : weights) {
				writer.write(String.format("%f\n", weight));
			}
			writer.flush();
			writer.close();
		} catch(IOException ex) {
			System.err.println("Failed to write weights to file. Reason: " + ex.getMessage());
		}
	}

	/**
	 * DO NOT CHANGE THIS!
	 *
	 * This function will load the weights stored at agent_weights/weights.txt. The contents of this file
	 * can be created using the saveWeights function. You will use this function if the load weights argument
	 * of the agent is set to 1.
	 *
	 * @return The array of weights
	 */
	public Double[] loadWeights() {
		File path = new File("agent_weights/weights.txt");
		if (!path.exists()) {
			System.err.println("Failed to load weights. File does not exist");
			return null;
		}

		try {
			BufferedReader reader = new BufferedReader(new FileReader(path));
			String line;
			List<Double> weights = new LinkedList<>();
			while((line = reader.readLine()) != null) {
				weights.add(Double.parseDouble(line));
			}
			reader.close();

			return weights.toArray(new Double[weights.size()]);
		} catch(IOException ex) {
			System.err.println("Failed to load weights from file. Reason: " + ex.getMessage());
		}
		return null;
	}

	@Override
	public void savePlayerData(OutputStream outputStream) {}
	@Override
	public void loadPlayerData(InputStream inputStream) {}
}
