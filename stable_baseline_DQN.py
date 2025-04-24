from stable_baselines3 import DQN
from configuration_stable_baseline import config_dict, make_env
from utilities import test_sb3, plot_test_stats

training_steps = 50000
eval_episodes = 50
train = False
test = True
model_name = "dqn_roundabout2"

if train:
    env = make_env()
    env.reset()

    model = DQN(
        "MlpPolicy",    # Politique
        env,    # Environnement
        policy_kwargs=dict(net_arch=[256, 256]),    # Architecture du réseau
        learning_rate=5e-4,  # Taux d'apprentissage
        buffer_size=20000,  # Taille du replay buffer
        learning_starts=500,    # Steps avant le début de l’apprentissage
        batch_size=128,  # Taille du batch tiré du buffer
        gamma=0.9,  # Discount factor (importance du futur)
        train_freq=1,   # Fréquence d'entraînement (chaque step)
        gradient_steps=1,   # Nb de mises à jour par step d'entraînement
        target_update_interval=50,  # Fréquence de mise à jour du réseau cible
        # % du training passé en exploration (epsilon décroît)
        exploration_fraction=0.2,
        verbose=1,  # Affiche les logs d’entraînement
        tensorboard_log="./dqn_roundabout_tensorboard/"
    )

    model.learn(total_timesteps=training_steps)

    # Visualisation tensorboard
    # tensorboard --logdir ./dqn_roundabout_tensorboard/

    model.save(model_name)

if test:
    model = DQN.load(model_name)

    test_env = make_env()
    obs, _ = test_env.reset()

    ### Test the agent ###
    test_env = test_sb3(model, test_env, n_episodes=50, display=True)

    ### Plot the testing results ###
    plot_test_stats(test_env, model, plot=True)


# RESULTATS :

# 1 : train 50000 episodes
# buffer_size = 20000
# exploration_fraction = 0.2
# batch_size = 128
# gamma = 0.9
# learning_start = 500
# # Mean return over n=50 episodes : 7.626363636363635 +/- 0.5125181552922121
# Mean episode length over n=50 episodes : 8.78

# 2 : train 50000 episodes
# buffer_size = 15000
# exploration_fraction = 0.7
# batch_size = 32
# gamma = 0.8
# learning_start = 200
# Mean return over n=50 episodes : 6.840909090909091 +/- 0.5556341771732324
# Mean episode length over n=50 episodes : 7.98

# 3 : Same as 1 but different reward
# Mean return over n = 50 episodes: 7.488910891089109 + /- 0.5208157875866947
# Mean episode length over n = 50 episodes: 8.42
