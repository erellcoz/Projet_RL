from stable_baselines3 import PPO
from configuration_stable_baseline import config_dict, make_env
from utilities import test_sb3, plot_test_stats

training_steps = 20000
eval_episodes = 50
train = True
test = True
model_name = "ppo_roundabout2"

if train:
    env = make_env()
    env.reset()

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.8,
        tensorboard_log="./ppo_roundabout_tensorboard/",
        verbose=1,
    )

    model.learn(total_timesteps=training_steps)

    # Visualisation tensorboard
    # tensorboard --logdir ./ppo_roundabout_tensorboard/

    model.save(model_name)

if test:
    model = PPO.load(model_name)

    test_env = make_env()
    obs, _ = test_env.reset()

    ### Test the agent ###
    test_env = test_sb3(model, test_env, n_episodes=50, display=True)

    ### Plot the testing results ###
    plot_test_stats(test_env, model, plot=True)

    # RESULTATS :

    # 1. train 50000 episodes
    # learning_rate=8e-4,
    # n_steps=1024,
    # batch_size=256,
    # n_epochs=10,
    # gamma=0.90,
    # gae_lambda=0.90,
    # clip_range=0.25,
    # ent_coef=0.001,
    # vf_coef=0.5,
    # max_grad_norm=0.5,
    # tensorboard_log="./ppo_roundabout_tensorboard/",
    # target_kl=0.03,
    # verbose=1,
    # Mean return over n=50 episodes : 6.894545454545454 +/- 0.551952866594738
    # Mean episode length over n=50 episodes : 8.02

    # 2. train 20000 episodes
    # learning_rate=5e-4,
    # n_steps=512,
    # batch_size=64,
    # n_epochs=10,
    # gamma=0.8,
    # tensorboard_log="./ppo_roundabout_tensorboard/",
    # verbose=1,
    # Mean return over n=50 episodes : 6.501818181818181 +/- 0.5240322998057174
    # Mean episode length over n=50 episodes : 7.8
