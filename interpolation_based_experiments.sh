#python run.py --type self_paced --learner ppois_feasible --env point_mass_2d --seed $1
#python run.py --type wasserstein --learner ppo --env point_mass_2d --seed $1
#
#python run.py --type wasserstein --learner sac --env maze --seed $1
#python run.py --type self_paced --learner sac --env maze --seed $1

python run.py --type wasserstein --learner ppo --env minigrid --seed 1
python run.py --type wasserstein --learner ppo --env minigrid --seed 2
python run.py --type wasserstein --learner ppo --env minigrid --seed 3
python run.py --type wasserstein --learner ppo --env minigrid --seed 4
python run.py --type wasserstein --learner ppo --env minigrid --seed 5

#python run.py --type self_paced --learner ppo --env minigrid --seed 1
#python run.py --type self_paced --learner ppo --env minigrid --seed 2
#python run.py --type self_paced --learner ppo --env minigrid --seed 3
#python run.py --type self_paced --learner ppo --env minigrid --seed 4
#python run.py --type self_paced --learner ppo --env minigrid --seed 5
