# Help the Human

## Running the Repo
Use Python 3.12 (Dataclasses).

Create an environment from the requirements.txt using
```{python}
pip install -r requirements.txt
```
To update the requirements.txt after you added an import, use
```{python}
pip freeze > requirements.txt
```

Debug without logging:
```{bash}
python run_ppo.py --log_to_wandb=False
```

## Citation for PettingZoo

https://pettingzoo.farama.org/tutorials/custom_environment/1-project-structure/

To cite PettingZoo in publication, please use

```
@article{terry2021pettingzoo,
  title={Pettingzoo: Gym for multi-agent reinforcement learning},
  author={Terry, J and Black, Benjamin and Grammel, Nathaniel and Jayakumar, Mario and Hari, Ananth and Sullivan, Ryan and Santos, Luis S and Dieffendahl, Clemens and Horsch, Caroline and Perez-Vicente, Rodrigo and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={15032--15043},
  year={2021}
}
```



Warning: Gym version v0.24.1 has a number of critical issues with `gym.make` such that environment observation and action spaces are incorrectly evaluated, raising incorrect errors and warning . It is recommend to downgrading to v0.23.1 or upgrading to v0.25.1