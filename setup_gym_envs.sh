#!/usr/bin/env bash


git clone https://github.com/openai/gym.git
pip install 'gym/[all]'
rm -rf gym

git clone https://github.com/maximecb/gym-minigrid.git
pip install gym-minigrid
rm -rf gym-minigrid