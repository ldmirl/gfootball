# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from . import *


def build_scenario(builder):
    # 100step = 180s in game
    builder.config().game_duration = 210
    builder.config().deterministic = False
    builder.config().offsides = False
    builder.config().end_episode_on_score = True
    builder.config().end_episode_on_out_of_play = True
    builder.config().end_episode_on_possession_change = True
    builder.config().right_team_difficulty = 0.95

    builder.SetBallPosition(0.0, 0.0)


    builder.SetTeam(Team.e_Left)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK,lazy=True,controllable=False)  # if lazy is True the goalkeeper will not move
    # builder.AddPlayer(-0.45, 0.0, e_PlayerRole_CB)
    builder.AddPlayer(-0.35, -0.1, e_PlayerRole_RB)
    builder.AddPlayer(-0.35, 0.1, e_PlayerRole_LB)
    
    builder.AddPlayer(-0.30, -0.3, e_PlayerRole_CM)
    builder.AddPlayer(-0.30, 0.3, e_PlayerRole_CM)

#    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK,lazy=True,controllable=False)
#    builder.AddPlayer(0.000000,  0.020000, e_PlayerRole_RM)
#    builder.AddPlayer(0.000000, -0.020000, e_PlayerRole_CF)
#    builder.AddPlayer(-0.1, -0.1, e_PlayerRole_LB)
#    builder.AddPlayer(-0.1,  0.1, e_PlayerRole_CB)

    builder.SetTeam(Team.e_Right)
    builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
    builder.AddPlayer(0.0, -0.02, e_PlayerRole_CF)
    builder.AddPlayer(0.0, 0.02, e_PlayerRole_CF)
    builder.AddPlayer(-0.2, 0.0, e_PlayerRole_CM)
    
    builder.AddPlayer(-0.1, 0.1, e_PlayerRole_CF)
    builder.AddPlayer(-0.1, -0.1, e_PlayerRole_CF)
#    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)
#    builder.AddPlayer(-0.04,  0.040000, e_PlayerRole_RM)
#    builder.AddPlayer(-0.04, -0.040000, e_PlayerRole_CF)
#    builder.AddPlayer(-0.1, -0.1, e_PlayerRole_LB)
#    builder.AddPlayer(-0.1,  0.1, e_PlayerRole_CB)

